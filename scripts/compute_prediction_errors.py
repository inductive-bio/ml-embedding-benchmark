"""Fit one regression head to one embedding, across all OpenADMET endpoints,
and save the per-compound test-set prediction error for each endpoint.

Deliberately independent of this repo's Hydra-based embed.py/score.py: dataset
configs here are flat (no `defaults:` composition), so a plain YAML read is
enough and avoids Hydra's cwd-switching, which would just complicate reuse
from Modal.

Per (endpoint, embedding, method), fits on the endpoint's existing train+valid
split (hyperparameters chosen via CV_SPLITS-fold GridSearchCV, scored with
that endpoint's own `metric` from config/dataset/<name>.yaml -- spearmancorr
for every OpenADMET endpoint today), predicts on the existing test split, and
stores {molecule_id, y_true, y_pred, error} for every test compound -- these
three arrays are the sufficient statistic this is collecting; everything else
(MAE, RMSE, paired per-compound comparisons across embeddings) is derivable
from them later.

Usage:
    python scripts/compute_prediction_errors.py --embedding ECFP --method rf
    python scripts/compute_prediction_errors.py --embedding coati --method knn --override
    python scripts/compute_prediction_errors.py --embedding coati --method tabicl
    python scripts/compute_prediction_errors.py --embedding coati --method tabpfn
    python scripts/compute_prediction_errors.py --embedding coati --method tabpfn_v2
    python scripts/compute_prediction_errors.py --list-embeddings
"""

import argparse
import glob
import logging as log
import os
import sys

import joblib
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.types import EmbeddedDataset
from src.eval.supervised.regression import fit_regressor

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")


def discover_dataset_configs(config_dir: str, pattern: str) -> list:
    configs = []
    for path in sorted(glob.glob(os.path.join(config_dir, f"{pattern}.yaml"))):
        with open(path) as f:
            configs.append(yaml.safe_load(f))
    return configs


def discover_embeddings(embedded_dir: str) -> list:
    """Union of model stems (filename without extension) across every endpoint
    directory -- not every endpoint necessarily has every model."""
    stems = set()
    for endpoint_dir in glob.glob(os.path.join(embedded_dir, "*")):
        for path in glob.glob(os.path.join(endpoint_dir, "*")):
            stem, ext = os.path.splitext(os.path.basename(path))
            if ext in (".joblib", ".json"):
                stems.add(stem)
    return sorted(stems)


def load_embedded_dataset(embedded_dir: str, dataset_name: str, embedding: str):
    joblib_path = os.path.join(embedded_dir, dataset_name, f"{embedding}.joblib")
    legacy_path = os.path.join(embedded_dir, dataset_name, f"{embedding}.json")
    try:
        if os.path.exists(joblib_path):
            return joblib.load(joblib_path)
        if os.path.exists(legacy_path):
            return EmbeddedDataset.deserialize_legacy(legacy_path)
    except Exception as e:
        log.error(f"Failed to load embedding '{embedding}' for '{dataset_name}': {e}")
        return None
    log.warning(f"No embedding found for '{embedding}' on '{dataset_name}' ({joblib_path})")
    return None


def split_indices(splits: dict, keys: list) -> list:
    indices = []
    for key in keys:
        idx = splits.get(key, [])
        indices.extend(idx.tolist() if isinstance(idx, np.ndarray) else list(idx))
    return indices


def compute_errors(
    dataset_config: dict,
    embedding: str,
    method: str,
    embedded_dir: str,
    prepared_dir: str,
    n_jobs: int,
) -> dict:
    dataset_name = dataset_config["name"]
    embedded = load_embedded_dataset(embedded_dir, dataset_name, embedding)
    if embedded is None:
        return None

    prepared_path = os.path.join(prepared_dir, f"{dataset_name}.joblib")
    prepared = joblib.load(prepared_path)
    # Read the target fresh from data/prepared rather than trusting
    # embedded.y_np -- the y baked into data/embedded/*/*.joblib was snapshotted
    # at embed time (embed.py's `y=dataset.labels`), so it goes stale if the
    # target transform in data/prepared is fixed/changed later without
    # re-running embeddings (X only depends on SMILES, so there's no need to).
    value_col = dataset_config["source"]["target_column"]

    X = embedded.X
    if len(X.shape) == 1:
        log.warning(f"'{embedding}'/'{dataset_name}': 1D X, reshaping")
        X = X.reshape(len(embedded.y), -1)

    if X.shape[0] != len(embedded.y):
        log.error(
            f"'{embedding}'/'{dataset_name}': X row count ({X.shape[0]}) doesn't "
            f"match embedded.y row count ({len(embedded.y)}) -- skipping"
        )
        return None

    # remove_failed_embeddings() (src/common/types.py) drops rows at embed
    # time but its .iloc-based filtering preserves each surviving row's
    # original position as embedded.y's DataFrame index -- use that to pick
    # out the matching molecule_id/target from the full prepared dataset, in
    # the same order as X's rows. When nothing was dropped this is just
    # [0, 1, ..., N-1], so it's correct in the common case too.
    surviving_positions = embedded.y.index.to_numpy()
    molecule_ids = prepared.data["molecule_id"].to_numpy()[surviving_positions]
    y = prepared.data[value_col].to_numpy(dtype=float)[surviving_positions]

    train_idx = split_indices(embedded.splits, ["train", "valid"])
    test_idx = split_indices(embedded.splits, ["test"])

    # train_idx/test_idx come from embedded.splits (surviving-row index space);
    # X is already in that same space. If remove_failed_embeddings() ever
    # stopped remapping splits correctly, this would silently mix coordinate
    # spaces instead of raising -- guard it explicitly.
    all_idx = sorted(train_idx + test_idx)
    assert all_idx == list(range(len(X))), (
        f"'{embedding}'/'{dataset_name}': splits do not partition surviving rows: "
        f"{len(all_idx)} split indices over {len(X)} rows"
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_molecule_ids = molecule_ids[test_idx]

    log.info(
        f"'{embedding}'/'{dataset_name}'/'{method}': train={X_train.shape}, test={X_test.shape}"
    )

    fit_result = fit_regressor(
        X_train, y_train, method=method, metric_name=dataset_config["metric"], n_jobs=n_jobs
    )
    y_pred = fit_result["model_obj"].predict(X_test)

    return {
        "dataset": dataset_name,
        "embedder": embedding,
        "method": method,
        "metric_name": dataset_config["metric"],
        "molecule_id": test_molecule_ids,
        "y_true": np.asarray(y_test, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "error": np.asarray(y_pred, dtype=float) - np.asarray(y_test, dtype=float),
        "best_params": fit_result["best_params"],
        "cv_score": fit_result["cv_score"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding", help="Embedding model stem, e.g. 'ECFP', 'coati' (matches data/embedded/<endpoint>/<embedding>.joblib)")
    parser.add_argument("--method", choices=["rf", "ridge", "knn", "tabicl", "tabpfn", "tabpfn_v2"])
    parser.add_argument("--datasets", default="reg_openadmet_*", help="Glob (without .yaml) over config/dataset/ to select endpoints")
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--embedded-dir", default="data/embedded")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--output-dir", default="data/errors")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--list-embeddings", action="store_true", help="Print discovered embedding stems and exit")
    args = parser.parse_args()

    if args.list_embeddings:
        for stem in discover_embeddings(args.embedded_dir):
            print(stem)
        return

    if not args.embedding or not args.method:
        parser.error("--embedding and --method are required (unless --list-embeddings)")

    dataset_configs = discover_dataset_configs(args.config_dir, args.datasets)
    log.info(f"Found {len(dataset_configs)} dataset config(s) matching '{args.datasets}'")
    if not dataset_configs:
        raise SystemExit(f"No dataset configs matched '--datasets {args.datasets}' under {args.config_dir}")

    produced, already_exists, failed = 0, 0, 0
    for dataset_config in dataset_configs:
        dataset_name = dataset_config["name"]
        output_dir = os.path.join(args.output_dir, dataset_name)
        output_path = os.path.join(output_dir, f"{args.embedding}__{args.method}.joblib")

        if os.path.exists(output_path) and not args.override:
            log.info(f"Skipping (already exists): {output_path}")
            already_exists += 1
            continue

        result = compute_errors(
            dataset_config, args.embedding, args.method,
            args.embedded_dir, args.prepared_dir, args.n_jobs,
        )
        if result is None:
            failed += 1
            continue

        os.makedirs(output_dir, exist_ok=True)
        # joblib.dump() straight to output_path isn't atomic -- a preempted or
        # timed-out run could leave a truncated file that a resumed run (the
        # os.path.exists() check above) then skips as complete. Write to a
        # temp path and rename, since os.replace() is atomic on the same
        # filesystem.
        tmp_path = output_path + ".tmp"
        joblib.dump(result, tmp_path)
        os.replace(tmp_path, output_path)
        produced += 1
        log.info(f"Saved {output_path} ({len(result['molecule_id'])} test compounds)")

    log.info(f"produced={produced} already_exists={already_exists} failed={failed} (of {len(dataset_configs)})")
    if produced == 0 and failed > 0:
        # Distinct from produced==0-with-failed==0, which just means every
        # endpoint's output already existed -- a legitimate idempotent resume,
        # not the silent-failure case this is meant to catch.
        raise SystemExit(f"{failed} endpoint(s) failed to produce output and none succeeded")


if __name__ == "__main__":
    main()
