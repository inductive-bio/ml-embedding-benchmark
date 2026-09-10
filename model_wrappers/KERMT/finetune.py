"""Fine-tune the KERMT pretrained checkpoint end-to-end (encoder + FFN head
trained jointly -- KERMT's own default `--fine_tune_coff 1.0`, not a frozen-
encoder linear probe) on one OpenADMET endpoint, and score it on the same
held-out test split the frozen-embedding path uses.

Shells out to KERMT's own `main.py finetune` CLI (and reads its
`test_result.csv` output) rather than reimplementing training in-process --
KERMT's CLI already handles the encoder/FFN joint optimizer, checkpointing,
and MTL/task-wise head machinery end-to-end; see
tests/integration/test_pretrain_finetune.py for the flags this mirrors.

Must run inside an environment with KERMT's own dependencies installed
(cuik-molmaker is conda-only -- see model_wrappers/KERMT/init.sh), e.g. the
image scripts/modal_finetune_kermt.py builds.

Usage:
    python model_wrappers/KERMT/finetune.py --dataset reg_openadmet_pxr_pEC50
"""

import argparse
import logging as log
import os
import subprocess
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

_MODEL_NAME = "kermt"
_KERMT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHECKPOINT = os.path.join(_KERMT_DIR, "checkpoints", "kermt_contrastive_v2.0.pt")

# KERMT's --metric choices don't include our dataset configs' "spearmancorr"
# spelling -- map the ones actually used by config/dataset/reg_openadmet_*.yaml.
# This only picks which metric KERMT logs/uses for its own best-epoch
# selection; the saved result always reports both mae and spearmancorr
# (see save_finetune_result) regardless of this choice.
_METRIC_MAP = {"spearmancorr": "spearmanr", "mae": "mae", "rmse": "rmse", "r2": "r2"}

# OADMET-0003727 (reg_openadmet_pxr_pEC50, train split) is auranofin, a
# gold-phosphine complex -- RDKit's Gasteiger partial-charge descriptors
# (part of the rdkit_2d_normalized feature set fed to the FFN head) are
# undefined for its Au atom and come out NaN even from plain descriptastorus.
# With --no_features_scaling that NaN feeds straight into the network; one
# NaN gradient update poisons every subsequent epoch's loss (confirmed:
# excluding this molecule alone fixed an all-epoch NaN loss run).
_EXCLUDE_MOLECULE_IDS = {"OADMET-0003727"}


def _write_kermt_csv(df: pd.DataFrame, target_column: str, path: str) -> None:
    # KERMT's MoleculeDatapoint parses CSVs positionally (column 0 = smiles,
    # remaining columns = targets) -- molecule_id must not be included here.
    df[["smiles", target_column]].to_csv(path, index=False)


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    checkpoint_path: str,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    seed: int,
    val_fraction: float,
    run_tag: str = None,
) -> str:
    target_column = dataset_config["source"]["target_column"]
    dataset_type = "regression" if dataset_config["task"] == "regression" else "classification"
    metric = _METRIC_MAP.get(dataset_config["metric"], "mae")

    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        excluded = split_df["molecule_id"].isin(_EXCLUDE_MOLECULE_IDS)
        if excluded.any():
            log.warning(f"Excluding {excluded.sum()} molecule(s) from {split_name} split: {split_df.loc[excluded, 'molecule_id'].tolist()}")
    train_df = train_df[~train_df["molecule_id"].isin(_EXCLUDE_MOLECULE_IDS)].reset_index(drop=True)
    val_df = val_df[~val_df["molecule_id"].isin(_EXCLUDE_MOLECULE_IDS)].reset_index(drop=True)
    test_df = test_df[~test_df["molecule_id"].isin(_EXCLUDE_MOLECULE_IDS)].reset_index(drop=True)

    # Two different molecule_ids can canonicalize to the same SMILES (a
    # replicate compound, e.g. reg_openadmet_polaris_admet_HLM's test split
    # has 2 such pairs) -- main.py finetune's test_result.csv is indexed by
    # SMILES, so a duplicate there can't be mapped back to a unique
    # molecule_id afterward. Drop the repeat up front (keep first).
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        dupes = split_df["smiles"].duplicated(keep="first")
        if dupes.any():
            log.warning(f"Dropping {dupes.sum()} duplicate-SMILES molecule(s) from {split_name} split: {split_df.loc[dupes, 'molecule_id'].tolist()}")
    train_df = train_df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    val_df = val_df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    test_df = test_df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    run_name = f"{_MODEL_NAME}_{run_tag}" if run_tag else _MODEL_NAME
    save_dir = os.path.abspath(os.path.join(checkpoint_dir, dataset_name, run_name))
    os.makedirs(save_dir, exist_ok=True)

    train_csv = os.path.join(save_dir, "train.csv")
    val_csv = os.path.join(save_dir, "val.csv")
    test_csv = os.path.join(save_dir, "test.csv")
    _write_kermt_csv(train_df, target_column, train_csv)
    _write_kermt_csv(val_df, target_column, val_csv)
    _write_kermt_csv(test_df, target_column, test_csv)

    log.info(
        f"Fine-tuning '{_MODEL_NAME}' on '{dataset_name}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    cmd = [
        sys.executable, "main.py", "finetune",
        "--data_path", train_csv,
        "--separate_val_path", val_csv,
        "--separate_test_path", test_csv,
        "--save_dir", save_dir,
        "--checkpoint_path", os.path.abspath(checkpoint_path),
        "--dataset_type", dataset_type,
        "--ensemble_size", "1",
        "--num_folds", "1",
        "--no_features_scaling",
        "--ffn_hidden_size", "700",
        "--ffn_num_layers", "3",
        "--bond_drop_rate", "0.1",
        "--epochs", str(epochs),
        "--metric", metric,
        "--self_attention",
        "--dist_coff", "0.15",
        "--max_lr", "1e-4",
        "--final_lr", "2e-5",
        "--dropout", "0.0",
        "--use_cuikmolmaker_featurization",
        "--features_generator", "rdkit_2d_normalized_cuik_molmaker",
        # "fast" (not exercised by KERMT's own integration test, unlike
        # "descriptastorus") produced all-NaN train/val loss from epoch 0
        # through epoch 29 in a real run against this repo's data -- match
        # the test's proven-working value instead of the untested one.
        "--rdkit2D_normalization_type", "descriptastorus",
        "--seed", str(seed),
    ]
    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # main.py forces deterministic algorithms on

    # cwd=_KERMT_DIR: main.py's own imports (`from kermt...`, `from task...`)
    # are repo-relative, not installed as a package.
    result = subprocess.run(cmd, cwd=_KERMT_DIR, env=env, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"main.py finetune failed for dataset={dataset_name} (exit {result.returncode})")

    # test_result.csv (task/train.py) is indexed by smiles, with a 2-level
    # ('preds'|'targets', task_name) column MultiIndex -- map back to
    # molecule_id via the test split written above rather than assuming row
    # order matches (KERMT may drop/reorder rows it can't featurize).
    # cross_validate.py rewrites args.save_dir to <save_dir>/fold_{n} before
    # calling run_training (kermt/task/cross_validate.py:102) -- with
    # --num_folds 1 that's always fold_0, so the file lands one level deeper
    # than the --save_dir we passed on the CLI.
    test_result = pd.read_csv(os.path.join(save_dir, "fold_0", "test_result.csv"), header=[0, 1], index_col=0)
    y_pred = test_result[("preds", target_column)].to_numpy()
    y_true = test_result[("targets", target_column)].to_numpy()
    smiles_to_id = dict(zip(test_df["smiles"], test_df["molecule_id"]))
    molecule_id = [smiles_to_id[s] for s in test_result.index]

    out_path = save_finetune_result(
        dataset_name=dataset_name,
        model_name=_MODEL_NAME,
        molecule_id=molecule_id,
        y_true=y_true,
        y_pred=y_pred,
        metric_name=dataset_config["metric"],
        output_dir=output_dir,
        tag=run_tag,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. 'reg_openadmet_pxr_pEC50' (config/dataset/<name>.yaml)")
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--checkpoint-path", default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=30)  # agent/config/defaults_finetune.json's own default
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix (e.g. 'epochs50') so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"{_MODEL_NAME}_{args.run_tag}" if args.run_tag else _MODEL_NAME
    output_path = os.path.join(args.output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        checkpoint_path=args.checkpoint_path,
        prepared_dir=args.prepared_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        seed=args.seed,
        val_fraction=args.val_fraction,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
