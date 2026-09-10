"""Fine-tune a GEM (GeoGNN) checkpoint end-to-end on one OpenADMET endpoint
and score it on the same held-out test split the frozen-embedding path uses.

GEM's own finetune_regr.py CLI hardcodes `--dataset_name` to a small
MoleculeNet list (esol/freesolv/lipophilicity/qm7/qm8/qm9/qm9_gdb) and does
its own internal 80/10/10 split inside main() -- it has no way to accept an
arbitrary CSV plus a pre-defined train/test split. Rather than fight that
through the CLI (e.g. by faking one of those dataset names), this imports
finetune_regr.py's own train()/evaluate() functions and model classes
directly and drives them against load_finetune_splits()'s train/val/test
partition, so GEM gets scored on the exact same held-out test set every
frozen-embedding result uses -- unlike a naive `--dataset_name=esol`-style
invocation, which would silently re-split the data and break comparability.

Must be run with this wrapper's own venv (model_wrappers/gem/venv, Python
3.8, paddlepaddle installed -- CPU-only wheel, see requirements.txt; there is
no GPU benefit here, unlike Uni-Mol).

Usage:
    python model_wrappers/gem/finetune.py --dataset reg_openadmet_pxr_pEC50
"""

import argparse
import copy
import logging as log
import os
import sys
import types

import numpy as np
import yaml

GEM_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(GEM_ROOT))
GEM_APP_DIR = os.path.join(GEM_ROOT, "gem_repo", "apps", "pretrained_compound", "ChemRL", "GEM")

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

COMPOUND_ENCODER_CONFIG_PATH = os.path.join(GEM_APP_DIR, "model_configs", "geognn_l8.json")
MODEL_CONFIG_PATH = os.path.join(GEM_APP_DIR, "model_configs", "down_mlp2.json")
PRETRAINED_MODEL_PATH = os.path.join(GEM_ROOT, "pretrain_models-chemrl_gem", "regr.pdparams")


# GEM's vendored repo has its own top-level `src` package (gem_repo/.../GEM/
# src/, a *namespace* package -- no __init__.py) which collides by name with
# this repo's own `src` package (a *regular* package, has __init__.py). A
# regular package always wins over a namespace portion for a plain `import
# src`, REGARDLESS of sys.path order -- so as long as both REPO_ROOT and
# GEM_APP_DIR are ever on sys.path at the same time, GEM's own internal
# `from src.model import ...`-style imports (inside finetune_regr.py and its
# siblings) resolve to *our* src package instead of GEM's, and crash with
# e.g. `No module named 'src.featurizer'` (confirmed live). Conversely,
# unpickling data/prepared/<name>.joblib needs `src.common.types.Dataset`
# to resolve to *our* real package -- so the two needs are incompatible
# within a single `sys.modules['src']` cache entry. Fixed by doing them in
# strict sequence: load our data first (only REPO_ROOT on path, nothing
# GEM-related imported yet), then purge every sys.modules['src'*] entry
# before switching to GEM_APP_DIR and importing GEM's own modules -- see
# _load_shared_infra(), _purge_src_modules(), and _import_gem() below.
def _load_shared_infra():
    sys.path.insert(0, REPO_ROOT)
    import src.common.finetune_data as finetune_data
    import src.common.finetune_results as finetune_results
    return finetune_data.load_finetune_splits, finetune_results.save_finetune_result


def _purge_src_modules():
    for key in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[key]
    if REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)


def _import_gem():
    sys.path.insert(0, os.path.join(GEM_ROOT, "gem_repo"))  # for pahelix, pgl imports
    sys.path.insert(0, GEM_APP_DIR)  # for `from src.model import ...` (finetune_regr.py's own imports)

    import paddle
    import paddle.nn as nn
    from pahelix.datasets.inmemory_dataset import InMemoryDataset
    from pahelix.model_zoo.gem_model import GeoGNNModel
    from pahelix.utils import load_json_config

    from src.featurizer import DownstreamCollateFn, DownstreamTransformFn
    from src.model import DownstreamModel
    from src.utils import exempt_parameters

    # finetune_regr.py's own train()/evaluate() loops, reused as-is rather
    # than re-implemented -- only args.batch_size/args.num_workers are read
    # off `args`.
    from finetune_regr import evaluate as gem_evaluate
    from finetune_regr import train as gem_train_epoch

    return types.SimpleNamespace(**{
        k: v for k, v in locals().items() if not k.startswith("_")
    })


def _to_data_list(df, target_column):
    return [
        {"smiles": s, "label": np.array([v], dtype="float32")}
        for s, v in zip(df["smiles"], df[target_column])
    ]


def _build_train_dataset(df, target_column, transform_fn, num_workers):
    dataset = InMemoryDataset(_to_data_list(df, target_column))
    dataset.transform(transform_fn, num_workers=num_workers, drop_none=True)
    return dataset


def _build_test_dataset(df, target_column, transform_fn):
    # drop_none=True (used for train/val) doesn't preserve which original
    # rows survived -- for the test set we need molecule_id alignment, so
    # transform is applied one at a time here (single-process; test splits
    # are small enough that this isn't a bottleneck) and failures are
    # filtered explicitly, tracking the surviving positions.
    raw_list = _to_data_list(df, target_column)
    transformed = [transform_fn(d) for d in raw_list]
    keep = [i for i, d in enumerate(transformed) if d is not None]
    dataset = InMemoryDataset([transformed[i] for i in keep])
    molecule_ids = df["molecule_id"].to_numpy()[keep]
    return dataset, molecule_ids


def _predict(model, label_mean, label_std, dataset, collate_fn, batch_size):
    data_gen = dataset.get_data_loader(
        batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn
    )
    model.eval()
    total_pred = []
    for atom_bond_graphs, bond_angle_graphs, _labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        scaled_preds = model(atom_bond_graphs, bond_angle_graphs)
        preds = scaled_preds.numpy() * label_std + label_mean
        total_pred.append(preds)
    return np.concatenate(total_pred, axis=0).reshape(-1)


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    encoder_lr: float,
    head_lr: float,
    dropout_rate: float,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    run_tag: str = None,
) -> str:
    model_name = "GEM"
    target_column = dataset_config["source"]["target_column"]

    # Strict ordering required -- see the big comment above _load_shared_infra().
    load_finetune_splits, save_finetune_result = _load_shared_infra()
    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    _purge_src_modules()
    globals().update(vars(_import_gem()))

    transform_fn = DownstreamTransformFn()
    train_dataset = _build_train_dataset(train_df, target_column, transform_fn, num_workers)
    val_dataset = _build_train_dataset(val_df, target_column, transform_fn, num_workers)
    test_dataset, test_molecule_ids = _build_test_dataset(test_df, target_column, transform_fn)

    log.info(
        f"Fine-tuning GEM on '{dataset_name}': train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    # label_mean/std computed from train only (GEM's own finetune_regr.py
    # computes this from the *entire* dataset before splitting -- using only
    # train here avoids leaking val/test statistics into the normalization).
    train_labels = np.array([d["label"] for d in train_dataset.data_list])
    label_mean = np.reshape(np.mean(train_labels, axis=0), [1, -1])
    label_std = np.reshape(np.std(train_labels, axis=0), [1, -1])

    compound_encoder_config = load_json_config(COMPOUND_ENCODER_CONFIG_PATH)
    model_config = load_json_config(MODEL_CONFIG_PATH)
    model_config["task_type"] = "regr"
    model_config["num_tasks"] = 1
    model_config["dropout_rate"] = dropout_rate

    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    compound_encoder.set_state_dict(paddle.load(PRETRAINED_MODEL_PATH))

    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(head_lr, parameters=head_params)
    criterion = nn.L1Loss()

    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config["atom_names"],
        bond_names=compound_encoder_config["bond_names"],
        bond_float_names=compound_encoder_config["bond_float_names"],
        bond_angle_float_names=compound_encoder_config["bond_angle_float_names"],
        task_type="regr",
    )
    train_args = types.SimpleNamespace(batch_size=batch_size, num_workers=num_workers)

    best_val_mae, best_state = None, None
    for epoch in range(epochs):
        train_loss = gem_train_epoch(
            train_args, model, label_mean, label_std,
            train_dataset, collate_fn, criterion, encoder_opt, head_opt,
        )
        val_mae = gem_evaluate(
            train_args, model, label_mean, label_std, val_dataset, collate_fn, metric="mae"
        )
        log.info(f"epoch {epoch}: train_loss={train_loss:.4f} val_mae={val_mae:.4f}")
        if best_val_mae is None or val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())

    model.set_state_dict(best_state)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    save_path = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)
    paddle.save(model.state_dict(), os.path.join(save_path, "model.pdparams"))

    y_pred = _predict(model, label_mean, label_std, test_dataset, collate_fn, batch_size)
    y_true = np.array([d["label"][0] for d in test_dataset.data_list])

    out_path = save_finetune_result(
        dataset_name=dataset_name,
        model_name=model_name,
        molecule_id=test_molecule_ids,
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
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    # Defaults mirror GEM's own scripts/finetune_regr.sh (batch_size=32,
    # encoder_lr=head_lr=1e-3, dropout=0.2, its first model config). Its
    # max_epoch=100 default is a long CPU-only run per dataset (this
    # wrapper's paddlepaddle install is CPU-only) -- override with --epochs
    # for a cheaper smoke test.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--encoder-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"GEM_{args.run_tag}" if args.run_tag else "GEM"
    output_path = os.path.join(args.output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        prepared_dir=args.prepared_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
