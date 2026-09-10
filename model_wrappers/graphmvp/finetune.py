"""Fine-tune a GraphMVP checkpoint end-to-end on one OpenADMET endpoint and
score it on the same held-out test split the frozen-embedding path uses.

Only `GraphMVP_AM`/`GraphMVP_CP` are fine-tunable this way -- only those two
ship a "complete-feature" checkpoint (`weights/regression/<variant>/
pretraining_model.pth`, compatible with `GNNComplete`/`GNN_graphpredComplete`
below). The other 8 baseline strategies (AttrMask/ContextPred/GraphCL/
EdgePred/GPT-GNN/Motif/Infomax) only ship simple-feature checkpoints
(`weights/<name>.pth`, for the plain `GNN` class `wrapper.py`'s
`GeneralGNNWrapper` uses) -- loading one into `GNNComplete` would hit the
same state_dict key mismatch `wrapper.py` already documents for the
embedding path.

`graphmvp_repo/src_regression/molecule_finetune_regression.py` is a real,
working scalar-regression training script (`torch.nn.MSELoss`, MAE/RMSE
eval) -- but its `MoleculeDatasetComplete` (a PyG `InMemoryDataset`) only
knows how to build a processed dataset from one of a fixed list of named
MoleculeNet-style raw-CSV schemas (tox21/hiv/bace/.../esol/freesolv/
lipophilicity/malaria/cep/...), not an arbitrary SMILES+float CSV. Every one
of the *regression* schemas (esol/freesolv/lipophilicity/malaria/cep) is
structurally identical: read `smiles` + one fixed-name label column. This
file reuses the `freesolv` schema (label column `expt`) as a vessel for our
own OpenADMET data -- write our train/val/test splits as three separate
`smiles,expt` CSVs, each under its own PyG dataset root's `raw/` dir (so
each root's own `process()` call builds a distinct processed dataset from
just that split, bypassing `molecule_finetune_regression.py`'s own internal
random/scaffold splitter entirely, since we need the exact same split every
other wrapper is scored against).

Two vendored-script quirks handled here, not fixable upstream without
forking the repo: (1) `molecule_finetune_regression.py` does `sys.path.insert
(0, '../src_classification')` -- a path relative to CWD at import time, so
this module `os.chdir()`s into `src_regression/` before importing it; (2) it
does `from config import args` and reads `reg_criterion`/`args` as module-
level globals that only get assigned inside its own `if __name__ ==
"__main__":` block -- which never runs on a plain `import`, so this module
sets both by hand afterward (`sys.argv` also has to be cleared first, since
`config.py` calls `parser.parse_args()` -- with no override, at import time
-- which would otherwise try to parse this file's own CLI arguments).

`process()` drops molecules RDKit can't parse and single-atom/no-bond
molecules (its own comment: "For ESOL and FreeSOlv, there are molecules
with single atoms and empty edges") -- for molecule_id alignment on the test
split, this reads back the `smiles.csv` `process()` itself writes next to
the processed tensors (the *survived*, final-order SMILES list) rather than
assuming the test split's row count/order is preserved.

Must be run with this wrapper's own Python 3.7.17 environment (local pyenv
venv, or the side-by-side interpreter Dockerfile.modal builds for Modal --
see model_wrappers/molbert/Dockerfile.modal for the pattern this mirrors).
graphmvp's own pin is `torch==1.9.0+cpu` (CPU-only by design, matching CPU-
only torch_scatter/torch_sparse wheels) -- no GPU path here.

Usage:
    python model_wrappers/graphmvp/finetune.py --dataset reg_openadmet_pxr_pEC50 --variant GraphMVP_CP
"""

import argparse
import copy
import logging as log
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPHMVP_ROOT = os.path.dirname(os.path.abspath(__file__))
# Overridable: on Modal, graphmvp_repo/ is baked into the image at /opt/
# graphmvp_repo (see Dockerfile.modal) rather than living as a sibling of
# this file, since only finetune.py itself (not the whole wrapper dir) is
# added_local_file'd onto the Dockerfile-built base image.
DEFAULT_GRAPHMVP_REPO_DIR = os.path.join(GRAPHMVP_ROOT, "graphmvp_repo")

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Reused verbatim from molecule_datasets.py's own `_load_freesolv_dataset` --
# any of esol/freesolv/lipophilicity/malaria/cep would work identically,
# freesolv picked arbitrarily.
DATASET_KEY = "freesolv"
LABEL_COLUMN = "expt"

VARIANT_CHECKPOINTS = {
    "GraphMVP_AM": os.path.join(GRAPHMVP_ROOT, "weights", "regression", "GraphMVP_AM", "pretraining_model.pth"),
    "GraphMVP_CP": os.path.join(GRAPHMVP_ROOT, "weights", "regression", "GraphMVP_CP", "pretraining_model.pth"),
}


def _load_shared_infra():
    sys.path.insert(0, REPO_ROOT)
    import src.common.finetune_data as finetune_data
    import src.common.finetune_results as finetune_results
    return finetune_data.load_finetune_splits, finetune_results.save_finetune_result


def _write_pyg_raw_csv(df: pd.DataFrame, target_column: str, raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    df[["smiles", target_column]].rename(columns={target_column: LABEL_COLUMN}).to_csv(
        os.path.join(raw_dir, "data.csv"), index=False
    )


def _import_graphmvp_regression(graphmvp_repo_dir: str):
    """See module docstring's "Two vendored-script quirks" paragraph."""
    src_regression_dir = os.path.join(graphmvp_repo_dir, "src_regression")
    os.chdir(src_regression_dir)
    sys.path.insert(0, src_regression_dir)
    old_argv = sys.argv
    sys.argv = ["molecule_finetune_regression.py"]
    try:
        import molecule_finetune_regression as mfr
    finally:
        sys.argv = old_argv

    import torch.nn as nn
    mfr.reg_criterion = nn.MSELoss()
    return mfr


def _build_split_dataset(mfr, run_dir: str, split_name: str, split_df: pd.DataFrame, target_column: str):
    root = os.path.join(run_dir, f"pyg_{split_name}")
    _write_pyg_raw_csv(split_df, target_column, os.path.join(root, "raw"))
    dataset = mfr.MoleculeDatasetComplete(root, dataset=DATASET_KEY)
    # process()'s shared_extractor() does `torch.tensor(labels[i])` straight
    # off a pandas column read from our CSV, which pandas parses as float64
    # -- the model (and the other regression datasets' own labels, likely
    # float32 upstream of a CSV round-trip) is float32, so torch's autograd
    # rejects the dtype mismatch on .backward(). Cast in place rather than
    # patch the vendored extractor.
    dataset.data.y = dataset.data.y.float()
    return dataset


def _test_predictions_with_molecule_id(mfr, model, device, test_dataset, run_dir, test_df):
    test_loader = mfr.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    _metrics, y_true, y_pred = mfr.eval(model, device, test_loader)

    # process() drops unparseable/single-atom molecules -- read back the
    # exact surviving-SMILES order it wrote itself rather than assuming the
    # split's row count/order is preserved (see module docstring).
    smiles_csv = os.path.join(run_dir, "pyg_test", "processed", "smiles.csv")
    survived_smiles = pd.read_csv(smiles_csv, header=None)[0].tolist()
    n_dropped = len(test_df) - len(survived_smiles)
    if n_dropped:
        log.warning(f"process() dropped {n_dropped} test row(s) (unparseable or single-atom/no-bond molecules)")

    smiles_to_id = dict(zip(test_df["smiles"], test_df["molecule_id"]))
    molecule_id = np.asarray([smiles_to_id[s] for s in survived_smiles])
    return molecule_id, y_true, y_pred


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    variant: str,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    lr: float,
    lr_scale: float,
    decay: float,
    batch_size: int,
    dropout_ratio: float,
    patience: int,
    val_fraction: float,
    seed: int,
    checkpoint_path: str = None,
    graphmvp_repo_dir: str = DEFAULT_GRAPHMVP_REPO_DIR,
    run_tag: str = None,
) -> str:
    model_name = f"graphmvp_{variant}"
    target_column = dataset_config["source"]["target_column"]
    checkpoint_path = checkpoint_path or VARIANT_CHECKPOINTS[variant]

    # Strict ordering: our shared harness must be imported (and the splits
    # derived) BEFORE _import_graphmvp_regression() chdir()s into
    # src_regression/ and pollutes sys.path with the vendored repo's own
    # (differently-named) modules.
    load_finetune_splits, save_finetune_result = _load_shared_infra()
    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )

    # Two different molecule_ids can canonicalize to the same SMILES (a
    # replicate compound -- several reg_openadmet_polaris_* test splits have
    # a handful of these). _test_predictions_with_molecule_id() below maps
    # process()'s surviving-SMILES list back to molecule_id via a smiles-keyed
    # dict, which can't disambiguate two rows sharing one SMILES. Drop the
    # repeat up front (keep first) rather than risk assigning both rows the
    # same molecule_id.
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        dupes = split_df["smiles"].duplicated(keep="first")
        if dupes.any():
            log.warning(f"Dropping {dupes.sum()} duplicate-SMILES molecule(s) from {split_name} split: {split_df.loc[dupes, 'molecule_id'].tolist()}")
    train_df = train_df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    val_df = val_df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    test_df = test_df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    run_dir = os.path.join(os.path.abspath(checkpoint_dir), dataset_name, checkpoint_name)
    os.makedirs(run_dir, exist_ok=True)

    log.info(
        f"Fine-tuning GraphMVP ({variant}) on '{dataset_name}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    mfr = _import_graphmvp_regression(graphmvp_repo_dir)
    import torch

    train_dataset = _build_split_dataset(mfr, run_dir, "train", train_df, target_column)
    valid_dataset = _build_split_dataset(mfr, run_dir, "valid", val_df, target_column)
    test_dataset = _build_split_dataset(mfr, run_dir, "test", test_df, target_column)

    args = mfr.args
    args.dataset = DATASET_KEY
    args.batch_size = batch_size
    args.epochs = epochs
    args.lr = lr
    args.lr_scale = lr_scale
    args.decay = decay
    args.dropout_ratio = dropout_ratio
    args.input_model_file = checkpoint_path
    args.output_model_dir = run_dir
    args.num_workers = 0
    args.eval_train = False

    device = torch.device("cpu")  # graphmvp's own torch pin is CPU-only by design, see module docstring

    train_loader = mfr.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = mfr.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    molecule_model = mfr.GNNComplete(
        num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK,
        drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type,
    )
    model = mfr.GNN_graphpredComplete(args=args, num_tasks=1, molecule_model=molecule_model)
    # Not model.from_pretrained(...) -- it does a bare torch.load() with no
    # map_location, and these checkpoints were saved from a CUDA device;
    # loading on a CPU-only machine (this wrapper's torch pin is +cpu by
    # design) crashes outright. Same load_state_dict() call, just CPU-mapped.
    model.molecule_model.load_state_dict(torch.load(args.input_model_file, map_location=torch.device("cpu")))
    model.to(device)

    model_param_group = [
        {"params": model.molecule_model.parameters()},
        {"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale},
    ]
    optimizer = torch.optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    best_val_mae, best_state, epochs_since_improvement = None, None, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = mfr.train(model, device, train_loader, optimizer)
        val_metrics, _val_true, _val_pred = mfr.eval(model, device, val_loader)
        log.info(f"epoch {epoch}: train_loss={train_loss:.4f} val_mae={val_metrics['MAE']:.4f}")
        if best_val_mae is None or val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                log.info(f"No val_mae improvement for {patience} epochs, stopping early at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pt"))

    molecule_id, y_true, y_pred = _test_predictions_with_molecule_id(mfr, model, device, test_dataset, run_dir, test_df)

    out_path = save_finetune_result(
        dataset_name=dataset_name,
        model_name=model_name,
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
    parser.add_argument("--variant", default="GraphMVP_CP", choices=list(VARIANT_CHECKPOINTS))
    parser.add_argument("--checkpoint-path", default=None, help="Override for containers where weights/ isn't a sibling of this file (e.g. Modal, see model_wrappers/graphmvp/Dockerfile.modal); defaults to VARIANT_CHECKPOINTS[--variant]")
    parser.add_argument("--graphmvp-repo-dir", default=DEFAULT_GRAPHMVP_REPO_DIR, help="Override for containers where graphmvp_repo/ isn't a sibling of this file (e.g. Modal, baked in at /opt/graphmvp_repo)")
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=100, help="Default: config.py's own --epochs default")
    parser.add_argument("--lr", type=float, default=0.001, help="Default: config.py's own --lr default")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Default: config.py's own --lr_scale default")
    parser.add_argument("--decay", type=float, default=0.0, help="Default: config.py's own --decay default")
    parser.add_argument("--batch-size", type=int, default=256, help="Default: config.py's own --batch_size default")
    parser.add_argument("--dropout-ratio", type=float, default=0.5, help="Default: config.py's own --dropout_ratio default")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping on val MAE (mirrors model_wrappers/huggingmolecules/finetune.py's default)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    with open(os.path.join(config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"graphmvp_{args.variant}_{args.run_tag}" if args.run_tag else f"graphmvp_{args.variant}"
    output_dir = os.path.abspath(args.output_dir)
    output_path = os.path.join(output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        variant=args.variant,
        prepared_dir=os.path.abspath(args.prepared_dir),
        checkpoint_dir=os.path.abspath(args.checkpoint_dir),
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        lr_scale=args.lr_scale,
        decay=args.decay,
        batch_size=args.batch_size,
        dropout_ratio=args.dropout_ratio,
        patience=args.patience,
        val_fraction=args.val_fraction,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        graphmvp_repo_dir=args.graphmvp_repo_dir,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
