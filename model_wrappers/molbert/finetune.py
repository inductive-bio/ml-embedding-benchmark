"""Fine-tune a MolBERT checkpoint end-to-end on one OpenADMET endpoint and
score it on the same held-out test split the frozen-embedding path uses.

Must be run with this wrapper's own Python 3.7.17 / pytorch-lightning==0.8.4 /
torch==1.4.0 environment -- either the local pyenv venv (built by
embed_wrapper.sh) or, on Modal, the
side-by-side interpreter Dockerfile.modal builds (see
scripts/modal_finetune_molbert.py). torch==1.4.0's PyPI wheel bundles CUDA
support directly (confirmed against both a local GPU and a Modal T4), so
`--gpus 1` works on either.

Reuses `FinetuneSmilesMolbertApp.get_model()` (pretrained-weight loading +
optional layer freezing) and `FinetuneSmilesMolbertModel`'s dataset/task
classes as-is, but drives training/testing with our own `pl.Trainer` rather
than `BaseMolbertApp.run()`: (1) `run()` only calls `trainer.fit(model)`,
with no path to also score a held-out test split with per-molecule
predictions (`test_epoch_end()` only writes aggregate MAE/RMSE/R2 to
`metrics.json`, no per-row output); (2) `Trainer.test()`'s default
`ckpt_path='best'` reloads via `LightningModule.load_from_checkpoint()`,
which `BaseMolbertApp.load_model_weights`'s own docstring flags as broken
for this codebase -- so the best checkpoint is instead reloaded manually via
that same `load_model_weights()` helper, the one method known to work.

Test-set scoring bypasses `MolbertDataLoader` (still used for train/val
dataloaders via `trainer.fit(model)`) and iterates the raw dataset directly:
`MolbertDataLoader` pads out invalid batches by re-sampling valid ones so
every epoch sees a fixed batch count -- correct for training, but wrong for
a single scoring pass over a test set, where it could silently re-yield
already-scored rows instead of terminating cleanly. Each
`BertFinetuneSmilesDataset` item is always padded to a fixed
`max_seq_length`, so manually re-batching kept items (via `torch.stack`)
needs no extra collate/padding logic.

`BertFinetuneSmilesDataset.load_sequences()` hardcodes reading a column
literally named `SMILES` -- the CLI's own `--smiles_column` flag exists but
is never actually read by the dataset. Every split CSV below is written with
that exact column name for this reason.

Defaults follow the authors' scripts/run_finetuning.py: max_seq_length 512,
50 epochs with no early stopping, validation every quarter epoch, the best
valid_loss checkpoint scored on test, and batch size 16 (run here as 2
accumulated micro-batches of 8 to fit a T4, the only Modal GPU torch==1.4.0's
CUDA 10 build supports). The authors never published their learning rate or
freeze level, so those keep apps/args.py's 3e-5 and a full fine-tune.

Usage:
    python model_wrappers/molbert/finetune.py --dataset reg_openadmet_pxr_pEC50
"""

import argparse
import logging as log
import os
import sys

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "molbert_repo"))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result

from molbert.apps.base import BaseMolbertApp
from molbert.apps.finetune import FinetuneSmilesMolbertApp
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from rdkit import RDLogger

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")
# molfeaturizer.py logs rejected SMILES verbatim through the root logger;
# structures must not reach logs.
log.getLogger().addFilter(lambda record: not record.pathname.endswith("molfeaturizer.py"))
# RDKit's C++ parser writes rejected SMILES straight to stderr.
RDLogger.DisableLog("rdApp.error")

MOLBERT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_PATH = os.path.join(MOLBERT_ROOT, "weights", "molbert_100epochs", "checkpoints", "last.ckpt")
TEST_BATCH_SIZE = 64
# run_finetuning.py's value; the checkpoint has 512 position embeddings.
MAX_SEQ_LENGTH = 512


def _drop_unencodable(df, target_column: str, max_seq_length: int = MAX_SEQ_LENGTH):
    """`BertFinetuneSmilesDataset.get_sample()` is supposed to mark an
    over-length SMILES invalid before it ever reaches
    `convert_example_to_features()` -- but its length check compares the
    *character* length of the standardised SMILES string against
    `featurizer.max_length`, not the actual token count `encode()` produces
    (a multi-char -> single-char substitution, so `len(encode(s))` is the
    eventual token count once `[CLS]`/`[SEP]` are added). A short-looking
    SMILES can still tokenize past `max_seq_length`, and
    `convert_example_to_features()` has no truncation path -- only pads if
    already under length -- so it crashes on `assert len(input_ids) ==
    max_seq_length` instead of silently truncating (confirmed against a
    real OpenADMET row, a gold-phosphine complex). Filtering here, before
    any of the three splits are ever written to CSV, means this repo's own
    drop is the only one that happens -- the vendored dataset's own (broken)
    check never gets a chance to run on rows we've already removed.
    """
    featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(max_seq_length)

    def encodable(smiles: str) -> bool:
        standardised = featurizer.standardise(smiles)
        if standardised is None or not featurizer.is_legal(standardised):
            return False
        return len(featurizer.encode(standardised)) + 2 <= max_seq_length  # +2 for [CLS]/[SEP]

    keep = df["smiles"].map(encodable)
    n_dropped = (~keep).sum()
    if n_dropped:
        log.warning(f"Dropping {n_dropped}/{len(df)} row(s) MolBERT's featurizer can't encode (invalid SMILES or tokenizes past max_seq_length={max_seq_length})")
    return df[keep].reset_index(drop=True)


def _write_split_csv(df, target_column: str, path: str) -> None:
    df[["smiles", target_column]].rename(columns={"smiles": "SMILES"}).to_csv(path, index=False)


def _test_set_predictions(model, test_df, target_column: str):
    dataset = model.datasets["test"]
    device = next(model.parameters()).device
    kept_idx, preds, pending = [], [], []

    def flush():
        if not pending:
            return
        stacked = {k: torch.stack([item[k] for item in pending]).to(device) for k in pending[0]}
        with torch.no_grad():
            y_hat = model.forward(stacked)["finetune"]
        preds.extend(y_hat.squeeze(-1).cpu().tolist())
        pending.clear()

    model.eval()
    for i in range(len(dataset)):
        (inputs, _labels), valid = dataset[i]
        if not valid:
            continue
        kept_idx.append(i)
        pending.append(inputs)
        if len(pending) == TEST_BATCH_SIZE:
            flush()
    flush()

    n_dropped = len(dataset) - len(kept_idx)
    if n_dropped:
        log.warning(f"Dropped {n_dropped} test row(s) MolBERT's own featurizer marked invalid (unparseable or over max_seq_length)")

    molecule_id = test_df["molecule_id"].to_numpy()[kept_idx]
    y_true = test_df[target_column].to_numpy(dtype=float)[kept_idx]
    y_pred = np.asarray(preds, dtype=float)
    return molecule_id, y_true, y_pred


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    max_epochs: int,
    min_epochs: int,
    lr: float,
    batch_size: int,
    accumulate_grad_batches: int,
    val_check_interval: float,
    patience: int,
    freeze_level: int,
    val_fraction: float,
    seed: int,
    pretrained_model_path: str = PRETRAINED_MODEL_PATH,
    gpus: int = 0,
    run_tag: str = None,
) -> str:
    model_name = "molbert"
    target_column = dataset_config["source"]["target_column"]

    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_df = _drop_unencodable(train_df, target_column)
    val_df = _drop_unencodable(val_df, target_column)
    test_df = _drop_unencodable(test_df, target_column)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    run_dir = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(run_dir, exist_ok=True)

    train_csv = os.path.join(run_dir, "train.csv")
    valid_csv = os.path.join(run_dir, "valid.csv")
    test_csv = os.path.join(run_dir, "test.csv")
    _write_split_csv(train_df, target_column, train_csv)
    _write_split_csv(val_df, target_column, valid_csv)
    _write_split_csv(test_df, target_column, test_csv)

    log.info(
        f"Fine-tuning MolBERT on '{dataset_name}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    argv = [
        "--train_file", train_csv,
        "--valid_file", valid_csv,
        "--test_file", test_csv,
        "--label_column", target_column,
        "--mode", "regression",
        "--pretrained_model_path", pretrained_model_path,
        "--output_size", "1",
        "--freeze_level", str(freeze_level),
        "--max_epochs", str(max_epochs),
        "--min_epochs", str(min_epochs),
        "--learning_rate", str(lr),
        "--batch_size", str(batch_size),
        "--accumulate_grad_batches", str(accumulate_grad_batches),
        "--max_seq_length", str(MAX_SEQ_LENGTH),
        "--gpus", str(gpus),
        "--num_workers", "0",
        "--val_check_interval", str(val_check_interval),
        "--default_root_dir", run_dir,
        "--progress_bar_refresh_rate", "0",
        "--seed", str(seed),
    ]
    app = FinetuneSmilesMolbertApp()
    args = app.parse_args(argv)
    seed_everything(args.seed)
    model = app.get_model(args)

    # period=0 so every intra-epoch validation is a checkpoint candidate; with
    # PL 0.8.4's default period=1 only the first check of each epoch is.
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(run_dir, "checkpoints"), monitor="valid_loss", save_top_k=1, period=0,
    )
    early_stop_callback = EarlyStopping(monitor="valid_loss", patience=patience) if patience else False

    trainer = Trainer(
        default_root_dir=run_dir,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        val_check_interval=args.val_check_interval,
        # PL 0.8.4 expands an int k to {0: 1, 1: k}, skipping accumulation in epoch 0.
        accumulate_grad_batches={0: args.accumulate_grad_batches},
        gpus=gpus,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model)

    if checkpoint_callback.best_model_path:
        log.info(f"Reloading best checkpoint: {checkpoint_callback.best_model_path}")
        model = BaseMolbertApp.load_model_weights(model=model, checkpoint_file=checkpoint_callback.best_model_path)
    else:
        log.warning("No best checkpoint recorded -- scoring with final in-memory weights")

    molecule_id, y_true, y_pred = _test_set_predictions(model, test_df, target_column)

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
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--max-epochs", type=int, default=50, help="Default: run_finetuning.py's max_epochs")
    parser.add_argument("--min-epochs", type=int, default=1, help="Default: apps/args.py's own --min_epochs default")
    parser.add_argument("--lr", type=float, default=3e-5, help="Default: apps/args.py's own --learning_rate default")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size; times --accumulate-grad-batches gives run_finetuning.py's batch size of 16")
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    parser.add_argument("--val-check-interval", type=float, default=0.25, help="Default: apps/args.py's default, which run_finetuning.py uses")
    parser.add_argument("--patience", type=int, default=None, help="EarlyStopping patience on valid_loss checks. Default: none, as in run_finetuning.py")
    parser.add_argument("--freeze-level", type=int, default=0, help="0 = full fine-tune (apps/finetune.py's own default); see FinetuneSmilesMolbertApp.freeze_network for other values")
    parser.add_argument("--pretrained-model-path", default=PRETRAINED_MODEL_PATH, help="Override for containers where weights/ isn't a sibling of this file (e.g. Modal, see model_wrappers/molbert/Dockerfile.modal)")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs for pytorch_lightning.Trainer (0 = CPU). torch==1.4.0's PyPI wheel has working CUDA support, confirmed against both a local GPU and a Modal T4.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"molbert_{args.run_tag}" if args.run_tag else "molbert"
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
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        patience=args.patience,
        freeze_level=args.freeze_level,
        pretrained_model_path=args.pretrained_model_path,
        gpus=args.gpus,
        val_fraction=args.val_fraction,
        seed=args.seed,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
