"""Fine-tune SimSon's pretrained Xtransformer encoder end-to-end on one
OpenADMET endpoint and score it on the same held-out test split the
frozen-embedding path uses.

simson_repo/main.py's own `--task downstream` path (`Trainer.pre_train()` +
`Trainer.test()`) can't be invoked as-is against arbitrary SMILES+float data:
its data loading (`tokenization.py::load_lists_from_url`) is hardcoded to a
fixed list of MoleculeNet benchmark names, each with its own hardcoded S3
URL and column names -- there's no generic CSV ingestion path. But the
underlying pieces it wires together are architecture-generic and reusable
directly: `GenerateDataset` (dataloader.py) already accepts explicit
`train_idx`/`valid_idx`/`test_idx` (used today for its scaffold-split
branch), `Classifier(args, encoder, num_classes=1)` is a plain
`nn.Linear(max_len, num_classes)` regression head with no
classification-specific wiring, and `RMSELoss` (`sqrt(MSELoss)`, from
losses.py) is a genuine scalar-regression loss (docs' `--criterion mse` is
imprecise; the actual choices are `contrastive`/`rmse`/`bce`, and `rmse` is
what's used here). So this file bypasses `main.py`/`Trainer` entirely and
drives `Xtransformer_Encoder`/`Classifier`/`GenerateDataset`/`RMSELoss`/
`Optimizer`/`Scheduler` directly against `load_finetune_splits()`'s
train/val/test DataFrames, mirroring `Trainer.pre_train_one_epoch()`/
`pre_validate()`/`test()`'s logic (which themselves only report an
aggregate metric, not per-molecule predictions) closely enough to reuse
almost verbatim, but keeping the raw per-molecule predictions this file
needs for `save_finetune_result`.

`GenerateDataset` itself is NOT reused, despite otherwise fitting: its
phase-dispatch has a real bug -- `elif phase == 'valid' or 'val':` parses as
`elif (phase == 'valid') or ('val'):`, and the string literal `'val'` is
always truthy, so this branch is taken for ANY phase other than `'train'`.
The `elif phase == 'test':` branch is dead code -- requesting `phase='test'`
silently returns the *validation* split instead. Plain `TensorDataset`
slices over explicit index arrays are used here instead, avoiding the bug
entirely rather than working around it in place.

`Tokenizer.enable_truncation(max_length=...)` (configured once below) means
an over-length SMILES gets silently truncated, not dropped or crashed on --
unlike molbert's hand-rolled length assertion (see
model_wrappers/molbert/finetune.py's `_drop_unencodable`).

Usage:
    python model_wrappers/simson/finetune.py --dataset reg_openadmet_pxr_pEC50
"""

import argparse
import copy
import logging as log
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error
from tokenizers import Tokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "simson_repo"))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result

from losses import RMSELoss
from model import Classifier, Xtransformer_Encoder

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

SIMSON_ROOT = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_PATH = os.path.join(SIMSON_ROOT, "pretrained_best_model.pth")
TOKENIZER_PATH = os.path.join(SIMSON_ROOT, "pubchem_part_tokenizer.json")

# Fixed architecture params -- must match the pretrained checkpoint exactly
# (also model_wrappers/simson/wrapper.py's embedding-path config).
DIC_SIZE = 300
D_MODEL = 768
N_LAYERS = 2
N_HEAD = 8
MAX_LEN = 512


class Args:
    """Xtransformer_Encoder/Classifier read hyperparameters off a plain
    attribute-having object (simson_repo/config.py's own argparse
    Namespace) rather than taking them as constructor kwargs directly."""

    def __init__(self, **kwargs):
        self.dic_size = DIC_SIZE
        self.d_model = D_MODEL
        self.nlayers = N_LAYERS
        self.nhead = N_HEAD
        self.max_len = MAX_LEN
        self.dropout = 0.1
        self.seed = 0
        for k, v in kwargs.items():
            setattr(self, k, v)


def _set_seed(seed: int) -> None:
    """Same seeding as simson_repo/main.py::set_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_LEN)
    tokenizer.enable_truncation(max_length=MAX_LEN)
    return tokenizer


def _tokenize(smiles_list, tokenizer: Tokenizer) -> torch.LongTensor:
    return torch.LongTensor([tokenizer.encode(s).ids for s in smiles_list])


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    weight_decay: float,
    clipping: float,
    val_fraction: float,
    seed: int,
    run_tag: str = None,
) -> str:
    model_name = "simson"
    target_column = dataset_config["source"]["target_column"]
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    log.info(
        f"Fine-tuning SimSon on '{dataset_name}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    tokenizer = _load_tokenizer()
    all_smiles = train_df["smiles"].tolist() + val_df["smiles"].tolist() + test_df["smiles"].tolist()
    all_labels = (
        train_df[target_column].tolist() + val_df[target_column].tolist() + test_df[target_column].tolist()
    )
    smiles_tensor = _tokenize(all_smiles, tokenizer)
    labels_tensor = torch.FloatTensor([[v] for v in all_labels])

    n_train, n_val = len(train_df), len(val_df)
    train_idx = slice(0, n_train)
    valid_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, len(all_smiles))

    args = Args(seed=seed)
    train_dataset = torch.utils.data.TensorDataset(smiles_tensor[train_idx], labels_tensor[train_idx])
    val_dataset = torch.utils.data.TensorDataset(smiles_tensor[valid_idx], labels_tensor[valid_idx])
    test_dataset = torch.utils.data.TensorDataset(smiles_tensor[test_idx], labels_tensor[test_idx])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    encoder = Xtransformer_Encoder(args)
    encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    model = Classifier(args, encoder=encoder, num_classes=1).to(device)

    criterion = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for smiles, labels in loader:
                smiles, labels = smiles.to(device), labels.to(device)
                if train:
                    optimizer.zero_grad()
                output = model(smiles)
                loss = criterion(output, labels)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), clipping)
                    optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        preds = []
        for smiles, _labels in loader:
            preds.extend(model(smiles.to(device)).cpu().numpy().reshape(-1).tolist())
        return np.asarray(preds, dtype=float)

    best_val_loss, best_state, epochs_since_improvement = None, None, 0
    for epoch in range(epochs):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        log.info(f"epoch {epoch}: train_rmse={train_loss:.4f} val_rmse={val_loss:.4f}")
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                log.info(f"No val_rmse improvement for {patience} epochs, stopping early at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    run_dir = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    y_pred = predict(test_loader)

    out_path = save_finetune_result(
        dataset_name=dataset_name,
        model_name=model_name,
        molecule_id=test_df["molecule_id"].to_numpy(),
        y_true=test_df[target_column].to_numpy(dtype=float),
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
    parser.add_argument("--epochs", type=int, default=500, help="Default: config.py's own --epochs default (early stopping via --patience makes this an upper bound, not a target)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Default: downstream.sh's LR for ESOL and Lipophilicity (config.py's 1e-6 is overridden for every downstream task)")
    parser.add_argument("--batch-size", type=int, default=128, help="Default: config.py's own --batch_size default")
    parser.add_argument("--patience", type=int, default=10, help="Default: config.py's own --patience default")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Default: config.py's own --weight_decay default")
    parser.add_argument("--clipping", type=float, default=2.0, help="Default: config.py's own --clipping default")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"simson_{args.run_tag}" if args.run_tag else "simson"
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
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        weight_decay=args.weight_decay,
        clipping=args.clipping,
        val_fraction=args.val_fraction,
        seed=args.seed,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
