"""Fine-tune a MAT/R-MAT/GROVER checkpoint end-to-end on one OpenADMET
endpoint and score it on the same held-out test split the frozen-embedding
path uses.

Writes a manual PyTorch training loop rather than reusing
huggingmolecules_repo/experiments/src/training's PyTorch-Lightning
TrainingModule: that module's own metrics (training_metrics.py) import
`pytorch_lightning.metrics`, an API removed after PL~1.3 in favor of
torchmetrics, and experiments/requirements.txt pins `pytorch_lightning==1.0.6`
-- installing that alongside this repo's torch==2.8 (needed by every other
wrapper sharing this venv) is not a realistic combination. The three
models' own forward passes and pretrained weight loading (huggingmolecules'
`MatModel`/`RMatModel`/`GroverModel` + their featurizers) are reused as-is;
only the training loop around them is new.

Hyperparameter defaults (lr, batch_size, epochs) below are read directly off
this repo's own experiments/configs/models/{mat,rmat,grover}.gin --
experiments/configs/tasks/regression.gin also confirms MSELoss as the
intended regression loss, and GROVER's own gin file swaps in `GroverLoss`
(pred loss on each of its two FFN heads plus a small agreement penalty
between them) -- reimplemented here as `_grover_loss` since the original
class depends on `gin.query_parameter(...)`, which needs gin's runtime state
initialized, not worth pulling in for one lookup. Not reproduced: the gin
configs' `NoamLRScheduler` (a warmup+decay schedule) -- a constant LR is used
instead, a simplification consistent with docs/finetuning_plan.md's
guidance to start from documented defaults rather than replicate every
training detail.

R-MAT memory note: `wrapper.py`'s embedding-only (no backprop) path already
needed batch_size=2 on a T4 to avoid OOM, well below rmat.gin's own
batch_size=16 (documented for whatever GPU ardigen used). Training needs
backprop activations on top of that forward pass, so the default batch size
below is deliberately lower than rmat.gin's -- override with --batch-size if
running on a bigger GPU.

Usage:
    python model_wrappers/huggingmolecules/finetune.py --dataset reg_openadmet_pxr_pEC50 --model-name mat_masking_20M
    python model_wrappers/huggingmolecules/finetune.py --dataset reg_openadmet_pxr_pEC50 --model-name grover_large
"""

import argparse
import copy
import logging as log
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result
from src.common.utils import get_device

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingmolecules_repo", "src"))
from huggingmolecules import MatFeaturizer, MatModel, RMatFeaturizer, RMatModel, GroverFeaturizer, GroverModel

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

# (featurizer_cls, model_cls, default_lr, default_batch_size, default_weight_decay)
# -- lr/batch_size/weight_decay read off experiments/configs/models/{mat,rmat,grover}.gin,
# except R-MAT's batch_size (see module docstring).
VARIANTS = {
    "mat": dict(feat_cls=MatFeaturizer, model_cls=MatModel, lr=5e-5, batch_size=32, weight_decay=0.0),
    "rmat": dict(feat_cls=RMatFeaturizer, model_cls=RMatModel, lr=5e-5, batch_size=4, weight_decay=0.0),
    "grover": dict(feat_cls=GroverFeaturizer, model_cls=GroverModel, lr=1e-4, batch_size=32, weight_decay=2e-7),
}


def _variant_for(model_name: str) -> str:
    name = model_name.lower()
    if "rmat" in name:
        return "rmat"
    if "mat" in name:
        return "mat"
    if "grover" in name:
        return "grover"
    raise ValueError(f"Unknown huggingmolecules model name: {model_name}")


def _grover_loss(preds, target, dist_coff: float = 0.1):
    """Port of experiments/src/training/training_loss_fn.py::GroverLoss for
    the regression task, minus its `gin.query_parameter('task.name')` lookup
    (always MSELoss here, since this wrapper is regression-only)."""
    pred_1, pred_2 = preds
    pred_loss = nn.functional.mse_loss(pred_1, target) + nn.functional.mse_loss(pred_2, target)
    dist_loss = nn.functional.mse_loss(torch.sigmoid(pred_1), torch.sigmoid(pred_2))
    return pred_loss + dist_coff * dist_loss


def _forward_preds(output):
    """GROVER's forward() returns a (atom_ffn_output, bond_ffn_output) tuple;
    MAT/R-MAT return a single tensor. Averaging the two GROVER heads for a
    point prediction mirrors how the vendored TrainingModule._step handles
    the same tuple output for its own logged metric."""
    if isinstance(output, tuple):
        return torch.mean(torch.stack(output), dim=0)
    return output


def _run_epoch(model, loader, loss_fn, optimizer, variant: str, device: torch.device) -> None:
    model.train()
    for encoded_batch in loader:
        encoded_batch = encoded_batch.to(device)
        optimizer.zero_grad()
        output = model(encoded_batch)
        loss = _grover_loss(output, encoded_batch.y) if variant == "grover" else loss_fn(_forward_preds(output), encoded_batch.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _predict(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for encoded_batch in loader:
        encoded_batch = encoded_batch.to(device)
        preds.append(_forward_preds(model(encoded_batch)).cpu().numpy())
    return np.concatenate(preds, axis=0).reshape(-1)


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    model_name: str,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    patience: int,
    val_fraction: float,
    seed: int,
    run_tag: str = None,
) -> str:
    variant = _variant_for(model_name)
    spec = VARIANTS[variant]
    lr = spec["lr"] if lr is None else lr
    batch_size = spec["batch_size"] if batch_size is None else batch_size
    weight_decay = spec["weight_decay"] if weight_decay is None else weight_decay

    target_column = dataset_config["source"]["target_column"]
    device = get_device()

    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    log.info(
        f"Fine-tuning '{model_name}' on '{dataset_name}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    featurizer = spec["feat_cls"].from_pretrained(model_name)
    model = spec["model_cls"].from_pretrained(model_name).to(device)

    def loader_for(df, shuffle):
        encodings = featurizer.encode_smiles_list(df["smiles"].tolist(), df[target_column].astype(float).tolist())
        return featurizer.get_data_loader(encodings, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader_for(train_df, shuffle=True)
    val_loader = loader_for(val_df, shuffle=False)
    test_loader = loader_for(test_df, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_mae, best_state, epochs_since_improvement = None, None, 0
    for epoch in range(epochs):
        _run_epoch(model, train_loader, loss_fn, optimizer, variant, device)
        val_pred = _predict(model, val_loader, device)
        val_mae = mean_absolute_error(val_df[target_column].to_numpy(dtype=float), val_pred)
        log.info(f"epoch {epoch}: val_mae={val_mae:.4f}")
        if best_val_mae is None or val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                log.info(f"No val_mae improvement for {patience} epochs, stopping early at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    save_path = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    y_pred = _predict(model, test_loader, device)

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
    parser.add_argument(
        "--model-name", required=True,
        choices=["mat_masking_200k", "mat_masking_2M", "mat_masking_20M", "rmat_4M", "rmat_4M_rdkit", "grover_base", "grover_large"],
    )
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=100, help="Default: this model family's own experiments/configs/models/*.gin num_epochs")
    parser.add_argument("--lr", type=float, default=None, help="Default: this model family's own experiments/configs/models/*.gin lr, see VARIANTS")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: this model family's own VARIANTS entry (R-MAT lowered from its gin default, see module docstring)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Default: this model family's own VARIANTS entry")
    parser.add_argument("--patience", type=int, default=5, help="Stop early after this many epochs with no val_mae improvement (mirrors model_wrappers/unimol/finetune.py's --early-stopping default)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"{args.model_name}_{args.run_tag}" if args.run_tag else args.model_name
    output_path = os.path.join(args.output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        model_name=args.model_name,
        prepared_dir=args.prepared_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        patience=args.patience,
        val_fraction=args.val_fraction,
        seed=args.seed,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
