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

MAT and R-MAT follow their papers' fine-tuning recipes (MAT App. D; R-MAT
App. C.4): 100 epochs with no early stopping and the best-validation epoch
kept, batch size 32, Adam with the Noam schedule from
experiments/src/training/training_callbacks.py, train-set standardized
targets, and a fixed Noam base LR of 5e-5 from huggingmolecules'
mat.gin/rmat.gin (the papers' per-task LR search is not run). R-MAT also
gets the paper's 2-layer prediction MLP in place of the released config's
single linear layer. GROVER keeps its earlier recipe (constant LR, raw
targets, patience 5) since it is being dropped from the benchmark.

experiments/configs/tasks/regression.gin confirms MSELoss as the regression
loss. GROVER's gin file swaps in `GroverLoss`, reimplemented here as
`_grover_loss` since the original class depends on gin's runtime state.

R-MAT memory note: `wrapper.py`'s embedding-only (no backprop) path already
needed batch_size=2 on a T4 to avoid OOM. R-MAT therefore trains on micro-
batches of 4 with 8 accumulation steps, for the paper's effective batch of 32.

Usage:
    python model_wrappers/huggingmolecules/finetune.py --dataset reg_openadmet_pxr_pEC50 --model-name mat_masking_20M
    python model_wrappers/huggingmolecules/finetune.py --dataset reg_openadmet_pxr_pEC50 --model-name grover_large
"""

import argparse
import functools
import logging as log
import math
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

# noam_model_size and noam_warmup_factor: mat.gin (MAT) and the R-MAT paper's
# 30% warmup (rmat.gin uses 0.2). noam_model_size=None means a constant LR.
# patience=None trains every epoch, as both papers do.
VARIANTS = {
    "mat": dict(
        feat_cls=MatFeaturizer, model_cls=MatModel, lr=5e-5, batch_size=32, grad_accum=1,
        weight_decay=0.0, noam_model_size=1024, noam_warmup_factor=0.2, standardize_targets=True,
        patience=None, head_hidden=None,
    ),
    "rmat": dict(
        feat_cls=RMatFeaturizer, model_cls=RMatModel, lr=5e-5, batch_size=4, grad_accum=8,
        weight_decay=0.0, noam_model_size=768, noam_warmup_factor=0.3, standardize_targets=True,
        patience=None, head_hidden=1024,
    ),
    "grover": dict(
        feat_cls=GroverFeaturizer, model_cls=GroverModel, lr=1e-4, batch_size=32, grad_accum=1,
        weight_decay=2e-7, noam_model_size=None, noam_warmup_factor=None, standardize_targets=False,
        patience=5, head_hidden=None,
    ),
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


def _noam_lr(base_lr: float, step: int, model_size: int, warmup_steps: float) -> float:
    """Port of training_callbacks.py::NoamLRScheduler.get_lr."""
    step += 1
    return base_lr * 100 * (model_size ** (-0.5) * min(step ** (-0.5), step * (1e-6 + warmup_steps) ** (-1.5)))


def _paper_rmat_head(proj: nn.Linear, hidden: int) -> nn.Module:
    """R-MAT paper App. C.1 prediction MLP: one hidden layer, LeakyReLU(0.1),
    dropout 0.1. Replaces a head that is randomly initialized anyway."""
    return nn.Sequential(
        nn.Linear(proj.in_features, hidden),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Dropout(0.1),
        nn.Linear(hidden, proj.out_features),
    )


def _run_epoch(model, loader, loss_fn, optimizer, variant: str, device: torch.device, grad_accum: int, lr_for_step, step: int) -> int:
    """Returns the updated optimizer step count. lr_for_step maps a step to
    its learning rate, or is None for a constant LR."""
    model.train()
    optimizer.zero_grad()
    n_batches = len(loader)
    for i, encoded_batch in enumerate(loader):
        encoded_batch = encoded_batch.to(device)
        output = model(encoded_batch)
        loss = _grover_loss(output, encoded_batch.y) if variant == "grover" else loss_fn(_forward_preds(output), encoded_batch.y)
        (loss / grad_accum).backward()
        if (i + 1) % grad_accum == 0 or i + 1 == n_batches:
            if lr_for_step is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr_for_step(step)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
    return step


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
    patience = spec["patience"] if patience is None else patience
    grad_accum = spec["grad_accum"]

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

    y_train = train_df[target_column].to_numpy(dtype=float)
    y_mean, y_std = (y_train.mean(), y_train.std()) if spec["standardize_targets"] else (0.0, 1.0)

    featurizer = spec["feat_cls"].from_pretrained(model_name)

    def loader_for(df, shuffle):
        y = ((df[target_column].to_numpy(dtype=float) - y_mean) / y_std).tolist()
        encodings = featurizer.encode_smiles_list(df["smiles"].tolist(), y)
        return featurizer.get_data_loader(encodings, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader_for(train_df, shuffle=True)
    val_loader = loader_for(val_df, shuffle=False)
    test_loader = loader_for(test_df, shuffle=False)
    y_val = val_df[target_column].to_numpy(dtype=float)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    loss_fn = nn.MSELoss()

    torch.manual_seed(seed)
    model = spec["model_cls"].from_pretrained(model_name)
    if spec["head_hidden"]:
        model.generator.proj = _paper_rmat_head(model.generator.proj, spec["head_hidden"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_for_step = None
    if spec["noam_model_size"]:
        lr_for_step = functools.partial(
            _noam_lr, lr, model_size=spec["noam_model_size"],
            warmup_steps=spec["noam_warmup_factor"] * steps_per_epoch * epochs,
        )

    step, best_val_mae, best_state, epochs_since_improvement = 0, None, None, 0
    for epoch in range(epochs):
        step = _run_epoch(model, train_loader, loss_fn, optimizer, variant, device, grad_accum, lr_for_step, step)
        val_pred = _predict(model, val_loader, device) * y_std + y_mean
        if not np.isfinite(val_pred).all():
            log.warning(f"epoch {epoch}: non-finite val predictions, stopping")
            break
        val_mae = mean_absolute_error(y_val, val_pred)
        log.info(f"epoch {epoch}: val_mae={val_mae:.4f}")
        if best_val_mae is None or val_mae < best_val_mae:
            best_val_mae, epochs_since_improvement = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_improvement += 1
            if patience is not None and epochs_since_improvement >= patience:
                log.info(f"No val_mae improvement for {patience} epochs, stopping early at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training diverged before the first validation check")
    model.load_state_dict(best_state)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    save_path = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    y_pred = _predict(model, test_loader, device) * y_std + y_mean

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
    parser.add_argument("--epochs", type=int, default=100, help="Default: 100, as in both papers and experiments/configs/models/*.gin")
    parser.add_argument("--lr", type=float, default=None, help="Default: this model family's VARIANTS lr (the Noam base rate for MAT/R-MAT, see _noam_lr)")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: this model family's own VARIANTS entry (micro-batch size; R-MAT accumulates 8 of them)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Default: this model family's own VARIANTS entry")
    parser.add_argument("--patience", type=int, default=None, help="Stop a run early after this many epochs with no val_mae improvement. Default: the VARIANTS entry (none for MAT/R-MAT)")
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
