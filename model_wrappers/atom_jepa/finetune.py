"""Fine-tune the pretrained Atom-JEPA `molecules` encoder end-to-end (plus a
linear regression head) on one OpenADMET endpoint and score it on the same
held-out test split the frozen-embedding path uses.

atom-jepa ships no training CLI, only an nn.Module, so this is a plain
PyTorch loop following the package README: AdamW with a small encoder lr and
a larger head lr, MSE on train-standardized targets, early stopping on
validation MAE, best-validation weights restored before test inference.

Must be run with this wrapper's own environment (model_wrappers/atom_jepa/
init.sh) -- see scripts/modal_finetune_atom_jepa.py for the Modal image.

Usage:
    python model_wrappers/atom_jepa/finetune.py --dataset reg_openadmet_pxr_pEC50
    python model_wrappers/atom_jepa/finetune.py --dataset reg_openadmet_pxr_pEC50 --fp32
"""

import argparse
import copy
import logging as log
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result

from atom_jepa import AtomJEPA, to_sample

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = "atom_jepa"


def build_samples(df, target_column, mean, std):
    """Returns (samples, kept_positions). Molecules whose conformer
    generation fails are skipped."""
    samples, kept = [], []
    for i, (smiles, y) in enumerate(zip(df["smiles"], df[target_column])):
        try:
            sample = to_sample(smiles)
        except Exception as e:
            log.warning(f"Skipping row {i}: featurization failed: {e}")
            continue
        samples.append({**sample, "y": torch.tensor([(y - mean) / std], dtype=torch.float32)})
        kept.append(i)
    return samples, np.asarray(kept)


@torch.no_grad()
def predict(model, head, samples, batch_size):
    model.eval()
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=model.collate)
    preds = []
    for batch in loader:
        preds.append(head(model(batch).float()).squeeze(-1).float().cpu())
    return torch.cat(preds).numpy()


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    prepared_dir: str,
    output_dir: str,
    epochs: int,
    encoder_lr: float,
    head_lr: float,
    batch_size: int,
    patience: int,
    val_fraction: float,
    seed: int,
    fp32: bool,
    run_tag: str = None,
) -> str:
    target_column = dataset_config["source"]["target_column"]
    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    log.info(f"Fine-tuning '{MODEL_NAME}' on '{dataset_name}': train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    mean, std = float(train_df[target_column].mean()), float(train_df[target_column].std())
    train_samples, _ = build_samples(train_df, target_column, mean, std)
    val_samples, val_kept = build_samples(val_df, target_column, mean, std)
    test_samples, test_kept = build_samples(test_df, target_column, mean, std)
    log.info(f"Featurized: train={len(train_samples)}/{len(train_df)}, val={len(val_samples)}/{len(val_df)}, test={len(test_samples)}/{len(test_df)}")

    load_kwargs = {"precision": "fp32", "cuequivariance": False} if fp32 else {}
    model = AtomJEPA.from_pretrained("molecules", **load_kwargs)
    head = torch.nn.Linear(model.embedding_dim, 1).to(model.device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": encoder_lr},
        {"params": head.parameters(), "lr": head_lr},
    ])

    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, collate_fn=model.collate)
    y_val = val_df[target_column].to_numpy()[val_kept]

    best_val_mae, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            pred = head(model(batch).float())
            loss = torch.nn.functional.mse_loss(pred, batch["y"].to(model.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_pred = predict(model, head, val_samples, batch_size) * std + mean
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        log.info(f"epoch {epoch + 1}/{epochs} train_loss={np.mean(losses):.4f} val_mae={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae, bad_epochs = val_mae, 0
            best_state = copy.deepcopy({"model": model.state_dict(), "head": head.state_dict()})
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"Early stopping at epoch {epoch + 1} (best val_mae={best_val_mae:.4f})")
                break

    model.load_state_dict(best_state["model"])
    head.load_state_dict(best_state["head"])

    # Test molecules that failed featurization get the train-mean prediction
    # so every row stays scored against the same test set as other models.
    test_pred = np.full(len(test_df), mean)
    test_pred[test_kept] = predict(model, head, test_samples, batch_size) * std + mean
    n_fallback = len(test_df) - len(test_kept)
    if n_fallback:
        log.warning(f"{n_fallback}/{len(test_df)} test molecules failed featurization and got the train-mean prediction")

    return save_finetune_result(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        molecule_id=test_df["molecule_id"].to_numpy(),
        y_true=test_df[target_column].to_numpy(),
        y_pred=test_pred,
        metric_name=dataset_config["metric"],
        output_dir=output_dir,
        tag=run_tag,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. 'reg_openadmet_expansionrx_LogD' (config/dataset/<name>.yaml)")
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32", action="store_true", help="fp32 without cuEquivariance, required on pre-Ampere GPUs (T4, V100)")
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a sweep run is saved alongside the default result")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"{MODEL_NAME}_{args.run_tag}" if args.run_tag else MODEL_NAME
    output_path = os.path.join(args.output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        patience=args.patience,
        val_fraction=args.val_fraction,
        seed=args.seed,
        fp32=args.fp32,
        run_tag=args.run_tag,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
