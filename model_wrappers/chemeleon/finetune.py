"""Fine-tune a pretrained Chemeleon encoder end-to-end (encoder + a fresh
single-task regression head, no freezing) on one OpenADMET endpoint.

Reuses the same pretrained checkpoint wrapper.py loads for frozen
embeddings -- unlike wrapper.py, which only ever calls .fingerprint() on a
frozen encoder, this attaches a fresh RegressionFFN(n_tasks=1) and trains the
whole model (encoder + head) jointly on ADMET labels.

Must be run with this wrapper's own venv (model_wrappers/chemeleon/venv,
Python 3.11, chemprop==2.2.4 + checkpoints already installed by init.sh).

Usage:
    python model_wrappers/chemeleon/finetune.py --dataset reg_openadmet_polaris_admet_KSOL --model-name chemeleon_original
"""

import argparse
import logging as log
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.finetune_data import load_finetune_splits
from src.common.finetune_results import save_finetune_result

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models.model import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN, UnscaleTransform

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Local checkpoint filenames, as downloaded by init.sh -- kept in sync with
# wrapper.py's own _VARIANTS by hand.
_VARIANTS = {
    "chemeleon_original": "chemeleon_mp.pt",
}

# Matches the CheMeleon authors' fine-tuning script
# (JacksonBurns/chemeleon models/chemprop_mpnn/evaluate.py): RegressionFFN
# with hidden_dim=256 and otherwise chemprop defaults, MPNN's default LR
# schedule. No mpnn_* keys: the encoder comes from the pretrained checkpoint.
DEFAULT_HPARAMS = dict(
    ffn_hidden_dim=256, ffn_n_layers=1, ffn_dropout=0.0, ffn_activation="relu",
    warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
)


def _load_pretrained_encoder(checkpoint_path: str):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mp = BondMessagePassing(**raw["hyper_parameters"])
    mp.load_state_dict(raw["state_dict"])
    return mp


def build_model(checkpoint_path: str, hparams: dict, output_transform: UnscaleTransform) -> MPNN:
    mp = _load_pretrained_encoder(checkpoint_path)
    agg = MeanAggregation()
    predictor = RegressionFFN(
        n_tasks=1,
        output_transform=output_transform,
        input_dim=mp.output_dim,
        hidden_dim=hparams["ffn_hidden_dim"],
        n_layers=hparams["ffn_n_layers"],
        dropout=hparams["ffn_dropout"],
        activation=hparams["ffn_activation"],
    )
    return MPNN(
        mp, agg, predictor,
        warmup_epochs=hparams["warmup_epochs"],
        init_lr=hparams["init_lr"],
        max_lr=hparams["max_lr"],
        final_lr=hparams["final_lr"],
    )


def _build_datapoints(df: pd.DataFrame, target_column: str) -> list:
    return [
        MoleculeDatapoint.from_smi(row["smiles"], y=np.array([row[target_column]], dtype=np.float32))
        for _, row in df.iterrows()
    ]


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    model_name: str,
    checkpoint_path: str,
    prepared_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    patience: int,
    val_fraction: float,
    seed: int,
    run_tag: str = None,
    hparam_overrides: dict = None,
) -> str:
    pl.seed_everything(seed, workers=True)
    target_column = dataset_config["source"]["target_column"]

    train_df, val_df, test_df = load_finetune_splits(
        dataset_name=dataset_name,
        target_column=target_column,
        prepared_dir=prepared_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    log.info(f"'{dataset_name}' ({model_name}): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_data = MoleculeDataset(_build_datapoints(train_df, target_column))
    val_data = MoleculeDataset(_build_datapoints(val_df, target_column))
    test_data = MoleculeDataset(_build_datapoints(test_df, target_column))
    # Train on standardized targets; the predictor's UnscaleTransform maps
    # eval-mode outputs back to original units, so test predictions need no
    # extra handling.
    scaler = train_data.normalize_targets()
    val_data.normalize_targets(scaler)

    train_loader = build_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(test_data, batch_size=batch_size, shuffle=False)

    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    save_path = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)

    hparams = {**DEFAULT_HPARAMS, **(hparam_overrides or {})}
    model = build_model(checkpoint_path, hparams, UnscaleTransform.from_standard_scaler(scaler))

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=save_path,
        callbacks=[
            ModelCheckpoint(dirpath=save_path, filename="best", monitor="val_loss", mode="min", save_top_k=1),
            EarlyStopping(monitor="val_loss", mode="min", patience=patience),
        ],
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        model = MPNN.load_from_checkpoint(best_path)

    preds = trainer.predict(model, test_loader)
    y_pred = torch.cat(preds, dim=0).squeeze(-1).cpu().numpy()

    out_path = save_finetune_result(
        dataset_name=dataset_name,
        model_name=model_name,
        molecule_id=test_df["molecule_id"].to_numpy(),
        y_true=test_df[target_column].to_numpy(),
        y_pred=y_pred,
        metric_name=dataset_config["metric"],
        output_dir=output_dir,
        tag=run_tag,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. 'reg_openadmet_polaris_admet_KSOL' (config/dataset/<name>.yaml)")
    parser.add_argument("--model-name", required=True, choices=sorted(_VARIANTS))
    parser.add_argument("--checkpoint-path", default=None, help="Override the default model_wrappers/chemeleon/checkpoints/<file> path (used by the Modal script)")
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=50, help="Default: the authors' evaluate.py max_epochs (also sets the LR decay horizon)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5, help="Default: the authors' evaluate.py EarlyStopping patience")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Override DEFAULT_HPARAMS' warmup_epochs (Noam LR schedule)")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    hparam_overrides = {}
    if args.warmup_epochs is not None:
        hparam_overrides["warmup_epochs"] = args.warmup_epochs

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    stem = f"{args.model_name}_{args.run_tag}" if args.run_tag else args.model_name
    output_path = os.path.join(args.output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    checkpoint_path = args.checkpoint_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints", _VARIANTS[args.model_name]
    )

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        prepared_dir=args.prepared_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        val_fraction=args.val_fraction,
        seed=args.seed,
        run_tag=args.run_tag,
        hparam_overrides=hparam_overrides,
    )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
