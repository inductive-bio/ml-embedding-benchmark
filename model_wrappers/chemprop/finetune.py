"""Train a chemprop MPNN end-to-end (from scratch, no pretraining) on one
OpenADMET endpoint and score it on the same held-out test split the
frozen-embedding path uses. See docs/e2e_chemprop_chemeleon_plan.md.

Two feature variants, both single-task (this repo's 17 endpoints are already
split one-per-file, unlike ensemble-aware-admet's 9-task ExpansionRx setup --
so there's no missing-label masking to do):

- `--features 2d`: chemprop's own RDKit-2D featurizer (217-dim) as extra
  per-molecule features (`x_d`), matching ensemble-aware-admet's
  baselines/chemprop_baseline.py ("chemprop-2d").
- `--features 3d`: RDKit-3D descriptors (912-dim, model_wrappers/descriptors/
  rdkit_3d.py::compute_rdkit_3d) computed from the lowest-energy conformer in
  this repo's own precomputed cache (data/conformers/<challenge>_mmff94s.joblib
  -- see scripts/modal_generate_conformers.py / modal_optimize_conformers.py),
  matching ensemble-aware-admet's src/models/chemprop_3d.py ("chemprop-3d").
  Molecules missing a usable cached conformer are dropped (logged).

Must be run with this wrapper's own venv (model_wrappers/chemprop/venv,
Python 3.11, chemprop==2.2.4 installed) -- see docs/openadmet_migration_status.md's
"Local run recipe" for how to build it, substituting this wrapper's directory.

Usage:
    python model_wrappers/chemprop/finetune.py --dataset reg_openadmet_polaris_admet_KSOL --features 2d
    python model_wrappers/chemprop/finetune.py --dataset reg_openadmet_polaris_admet_KSOL --features 3d
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "descriptors"))
from rdkit_3d import compute_rdkit_3d, N_3D_FEATURES  # noqa: E402

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from rdkit import Chem
from rdkit.Geometry import Point3D
from sklearn.preprocessing import RobustScaler

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import MoleculeFeaturizerRegistry
from chemprop.models.model import MPNN
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN

log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Dataset name -> challenge (unique compound pool), matching
# scripts/modal_generate_conformers.py::CHALLENGES / modal_embed_aggregated_conformers.py.
CHALLENGE_PATTERNS = [
    ("reg_openadmet_expansionrx_", "expansionrx"),
    ("reg_openadmet_polaris_admet_", "polaris_antiviral"),
    ("reg_openadmet_polaris_potency_", "polaris_antiviral"),
    ("reg_openadmet_pxr_", "pxr"),
]

# Starting hyperparameters ported from ensemble-aware-admet's own configs
# (baselines/configs/chemprop_2d_small.yaml, configs/chemprop_3d.yaml) -- not
# tuned here, see docs/e2e_chemprop_chemeleon_plan.md.
DEFAULT_HPARAMS = {
    "2d": dict(
        mpnn_hidden_dim=256, mpnn_bias=False, mpnn_depth=3, mpnn_dropout=0.2, mpnn_activation="leakyrelu",
        aggr_norm=100.0, ffn_hidden_dim=256, ffn_n_layers=2, ffn_dropout=0.2, ffn_activation="leakyrelu",
        warmup_epochs=20, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
    ),
    "3d": dict(
        mpnn_hidden_dim=300, mpnn_bias=False, mpnn_depth=3, mpnn_dropout=0.1, mpnn_activation="relu",
        aggr_norm=100.0, ffn_hidden_dim=512, ffn_n_layers=2, ffn_dropout=0.1, ffn_activation="relu",
        warmup_epochs=20, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
    ),
}


def _challenge_for_dataset(dataset_name: str) -> str:
    for prefix, challenge in CHALLENGE_PATTERNS:
        if dataset_name.startswith(prefix):
            return challenge
    raise ValueError(f"No known challenge for dataset '{dataset_name}'")


def _mol_from_conformer(smiles: str, atomic_nums: np.ndarray, coords: np.ndarray) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != len(atomic_nums):
        raise ValueError("atom count mismatch reconstructing mol from SMILES")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


def _add_2d_features(df: pd.DataFrame) -> tuple:
    featurizer = MoleculeFeaturizerRegistry["rdkit_2d"]()
    xs, keep, n_failed = [], [], 0
    for i, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row["smiles"])
            xs.append(featurizer(mol))
            keep.append(i)
        except Exception as e:
            log.warning(f"2D featurization failed for molecule_id={row['molecule_id']}: {e}")
            n_failed += 1
    if n_failed:
        log.info(f"Dropped {n_failed}/{len(df)} molecules failing 2D featurization")
    kept_df = df.loc[keep].reset_index(drop=True)
    x_d = np.stack(xs).astype(np.float32) if xs else np.zeros((0, 217), dtype=np.float32)
    return kept_df, x_d


def _add_3d_features(df: pd.DataFrame, challenge: str, conformer_dir: str) -> tuple:
    conformer_cache = _load_conformer_cache(challenge, conformer_dir)
    xs, keep, n_missing = [], [], 0
    for i, row in df.iterrows():
        entry = conformer_cache.get(Chem.CanonSmiles(row["smiles"]))
        if entry is None or "error" in entry:
            n_missing += 1
            continue
        try:
            idx = int(np.argmin(entry["energies"]))
            mol = _mol_from_conformer(row["smiles"], entry["atomic_nums"], entry["coords"][idx])
            xs.append(compute_rdkit_3d(mol))
            keep.append(i)
        except Exception as e:
            log.warning(f"3D featurization failed for molecule_id={row['molecule_id']}: {e}")
            n_missing += 1
    if n_missing:
        log.info(f"Dropped {n_missing}/{len(df)} molecules missing a usable cached conformer")
    kept_df = df.loc[keep].reset_index(drop=True)
    x_d = np.stack(xs).astype(np.float32) if xs else np.zeros((0, N_3D_FEATURES), dtype=np.float32)
    return kept_df, x_d


_conformer_cache_memo = {}


def _load_conformer_cache(challenge: str, conformer_dir: str) -> dict:
    if challenge not in _conformer_cache_memo:
        import joblib
        path = os.path.join(conformer_dir, f"{challenge}_mmff94s.joblib")
        log.info(f"Loading conformer cache: {path}")
        _conformer_cache_memo[challenge] = joblib.load(path)
    return _conformer_cache_memo[challenge]


def _build_datapoints(df: pd.DataFrame, target_column: str, x_d: np.ndarray) -> list:
    return [
        MoleculeDatapoint.from_smi(
            row["smiles"],
            y=np.array([row[target_column]], dtype=np.float32),
            x_d=x_d[i],
        )
        for i, row in df.iterrows()
    ]


def build_model(x_d_width: int, hparams: dict) -> MPNN:
    mp = BondMessagePassing(
        d_h=hparams["mpnn_hidden_dim"],
        bias=hparams["mpnn_bias"],
        depth=hparams["mpnn_depth"],
        dropout=hparams["mpnn_dropout"],
        activation=hparams["mpnn_activation"],
    )
    agg = NormAggregation(norm=hparams["aggr_norm"])
    predictor = RegressionFFN(
        n_tasks=1,
        input_dim=mp.output_dim + x_d_width,
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


def run_finetune(
    dataset_name: str,
    dataset_config: dict,
    features: str,
    prepared_dir: str,
    conformer_dir: str,
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
    log.info(f"'{dataset_name}' ({features}): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    if features == "2d":
        train_df, train_x_d = _add_2d_features(train_df)
        val_df, val_x_d = _add_2d_features(val_df)
        test_df, test_x_d = _add_2d_features(test_df)
    elif features == "3d":
        challenge = _challenge_for_dataset(dataset_name)
        train_df, train_x_d = _add_3d_features(train_df, challenge, conformer_dir)
        val_df, val_x_d = _add_3d_features(val_df, challenge, conformer_dir)
        test_df, test_x_d = _add_3d_features(test_df, challenge, conformer_dir)
    else:
        raise ValueError(f"Unknown --features '{features}'")

    # RDKit-2D/3D descriptors can be wildly non-Gaussian (RDKit2DFeaturizer's
    # own warning recommends against StandardScaler); RobustScaler + a clip
    # matches the recipe ensemble-aware-admet's chemprop_baseline.py/
    # chemprop_3d.py already use.
    scaler = RobustScaler().fit(train_x_d)
    train_x_d = np.clip(scaler.transform(train_x_d), -10, 10).astype(np.float32)
    val_x_d = np.clip(scaler.transform(val_x_d), -10, 10).astype(np.float32)
    test_x_d = np.clip(scaler.transform(test_x_d), -10, 10).astype(np.float32)

    train_data = MoleculeDataset(_build_datapoints(train_df, target_column, train_x_d))
    val_data = MoleculeDataset(_build_datapoints(val_df, target_column, val_x_d))
    test_data = MoleculeDataset(_build_datapoints(test_df, target_column, test_x_d))

    train_loader = build_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(test_data, batch_size=batch_size, shuffle=False)

    model_name = f"chemprop_{features}"
    checkpoint_name = f"{model_name}_{run_tag}" if run_tag else model_name
    save_path = os.path.join(checkpoint_dir, dataset_name, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)

    hparams = {**DEFAULT_HPARAMS[features], **(hparam_overrides or {})}
    model = build_model(x_d_width=train_x_d.shape[1], hparams=hparams)

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
    parser.add_argument("--features", default="2d", choices=["2d", "3d"])
    parser.add_argument("--config-dir", default="config/dataset")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--conformer-dir", default="data/conformers")
    parser.add_argument("--checkpoint-dir", default="data/finetune_checkpoints")
    parser.add_argument("--output-dir", default="data/finetune_errors")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default=None, help="Optional suffix (e.g. 'epochs30') so a hyperparameter-sweep run is saved alongside the default config's result instead of overwriting it")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Override DEFAULT_HPARAMS' warmup_epochs (Noam LR schedule)")
    parser.add_argument("--mpnn-hidden-dim", type=int, default=None, help="Override DEFAULT_HPARAMS' mpnn_hidden_dim")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    hparam_overrides = {}
    if args.warmup_epochs is not None:
        hparam_overrides["warmup_epochs"] = args.warmup_epochs
    if args.mpnn_hidden_dim is not None:
        hparam_overrides["mpnn_hidden_dim"] = args.mpnn_hidden_dim

    with open(os.path.join(args.config_dir, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)

    model_name = f"chemprop_{args.features}"
    stem = f"{model_name}_{args.run_tag}" if args.run_tag else model_name
    output_path = os.path.join(args.output_dir, args.dataset, f"{stem}__finetuned.joblib")
    if os.path.exists(output_path) and not args.override:
        log.info(f"Skipping (already exists): {output_path}")
        return

    out_path = run_finetune(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        features=args.features,
        prepared_dir=args.prepared_dir,
        conformer_dir=args.conformer_dir,
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
