"""Shared train/val/test split loading for end-to-end fine-tuning, used by
every model_wrappers/<name>/finetune.py.

Reads data/prepared/<dataset_name>.joblib rather than
data/processed/<source>/<endpoint>.csv directly, since that's the same
canonicalized Dataset (src/common/data_v2.py::build_dataset) every embedding
is already scored against (scripts/compute_prediction_errors.py) -- keeping
a fine-tuned model's molecule_id/smiles/split alignment identical to the
frozen-embedding results it's compared against.

Every Tier 2 wrapper (molbert, and any future graphfp/mol-r/graphmvp-style
adapter) runs under a legacy Python (3.7-3.9) venv whose newest installable
pandas predates the one that wrote data/prepared/*.joblib (this repo's own
venv, pandas 2.3.3) -- unpickling a pandas 2.x DataFrame under pandas
<=1.3.5 (the last cp37 wheel) fails outright (`AttributeError: Can't get
attribute '_unpickle_block'...`, confirmed live against
model_wrappers/molbert); this is a hard pickle-format incompatibility no
pin can fix. So the split is cached to plain CSVs
(data/finetune_splits/<dataset>/val<f>_seed<s>/{train,valid,test}.csv) the
first time it's computed -- CSV has no such version dependency. The cache
must be populated by a call running under a pandas new enough to read the
source joblib (this repo's own venv, or any Tier 1 wrapper's -- all
>=3.9); a Tier 2 wrapper's own finetune.py should never be the first
caller for a given dataset (see scripts/modal_finetune_molbert.py, which
pre-populates the cache from the Modal function's own >=3.10 driver before
invoking the legacy subprocess).
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 0
DEFAULT_CACHE_DIR = "data/finetune_splits"


def load_finetune_splits(
    dataset_name: str,
    target_column: str,
    prepared_dir: str = "data/prepared",
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one OpenADMET endpoint and carve a fixed validation split out of
    its existing `train` rows.

    The frozen-embedding scoring path has no separate validation split (its
    sklearn heads pick hyperparameters via GridSearchCV's own internal CV
    instead) -- this val split exists only for fine-tuning's early
    stopping/model selection. Every wrapper's finetune.py must call this with
    the same val_fraction/seed rather than re-deriving its own split, so
    every model is compared on the identical partition. A model with its own
    internal validation splitting (e.g. unimol_tools.MolTrain's built-in
    k-fold CV) can just concatenate train_df+val_df back together before
    fitting -- see model_wrappers/unimol/finetune.py.

    Returns:
        (train_df, val_df, test_df), each with columns
        [molecule_id, smiles, <target_column>].
    """
    cache_subdir = os.path.join(cache_dir, dataset_name, f"val{val_fraction}_seed{seed}")
    cache_paths = {split: os.path.join(cache_subdir, f"{split}.csv") for split in ("train", "valid", "test")}
    if all(os.path.exists(p) for p in cache_paths.values()):
        return tuple(pd.read_csv(p) for p in cache_paths.values())

    prepared = joblib.load(os.path.join(prepared_dir, f"{dataset_name}.joblib"))
    df = prepared.data

    train_idx = np.asarray(prepared.splits["train"])
    test_idx = np.asarray(prepared.splits["test"])

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(len(train_idx))
    n_val = int(round(len(shuffled) * val_fraction))
    val_idx = train_idx[shuffled[:n_val]]
    train_only_idx = train_idx[shuffled[n_val:]]

    keep_cols = ["molecule_id", "smiles", target_column]
    train_df = df.iloc[train_only_idx][keep_cols].reset_index(drop=True)
    val_df = df.iloc[val_idx][keep_cols].reset_index(drop=True)
    test_df = df.iloc[test_idx][keep_cols].reset_index(drop=True)

    os.makedirs(cache_subdir, exist_ok=True)
    train_df.to_csv(cache_paths["train"], index=False)
    val_df.to_csv(cache_paths["valid"], index=False)
    test_df.to_csv(cache_paths["test"], index=False)
    return train_df, val_df, test_df
