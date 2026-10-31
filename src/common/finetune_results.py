"""Shared result scoring/saving for end-to-end fine-tuning, used by every
model_wrappers/<name>/finetune.py.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


def save_finetune_result(
    dataset_name: str,
    model_name: str,
    molecule_id: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    output_dir: str = "data/finetune_errors",
    tag: str = None,
) -> str:
    """Score one fine-tuned model's held-out test-set predictions and save
    them in the same {molecule_id, y_true, y_pred, error} schema
    scripts/compute_prediction_errors.py writes to data/errors/ (method
    fixed to "finetuned" rather than rf/ridge/knn/tabicl), so
    scripts/blam_analysis.py's ranking model can treat a fine-tuned model as
    just another (dataset, method) entry with no changes to its code.

    Returns:
        Path the result was written to.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "dataset": dataset_name,
        "embedder": model_name,
        "method": "finetuned",
        "run_tag": tag,
        "metric_name": metric_name,
        "molecule_id": np.asarray(molecule_id),
        "y_true": y_true,
        "y_pred": y_pred,
        "error": y_pred - y_true,
        "mae": mean_absolute_error(y_true, y_pred),
        "spearmancorr": spearmanr(y_true, y_pred).correlation,
    }

    # tag lets a hyperparameter-sweep run write alongside the default config's
    # result instead of overwriting it (e.g. "epochs30_lr5e-05").
    stem = f"{model_name}_{tag}" if tag else model_name
    out_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}__finetuned.joblib")
    joblib.dump(result, out_path)
    return out_path
