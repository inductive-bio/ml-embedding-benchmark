"""Direct head-to-head comparison of 'tabpfn' (v3, official) vs 'tabpfn_v2'
(internal-only, Apache-licensed) prediction errors, per (embedding, endpoint).

Reads the per-compound error joblibs scripts/compute_prediction_errors.py
already wrote for both methods (data/errors/<endpoint>/<embedding>__tabpfn.joblib
and .../<embedding>__tabpfn_v2.joblib) and reports, for every pair that has
both: each version's CV score (spearmancorr, computed during training) and
test-set spearmancorr/MAE, plus the v2-v3 deltas. This is a direct method
comparison only -- it does not touch scripts/blam_analysis.py's embedding
ranking, which stays on 'tabpfn' (v3) as the official method.

Usage:
    python scripts/compare_tabpfn_versions.py
    python scripts/compare_tabpfn_versions.py --errors-dir data/errors --output data/tabpfn_v2_vs_v3.csv
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_result(path: str) -> dict:
    return joblib.load(path)


def test_spearman(result: dict) -> float:
    rho, _ = spearmanr(result["y_true"], result["y_pred"])
    return rho


def test_mae(result: dict) -> float:
    return float(np.mean(np.abs(result["error"])))


def compare(errors_dir: str) -> pd.DataFrame:
    rows = []
    for endpoint_dir in sorted(glob.glob(os.path.join(errors_dir, "reg_openadmet_*"))):
        if not os.path.isdir(endpoint_dir):
            continue
        endpoint = os.path.basename(endpoint_dir)
        for v3_path in sorted(glob.glob(os.path.join(endpoint_dir, "*__tabpfn.joblib"))):
            embedding = os.path.basename(v3_path)[: -len("__tabpfn.joblib")]
            v2_path = os.path.join(endpoint_dir, f"{embedding}__tabpfn_v2.joblib")
            if not os.path.exists(v2_path):
                continue
            v3 = load_result(v3_path)
            v2 = load_result(v2_path)
            # Both methods are fit on the same embedding-derived train/test split
            # (compute_prediction_errors.py), so their test sets should agree --
            # a mismatch means one of the two runs used a different data snapshot.
            if not np.array_equal(np.sort(v3["molecule_id"]), np.sort(v2["molecule_id"])):
                raise ValueError(
                    f"'{embedding}'/'{endpoint}': tabpfn and tabpfn_v2 disagree on test-set molecule_ids"
                )
            v3_spearman, v2_spearman = test_spearman(v3), test_spearman(v2)
            v3_mae, v2_mae = test_mae(v3), test_mae(v2)
            rows.append({
                "endpoint": endpoint,
                "embedding": embedding,
                "cv_score_v3": v3["cv_score"],
                "cv_score_v2": v2["cv_score"],
                "cv_score_delta": v2["cv_score"] - v3["cv_score"],
                "test_spearman_v3": v3_spearman,
                "test_spearman_v2": v2_spearman,
                "test_spearman_delta": v2_spearman - v3_spearman,
                "test_mae_v3": v3_mae,
                "test_mae_v2": v2_mae,
                "test_mae_delta": v2_mae - v3_mae,
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> tuple:
    per_endpoint = df.groupby("endpoint").agg(
        n_pairs=("embedding", "count"),
        cv_score_delta_mean=("cv_score_delta", "mean"),
        test_spearman_delta_mean=("test_spearman_delta", "mean"),
        test_mae_delta_mean=("test_mae_delta", "mean"),
    ).reset_index()
    rank_corr, _ = spearmanr(df["test_spearman_v3"], df["test_spearman_v2"])
    overall = {
        "n_pairs": len(df),
        "cv_score_delta_mean": df["cv_score_delta"].mean(),
        "cv_score_delta_median": df["cv_score_delta"].median(),
        "test_spearman_delta_mean": df["test_spearman_delta"].mean(),
        "test_spearman_delta_median": df["test_spearman_delta"].median(),
        "test_mae_delta_mean": df["test_mae_delta"].mean(),
        "test_mae_delta_median": df["test_mae_delta"].median(),
        "rank_corr_v2_vs_v3_test_spearman": rank_corr,
    }
    return per_endpoint, overall


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--errors-dir", default="data/errors")
    parser.add_argument("--output", default="data/tabpfn_v2_vs_v3.csv")
    args = parser.parse_args()

    df = compare(args.errors_dir)
    if df.empty:
        raise SystemExit("No (embedding, endpoint) pairs have both tabpfn and tabpfn_v2 results")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)

    per_endpoint, overall = summarize(df)
    print(f"Wrote {len(df)} (embedding, endpoint) comparisons to {args.output}")
    print("\nPer-endpoint mean deltas (tabpfn_v2 - tabpfn/v3):")
    print(per_endpoint.to_string(index=False))
    print(f"\nOverall ({overall['n_pairs']} pairs):")
    print(f"  cv_score delta:      mean={overall['cv_score_delta_mean']:.4f}  median={overall['cv_score_delta_median']:.4f}")
    print(f"  test spearman delta: mean={overall['test_spearman_delta_mean']:.4f}  median={overall['test_spearman_delta_median']:.4f}")
    print(f"  test MAE delta:      mean={overall['test_mae_delta_mean']:.4f}  median={overall['test_mae_delta_median']:.4f}")
    print(f"  rank correlation (v2 vs v3 test spearman across pairs): {overall['rank_corr_v2_vs_v3_test_spearman']:.4f}")


if __name__ == "__main__":
    main()
