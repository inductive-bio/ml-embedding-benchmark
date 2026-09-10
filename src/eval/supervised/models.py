import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from skfp.distances import tanimoto_count_distance

# RF/KNN below use n_jobs=1: callers wrap these in GridSearchCV with their own
# n_jobs, and RandomForest/KNeighbors' own n_jobs=-1 would oversubscribe CPUs
# under that outer parallelism.

RF_REG = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    # 500 unconstrained-depth trees timed out past 4h on the largest
    # datasets (~5-7k rows) for the widest embeddings (ChemFM-1B/3B at
    # 2048-dim, grover_large at 2400-dim, rmat_4M at 3072-dim, mol_r_sage_1024
    # at 1024-dim float64) -- max_depth caps runaway tree growth (the actual
    # cost driver at this N, more than estimator count), n_estimators trimmed
    # too for headroom.
    "clf__n_estimators": [200],
    "clf__max_depth": [20],
    "clf__criterion": ["squared_error"],
}

RIDGE_REG = {
    "clf__alpha": np.logspace(-2, 3, 10),
    "clf__max_iter": [5000],
    # "lbfgs" requires positive=True in sklearn's Ridge; "auto" (Ridge's own
    # default) picks cholesky/svd/etc. as appropriate and has no such constraint.
    "clf__solver": ["auto"],
}

KNN_REG = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}


def get_knn_distance(embeddings_dtype):
    if np.issubdtype(embeddings_dtype, np.integer):
        return tanimoto_count_distance
    elif np.issubdtype(embeddings_dtype, np.floating):
        return "cosine"
    else:
        raise ValueError(f"Unsupported embeddings dtype: {embeddings_dtype}. Expected integer or floating point type.")


def get_reg_models(embeddings_dtype):
    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestRegressor(n_jobs=1))]),
            "params": RF_REG.copy(),
        },
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", Ridge()),
                ]
            ),
            "params": RIDGE_REG.copy(),
        },
        "knn": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsRegressor(n_jobs=1, metric=get_knn_distance(embeddings_dtype))),
                ]
            ),
            "params": KNN_REG.copy(),
        }
    }
