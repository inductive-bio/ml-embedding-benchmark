import numpy as np
from sklearn.metrics import get_scorer, get_scorer_names, make_scorer, mean_absolute_error
from scipy.stats import spearmanr

# Shared by tabicl_head.py and tabpfn_head.py's preprocessing pipelines (the
# published TFM-on-molecules recipe: standardize, then clamp to [-CLAMP,
# CLAMP]). Lives here rather than in tabicl_head.py so importing it doesn't
# also trigger tabicl_head.py's `from tabicl import TabICLRegressor` --
# tabpfn's dedicated Modal image installs the tabpfn package only, not tabicl.
CLAMP = 6.0


def clamp(X, limit):
    return np.clip(X, -limit, limit)


def spearman_corr(x, y):
    rho, _ = spearmanr(x, y)
    return rho


def get_sklearn_scorer(metric_name: str):
    if metric_name in get_scorer_names():
        return get_scorer(metric_name)
    
    if metric_name == "spearmancorr":
        return make_scorer(
            spearman_corr,
            greater_is_better=True
        )
    
    if metric_name == "mae":
        return make_scorer(
            mean_absolute_error,
            greater_is_better=False
        )
        
    raise ValueError(f"Unknown metric name: {metric_name}")