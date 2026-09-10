"""TabPFN regression head (https://github.com/PriorLabs/TabPFN): a pretrained
tabular in-context-learning transformer, used here as a drop-in alternative
to rf/ridge/knn/tabicl for OpenADMET endpoint regression.

Like tabicl, TabPFN has no hyperparameters worth tuning -- `fit()` just stores
the training data and loads the pretrained checkpoint, and prediction happens
via in-context attention over that stored data at `predict()` time. This
bypasses GridSearchCV for the same reason tabicl_head.py does: cross_val_score
for a comparable CV score, then one plain fit on all the training data for the
returned model. It also shares tabicl_head.py's scale+clamp preprocessing --
see that module's docstring for why.

Two methods share this module: 'tabpfn' (TabPFNRegressor's own "auto" default,
which resolves to the v3 model today -- our official, most-up-to-date
results) and 'tabpfn_v2' (explicitly pinned to the v2 model via
create_default_for_version, whose Apache-licensed checkpoint has no
commercial-licensing restriction -- used for internal-only benchmarking
comparisons alongside the v3 numbers, not as a replacement for them).
"""
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from .const import CV_SPLITS
from .utils import CLAMP, clamp, get_sklearn_scorer

# TabPFN's default checkpoint validates against a 2,000-feature pretraining
# limit; this repo's widest embeddings (up to ~3072-dim) exceed that, so
# ignore_pretraining_limits=True is required to fit them at all.
_MODEL_KWARGS = {"ignore_pretraining_limits": True, "random_state": 0}


def _make_model(method: str) -> TabPFNRegressor:
    if method == "tabpfn_v2":
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2, **_MODEL_KWARGS)
    return TabPFNRegressor(**_MODEL_KWARGS)


def _build(method: str) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clamp", FunctionTransformer(clamp, kw_args={"limit": CLAMP})),
        ("model", _make_model(method)),
    ])


def fit_tabpfn_regressor(X: np.ndarray, y: np.ndarray, metric_name: str, method: str = "tabpfn") -> dict:
    assert np.isfinite(X).all(), "TabPFN input X contains NaN/Inf -- remove_failed_embeddings() should have dropped these rows"
    assert np.isfinite(y).all(), "TabPFN input y contains NaN/Inf"
    X = X.astype(np.float32)

    scorer = get_sklearn_scorer(metric_name)
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=0)
    cv_score = np.mean(cross_val_score(
        _build(method), X, y, cv=cv, scoring=scorer, n_jobs=1,
    ))

    model = _build(method)
    model.fit(X, y)

    return {
        "method": method,
        "model_obj": model,
        "best_params": {},
        "cv_score": cv_score,
    }
