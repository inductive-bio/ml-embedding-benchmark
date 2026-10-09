"""TabICL regression head (https://github.com/soda-inria/tabicl): a pretrained
tabular in-context-learning transformer, used here as a drop-in alternative
to rf/ridge/knn for OpenADMET endpoint regression.

Unlike those heads, TabICL has no hyperparameters worth tuning -- `fit()` just
stores the training data and loads the pretrained checkpoint, and prediction
happens via in-context attention over that stored data at `predict()` time.
Routing it through fit_regressor()'s GridSearchCV (a single-candidate param
grid, since there's nothing to search) would still reload/re-fit the
checkpoint once per CV fold for no benefit over calling cross_val_score
directly, so this bypasses GridSearchCV: cross_val_score for a comparable CV
score, then one plain fit on all the training data for the returned model.

Unlike rf/ridge/knn (models.py), where ridge/knn already get a StandardScaler
via their own sklearn Pipelines, TabICL previously saw raw embeddings --
`_build` standardizes on the training portion and clamps to [-CLAMP, CLAMP]
(the published TFM-on-molecules preprocessing recipe) ahead of TabICL's own
internal normalization.
"""
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tabicl import TabICLRegressor

from .const import CV_SPLITS
from .utils import CLAMP, clamp, get_sklearn_scorer

# TabICLRegressor's default batch_size=8 processes all 8 ensemble members'
# column-wise embeddings (shape n_rows x n_columns x embed_dim each) at once.
# On this repo's widest embeddings (up to ~3072-dim, ~5-7k rows), that OOM-
# killed a 30GB CPU host well before finishing one fit; batch_size=1 trades
# some speed to keep peak memory bounded regardless of embedding width.
_BATCH_SIZE = 1


def _build() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clamp", FunctionTransformer(clamp, kw_args={"limit": CLAMP})),
        ("model", TabICLRegressor(batch_size=_BATCH_SIZE)),
    ])


def fit_tabicl_regressor(X: np.ndarray, y: np.ndarray, metric_name: str) -> dict:
    assert np.isfinite(X).all(), "TabICL input X contains NaN/Inf -- remove_failed_embeddings() should have dropped these rows"
    assert np.isfinite(y).all(), "TabICL input y contains NaN/Inf"
    X = X.astype(np.float32)

    scorer = get_sklearn_scorer(metric_name)
    # data/prepared rows are ordered (train block sorted by molecule_id, i.e.
    # registration order), so an integer `cv` here would resolve to
    # KFold(shuffle=False) and fold by that ordering rather than at random.
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=0)
    cv_score = np.mean(cross_val_score(
        _build(), X, y, cv=cv, scoring=scorer, n_jobs=1,
    ))

    model = _build()
    model.fit(X, y)

    return {
        "method": "tabicl",
        "model_obj": model,
        "best_params": {},
        "cv_score": cv_score,
    }
