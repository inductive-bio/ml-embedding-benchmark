import numpy as np
import logging as log

from sklearn.model_selection import GridSearchCV, KFold

from .models import get_reg_models
from .const import CV_SPLITS
from .utils import get_sklearn_scorer


def fit_regressor(X: np.ndarray, y: np.ndarray, method: str, metric_name: str, n_jobs: int = -1) -> dict:
    """Grid-search one regression head's hyperparameters (scored via CV_SPLITS-fold
    CV on the given data) and refit on all of it.

    Returns:
        dict with the fitted estimator, its chosen hyperparameters, and the
        CV score used to select them.
    """
    if method == "tabicl":
        # Imported lazily -- tabicl is only installed on modal_score_errors.py's
        # GPU image; the rf/ridge/knn CPU path (and any local venv without
        # tabicl) must stay importable without it.
        from .tabicl_head import fit_tabicl_regressor
        # No hyperparameters to grid-search -- see tabicl_head.py.
        return fit_tabicl_regressor(X, y, metric_name)

    if method in ("tabpfn", "tabpfn_v2"):
        # Imported lazily -- same rationale as tabicl above.
        from .tabpfn_head import fit_tabpfn_regressor
        # No hyperparameters to grid-search -- see tabpfn_head.py.
        return fit_tabpfn_regressor(X, y, metric_name, method=method)

    models = get_reg_models(X.dtype)
    if method not in models:
        raise ValueError(f"Unknown regression method: {method}. Available: {list(models)}")
    model = models[method]
    scorer = get_sklearn_scorer(metric_name)

    # data/prepared rows are ordered (train block sorted by molecule_id, i.e.
    # registration order), so an integer `cv` here would resolve to
    # KFold(shuffle=False) and fold by that ordering rather than at random.
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=0)
    grid_search = GridSearchCV(
        model["model"],
        model["params"],
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        refit=True,
    )
    grid_search.fit(X, y)

    return {
        "method": method,
        "model_obj": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
    }
