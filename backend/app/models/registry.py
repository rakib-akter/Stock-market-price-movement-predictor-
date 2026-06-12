"""Model registry — map a short name to a scikit-learn compatible estimator.

When to use each model
----------------------
* **logistic**  Baseline. Linear, fast, interpretable, hard to overfit. ALWAYS
  fit this first — if a complex model can't beat it out-of-sample, it is overfit.
* **random_forest**  Captures non-linear interactions; robust to feature scaling;
  good default when you have many features and moderate data.
* **gradient_boosting**  sklearn's boosting; often stronger than RF but slower and
  easier to overfit — keep it shallow.
* **xgboost / lightgbm**  State-of-the-art for tabular data. Use when you have
  enough data and want maximum signal; tune with early stopping. LightGBM is much
  faster on large panels.

Tree models go in a pipeline *without* scaling; linear models get a StandardScaler.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.config import settings

SEED = settings.random_seed


def _logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


def _gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        random_state=SEED,
    )


def _xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=SEED,
    )


def _lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )


_BUILDERS: dict[str, Callable[[], object]] = {
    "logistic": _logistic,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}


def available_models() -> list[str]:
    """Return the list of registered model names."""
    return list(_BUILDERS)


def build_estimator(name: str):
    """Construct a fresh estimator by registry name.

    Optional dependencies (xgboost/lightgbm) raise a clear error if missing.
    """
    if name not in _BUILDERS:
        raise KeyError(f"Unknown model '{name}'. Available: {available_models()}")
    try:
        return _BUILDERS[name]()
    except ImportError as exc:  # pragma: no cover - optional deps
        raise ImportError(
            f"Model '{name}' requires an optional dependency that is not installed: {exc}"
        ) from exc
