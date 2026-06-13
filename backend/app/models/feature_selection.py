"""Feature-importance *stability* analysis.

A feature that looks important on one slice of history but not others is probably
fitting noise. Here we refit the model across walk-forward folds and measure how
*consistent* each feature's importance is. Stable, consistently-important features
are the ones worth trusting; high-variance ones are candidates to drop.

Importance source:
    * tree models  → ``feature_importances_``
    * linear models → ``|coef|`` (after the pipeline's scaler, so it is comparable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.data.splits import walk_forward_folds
from backend.app.features.pipeline import split_X_y
from backend.app.models.registry import build_estimator
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_importance(estimator, feature_names: list[str]) -> pd.Series | None:
    """Pull a per-feature importance vector from a fitted estimator (or None)."""
    model = estimator
    if hasattr(estimator, "named_steps"):  # sklearn Pipeline
        model = estimator.named_steps.get("clf", estimator)
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        return None
    if len(vals) != len(feature_names):
        return None
    return pd.Series(vals, index=feature_names)


def importance_stability(
    matrix: pd.DataFrame,
    model_name: str = "random_forest",
    horizon: int = 1,
    n_splits: int = 5,
    embargo: int | None = None,
) -> pd.DataFrame:
    """Refit across walk-forward folds and summarize importance stability.

    Returns a DataFrame indexed by feature with columns:
        * ``mean_importance``  average normalized importance across folds
        * ``std_importance``   spread across folds (lower = more stable)
        * ``mean_rank``        average rank (1 = most important)
        * ``stability``        ``mean / (std + eps)`` — high = strong and consistent
        * ``n_folds``          folds the feature was scored in
    Sorted by ``stability`` descending.
    """
    target = f"y_dir_{horizon}"
    embargo = horizon if embargo is None else embargo
    folds = walk_forward_folds(matrix, n_splits=n_splits, embargo=embargo)
    if not folds:
        raise ValueError("Not enough data to build walk-forward folds.")

    per_fold: list[pd.Series] = []
    rank_rows: list[pd.Series] = []
    for fold in folds:
        train = matrix.loc[fold.train_idx]
        X_tr, y_tr = split_X_y(train, target)
        est = build_estimator(model_name)
        est.fit(X_tr, y_tr.astype(int))
        imp = _extract_importance(est, list(X_tr.columns))
        if imp is None:
            raise ValueError(f"Model '{model_name}' exposes no importances/coef_.")
        # Normalize so folds are comparable regardless of scale.
        total = imp.sum()
        imp = imp / total if total > 0 else imp
        per_fold.append(imp)
        rank_rows.append(imp.rank(ascending=False))

    imp_df = pd.concat(per_fold, axis=1)
    rank_df = pd.concat(rank_rows, axis=1)
    eps = 1e-9

    out = pd.DataFrame(
        {
            "mean_importance": imp_df.mean(axis=1),
            "std_importance": imp_df.std(axis=1, ddof=0),
            "mean_rank": rank_df.mean(axis=1),
            "n_folds": imp_df.notna().sum(axis=1).astype(int),
        }
    )
    out["stability"] = out["mean_importance"] / (out["std_importance"] + eps)
    return out.sort_values("stability", ascending=False)


def select_stable_features(
    matrix: pd.DataFrame,
    model_name: str = "random_forest",
    horizon: int = 1,
    n_splits: int = 5,
    top_k: int = 15,
) -> list[str]:
    """Return the ``top_k`` features ranked by stability (strong + consistent)."""
    table = importance_stability(
        matrix, model_name=model_name, horizon=horizon, n_splits=n_splits
    )
    return table.head(top_k).index.tolist()
