"""Hyperparameter tuning with purged, embargoed, walk-forward cross-validation.

Standard ``GridSearchCV`` shuffles and leaks in time series. Here we score each
candidate parameter set on the project's own embargoed walk-forward folds, so the
selected hyperparameters are chosen the same honest way the final model is judged.

This is deliberately a small, dependency-free search (grid or random sampling).
For larger searches, swap the loop body for Optuna later — the fold logic stays.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from backend.app.data.splits import walk_forward_folds
from backend.app.features.pipeline import split_X_y
from backend.app.models.registry import build_estimator
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TuningResult:
    best_params: dict
    best_score: float
    leaderboard: pd.DataFrame
    scored: list[dict] = field(default_factory=list)


def _param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _set_params(estimator, params: dict):
    """Apply params, handling both bare estimators and sklearn Pipelines."""
    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        estimator.set_params(**{f"clf__{k}": v for k, v in params.items()})
    else:
        estimator.set_params(**params)
    return estimator


def tune_hyperparameters(
    matrix: pd.DataFrame,
    model_name: str = "random_forest",
    horizon: int = 1,
    param_grid: dict[str, list] | None = None,
    n_splits: int = 4,
    embargo: int | None = None,
    max_candidates: int | None = None,
    seed: int = 42,
) -> TuningResult:
    """Search ``param_grid`` using embargoed walk-forward CV, scoring by mean OOS AUC.

    Parameters
    ----------
    param_grid:
        Mapping of hyperparameter name → list of values. Defaults to a small,
        sensible grid for the chosen model.
    max_candidates:
        If set and smaller than the full grid, randomly sample this many
        combinations (random search) for speed.
    """
    target = f"y_dir_{horizon}"
    embargo = horizon if embargo is None else embargo
    param_grid = param_grid or _default_grid(model_name)
    candidates = _param_grid(param_grid)

    if max_candidates is not None and max_candidates < len(candidates):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[i] for i in idx]

    folds = walk_forward_folds(matrix, n_splits=n_splits, embargo=embargo)
    if not folds:
        raise ValueError("Not enough data to build walk-forward folds for tuning.")

    scored: list[dict] = []
    for params in candidates:
        fold_aucs: list[float] = []
        for fold in folds:
            train, test = matrix.loc[fold.train_idx], matrix.loc[fold.test_idx]
            X_tr, y_tr = split_X_y(train, target)
            X_te, y_te = split_X_y(test, target)
            if y_te.astype(int).nunique() < 2:
                continue
            est = _set_params(clone(build_estimator(model_name)), params)
            est.fit(X_tr, y_tr.astype(int))
            prob = est.predict_proba(X_te)[:, 1]
            fold_aucs.append(roc_auc_score(y_te.astype(int), prob))

        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        scored.append({**params, "mean_auc": mean_auc, "n_folds": len(fold_aucs)})
        logger.info("Tuned %s %s → AUC %.4f", model_name, params, mean_auc)

    leaderboard = (
        pd.DataFrame(scored).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    )
    best_row = leaderboard.iloc[0].to_dict()
    best_score = best_row.pop("mean_auc")
    best_row.pop("n_folds", None)

    return TuningResult(
        best_params=best_row,
        best_score=float(best_score),
        leaderboard=leaderboard,
        scored=scored,
    )


def _default_grid(model_name: str) -> dict[str, list]:
    grids: dict[str, dict[str, list]] = {
        "logistic": {"C": [0.1, 1.0, 10.0]},
        "random_forest": {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [10, 20],
        },
        "gradient_boosting": {
            "n_estimators": [200, 300],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.1],
        },
        "xgboost": {
            "n_estimators": [300, 500],
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.1],
        },
        "lightgbm": {
            "n_estimators": [300, 500],
            "num_leaves": [15, 31],
            "learning_rate": [0.03, 0.1],
        },
    }
    return grids.get(model_name, {})
