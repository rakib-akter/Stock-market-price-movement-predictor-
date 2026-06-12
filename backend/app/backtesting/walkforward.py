"""Walk-forward backtesting: the realistic way to evaluate a strategy.

Instead of a single train/test split, we repeatedly:
    1. train on an expanding (or rolling) window,
    2. predict the next out-of-sample block,
    3. concatenate those out-of-sample predictions,
then backtest the stitched-together OOS signal. Every prediction is therefore made
by a model that never saw that period — exactly how live trading would work.
"""

from __future__ import annotations

import pandas as pd

from backend.app.backtesting.engine import BacktestConfig, BacktestResult, backtest_signals
from backend.app.data.splits import walk_forward_folds
from backend.app.features.pipeline import split_X_y
from backend.app.models.registry import build_estimator
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


def walk_forward_predictions(
    matrix: pd.DataFrame,
    model_name: str = "logistic",
    horizon: int = 1,
    n_splits: int = 5,
    embargo: int | None = None,
    expanding: bool = True,
) -> pd.Series:
    """Return out-of-sample up-probabilities stitched across walk-forward folds."""
    target = f"y_dir_{horizon}"
    embargo = horizon if embargo is None else embargo
    folds = walk_forward_folds(
        matrix, n_splits=n_splits, embargo=embargo, expanding=expanding
    )
    if not folds:
        raise ValueError("Not enough data to build any walk-forward fold.")

    oos = []
    for i, fold in enumerate(folds, start=1):
        train = matrix.loc[fold.train_idx]
        test = matrix.loc[fold.test_idx]
        X_train, y_train = split_X_y(train, target)
        X_test, _ = split_X_y(test, target)

        est = build_estimator(model_name)
        est.fit(X_train, y_train.astype(int))
        proba = est.predict_proba(X_test)[:, 1]
        oos.append(pd.Series(proba, index=X_test.index))
        logger.info("Fold %d/%d  train=%d test=%d", i, len(folds), len(train), len(test))

    return pd.concat(oos).sort_index()


def walk_forward_backtest(
    matrix: pd.DataFrame,
    model_name: str = "logistic",
    horizon: int = 1,
    n_splits: int = 5,
    config: BacktestConfig | None = None,
    expanding: bool = True,
) -> BacktestResult:
    """Run a full walk-forward backtest and return the stitched-OOS result."""
    prob_up = walk_forward_predictions(
        matrix,
        model_name=model_name,
        horizon=horizon,
        n_splits=n_splits,
        expanding=expanding,
    )
    prices = matrix.loc[prob_up.index, ["adj_close"]]
    return backtest_signals(prices, prob_up, config=config)
