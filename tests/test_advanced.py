"""Tests for Phase-5 modules: comparison, calibration, tuning, portfolio."""

from __future__ import annotations

import pandas as pd

from backend.app.backtesting.portfolio import portfolio_backtest
from backend.app.features.pipeline import build_feature_matrix, split_X_y
from backend.app.models.calibration import (
    brier_score,
    calibrate_estimator,
    reliability_table,
)
from backend.app.models.evaluate import compare_models
from backend.app.models.tuning import tune_hyperparameters


def _matrix(ohlcv, horizon=1):
    return build_feature_matrix(ohlcv, horizon=horizon)


def test_compare_models_returns_ranked_table(ohlcv):
    matrix = _matrix(ohlcv)
    table = compare_models(
        matrix, model_names=["logistic", "random_forest"], n_splits=4
    )
    assert set(["accuracy", "brier", "bt_sharpe"]).issubset(table.columns)
    assert len(table) == 2
    # Sorted by the ranking column (descending) — first row is the best.
    rank_col = "roc_auc" if "roc_auc" in table.columns else "accuracy"
    assert table[rank_col].is_monotonic_decreasing


def test_calibration_produces_valid_probabilities(ohlcv):
    matrix = _matrix(ohlcv)
    X, y = split_X_y(matrix, "y_dir_1")
    # Use a chronological holdout so calibration never sees the tail.
    cut = int(len(X) * 0.8)
    model = calibrate_estimator("logistic", X.iloc[:cut], y.iloc[:cut], method="sigmoid", n_splits=4)
    prob = pd.Series(model.predict_proba(X.iloc[cut:])[:, 1], index=X.index[cut:])
    assert ((prob >= 0) & (prob <= 1)).all()
    score = brier_score(y.iloc[cut:], prob)
    assert 0.0 <= score <= 1.0


def test_reliability_table_shape(ohlcv):
    matrix = _matrix(ohlcv)
    X, y = split_X_y(matrix, "y_dir_1")
    model = calibrate_estimator("logistic", X, y, method="sigmoid", n_splits=4)
    prob = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    table = reliability_table(y, prob, n_bins=5)
    assert {"mean_predicted", "observed_rate", "count"}.issubset(table.columns)
    assert table["count"].sum() == len(y)


def test_tuning_returns_best_params(ohlcv):
    matrix = _matrix(ohlcv)
    result = tune_hyperparameters(
        matrix,
        model_name="random_forest",
        param_grid={"n_estimators": [100, 200], "max_depth": [3, 5]},
        n_splits=3,
    )
    assert "n_estimators" in result.best_params
    assert "max_depth" in result.best_params
    assert len(result.leaderboard) == 4  # 2 x 2 grid
    # Leaderboard is sorted best-first.
    assert result.leaderboard["mean_auc"].is_monotonic_decreasing


def test_portfolio_backtest_combines_assets(ohlcv, benchmark):
    matrices = {"AAA": _matrix(ohlcv), "BBB": _matrix(benchmark)}
    result = portfolio_backtest(matrices, model_name="logistic", n_splits=4)
    assert result.metrics["n_assets"] == 2
    assert len(result.equity) > 0
    # Weights never exceed 1 in total (no leverage) and are non-negative (long/flat).
    weight_sums = result.weights.sum(axis=1)
    assert (weight_sums <= 1.0 + 1e-9).all()
    assert (result.weights.values >= 0).all()
