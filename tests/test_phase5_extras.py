"""Tests for calibrated training, position sizing, and importance stability."""

from __future__ import annotations

import numpy as np

from backend.app.backtesting.engine import BacktestConfig, backtest_signals
from backend.app.features.pipeline import build_feature_matrix
from backend.app.models.feature_selection import (
    importance_stability,
    select_stable_features,
)
from backend.app.models.predict import predict_proba_frame
from backend.app.models.train import train_classifier


def _matrix(ohlcv, horizon=1):
    return build_feature_matrix(ohlcv, horizon=horizon)


# --------------------------------------------------------------------------- #
# Calibrated training path
# --------------------------------------------------------------------------- #
def test_calibrated_training_sets_flag_and_brier(ohlcv):
    matrix = _matrix(ohlcv)
    result = train_classifier(
        matrix, "TEST", model_name="logistic", calibrate=True, calibration_method="sigmoid"
    )
    assert result.artifact.calibrated is True
    assert "brier" in result.metrics
    # Calibrated estimator still produces valid probabilities through the artifact.
    frame = predict_proba_frame(result.artifact, matrix)
    assert ((frame["prob_up"] >= 0) & (frame["prob_up"] <= 1)).all()


def test_uncalibrated_training_flag_false(ohlcv):
    result = train_classifier(_matrix(ohlcv), "TEST", model_name="logistic")
    assert result.artifact.calibrated is False


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #
def test_confidence_sizing_scales_exposure(ohlcv):
    matrix = _matrix(ohlcv)
    close = matrix["adj_close"]
    # A graded probability so confidence sizing differs from binary.
    prob = (0.5 + 0.4 * np.sign(close.pct_change().fillna(0))).clip(0.01, 0.99)
    prices = matrix[["adj_close"]]

    binary = backtest_signals(prices, prob, BacktestConfig(sizing="binary"))
    conf = backtest_signals(prices, prob, BacktestConfig(sizing="confidence"))
    # Confidence exposure is partial, so average absolute exposure is lower.
    assert conf.metrics["avg_exposure"] <= binary.metrics["avg_exposure"] + 1e-9
    assert conf.metrics["sizing"] == "confidence"
    assert (conf.positions.abs() <= 1.0 + 1e-9).all()


def test_vol_target_caps_at_max_leverage(ohlcv):
    matrix = _matrix(ohlcv)
    prob = (matrix["adj_close"].pct_change().fillna(0) > 0).astype(float)
    prices = matrix[["adj_close"]]
    cfg = BacktestConfig(sizing="vol_target", target_annual_vol=0.15, max_leverage=1.0)
    result = backtest_signals(prices, prob, cfg)
    assert (result.positions.abs() <= 1.0 + 1e-9).all()
    assert result.metrics["sizing"] == "vol_target"


# --------------------------------------------------------------------------- #
# Importance stability
# --------------------------------------------------------------------------- #
def test_importance_stability_table(ohlcv):
    matrix = _matrix(ohlcv)
    table = importance_stability(matrix, model_name="random_forest", n_splits=4)
    assert {"mean_importance", "std_importance", "stability", "mean_rank"}.issubset(
        table.columns
    )
    # Sorted by stability descending.
    assert table["stability"].is_monotonic_decreasing
    # Normalized importances sum to ~1 on average across features.
    assert abs(table["mean_importance"].sum() - 1.0) < 0.1


def test_select_stable_features_returns_k(ohlcv):
    feats = select_stable_features(_matrix(ohlcv), model_name="random_forest", top_k=10)
    assert len(feats) == 10
    assert all(isinstance(f, str) for f in feats)
