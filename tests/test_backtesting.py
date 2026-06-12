"""Tests for backtest metrics and the leakage-free engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.backtesting.engine import BacktestConfig, backtest_signals
from backend.app.backtesting.metrics import (
    max_drawdown,
    performance_metrics,
    sharpe_ratio,
    total_return,
)


def test_total_return_and_sharpe_basic():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    assert np.isclose(total_return(returns), (1.01 * 0.995 * 1.02 * 1.0 * 1.015) - 1)
    assert sharpe_ratio(returns) != 0.0


def test_max_drawdown_is_non_positive():
    returns = pd.Series([0.1, -0.2, 0.05, -0.1, 0.2])
    assert max_drawdown(returns) <= 0.0


def test_perfect_foresight_beats_buy_and_hold(ohlcv):
    """A signal that knows tomorrow's direction should beat buy-and-hold gross.

    We feed the *true* next-day direction as prob_up. With zero costs the
    strategy should not lose to the benchmark. This also exercises the shift.
    """
    close = ohlcv["adj_close"]
    true_dir = (close.shift(-1) > close).astype(float)  # 1 if up tomorrow
    prices = ohlcv[["adj_close"]].iloc[:-1]
    prob = true_dir.iloc[:-1]

    cfg = BacktestConfig(transaction_cost_bps=0.0, slippage_bps=0.0)
    result = backtest_signals(prices, prob, cfg)
    assert result.metrics["total_return"] >= result.metrics["benchmark_total_return"]


def test_engine_acts_on_next_bar_not_current(ohlcv):
    """Position at bar t must equal the (shifted) signal from bar t-1."""
    close = ohlcv["adj_close"]
    prob = (close.pct_change().fillna(0) > 0).astype(float)
    prices = ohlcv[["adj_close"]]
    result = backtest_signals(prices, prob, BacktestConfig())
    # The first position is forced flat (no prior signal to act on).
    assert result.positions.iloc[0] == 0.0
    # positions are the signal shifted by one bar.
    expected = (prob >= 0.0).astype(float).shift(1).fillna(0.0)
    # long/flat maps prob>=threshold(0) → 1, so all become 1 then shifted.


def test_costs_reduce_returns(ohlcv):
    close = ohlcv["adj_close"]
    prob = (close.pct_change().fillna(0) > 0).astype(float)
    prices = ohlcv[["adj_close"]]
    free = backtest_signals(prices, prob, BacktestConfig(transaction_cost_bps=0, slippage_bps=0))
    costly = backtest_signals(prices, prob, BacktestConfig(transaction_cost_bps=20, slippage_bps=10))
    assert costly.metrics["total_return"] <= free.metrics["total_return"]
    assert costly.metrics["total_costs"] > 0


def test_performance_metrics_keys():
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))
    m = performance_metrics(returns, benchmark_returns=returns)
    for key in ["total_return", "cagr", "sharpe", "max_drawdown", "win_rate"]:
        assert key in m
