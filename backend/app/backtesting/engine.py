"""Signal → equity-curve backtester.

Leakage rule (critical)
------------------------
A prediction made using data up to the close of bar ``t`` can only be acted on at
bar ``t+1``. So the position held during bar ``t+1`` is decided at ``t``. We
implement this by shifting the signal forward one bar before multiplying by the
bar's return. Getting this wrong is the #1 way backtests lie.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.app.backtesting.costs import apply_costs, one_way_cost
from backend.app.backtesting.metrics import equity_curve, performance_metrics
from backend.app.config import settings


@dataclass
class BacktestConfig:
    initial_capital: float = settings.initial_capital
    transaction_cost_bps: float = settings.transaction_cost_bps
    slippage_bps: float = settings.slippage_bps
    allow_short: bool = False          # False = long/flat; True = long/short
    confidence_threshold: float = 0.0  # trade only when |edge| exceeds this


@dataclass
class BacktestResult:
    returns: pd.Series          # net strategy returns per bar
    positions: pd.Series        # exposure held each bar, in [-1, 1]
    equity: pd.Series           # strategy equity curve
    benchmark_equity: pd.Series # buy-and-hold equity curve
    metrics: dict
    signals: pd.DataFrame       # per-bar signal/position/return detail


def _signal_to_position(
    prob_up: pd.Series, allow_short: bool, threshold: float
) -> pd.Series:
    """Map probability-of-up into a target position in [-1, 1]."""
    edge = (prob_up - 0.5) * 2.0  # in [-1, 1]
    if allow_short:
        pos = edge.where(edge.abs() >= threshold, 0.0)
        return pos.clip(-1.0, 1.0).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    # Long/flat: go long when prob_up clears the threshold, else flat.
    return (edge >= threshold).astype(float)


def backtest_signals(
    prices: pd.DataFrame,
    prob_up: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest a long/flat (or long/short) strategy from up-probabilities.

    Parameters
    ----------
    prices:
        Frame with an ``adj_close`` column, indexed by date.
    prob_up:
        Predicted probability the next bar is up, aligned to ``prices`` index.
        The signal at ``t`` is acted on at ``t+1`` (handled internally).
    """
    config = config or BacktestConfig()

    df = prices.loc[prob_up.index].copy()
    bar_return = df["adj_close"].pct_change().fillna(0.0)

    # Decide target position from the signal, then SHIFT it forward one bar so we
    # only ever trade on information from the *previous* close. This is the
    # anti-look-ahead step.
    target_pos = _signal_to_position(
        prob_up, config.allow_short, config.confidence_threshold
    )
    positions = target_pos.shift(1).fillna(0.0)

    gross = positions * bar_return
    cost = one_way_cost(config.transaction_cost_bps, config.slippage_bps)
    net, costs = apply_costs(gross, positions, cost)

    strat_equity = equity_curve(net, config.initial_capital)
    bench_equity = equity_curve(bar_return, config.initial_capital)

    metrics = performance_metrics(net, benchmark_returns=bar_return)
    metrics["n_trades"] = int((positions.diff().abs() > 0).sum())
    metrics["avg_exposure"] = float(positions.abs().mean())
    metrics["total_costs"] = float(costs.sum())

    signals = pd.DataFrame(
        {
            "prob_up": prob_up,
            "position": positions,
            "bar_return": bar_return,
            "gross_return": gross,
            "cost": costs,
            "net_return": net,
        }
    )

    return BacktestResult(
        returns=net,
        positions=positions,
        equity=strat_equity,
        benchmark_equity=bench_equity,
        metrics=metrics,
        signals=signals,
    )
