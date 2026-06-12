"""Transaction-cost and slippage model.

Costs are charged on *position changes* (turnover), not on every bar. A round
trip (enter then exit) therefore pays the one-way cost twice — once on each side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.config import settings


def one_way_cost(
    transaction_cost_bps: float | None = None, slippage_bps: float | None = None
) -> float:
    """Combined one-way cost as a fraction of traded notional."""
    tc = settings.transaction_cost_bps if transaction_cost_bps is None else transaction_cost_bps
    sl = settings.slippage_bps if slippage_bps is None else slippage_bps
    return (tc + sl) / 10_000.0


def turnover_costs(positions: pd.Series, cost: float) -> pd.Series:
    """Cost incurred each bar from changing position.

    ``positions`` is the target exposure per bar in [-1, 1] (or [0, 1] for
    long/flat). Cost on bar ``t`` = |pos[t] - pos[t-1]| * cost.
    """
    turnover = positions.diff().abs().fillna(positions.abs())
    return turnover * cost


def apply_costs(
    gross_returns: pd.Series, positions: pd.Series, cost: float | None = None
) -> tuple[pd.Series, pd.Series]:
    """Return ``(net_returns, costs)`` given gross strategy returns and positions."""
    cost = one_way_cost() if cost is None else cost
    costs = turnover_costs(positions, cost)
    net = gross_returns - costs
    return net.replace([np.inf, -np.inf], np.nan).fillna(0.0), costs
