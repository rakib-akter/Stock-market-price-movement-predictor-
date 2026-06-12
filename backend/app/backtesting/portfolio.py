"""Multi-asset portfolio backtest.

Runs the single-name walk-forward strategy on several tickers and combines them
into one portfolio. Capital is split equally across whichever names are active
(long) on each bar, so the portfolio is always fully or partially invested without
leverage. Each leg already accounts for costs and the one-bar execution lag.

This is intentionally simple (equal-weight, long/flat). Volatility targeting and
risk parity are natural Phase-5 extensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.app.backtesting.engine import BacktestConfig, backtest_signals
from backend.app.backtesting.metrics import equity_curve, performance_metrics
from backend.app.backtesting.walkforward import walk_forward_predictions
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioResult:
    returns: pd.Series
    equity: pd.Series
    benchmark_equity: pd.Series
    metrics: dict
    per_asset_returns: pd.DataFrame
    weights: pd.DataFrame


def portfolio_backtest(
    matrices: dict[str, pd.DataFrame],
    model_name: str = "logistic",
    horizon: int = 1,
    n_splits: int = 5,
    config: BacktestConfig | None = None,
) -> PortfolioResult:
    """Equal-weight, long/flat portfolio over several tickers.

    Parameters
    ----------
    matrices:
        ``{ticker: feature_matrix}`` (each from ``build_feature_matrix``).
    """
    config = config or BacktestConfig()
    leg_net: dict[str, pd.Series] = {}
    leg_position: dict[str, pd.Series] = {}
    leg_bench: dict[str, pd.Series] = {}

    for ticker, matrix in matrices.items():
        try:
            prob = walk_forward_predictions(
                matrix, model_name=model_name, horizon=horizon, n_splits=n_splits
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s in portfolio: %s", ticker, exc)
            continue
        prices = matrix.loc[prob.index, ["adj_close"]]
        bt = backtest_signals(prices, prob, config)
        leg_net[ticker] = bt.returns
        leg_position[ticker] = bt.positions
        leg_bench[ticker] = prices["adj_close"].pct_change().fillna(0.0)

    if not leg_net:
        raise RuntimeError("No tickers produced a backtest; portfolio is empty.")

    net = pd.DataFrame(leg_net).sort_index()
    positions = pd.DataFrame(leg_position).reindex(net.index).fillna(0.0)
    bench = pd.DataFrame(leg_bench).reindex(net.index).fillna(0.0)

    # Equal weight across the names that are active (position != 0) each bar.
    active = (positions.abs() > 0).astype(float)
    n_active = active.sum(axis=1).replace(0, pd.NA)
    weights = active.div(n_active, axis=0).fillna(0.0)

    portfolio_ret = (net.fillna(0.0) * weights).sum(axis=1)
    # Benchmark: equal-weight buy-and-hold of the same universe.
    benchmark_ret = bench.mean(axis=1)

    metrics = performance_metrics(portfolio_ret, benchmark_returns=benchmark_ret)
    metrics["n_assets"] = int(net.shape[1])
    metrics["avg_active_names"] = float(active.sum(axis=1).mean())

    return PortfolioResult(
        returns=portfolio_ret,
        equity=equity_curve(portfolio_ret, config.initial_capital),
        benchmark_equity=equity_curve(benchmark_ret, config.initial_capital),
        metrics=metrics,
        per_asset_returns=net,
        weights=weights,
    )
