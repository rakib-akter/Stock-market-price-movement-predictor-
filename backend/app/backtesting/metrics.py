"""Performance metrics computed from a return series / equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.utils.timeutils import TRADING_DAYS_PER_YEAR


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Compound a return series into an equity curve."""
    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    growth = (1.0 + returns.fillna(0.0)).prod()
    if growth <= 0:
        return -1.0
    years = n / periods_per_year
    return float(growth ** (1 / years) - 1.0) if years > 0 else 0.0


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio. ``risk_free`` is a per-period rate."""
    excess = returns.fillna(0.0) - risk_free
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Like Sharpe but penalizes only downside volatility."""
    r = returns.fillna(0.0)
    downside = r[r < 0]
    dstd = downside.std(ddof=1)
    if dstd == 0 or np.isnan(dstd):
        return 0.0
    return float(r.mean() / dstd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve (negative number)."""
    curve = equity_curve(returns)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def win_rate(returns: pd.Series) -> float:
    """Fraction of *active* (non-zero) bars that were positive."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def performance_metrics(
    strat_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Bundle the headline metrics into a dict, optionally vs. a benchmark."""
    metrics = {
        "total_return": total_return(strat_returns),
        "cagr": cagr(strat_returns, periods_per_year),
        "sharpe": sharpe_ratio(strat_returns, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(strat_returns, periods_per_year),
        "max_drawdown": max_drawdown(strat_returns),
        "win_rate": win_rate(strat_returns),
        "volatility_annual": float(
            strat_returns.std(ddof=1) * np.sqrt(periods_per_year)
        ),
        "n_periods": int(len(strat_returns)),
    }
    if benchmark_returns is not None:
        metrics["benchmark_total_return"] = total_return(benchmark_returns)
        metrics["benchmark_sharpe"] = sharpe_ratio(
            benchmark_returns, periods_per_year=periods_per_year
        )
        metrics["excess_total_return"] = (
            metrics["total_return"] - metrics["benchmark_total_return"]
        )
    return metrics
