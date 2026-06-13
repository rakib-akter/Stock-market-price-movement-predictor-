"""Deterministic synthetic market data for **offline demo mode**.

This lets the entire product (API + frontend) run with zero network access and no
API keys — essential for a one-command demo. Data is generated from a seed derived
from the ticker symbol, so the same ticker always yields the same series (stable
charts across reloads), while different tickers look different.

The output obeys the same contract as the live fetcher
(:func:`backend.app.data.fetch.fetch_ohlcv`): a tz-naive ``DatetimeIndex`` named
``date`` and columns ``open, high, low, close, adj_close, volume``.

⚠️ This is *fabricated* data for UI/demo purposes only. It has no predictive value
and must never be confused with real market data. See ``docs/warnings.md``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from backend.app.config import settings


def _seed_for(ticker: str) -> int:
    """Stable 32-bit seed derived from the ticker symbol."""
    digest = hashlib.sha256(ticker.upper().encode()).hexdigest()
    return int(digest[:8], 16)


def generate_demo_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    periods: int | None = None,
) -> pd.DataFrame:
    """Generate a realistic-looking synthetic OHLCV series for ``ticker``.

    The model is a geometric random walk with slowly-varying drift and volatility
    regimes plus occasional shocks, so charts show trends, pullbacks, and calmer
    vs. choppier periods — enough to make the dashboard feel alive.
    """
    start = start or settings.data_start_date
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    dates = pd.bdate_range(start_ts, end_ts)
    if periods is not None:
        dates = dates[:periods]
    n = len(dates)
    if n == 0:
        raise ValueError(f"No business days between {start} and {end}.")

    rng = np.random.default_rng(_seed_for(ticker))

    # Base price differs per ticker so charts don't all start at 100.
    base_price = float(rng.uniform(25, 400))

    # Slowly varying drift (trend regimes) via a low-frequency sine + noise.
    t = np.arange(n)
    drift = (
        0.0003 * np.sin(2 * np.pi * t / rng.integers(180, 400))
        + rng.normal(0.0002, 0.0001)
    )

    # Volatility regimes: a smoothed random walk in log-vol space.
    raw_vol = rng.normal(0, 1, n)
    smooth_vol = pd.Series(raw_vol).ewm(span=30).mean().to_numpy()
    vol = 0.011 * np.exp(0.35 * (smooth_vol - smooth_vol.mean()))

    daily_ret = rng.normal(drift, vol)

    # Occasional shocks (earnings-like jumps).
    shock_mask = rng.random(n) < 0.01
    daily_ret[shock_mask] += rng.normal(0, 0.05, int(shock_mask.sum()))

    close = base_price * np.exp(np.cumsum(daily_ret))

    # Build OHLC around the close with plausible intrabar ranges.
    intraday = np.abs(rng.normal(0, vol, n))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = close * (1 + rng.normal(0, vol * 0.5, n))
    open_ = np.clip(open_, low, high)

    # Volume loosely scales with absolute returns (more action on big moves).
    base_vol = rng.uniform(1e6, 8e6)
    volume = base_vol * (1 + 4 * np.abs(daily_ret) / vol.mean()) * rng.uniform(0.6, 1.4, n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,  # demo data has no splits/dividends
            "volume": np.round(volume),
        },
        index=dates,
    )
    df.index.name = "date"
    return df
