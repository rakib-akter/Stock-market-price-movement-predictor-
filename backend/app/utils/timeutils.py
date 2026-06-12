"""Time and trading-calendar helpers.

Kept deliberately light — we use pandas' business-day logic rather than a full
exchange calendar so the project has no heavy dependencies. Swap in
``pandas_market_calendars`` later if precise holidays matter.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a sorted, tz-naive ``DatetimeIndex``."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def annualization_factor(periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Square-root-of-time factor used to annualize volatility/Sharpe."""
    return float(periods_per_year) ** 0.5


def today_iso() -> str:
    """Today's date as an ISO ``YYYY-MM-DD`` string."""
    return dt.date.today().isoformat()
