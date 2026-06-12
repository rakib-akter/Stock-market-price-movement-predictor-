"""Download OHLCV market data from a free source (yfinance).

The rest of the codebase only depends on the *shape* of the returned DataFrame,
so an alternative source (Stooq, Tiingo, Alpha Vantage) can be dropped in behind
this same contract.

Contract — :func:`fetch_ohlcv` returns a DataFrame with:
    * a tz-naive ``DatetimeIndex`` named ``date``
    * columns: ``open, high, low, close, adj_close, volume`` (lowercase)
"""

from __future__ import annotations

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.config import settings
from backend.app.utils.logging import get_logger
from backend.app.utils.timeutils import ensure_datetime_index

logger = get_logger(__name__)

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV for a single ticker.

    Parameters
    ----------
    ticker:
        Symbol, e.g. ``"AAPL"``.
    start, end:
        ISO date strings. ``start`` defaults to ``settings.data_start_date``.
    interval:
        Bar size (``"1d"``, ``"1h"`` …). Defaults to ``settings.data_interval``.
    """
    import yfinance as yf

    start = start or settings.data_start_date
    interval = interval or settings.data_interval

    logger.info("Fetching %s [%s → %s, %s]", ticker, start, end or "today", interval)

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   # keep raw close AND adj close so we control adjustment
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance returns a MultiIndex column frame for single tickers in some versions.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=_COLUMN_MAP)
    expected = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for {ticker}: {missing}")

    df = df[expected]
    df.index.name = "date"
    return ensure_datetime_index(df)


def fetch_many(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch several tickers, returning ``{ticker: DataFrame}``.

    Failures are logged and skipped so one bad symbol does not abort the batch.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            out[ticker] = fetch_ohlcv(ticker, start=start, end=end, interval=interval)
        except Exception as exc:  # noqa: BLE001 - we want to continue the batch
            logger.warning("Failed to fetch %s: %s", ticker, exc)
    return out
