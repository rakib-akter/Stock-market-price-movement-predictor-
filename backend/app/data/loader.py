"""High-level data loading: fetch + clean in one call, with a tiny disk cache.

The cache lives under ``data/raw`` (git-ignored) as parquet, so re-running the
pipeline during development does not hammer the data source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.config import settings
from backend.app.data.clean import clean_ohlcv
from backend.app.data.demo import generate_demo_ohlcv
from backend.app.data.fetch import fetch_ohlcv
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/raw")


def _cache_path(ticker: str, interval: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{interval}.parquet"


def load_price_panel(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
    refresh: bool = False,
    demo: bool | None = None,
) -> pd.DataFrame:
    """Return a cleaned OHLCV frame for ``ticker``.

    Parameters
    ----------
    demo:
        Force demo (synthetic) data on/off. If ``None``, uses ``settings.demo_mode``.
        Demo mode lets the whole product run offline with no API keys.
    refresh:
        Bypass and overwrite the on-disk cache (ignored in demo mode).
    """
    use_demo = settings.demo_mode if demo is None else demo

    if use_demo:
        logger.info("DEMO MODE — generating synthetic data for %s", ticker)
        return clean_ohlcv(generate_demo_ohlcv(ticker, start=start, end=end))

    path = _cache_path(ticker, interval)
    if use_cache and not refresh and path.exists():
        logger.info("Loading %s from cache %s", ticker, path)
        cached = pd.read_parquet(path)
        return clean_ohlcv(cached)

    try:
        raw = fetch_ohlcv(ticker, start=start, end=end, interval=interval)
    except Exception as exc:  # noqa: BLE001
        if settings.demo_fallback:
            logger.warning(
                "Live fetch for %s failed (%s); falling back to DEMO data.", ticker, exc
            )
            return clean_ohlcv(generate_demo_ohlcv(ticker, start=start, end=end))
        raise

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(path)
        logger.info("Cached %s → %s", ticker, path)

    return clean_ohlcv(raw)


def load_many(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    **kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Load and clean several tickers into ``{ticker: DataFrame}``."""
    panel: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            panel[ticker] = load_price_panel(
                ticker, start=start, end=end, interval=interval, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", ticker, exc)
    return panel
