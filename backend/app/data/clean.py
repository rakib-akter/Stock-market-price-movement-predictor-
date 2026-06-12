"""Clean raw OHLCV data.

Key decisions:
    * **Use adjusted close for returns.** Splits and dividends create artificial
      jumps in the raw close. We rescale OHLC by the adj_close/close ratio so the
      whole bar is dividend/split-adjusted and internally consistent.
    * **Do not forward-fill prices blindly.** Filling a price the market never
      printed is a subtle form of fabricating data. We drop fully-empty rows and
      only fill tiny interior gaps (≤ ``max_gap`` days) in volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.utils.logging import get_logger
from backend.app.utils.timeutils import ensure_datetime_index

logger = get_logger(__name__)

_PRICE_COLS = ["open", "high", "low", "close"]


def clean_ohlcv(df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """Return a cleaned, split/dividend-adjusted OHLCV frame.

    Parameters
    ----------
    df:
        Raw frame from :func:`backend.app.data.fetch.fetch_ohlcv`.
    max_gap:
        Maximum number of consecutive missing volume values to interpolate.
    """
    out = ensure_datetime_index(df)

    # Drop exact duplicate timestamps, keeping the last observation.
    out = out[~out.index.duplicated(keep="last")]

    # Rows where we have no usable close are useless — drop them.
    out = out.dropna(subset=["close", "adj_close"], how="any")

    # Adjust OHLC to be consistent with the adjusted close.
    ratio = out["adj_close"] / out["close"]
    for col in _PRICE_COLS:
        out[col] = out[col] * ratio
    out["adj_open"] = out["open"]
    out["adj_high"] = out["high"]
    out["adj_low"] = out["low"]

    # Volume: small interior gaps interpolated; long gaps left as NaN then zeroed.
    out["volume"] = (
        out["volume"]
        .interpolate(method="linear", limit=max_gap, limit_area="inside")
        .fillna(0.0)
    )

    # Guardrails: prices must be positive and finite.
    out = out.replace([np.inf, -np.inf], np.nan)
    bad = (out[_PRICE_COLS] <= 0).any(axis=1)
    if bad.any():
        logger.warning("Dropping %d rows with non-positive prices", int(bad.sum()))
        out = out[~bad]

    out = out.dropna(subset=_PRICE_COLS)
    return out


def add_missing_data_report(df: pd.DataFrame) -> pd.Series:
    """Return per-column NaN counts — useful for sanity checks and tests."""
    return df.isna().sum().rename("n_missing")
