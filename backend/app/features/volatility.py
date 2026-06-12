"""Volatility and volume-based features."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE = "adj_close"


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's Average True Range using adjusted OHLC."""
    high = df.get("adj_high", df["high"])
    low = df.get("adj_low", df["low"])
    close = df[PRICE]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_volatility_features(
    df: pd.DataFrame,
    vol_windows: tuple[int, ...] = (10, 21),
    volume_window: int = 5,
) -> pd.DataFrame:
    """Append realized volatility, ATR, and volume-change features.

    Requires ``ret_1`` (call :func:`add_return_features` first).
    """
    out = df.copy()
    if "ret_1" not in out.columns:
        out["ret_1"] = out[PRICE].pct_change()

    for w in vol_windows:
        out[f"vol_{w}"] = out["ret_1"].rolling(w).std()

    if len(vol_windows) >= 2:
        short, long = vol_windows[0], vol_windows[-1]
        out["vol_ratio"] = out[f"vol_{short}"] / out[f"vol_{long}"]

    out["atr_14"] = average_true_range(out, 14)

    # Volume change vs. its own recent average (spikes/contractions).
    vol_ma = out["volume"].rolling(volume_window).mean()
    out[f"vol_chg_{volume_window}"] = out["volume"] / vol_ma - 1.0

    # Dollar volume (liquidity proxy), log-scaled to tame heavy tails.
    out["dollar_vol"] = np.log1p(out[PRICE] * out["volume"])

    return out
