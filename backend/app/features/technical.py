"""Classic technical indicators implemented in pure pandas/numpy.

We avoid TA-Lib so the project installs cleanly everywhere. Formulas follow the
standard definitions (Wilder's RSI/ATR, MACD 12/26/9, Bollinger 20/2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE = "adj_close"


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index in [0, 100]."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_technical_features(
    df: pd.DataFrame,
    sma_windows: tuple[int, ...] = (10, 20, 50, 200),
    bb_window: int = 20,
    bb_k: float = 2.0,
) -> pd.DataFrame:
    """Append trend, MACD, RSI, and Bollinger Band features."""
    out = df.copy()
    price = out[PRICE]

    # Moving averages and a couple of normalized trend features.
    for w in sma_windows:
        out[f"sma_{w}"] = _sma(price, w)
    out["ema_12"] = _ema(price, 12)
    out["ema_26"] = _ema(price, 26)

    if 50 in sma_windows:
        out["price_to_sma_50"] = price / out["sma_50"] - 1.0
    if 20 in sma_windows and 50 in sma_windows:
        out["sma_20_50_cross"] = (out["sma_20"] - out["sma_50"]) / price

    # MACD (12, 26, 9).
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI.
    out["rsi_14"] = rsi(price, 14)

    # Bollinger Bands.
    mid = _sma(price, bb_window)
    std = price.rolling(bb_window).std()
    upper = mid + bb_k * std
    lower = mid - bb_k * std
    out[f"bb_mid_{bb_window}"] = mid
    out[f"bb_upper_{bb_window}"] = upper
    out[f"bb_lower_{bb_window}"] = lower
    # %B: where price sits within the bands. Width: relative band size (vol proxy).
    out["bb_pctb"] = (price - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / mid

    return out
