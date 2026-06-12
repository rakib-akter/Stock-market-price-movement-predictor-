"""Assemble the full feature matrix and the forward-looking labels.

This is the only module allowed to create columns that reference the future, and
it does so exclusively for the label (``y_*``) columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.features.market import add_market_features
from backend.app.features.returns import add_return_features
from backend.app.features.technical import add_technical_features
from backend.app.features.volatility import add_volatility_features

PRICE = "adj_close"

# Raw OHLCV columns are never used directly as model features (they are
# non-stationary price levels). They stay in the frame for charting/backtesting.
_NON_FEATURE_COLS = {
    "open", "high", "low", "close", "adj_close", "volume",
    "adj_open", "adj_high", "adj_low",
}


def make_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Append forward-looking labels for the given horizon (in bars).

    Adds:
        * ``y_dir_{h}``    — 1 if price rises over the next ``h`` bars, else 0
        * ``y_fwd_ret_{h}``— forward simple return over ``h`` bars
        * ``y_fwd_vol_{h}``— realized vol of the next ``h`` daily returns
    """
    out = df.copy()
    price = out[PRICE]

    fwd_ret = price.shift(-horizon) / price - 1.0
    out[f"y_fwd_ret_{horizon}"] = fwd_ret
    out[f"y_dir_{horizon}"] = (fwd_ret > 0).astype("Int64")

    daily_ret = price.pct_change()
    # Forward realized vol: std of returns over the *next* h bars. A window of at
    # least 2 is required for std to be defined (rolling(1).std() is always NaN).
    vol_window = max(horizon, 2)
    out[f"y_fwd_vol_{horizon}"] = (
        daily_ret.shift(-horizon).rolling(vol_window).std()
    )
    return out


def build_feature_matrix(
    price_df: pd.DataFrame,
    horizon: int = 1,
    benchmarks: dict[str, pd.DataFrame] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Run the full feature pipeline on a cleaned price frame.

    Returns a frame containing raw OHLCV (for plotting/backtest), all features,
    and the label columns. Rows with NaNs from indicator warm-up or the unknown
    forward label are dropped when ``dropna`` is True.
    """
    df = add_return_features(price_df)
    df = add_technical_features(df)
    df = add_volatility_features(df)
    df = add_market_features(df, benchmarks=benchmarks)
    df = make_labels(df, horizon=horizon)

    df = df.replace([np.inf, -np.inf], np.nan)
    if dropna:
        df = df.dropna()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model-input columns (exclude raw prices and labels)."""
    return [
        c
        for c in df.columns
        if c not in _NON_FEATURE_COLS and not c.startswith("y_")
    ]


def split_X_y(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a built matrix into feature frame ``X`` and target series ``y``."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in matrix. Available y_*: "
                       f"{[c for c in df.columns if c.startswith('y_')]}")
    cols = feature_columns(df)
    X = df[cols].astype(float)
    y = df[target]
    return X, y
