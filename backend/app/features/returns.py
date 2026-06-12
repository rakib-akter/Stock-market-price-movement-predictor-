"""Return-based features: simple/log returns, lags, and momentum.

All features are computed from the adjusted close (``adj_close``) and are strictly
backward-looking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE = "adj_close"


def add_return_features(
    df: pd.DataFrame,
    return_windows: tuple[int, ...] = (1, 5, 10, 21),
    momentum_windows: tuple[int, ...] = (10, 21, 63),
    n_lags: int = 5,
) -> pd.DataFrame:
    """Append return, lag, and momentum features.

    Parameters
    ----------
    return_windows:
        Horizons (in bars) for trailing simple returns, e.g. 5 ≈ one week.
    momentum_windows:
        Horizons for price momentum ``close[t]/close[t-n] - 1``.
    n_lags:
        Number of lagged 1-day returns to include as features.
    """
    out = df.copy()
    price = out[PRICE]

    # 1-day simple and log returns (the building blocks).
    out["ret_1"] = price.pct_change()
    out["log_ret_1"] = np.log(price).diff()

    # Trailing N-day simple returns.
    for w in return_windows:
        if w == 1:
            continue
        out[f"ret_{w}"] = price.pct_change(w)

    # Lagged 1-day returns (yesterday's, the day before's, ...).
    for lag in range(1, n_lags + 1):
        out[f"lag_ret_{lag}"] = out["ret_1"].shift(lag)

    # Momentum (same formula as ret_n but kept separate for clarity/feature naming).
    for w in momentum_windows:
        out[f"mom_{w}"] = price / price.shift(w) - 1.0

    return out
