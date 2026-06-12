"""Shared pytest fixtures.

We generate a deterministic synthetic OHLCV series so the test suite runs fully
offline (no yfinance calls) and is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    # Geometric random walk with a slight upward drift.
    rets = rng.normal(0.0004, 0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,  # no splits in synthetic data
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """A ~600-row clean synthetic OHLCV frame."""
    return _synthetic_ohlcv()


@pytest.fixture
def benchmark() -> pd.DataFrame:
    """A second synthetic series to act as SPY/QQQ."""
    return _synthetic_ohlcv(seed=21)
