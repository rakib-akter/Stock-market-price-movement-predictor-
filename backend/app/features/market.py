"""Market-relative features: how the stock behaves vs. indices and its sector.

These features capture whether a move is idiosyncratic or just the whole market
moving, which is often more predictive than raw returns.
"""

from __future__ import annotations

import pandas as pd

PRICE = "adj_close"

# A small, extensible ticker → sector-ETF map. Extend as needed.
SECTOR_ETF: dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "GOOGL": "XLC",
    "META": "XLC",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "JPM": "XLF",
    "XOM": "XLE",
    "JNJ": "XLV",
}


def _benchmark_returns(benchmark: pd.DataFrame) -> pd.Series:
    return benchmark[PRICE].pct_change().rename("bench_ret")


def add_market_features(
    df: pd.DataFrame,
    benchmarks: dict[str, pd.DataFrame] | None = None,
    primary: str = "SPY",
    beta_window: int = 63,
) -> pd.DataFrame:
    """Append market-relative features.

    Parameters
    ----------
    df:
        Single-ticker feature frame (must already contain ``ret_1``).
    benchmarks:
        ``{symbol: cleaned_price_frame}`` for index/sector ETFs (e.g. SPY, QQQ).
        If ``None``, market features are skipped gracefully.
    primary:
        The benchmark used for excess return and beta (default SPY).
    """
    out = df.copy()
    if "ret_1" not in out.columns:
        out["ret_1"] = out[PRICE].pct_change()
    if not benchmarks:
        return out

    aligned_primary: pd.Series | None = None
    for symbol, frame in benchmarks.items():
        bench_ret = _benchmark_returns(frame).reindex(out.index)
        out[f"mkt_ret_1_{symbol}"] = bench_ret
        if symbol == primary:
            aligned_primary = bench_ret

    if aligned_primary is not None:
        out["excess_ret_1"] = out["ret_1"] - aligned_primary

        # Rolling beta = cov(stock, mkt) / var(mkt).
        cov = out["ret_1"].rolling(beta_window).cov(aligned_primary)
        var = aligned_primary.rolling(beta_window).var()
        out[f"beta_{beta_window}"] = cov / var

        # Relative strength: stock vs. market cumulative growth, normalized.
        rs = (1 + out["ret_1"]).cumprod() / (1 + aligned_primary).cumprod()
        out[f"rel_strength_{primary}"] = rs / rs.rolling(beta_window).mean() - 1.0

    return out


def attach_sector_excess(
    df: pd.DataFrame, ticker: str, sector_frame: pd.DataFrame | None
) -> pd.DataFrame:
    """Add ``sector_excess_ret_1`` if a sector ETF frame is available."""
    out = df.copy()
    if sector_frame is None:
        return out
    sector_ret = sector_frame[PRICE].pct_change().reindex(out.index)
    out["sector_excess_ret_1"] = out["ret_1"] - sector_ret
    return out
