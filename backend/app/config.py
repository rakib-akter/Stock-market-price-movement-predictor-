"""Central application settings.

All configuration is read from environment variables (or a local `.env` file),
so the same code runs locally, in CI, and in production without edits.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Database ----
    database_url: str = "sqlite:///./quant.db"

    # ---- Data ----
    default_tickers: list[str] = Field(default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
    market_benchmarks: list[str] = Field(default=["SPY", "QQQ"])
    data_start_date: str = "2015-01-01"
    data_interval: str = "1d"

    # ---- Demo mode ----
    # When True, the app uses deterministic *synthetic* data instead of live
    # yfinance downloads, so the whole product runs offline with no API keys.
    demo_mode: bool = True
    # When True (and not in demo mode), fall back to demo data if a live fetch
    # fails — so the product never hard-crashes for a user without internet.
    demo_fallback: bool = True

    # ---- Modeling ----
    prediction_horizon_days: int = 1
    test_size: float = 0.2
    random_seed: int = 42

    # ---- Backtesting ----
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_capital: float = 10_000.0

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("default_tickers", "market_benchmarks", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated strings in the env file (e.g. ``AAPL,MSFT``)."""
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return value

    @property
    def cost_per_trade(self) -> float:
        """Combined one-way cost (fraction of notional) from fees + slippage."""
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the settings."""
    return Settings()


settings = get_settings()
