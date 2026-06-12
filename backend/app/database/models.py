"""ORM table definitions.

Schema overview
---------------
stocks            one row per tracked symbol (metadata)
price_data        cleaned OHLCV bars, one row per (stock, date)
features          serialized feature vector per (stock, date) + label
model_runs        one row per training run (params, metrics, artifact path)
predictions       model outputs per (model_run, stock, date) with confidence
backtest_results  summary metrics + equity curve per backtest
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.db import Base


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    prices: Mapped[list["PriceData"]] = relationship(
        back_populates="stock", cascade="all, delete-orphan"
    )


class PriceData(Base):
    __tablename__ = "price_data"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_price_stock_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    date: Mapped[dt.date] = mapped_column(index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    stock: Mapped[Stock] = relationship(back_populates="prices")


class FeatureRow(Base):
    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", "horizon", name="uq_feature_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    date: Mapped[dt.date] = mapped_column(index=True)
    horizon: Mapped[int] = mapped_column(Integer, default=1)
    # Feature vector and label stored as JSON for schema flexibility.
    values: Mapped[dict] = mapped_column(JSON)
    label_dir: Mapped[int | None] = mapped_column(Integer, nullable=True)
    label_fwd_ret: Mapped[float | None] = mapped_column(Float, nullable=True)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(32))
    horizon: Mapped[int] = mapped_column(Integer, default=1)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    backtests: Mapped[list["BacktestResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[dt.date] = mapped_column(index=True)
    horizon: Mapped[int] = mapped_column(Integer, default=1)
    predicted_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[ModelRun] = relationship(back_populates="predictions")


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_runs.id"), index=True, nullable=True
    )
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    strategy: Mapped[str] = mapped_column(String(64), default="long_flat")
    start_date: Mapped[dt.date | None] = mapped_column(nullable=True)
    end_date: Mapped[dt.date | None] = mapped_column(nullable=True)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    cagr: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    equity_curve: Mapped[dict] = mapped_column(JSON, default=dict)  # {iso_date: equity}
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[ModelRun | None] = relationship(back_populates="backtests")
