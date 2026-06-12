"""Thin CRUD helpers over the ORM models.

Kept deliberately small — they cover the inserts/queries the API and pipeline
need without becoming a full repository layer.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    BacktestResult,
    ModelRun,
    Prediction,
    PriceData,
    Stock,
)


# ---- Stocks ----------------------------------------------------------------
def get_or_create_stock(
    session: Session, ticker: str, name: str | None = None, sector: str | None = None
) -> Stock:
    stock = session.scalar(select(Stock).where(Stock.ticker == ticker.upper()))
    if stock is None:
        stock = Stock(ticker=ticker.upper(), name=name, sector=sector)
        session.add(stock)
        session.flush()
    return stock


def list_stocks(session: Session) -> list[Stock]:
    return list(session.scalars(select(Stock).order_by(Stock.ticker)))


# ---- Price data ------------------------------------------------------------
def upsert_price_data(session: Session, ticker: str, df: pd.DataFrame) -> int:
    """Insert price rows for a ticker, skipping dates already stored.

    Returns the number of new rows inserted.
    """
    stock = get_or_create_stock(session, ticker)
    existing = {
        d
        for (d,) in session.execute(
            select(PriceData.date).where(PriceData.stock_id == stock.id)
        )
    }
    inserted = 0
    for ts, row in df.iterrows():
        day = ts.date() if isinstance(ts, (pd.Timestamp, dt.datetime)) else ts
        if day in existing:
            continue
        session.add(
            PriceData(
                stock_id=stock.id,
                date=day,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                adj_close=float(row["adj_close"]),
                volume=float(row["volume"]),
            )
        )
        inserted += 1
    return inserted


def load_price_data(session: Session, ticker: str) -> pd.DataFrame:
    """Return stored price data for a ticker as a DataFrame indexed by date."""
    stock = session.scalar(select(Stock).where(Stock.ticker == ticker.upper()))
    if stock is None:
        return pd.DataFrame()
    rows = session.scalars(
        select(PriceData).where(PriceData.stock_id == stock.id).order_by(PriceData.date)
    ).all()
    if not rows:
        return pd.DataFrame()
    data = [
        {
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "adj_close": r.adj_close,
            "volume": r.volume,
        }
        for r in rows
    ]
    return pd.DataFrame(data).set_index("date")


# ---- Model runs ------------------------------------------------------------
def create_model_run(session: Session, **kwargs: object) -> ModelRun:
    run = ModelRun(**kwargs)
    session.add(run)
    session.flush()
    return run


def latest_run(session: Session, ticker: str) -> ModelRun | None:
    return session.scalar(
        select(ModelRun)
        .where(ModelRun.ticker == ticker.upper())
        .order_by(ModelRun.created_at.desc())
    )


# ---- Predictions -----------------------------------------------------------
def save_prediction(session: Session, **kwargs: object) -> Prediction:
    pred = Prediction(**kwargs)
    session.add(pred)
    session.flush()
    return pred


# ---- Backtests -------------------------------------------------------------
def save_backtest(session: Session, **kwargs: object) -> BacktestResult:
    result = BacktestResult(**kwargs)
    session.add(result)
    session.flush()
    return result
