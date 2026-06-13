"""`/fetch-data` and `/price-history` — OHLCV ingestion and chart series."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.schemas import (
    FetchDataRequest,
    FetchDataResponse,
    PriceHistoryResponse,
)
from backend.app.data.loader import load_price_panel
from backend.app.database.crud import upsert_price_data
from backend.app.database.db import get_session
from backend.app.service import price_history_for

router = APIRouter(tags=["data"])


@router.get("/price-history/{ticker}", response_model=PriceHistoryResponse)
def price_history(
    ticker: str,
    horizon: int = Query(1, ge=1, le=21),
    lookback: int = Query(250, ge=20, le=2000),
) -> PriceHistoryResponse:
    """Recent price + indicator series and the latest prediction, for charting."""
    try:
        data = price_history_for(ticker, horizon=horizon, lookback=lookback)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Price history failed: {exc}") from exc
    return PriceHistoryResponse(**data)


@router.post("/fetch-data", response_model=FetchDataResponse)
def fetch_data(req: FetchDataRequest) -> FetchDataResponse:
    try:
        df = load_price_panel(
            req.ticker, start=req.start, end=req.end, refresh=req.refresh
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {req.ticker}")

    with get_session() as session:
        inserted = upsert_price_data(session, req.ticker, df)

    return FetchDataResponse(
        ticker=req.ticker.upper(),
        rows=len(df),
        start=str(df.index.min().date()),
        end=str(df.index.max().date()),
        inserted=inserted,
    )
