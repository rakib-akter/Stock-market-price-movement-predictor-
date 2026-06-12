"""`/fetch-data` — download, clean, and persist OHLCV for a ticker."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api.schemas import FetchDataRequest, FetchDataResponse
from backend.app.data.loader import load_price_panel
from backend.app.database.crud import upsert_price_data
from backend.app.database.db import get_session

router = APIRouter(tags=["data"])


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
