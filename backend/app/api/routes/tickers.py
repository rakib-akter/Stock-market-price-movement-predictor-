"""`/tickers` — list configured and stored tickers."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import settings
from backend.app.database.crud import list_stocks
from backend.app.database.db import get_session

router = APIRouter(prefix="/tickers", tags=["tickers"])


@router.get("")
def get_tickers() -> dict:
    """Return default (configured) tickers, benchmarks, and any stored stocks."""
    with get_session() as session:
        stored = [s.ticker for s in list_stocks(session)]
    return {
        "default_tickers": settings.default_tickers,
        "benchmarks": settings.market_benchmarks,
        "stored": stored,
    }
