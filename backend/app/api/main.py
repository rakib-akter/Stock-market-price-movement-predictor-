"""FastAPI application entry point.

Run with::

    uvicorn backend.app.api.main:app --reload

Interactive docs are served at ``/docs`` (Swagger) and ``/redoc``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import backtest, data, models, predict, tickers
from backend.app.config import settings
from backend.app.database.db import init_db
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database…")
    init_db()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Quant Stock-Movement Predictor",
    description=(
        "Research/backtesting API for short-term stock movement prediction. "
        "⚠️ Research tool — not financial advice."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS open for local dashboard development. Tighten for any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickers.router)
app.include_router(data.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(backtest.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "Quant Stock-Movement Predictor",
        "version": "0.1.0",
        "docs": "/docs",
        "disclaimer": "Research tool only. Not financial advice. See docs/warnings.md.",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "database_url": settings.database_url.split("://")[0]}
