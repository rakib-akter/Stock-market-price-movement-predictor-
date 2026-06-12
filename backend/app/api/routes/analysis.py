"""`/compare-models` and `/portfolio-backtest` — Phase-5 research endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.schemas import (
    CompareModelsResponse,
    PortfolioBacktestRequest,
    PortfolioBacktestResponse,
)
from backend.app.service import compare_models_for, portfolio_backtest_for

router = APIRouter(tags=["analysis"])


def _curve_to_dict(curve) -> dict:
    return {
        str(idx.date()) if hasattr(idx, "date") else str(idx): float(v)
        for idx, v in curve.items()
    }


@router.get("/compare-models/{ticker}", response_model=CompareModelsResponse)
def compare_models_route(
    ticker: str,
    horizon: int = Query(1, ge=1, le=21),
    n_splits: int = Query(5, ge=2, le=20),
) -> CompareModelsResponse:
    try:
        table = compare_models_for(ticker, horizon=horizon, n_splits=n_splits)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Comparison failed: {exc}") from exc
    leaderboard = table.reset_index().to_dict(orient="records")
    return CompareModelsResponse(
        ticker=ticker.upper(), horizon=horizon, leaderboard=leaderboard
    )


@router.post("/portfolio-backtest", response_model=PortfolioBacktestResponse)
def portfolio_backtest_route(req: PortfolioBacktestRequest) -> PortfolioBacktestResponse:
    if not req.tickers:
        raise HTTPException(status_code=422, detail="Provide at least one ticker.")
    try:
        result = portfolio_backtest_for(
            req.tickers,
            model_name=req.model_name,
            horizon=req.horizon,
            n_splits=req.n_splits,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Portfolio backtest failed: {exc}") from exc

    return PortfolioBacktestResponse(
        tickers=[t.upper() for t in req.tickers],
        metrics=result.metrics,
        equity_curve=_curve_to_dict(result.equity),
        benchmark_curve=_curve_to_dict(result.benchmark_equity),
    )
