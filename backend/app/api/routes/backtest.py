"""`/backtest` — run a walk-forward backtest and persist the result."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api.schemas import BacktestRequest, BacktestResponse
from backend.app.database.crud import save_backtest
from backend.app.database.db import get_session
from backend.app.service import backtest_for

router = APIRouter(tags=["backtest"])


def _curve_to_dict(curve) -> dict:
    return {str(idx.date()) if hasattr(idx, "date") else str(idx): float(v)
            for idx, v in curve.items()}


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    try:
        result = backtest_for(
            req.ticker,
            model_name=req.model_name,
            horizon=req.horizon,
            n_splits=req.n_splits,
            allow_short=req.allow_short,
            confidence_threshold=req.confidence_threshold,
            sizing=req.sizing,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Backtest failed: {exc}") from exc

    equity = _curve_to_dict(result.equity)
    benchmark = _curve_to_dict(result.benchmark_equity)

    with get_session() as session:
        save_backtest(
            session,
            ticker=req.ticker.upper(),
            strategy="long_short" if req.allow_short else "long_flat",
            total_return=result.metrics.get("total_return"),
            cagr=result.metrics.get("cagr"),
            sharpe=result.metrics.get("sharpe"),
            max_drawdown=result.metrics.get("max_drawdown"),
            win_rate=result.metrics.get("win_rate"),
            benchmark_return=result.metrics.get("benchmark_total_return"),
            metrics=result.metrics,
            equity_curve=equity,
        )

    return BacktestResponse(
        ticker=req.ticker.upper(),
        model_name=req.model_name,
        metrics=result.metrics,
        equity_curve=equity,
        benchmark_curve=benchmark,
    )
