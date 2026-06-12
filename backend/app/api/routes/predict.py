"""`/predict` — latest movement prediction for a ticker."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.schemas import PredictResponse
from backend.app.service import predict_for

router = APIRouter(tags=["predict"])


@router.get("/predict/{ticker}", response_model=PredictResponse)
def predict(
    ticker: str,
    model_name: str = Query("logistic"),
    horizon: int = Query(1, ge=1, le=21),
) -> PredictResponse:
    try:
        result = predict_for(ticker, model_name=model_name, horizon=horizon)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictResponse(**result)
