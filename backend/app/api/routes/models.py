"""`/train-model` and `/metrics` — train a model and read its metrics."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.schemas import (
    FeatureStabilityResponse,
    MetricsResponse,
    TrainRequest,
    TrainResponse,
)
from backend.app.database.crud import create_model_run, latest_run
from backend.app.database.db import get_session
from backend.app.models.persistence import artifact_path
from backend.app.service import feature_stability_for, train_and_save

router = APIRouter(tags=["models"])


@router.post("/train-model", response_model=TrainResponse)
def train_model(req: TrainRequest) -> TrainResponse:
    try:
        result = train_and_save(
            req.ticker,
            model_name=req.model_name,
            horizon=req.horizon,
            calibrate=req.calibrate,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Training failed: {exc}") from exc

    target = f"y_dir_{req.horizon}"
    path = str(artifact_path(req.ticker, req.model_name, target))

    # Persist the run for experiment tracking.
    with get_session() as session:
        create_model_run(
            session,
            ticker=req.ticker.upper(),
            model_name=req.model_name,
            target=target,
            horizon=req.horizon,
            metrics=result.metrics,
            artifact_path=path,
        )

    return TrainResponse(
        ticker=req.ticker.upper(),
        model_name=req.model_name,
        horizon=req.horizon,
        calibrated=result.artifact.calibrated,
        metrics=result.metrics,
        artifact_path=path,
    )


@router.get("/feature-stability/{ticker}", response_model=FeatureStabilityResponse)
def feature_stability(
    ticker: str,
    model_name: str = Query("random_forest"),
    horizon: int = Query(1, ge=1, le=21),
    n_splits: int = Query(5, ge=2, le=20),
    top_k: int = Query(20, ge=1, le=100),
) -> FeatureStabilityResponse:
    try:
        table = feature_stability_for(
            ticker, model_name=model_name, horizon=horizon, n_splits=n_splits, top_k=top_k
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Stability analysis failed: {exc}") from exc
    features = table.reset_index(names="feature").to_dict(orient="records")
    return FeatureStabilityResponse(
        ticker=ticker.upper(), horizon=horizon, model_name=model_name, features=features
    )


@router.get("/metrics/{ticker}", response_model=MetricsResponse)
def get_metrics(ticker: str) -> MetricsResponse:
    with get_session() as session:
        run = latest_run(session, ticker)
        payload = (
            {
                "model_name": run.model_name,
                "target": run.target,
                "horizon": run.horizon,
                "metrics": run.metrics,
                "created_at": run.created_at.isoformat(),
            }
            if run
            else None
        )
    return MetricsResponse(ticker=ticker.upper(), latest_run=payload)
