"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TickerInfo(BaseModel):
    ticker: str
    name: str | None = None
    sector: str | None = None


class FetchDataRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start: str | None = Field(None, examples=["2015-01-01"])
    end: str | None = None
    refresh: bool = False


class FetchDataResponse(BaseModel):
    ticker: str
    rows: int
    start: str | None
    end: str | None
    inserted: int


class TrainRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    model_name: str = Field("logistic", examples=["logistic", "random_forest", "xgboost"])
    horizon: int = Field(1, ge=1, le=21)
    calibrate: bool = Field(
        False, description="Fit time-series-safe probability calibration."
    )


class TrainResponse(BaseModel):
    ticker: str
    model_name: str
    horizon: int
    calibrated: bool = False
    metrics: dict
    artifact_path: str | None = None


class FeatureStabilityResponse(BaseModel):
    ticker: str
    horizon: int
    model_name: str
    features: list[dict]  # one row per feature, most stable first


class PriceHistoryResponse(BaseModel):
    ticker: str
    horizon: int
    demo: bool
    points: list[dict]          # [{date, close, sma_20, sma_50, bb_upper, bb_lower, ...}]
    prediction: dict | None = None


class PredictResponse(BaseModel):
    ticker: str
    date: str
    model: str
    horizon: int
    direction: str
    predicted_class: int
    prob_up: float
    confidence: float


class BacktestRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    model_name: str = "logistic"
    horizon: int = Field(1, ge=1, le=21)
    n_splits: int = Field(5, ge=2, le=20)
    allow_short: bool = False
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
    sizing: Literal["binary", "confidence", "vol_target"] = "binary"


class BacktestResponse(BaseModel):
    ticker: str
    model_name: str
    metrics: dict
    equity_curve: dict  # {iso_date: equity}
    benchmark_curve: dict


class MetricsResponse(BaseModel):
    ticker: str
    latest_run: dict | None = None


class CompareModelsResponse(BaseModel):
    ticker: str
    horizon: int
    leaderboard: list[dict]  # one row per model, best first


class PortfolioBacktestRequest(BaseModel):
    tickers: list[str] = Field(..., examples=[["AAPL", "MSFT", "NVDA"]])
    model_name: str = "logistic"
    horizon: int = Field(1, ge=1, le=21)
    n_splits: int = Field(5, ge=2, le=20)


class PortfolioBacktestResponse(BaseModel):
    tickers: list[str]
    metrics: dict
    equity_curve: dict
    benchmark_curve: dict
