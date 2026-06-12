"""Orchestration layer tying data → features → model → backtest together.

The API routes and the CLI both call into this module so the end-to-end logic
lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.app.backtesting.engine import BacktestConfig, BacktestResult
from backend.app.backtesting.walkforward import walk_forward_backtest
from backend.app.config import settings
from backend.app.data.loader import load_price_panel
from backend.app.features.pipeline import build_feature_matrix
from backend.app.models.persistence import (
    ModelArtifact,
    artifact_path,
    load_model,
    save_model,
)
from backend.app.models.predict import predict_latest
from backend.app.models.train import TrainResult, train_classifier
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureBundle:
    ticker: str
    horizon: int
    matrix: pd.DataFrame


def build_features_for(
    ticker: str,
    horizon: int | None = None,
    start: str | None = None,
    end: str | None = None,
    with_benchmarks: bool = True,
) -> FeatureBundle:
    """Fetch, clean, and engineer the feature matrix for one ticker."""
    horizon = horizon or settings.prediction_horizon_days
    prices = load_price_panel(ticker, start=start, end=end)

    benchmarks: dict[str, pd.DataFrame] | None = None
    if with_benchmarks:
        benchmarks = {}
        for sym in settings.market_benchmarks:
            try:
                benchmarks[sym] = load_price_panel(sym, start=start, end=end)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Benchmark %s unavailable: %s", sym, exc)
        benchmarks = benchmarks or None

    matrix = build_feature_matrix(prices, horizon=horizon, benchmarks=benchmarks)
    logger.info("Built %d feature rows for %s", len(matrix), ticker)
    return FeatureBundle(ticker=ticker.upper(), horizon=horizon, matrix=matrix)


def train_and_save(
    ticker: str,
    model_name: str = "logistic",
    horizon: int | None = None,
    start: str | None = None,
) -> TrainResult:
    """Build features, train a classifier, persist the artifact, return results."""
    bundle = build_features_for(ticker, horizon=horizon, start=start)
    result = train_classifier(
        bundle.matrix, ticker=ticker, model_name=model_name, horizon=bundle.horizon
    )
    path = save_model(result.artifact)
    logger.info("Saved model artifact → %s", path)
    return result


def load_or_train(
    ticker: str, model_name: str = "logistic", horizon: int | None = None
) -> ModelArtifact:
    """Load a saved artifact if present, otherwise train and save one."""
    horizon = horizon or settings.prediction_horizon_days
    target = f"y_dir_{horizon}"
    path = artifact_path(ticker, model_name, target)
    if path.exists():
        return load_model(path)
    return train_and_save(ticker, model_name=model_name, horizon=horizon).artifact


def predict_for(
    ticker: str, model_name: str = "logistic", horizon: int | None = None
) -> dict:
    """Return the latest prediction for a ticker (training on demand if needed)."""
    bundle = build_features_for(ticker, horizon=horizon)
    artifact = load_or_train(ticker, model_name=model_name, horizon=bundle.horizon)
    return predict_latest(artifact, bundle.matrix)


def backtest_for(
    ticker: str,
    model_name: str = "logistic",
    horizon: int | None = None,
    n_splits: int = 5,
    allow_short: bool = False,
    confidence_threshold: float = 0.0,
) -> BacktestResult:
    """Run a walk-forward backtest for one ticker end-to-end."""
    bundle = build_features_for(ticker, horizon=horizon)
    config = BacktestConfig(
        allow_short=allow_short, confidence_threshold=confidence_threshold
    )
    return walk_forward_backtest(
        bundle.matrix,
        model_name=model_name,
        horizon=bundle.horizon,
        n_splits=n_splits,
        config=config,
    )


def compare_models_for(
    ticker: str,
    model_names: list[str] | None = None,
    horizon: int | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare several models on one ticker via walk-forward OOS metrics."""
    from backend.app.models.evaluate import compare_models

    bundle = build_features_for(ticker, horizon=horizon)
    return compare_models(
        bundle.matrix,
        model_names=model_names,
        horizon=bundle.horizon,
        n_splits=n_splits,
    )


def portfolio_backtest_for(
    tickers: list[str],
    model_name: str = "logistic",
    horizon: int | None = None,
    n_splits: int = 5,
):
    """Run an equal-weight portfolio backtest across several tickers."""
    from backend.app.backtesting.portfolio import portfolio_backtest

    matrices = {
        t: build_features_for(t, horizon=horizon).matrix for t in tickers
    }
    return portfolio_backtest(
        matrices, model_name=model_name, horizon=horizon or settings.prediction_horizon_days,
        n_splits=n_splits,
    )
