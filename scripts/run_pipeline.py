"""End-to-end CLI: fetch → features → train → walk-forward backtest.

Examples
--------
    python -m scripts.run_pipeline --ticker AAPL
    python -m scripts.run_pipeline --ticker MSFT --model xgboost --horizon 5
    python -m scripts.run_pipeline --ticker NVDA --stage features
"""

from __future__ import annotations

import argparse

from backend.app.backtesting.engine import BacktestConfig
from backend.app.backtesting.walkforward import walk_forward_backtest
from backend.app.database.crud import (
    create_model_run,
    save_backtest,
    upsert_price_data,
)
from backend.app.database.db import get_session, init_db
from backend.app.models.persistence import save_model
from backend.app.models.train import train_classifier
from backend.app.service import build_features_for
from backend.app.utils.logging import get_logger

logger = get_logger("pipeline")


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<24} {value:>12.4f}")
        else:
            print(f"  {key:<24} {value!s:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant predictor end-to-end pipeline")
    parser.add_argument("--ticker", required=True, help="Symbol, e.g. AAPL")
    parser.add_argument("--model", default="logistic", help="Model registry name")
    parser.add_argument("--horizon", type=int, default=1, help="Label horizon in days")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward folds")
    parser.add_argument(
        "--stage",
        choices=["features", "train", "backtest", "all"],
        default="all",
        help="How far to run the pipeline",
    )
    args = parser.parse_args()

    init_db()

    # ---- Stage 1: features ----
    bundle = build_features_for(args.ticker, horizon=args.horizon, start=args.start)
    print(f"\nBuilt feature matrix: {bundle.matrix.shape[0]} rows × "
          f"{bundle.matrix.shape[1]} cols for {bundle.ticker}")
    with get_session() as session:
        inserted = upsert_price_data(
            session, bundle.ticker, bundle.matrix[["open", "high", "low", "close", "adj_close", "volume"]]
        )
    print(f"Persisted {inserted} new price rows.")
    if args.stage == "features":
        return

    # ---- Stage 2: train ----
    result = train_classifier(
        bundle.matrix, ticker=bundle.ticker, model_name=args.model, horizon=args.horizon
    )
    path = save_model(result.artifact)
    _print_metrics(f"Hold-out metrics ({args.model})", result.metrics)
    print(f"Saved model → {path}")
    with get_session() as session:
        create_model_run(
            session,
            ticker=bundle.ticker,
            model_name=args.model,
            target=f"y_dir_{args.horizon}",
            horizon=args.horizon,
            metrics=result.metrics,
            artifact_path=str(path),
        )
    if args.stage == "train":
        return

    # ---- Stage 3: walk-forward backtest ----
    bt = walk_forward_backtest(
        bundle.matrix,
        model_name=args.model,
        horizon=args.horizon,
        n_splits=args.n_splits,
        config=BacktestConfig(),
    )
    _print_metrics("Walk-forward backtest (net of costs)", bt.metrics)
    with get_session() as session:
        save_backtest(
            session,
            ticker=bundle.ticker,
            strategy="long_flat",
            total_return=bt.metrics.get("total_return"),
            cagr=bt.metrics.get("cagr"),
            sharpe=bt.metrics.get("sharpe"),
            max_drawdown=bt.metrics.get("max_drawdown"),
            win_rate=bt.metrics.get("win_rate"),
            benchmark_return=bt.metrics.get("benchmark_total_return"),
            metrics=bt.metrics,
        )

    print("\nReminder: backtested performance is NOT a promise of future returns. "
          "See docs/warnings.md.\n")


if __name__ == "__main__":
    main()
