# Architecture

This document explains the design, the folder layout, and how data flows through
the system.

## Design principles

1. **No look-ahead, ever.** Features at time `t` may only use information available
   at or before the close of bar `t`. Labels look forward; features never do.
2. **Time-ordered everything.** Splits, cross-validation, and backtests respect the
   arrow of time. We never shuffle rows across time.
3. **Reproducibility.** A fixed `RANDOM_SEED`, versioned `model_runs`, and persisted
   artifacts mean any result can be regenerated.
4. **Separation of concerns.** Data, features, models, and backtesting are
   independent layers that communicate through plain DataFrames and the database.

## Layered data flow

```
fetch ──▶ clean ──▶ feature pipeline ──▶ time split ──▶ train ──▶ predict
                                                │                    │
                                                └──────▶ backtest ◀──┘
```

## Folder responsibilities

| Path | Responsibility |
| --- | --- |
| `backend/app/config.py` | Single source of truth for settings, read from `.env`. |
| `backend/app/data/fetch.py` | Download OHLCV from yfinance with retries. |
| `backend/app/data/clean.py` | Handle missing data, adjusted close, dtype hygiene. |
| `backend/app/data/splits.py` | Time-based train/test split + walk-forward folds. |
| `backend/app/data/loader.py` | High-level "give me a clean panel for these tickers". |
| `backend/app/features/returns.py` | Returns, log returns, lags, momentum. |
| `backend/app/features/technical.py` | MA, EMA, RSI, MACD, Bollinger Bands. |
| `backend/app/features/volatility.py` | Rolling/realized volatility, volume changes. |
| `backend/app/features/market.py` | SPY/QQQ + sector ETF relative features. |
| `backend/app/features/pipeline.py` | Assemble the full feature matrix + label. |
| `backend/app/models/base.py` | Model interface + factory. |
| `backend/app/models/registry.py` | Map model names → estimator builders. |
| `backend/app/models/train.py` | Fit a model, evaluate, return metrics. |
| `backend/app/models/predict.py` | Load a model and produce predictions + confidence. |
| `backend/app/models/persistence.py` | Save/load model artifacts with joblib. |
| `backend/app/backtesting/costs.py` | Transaction cost + slippage model. |
| `backend/app/backtesting/metrics.py` | Sharpe, drawdown, win rate, CAGR, etc. |
| `backend/app/backtesting/engine.py` | Turn signals → equity curve → metrics. |
| `backend/app/backtesting/walkforward.py` | Roll training/testing windows forward. |
| `backend/app/database/db.py` | SQLAlchemy engine/session factory. |
| `backend/app/database/models.py` | ORM tables (see `docs/` schema). |
| `backend/app/database/crud.py` | Insert/query helpers. |
| `backend/app/api/main.py` | FastAPI app + router registration. |
| `backend/app/api/routes/` | One module per resource group. |
| `backend/app/api/schemas.py` | Pydantic request/response models. |
| `frontend/streamlit_app.py` | Dashboard UI. |
| `scripts/run_pipeline.py` | End-to-end CLI: fetch → features → train → backtest. |

## Why this stack

- **pandas/numpy** — the lingua franca of quant research.
- **scikit-learn** — consistent estimator API; LR/RF/GBM out of the box.
- **XGBoost/LightGBM** — strong tabular performance; LightGBM is fast on large panels.
- **FastAPI** — typed, async, auto-documented REST API.
- **SQLAlchemy + SQLite** — zero-config local store; swap to Postgres via one env var.
- **Streamlit + Plotly** — fastest path to an interactive research dashboard.

## Extending the system

- **More data sources:** implement another fetcher behind the same DataFrame contract
  in `data/` (e.g. Stooq, Alpha Vantage, Tiingo). The rest of the stack is agnostic.
- **More models:** register a builder in `models/registry.py`; nothing else changes.
- **More features:** add a function returning a DataFrame indexed by date and wire it
  into `features/pipeline.py`.
