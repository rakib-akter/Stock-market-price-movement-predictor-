# Development Roadmap

A phased plan. Each phase produces something runnable and testable before moving on.

## Phase 1 — Data + Features (foundation)
**Goal:** a clean, leak-free feature matrix for any ticker.

- [ ] Fetch OHLCV via yfinance with retry/backoff (`data/fetch.py`).
- [ ] Clean: forward-fill gaps sensibly, drop non-trading rows, use adjusted close
      for returns (`data/clean.py`).
- [ ] Build features: returns, log returns, MAs, EMAs, RSI, MACD, Bollinger Bands,
      volatility, volume changes, momentum, lagged returns (`features/`).
- [ ] Add market-relative features (SPY/QQQ, sector ETF) (`features/market.py`).
- [ ] Persist raw + features to the database (`database/`).
- [ ] **Exit criterion:** `scripts/run_pipeline.py --ticker AAPL --stage features`
      writes a feature table with zero NaNs in the modeling window and no future leak.

## Phase 2 — Baseline models
**Goal:** an honest baseline you can beat.

- [ ] Time-ordered train/test split (`data/splits.py`).
- [ ] Logistic Regression baseline with standardized features.
- [ ] Random Forest + Gradient Boosting.
- [ ] XGBoost / LightGBM.
- [ ] Report accuracy, ROC-AUC, precision/recall **vs. a naive "always up" baseline**.
- [ ] **Exit criterion:** documented out-of-sample metrics for each model; pick the
      best by *validation* AUC, not test AUC.

## Phase 3 — Backtesting
**Goal:** turn predictions into a realistic equity curve.

- [ ] Signal generation (long/flat or long/short with a confidence threshold).
- [ ] Transaction costs + slippage (`backtesting/costs.py`).
- [ ] Walk-forward validation (`backtesting/walkforward.py`).
- [ ] Metrics: total return, CAGR, Sharpe, Sortino, max drawdown, win rate.
- [ ] Benchmark vs. buy-and-hold.
- [ ] **Exit criterion:** strategy and benchmark equity curves plotted on one chart;
      metrics reproducible from stored `backtest_results`.

## Phase 4 — Dashboard + API
**Goal:** make the research interactive.

- [ ] FastAPI endpoints: `/tickers`, `/fetch-data`, `/train-model`, `/predict`,
      `/backtest`, `/metrics`.
- [ ] Streamlit dashboard: price chart, indicators, prediction + confidence,
      equity curve, feature importance, recent signals.
- [ ] **Exit criterion:** select a ticker in the UI and see prediction + backtest.

## Phase 5 — Advanced
**Goal:** improve signal quality without fooling yourself.

- [x] Walk-forward model comparison (`models/evaluate.py`) — rank LR/RF/GBM/XGB/LGBM
      on the same OOS predictions by AUC, Brier, and backtest Sharpe.
- [x] Hyperparameter tuning with **purged, embargoed** time-series CV
      (`models/tuning.py`).
- [x] Probability calibration with time-series-safe CV + Brier score / reliability
      table (`models/calibration.py`).
- [x] Multi-asset equal-weight portfolio backtest (`backtesting/portfolio.py`).
- [x] Feature-importance **stability** analysis across walk-forward folds
      (`models/feature_selection.py`) + `/feature-stability` endpoint.
- [x] Position sizing by **confidence** and **volatility targeting**
      (`BacktestConfig.sizing`); calibrated probabilities make confidence honest.
- [x] Calibration wired into the training path and dashboard (toggle + Brier).
- [ ] Ensembling and regime awareness (volatility regimes, trend filters).
- [ ] Optional sequence model (LSTM/Temporal CNN) **only** if it beats the tree
      baseline on walk-forward — never as the default.

## Definition of done (project-level)
A reviewer can clone the repo, run one command, and reproduce: a trained model, an
out-of-sample metric table, and a cost-aware walk-forward equity curve vs. buy-and-hold.
