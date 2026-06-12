# 📈 Quant Stock-Movement Predictor

A **research and backtesting platform** that predicts short-term stock price
*movement* using machine learning and quantitative features.

> ⚠️ **This is a research tool, not financial advice.** The goal is to build a
> rigorous, leak-free ML + backtesting pipeline — **not** to guarantee profit.
> Read [`docs/warnings.md`](docs/warnings.md) before trusting any number this
> repo produces.

---

## 1. Concept

Markets are noisy and close to efficient. A model that is right **53–56% of the
time** on next-day direction — *after* realistic costs — is already a serious
result. This project is built around that reality:

- It treats prediction as a **scientific experiment**, not a money printer.
- Every result must survive **walk-forward backtesting** with transaction costs
  and slippage before it means anything.
- It is honest about **overfitting, look-ahead bias, and data leakage** — the
  three ways nearly every amateur quant project fools itself.

### Prediction targets

| Target | Type | Description | Difficulty |
| --- | --- | --- | --- |
| **Next-day direction** | Binary classification | Will tomorrow's close be higher than today's? | ⭐ (best first target) |
| Next-week direction | Binary classification | Will close in 5 trading days be higher? | ⭐⭐ |
| Percentage return | Regression | Predict the actual N-day forward return | ⭐⭐⭐ |
| Volatility | Regression | Predict N-day realized volatility | ⭐⭐ |

### 👉 Recommended first target: **next-day direction (binary classification)**

Why it is the right starting point for a *serious but beginner-friendly* project:

1. **Clean, balanced labels** — `1` if `close[t+1] > close[t]`, else `0`. No
   threshold tuning to start.
2. **Easy to evaluate** — accuracy, ROC-AUC, precision/recall, and a confusion
   matrix all apply directly.
3. **Directly tradable** — direction maps cleanly to a long/flat (or long/short)
   backtest.
4. **Forces the hard lessons early** — you immediately confront leakage and the
   ~50% baseline, which is exactly the discipline a quant needs.

Once next-day direction works end-to-end, graduate to **return regression** (predict
magnitude, size positions by confidence) and **volatility** (risk targeting).

---

## 2. Architecture

```
              ┌───────────────┐      ┌──────────────────┐
   yfinance ──▶  Data pipeline  ──▶   Feature pipeline    │
              │  (fetch/clean) │      │ (returns, RSI,    │
              └───────┬────────┘      │  MACD, BB, vol…)  │
                      │               └─────────┬─────────┘
                      ▼                         ▼
              ┌──────────────────────────────────────────┐
              │              SQLite / Postgres            │
              │  stocks · price_data · features ·         │
              │  model_runs · predictions · backtests     │
              └───────┬───────────────────────┬───────────┘
                      ▼                         ▼
              ┌───────────────┐        ┌────────────────────┐
              │  ML training  │        │   Backtester        │
              │ (LR/RF/GBM/   │        │ walk-forward, costs │
              │  XGB/LGBM)    │        │ Sharpe, drawdown…   │
              └───────┬───────┘        └─────────┬──────────┘
                      ▼                          ▼
              ┌──────────────────────────────────────────┐
              │           FastAPI service (REST)          │
              └───────────────────┬──────────────────────┘
                                  ▼
                      ┌────────────────────────┐
                      │  Streamlit dashboard    │
                      └────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full design.

---

## 3. Project structure

```
.
├── backend/
│   └── app/
│       ├── config.py            # central settings (pydantic-settings)
│       ├── data/                # fetch, clean, time-aware splits
│       ├── features/            # returns, technical indicators, market features
│       ├── models/              # train / predict / persistence / registry
│       ├── backtesting/         # engine, metrics, costs, walk-forward
│       ├── api/                 # FastAPI app + routes + schemas
│       ├── database/            # SQLAlchemy engine, ORM models, CRUD
│       └── utils/               # logging, time helpers
├── frontend/                    # Streamlit dashboard
├── notebooks/                   # exploratory research
├── scripts/                     # CLI entry points (end-to-end pipeline)
├── tests/                       # pytest suite
└── docs/                        # architecture, roadmap, warnings, data dict
```

Each folder is explained in [`docs/architecture.md`](docs/architecture.md).

---

## 4. Quickstart

```bash
# 1. Create environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

# 2. Install
pip install -r requirements-dev.txt

# 3. Configure
cp .env.example .env

# 4. Run the full pipeline for one ticker (fetch → features → train → backtest)
python -m scripts.run_pipeline --ticker AAPL

# 5. Start the API
uvicorn backend.app.api.main:app --reload

# 6. Start the dashboard (in a second terminal)
streamlit run frontend/streamlit_app.py
```

---

## 5. Development roadmap

| Phase | Goal | Status |
| --- | --- | --- |
| **1** | Data fetching + feature engineering | scaffolded |
| **2** | Baseline ML models (LR → RF → GBM → XGB/LGBM) | scaffolded |
| **3** | Walk-forward backtester with costs & slippage | scaffolded |
| **4** | Streamlit dashboard | scaffolded |
| **5** | Advanced models, tuning, ensembles, risk targeting | planned |

Full breakdown in [`docs/roadmap.md`](docs/roadmap.md).

---

## 6. ⚠️ Read before using

- [`docs/warnings.md`](docs/warnings.md) — overfitting, survivorship bias,
  look-ahead bias, data leakage, unrealistic backtests, and why this is research.
- [`docs/data_dictionary.md`](docs/data_dictionary.md) — every feature defined.

## License

MIT — see [`LICENSE`](LICENSE). Research/educational use only. **Not financial advice.**
