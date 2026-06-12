# Frontend — Streamlit dashboard

A single-file research dashboard (`streamlit_app.py`) that calls the backend
service layer directly.

## Run

```bash
pip install -r ../requirements.txt
streamlit run streamlit_app.py
```

(Run from the repo root so the `backend` package is importable, or set
`PYTHONPATH` to the repo root.)

## What it shows

| Panel | Source |
| --- | --- |
| Price + SMA/Bollinger chart | `build_feature_matrix` |
| Hold-out accuracy / AUC / precision | `train_classifier` |
| Equity curve vs. buy-and-hold | `walk_forward_backtest` |
| Sharpe / drawdown / win rate / trades | backtest metrics |
| Feature importance | model `feature_importances_` or `|coef|` |
| Recent signals table | backtest `signals` frame |

## Swapping in React later

The same numbers are available over REST from the FastAPI service
(`/predict`, `/backtest`, `/metrics`), so a React/Next.js frontend can replace
this dashboard without backend changes.
