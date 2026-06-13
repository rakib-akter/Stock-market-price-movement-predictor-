# Quant Predictor — React SPA

A Vite + React + TypeScript single-page app for the Quant Movement Predictor. It
talks to the FastAPI backend over `/api` (proxied to `http://localhost:8000` in
dev). Charts use [Recharts](https://recharts.org).

## Run (two terminals)

**1. Backend API** (from the repo root). Demo mode is on by default, so this needs
no internet or API keys:

```bash
pip install -r requirements.txt
uvicorn backend.app.api.main:app --reload
```

**2. Frontend** (from `frontend/web`):

```bash
npm install
npm run dev
```

Open the URL Vite prints (default http://localhost:5173).

> If `npm install` hits TLS/certificate errors on a corporate network, run it as
> `npm install --use-system-ca` (Node 18.20+/20.12+/22+) or point npm at your
> system certificate bundle.

## Tabs

| Tab | Backend endpoint |
| --- | --- |
| Overview | `GET /price-history/{ticker}` |
| Prediction | `GET /price-history/{ticker}` (prediction inline) |
| Backtest | `POST /backtest` |
| Portfolio | `POST /portfolio-backtest` |
| Features | `GET /feature-stability/{ticker}` |

The header badge shows **DEMO DATA** (synthetic) or **LIVE DATA** based on the
backend's `/health` response.

## Build

```bash
npm run build      # type-checks then emits static files to dist/
npm run preview    # serve the production build locally
```

## Configuration

- `VITE_API_TARGET` — override the backend URL the dev proxy forwards to
  (default `http://localhost:8000`).

## Notes

This is a **research tool, not financial advice.** In demo mode all data is
synthetic and has no predictive meaning — see [`../../docs/warnings.md`](../../docs/warnings.md).
