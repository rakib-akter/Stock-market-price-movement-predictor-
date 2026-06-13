import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { Backtest } from "./tabs/Backtest";
import { Features } from "./tabs/Features";
import { Overview } from "./tabs/Overview";
import { Portfolio } from "./tabs/Portfolio";
import { Prediction } from "./tabs/Prediction";
import type { Health } from "./types";
import { ErrorBox, Spinner } from "./ui";

const TABS = ["Overview", "Prediction", "Backtest", "Portfolio", "Features"] as const;
type Tab = (typeof TABS)[number];

const FALLBACK_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"];
const FALLBACK_MODELS = ["logistic", "random_forest", "gradient_boosting"];

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [ticker, setTicker] = useState("AAPL");
  const [model, setModel] = useState("logistic");
  const [horizon, setHorizon] = useState(1);
  const [tab, setTab] = useState<Tab>("Overview");

  useEffect(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        if (h.default_tickers?.length) setTicker(h.default_tickers[0]);
      })
      .catch((e) => setHealthError(String(e.message ?? e)));
  }, []);

  const tickers = health?.default_tickers?.length ? health.default_tickers : FALLBACK_TICKERS;
  const models = health?.models?.length ? health.models : FALLBACK_MODELS;

  const body = useMemo(() => {
    switch (tab) {
      case "Overview":
        return <Overview ticker={ticker} horizon={horizon} />;
      case "Prediction":
        return <Prediction ticker={ticker} horizon={horizon} />;
      case "Backtest":
        return <Backtest ticker={ticker} model={model} horizon={horizon} />;
      case "Portfolio":
        return <Portfolio universe={tickers} model={model} horizon={horizon} />;
      case "Features":
        return <Features ticker={ticker} model={model} horizon={horizon} />;
    }
  }, [tab, ticker, model, horizon, tickers]);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo">📈</div>
          <div>
            <h1>Quant Movement Predictor</h1>
            <small>Research dashboard · not financial advice</small>
          </div>
        </div>
        {health ? (
          <span className={`badge ${health.demo_mode ? "demo" : "live"}`}>
            {health.demo_mode ? "DEMO DATA (synthetic)" : "LIVE DATA"}
          </span>
        ) : (
          <span className="badge">connecting…</span>
        )}
      </header>

      {healthError && (
        <ErrorBox
          message={`Cannot reach the API at /api — start it with "uvicorn backend.app.api.main:app --reload". (${healthError})`}
        />
      )}

      <div className="controls">
        <div className="field">
          <label>Ticker</label>
          <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Custom symbol</label>
          <input
            type="text"
            placeholder="e.g. TSLA"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                const v = (e.target as HTMLInputElement).value.trim().toUpperCase();
                if (v) setTicker(v);
              }
            }}
          />
        </div>
        <div className="field">
          <label>Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Horizon: {horizon}d</label>
          <input
            type="range"
            min={1}
            max={21}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </div>
      </div>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t}
            className={`tab ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </nav>

      {!health && !healthError ? <Spinner label="Connecting to API…" /> : body}
    </div>
  );
}
