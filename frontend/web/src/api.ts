// Typed client for the FastAPI backend. All calls go through the Vite proxy at
// "/api", which forwards to http://localhost:8000 in development.

import type {
  BacktestResponse,
  FeatureStabilityResponse,
  Health,
  PortfolioResponse,
  PriceHistory,
  Sizing,
} from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),

  priceHistory: (ticker: string, horizon: number, lookback = 250) =>
    request<PriceHistory>(
      `/price-history/${encodeURIComponent(ticker)}?horizon=${horizon}&lookback=${lookback}`,
    ),

  backtest: (body: {
    ticker: string;
    model_name: string;
    horizon: number;
    n_splits: number;
    allow_short: boolean;
    confidence_threshold: number;
    sizing?: Sizing;
  }) =>
    request<BacktestResponse>("/backtest", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  portfolioBacktest: (body: {
    tickers: string[];
    model_name: string;
    horizon: number;
    n_splits: number;
  }) =>
    request<PortfolioResponse>("/portfolio-backtest", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  featureStability: (ticker: string, model_name: string, horizon: number, n_splits = 5) =>
    request<FeatureStabilityResponse>(
      `/feature-stability/${encodeURIComponent(ticker)}?model_name=${model_name}&horizon=${horizon}&n_splits=${n_splits}`,
    ),
};

// Convert a {iso_date: value} map to a sorted array for charting.
export function curveToSeries(
  curve: Record<string, number>,
  key: string,
): { date: string; [k: string]: number | string }[] {
  return Object.entries(curve)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, value]) => ({ date, [key]: value }));
}

// Merge strategy + benchmark equity curves into one array by date.
export function mergeCurves(
  strategy: Record<string, number>,
  benchmark: Record<string, number>,
): { date: string; strategy: number; benchmark: number }[] {
  const dates = Array.from(
    new Set([...Object.keys(strategy), ...Object.keys(benchmark)]),
  ).sort();
  return dates.map((date) => ({
    date,
    strategy: strategy[date],
    benchmark: benchmark[date],
  }));
}
