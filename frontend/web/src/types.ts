// Shared API response types (mirror backend/app/api/schemas.py).

export interface Health {
  status: string;
  database_url: string;
  demo_mode: boolean;
  default_tickers: string[];
  models: string[];
}

export interface Prediction {
  ticker: string;
  date: string;
  model: string;
  horizon: number;
  prob_up: number;
  predicted_class: number;
  direction: "UP" | "DOWN";
  confidence: number;
}

export interface PricePoint {
  date: string;
  close: number;
  sma_20?: number | null;
  sma_50?: number | null;
  bb_upper?: number | null;
  bb_lower?: number | null;
  rsi_14?: number | null;
  volume?: number | null;
}

export interface PriceHistory {
  ticker: string;
  horizon: number;
  demo: boolean;
  points: PricePoint[];
  prediction: Prediction | null;
}

export interface BacktestMetrics {
  total_return: number;
  cagr: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  win_rate: number;
  volatility_annual: number;
  benchmark_total_return?: number;
  excess_total_return?: number;
  n_trades?: number;
  avg_exposure?: number;
  total_costs?: number;
  sizing?: string;
  n_assets?: number;
  [key: string]: number | string | undefined;
}

export interface BacktestResponse {
  ticker: string;
  model_name: string;
  metrics: BacktestMetrics;
  equity_curve: Record<string, number>;
  benchmark_curve: Record<string, number>;
}

export interface PortfolioResponse {
  tickers: string[];
  metrics: BacktestMetrics;
  equity_curve: Record<string, number>;
  benchmark_curve: Record<string, number>;
}

export interface FeatureStabilityRow {
  feature: string;
  mean_importance: number;
  std_importance: number;
  mean_rank: number;
  stability: number;
  n_folds: number;
}

export interface FeatureStabilityResponse {
  ticker: string;
  horizon: number;
  model_name: string;
  features: FeatureStabilityRow[];
}

export type Sizing = "binary" | "confidence" | "vol_target";
