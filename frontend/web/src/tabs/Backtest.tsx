import { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, mergeCurves } from "../api";
import type { BacktestResponse, Sizing } from "../types";
import { ErrorBox, Metric, num, pct, Spinner } from "../ui";

export function Backtest({
  ticker,
  model,
  horizon,
}: {
  ticker: string;
  model: string;
  horizon: number;
}) {
  const [nSplits, setNSplits] = useState(5);
  const [sizing, setSizing] = useState<Sizing>("binary");
  const [allowShort, setAllowShort] = useState(false);
  const [threshold, setThreshold] = useState(0);
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setLoading(true);
    setError(null);
    api
      .backtest({
        ticker,
        model_name: model,
        horizon,
        n_splits: nSplits,
        allow_short: allowShort,
        confidence_threshold: threshold,
        sizing,
      })
      .then(setResult)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  const m = result?.metrics;
  const series = result ? mergeCurves(result.equity_curve, result.benchmark_curve) : [];

  return (
    <>
      <div className="card">
        <h3>Walk-forward backtest</h3>
        <div className="sub">
          Out-of-sample, cost-aware, leakage-free. Strategy vs. buy-and-hold.
        </div>
        <div className="controls">
          <div className="field">
            <label>Folds</label>
            <input
              type="number"
              min={2}
              max={20}
              value={nSplits}
              onChange={(e) => setNSplits(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>Sizing</label>
            <select value={sizing} onChange={(e) => setSizing(e.target.value as Sizing)}>
              <option value="binary">binary</option>
              <option value="confidence">confidence</option>
              <option value="vol_target">vol_target</option>
            </select>
          </div>
          <div className="field">
            <label>Confidence threshold: {threshold.toFixed(2)}</label>
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={allowShort}
              onChange={(e) => setAllowShort(e.target.checked)}
            />
            Allow shorts
          </label>
          <button onClick={run} disabled={loading}>
            {loading ? "Running…" : "Run backtest"}
          </button>
        </div>
      </div>

      {loading && <Spinner label="Running walk-forward folds…" />}
      {error && <ErrorBox message={error} />}

      {m && (
        <div className="card">
          <h3>Results</h3>
          <div className="sub">
            sizing <span className="pill">{m.sizing}</span> · avg exposure{" "}
            {num(m.avg_exposure)}
          </div>
          <div className="metrics">
            <Metric
              label="Total return"
              value={pct(m.total_return)}
              positive={(m.total_return ?? 0) >= 0}
              delta={`${pct(m.excess_total_return)} vs B&H`}
            />
            <Metric label="CAGR" value={pct(m.cagr)} />
            <Metric label="Sharpe" value={num(m.sharpe)} positive={(m.sharpe ?? 0) >= 0} />
            <Metric label="Max drawdown" value={pct(m.max_drawdown)} />
            <Metric label="Win rate" value={pct(m.win_rate)} />
            <Metric label="# Trades" value={`${m.n_trades ?? 0}`} />
          </div>
        </div>
      )}

      {result && (
        <div className="card">
          <h3>Equity curve</h3>
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={series} margin={{ top: 6, right: 8, bottom: 0, left: 8 }}>
              <CartesianGrid stroke="#262d3d" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#9aa4b8", fontSize: 11 }} minTickGap={56} />
              <YAxis tick={{ fill: "#9aa4b8", fontSize: 11 }} width={64} />
              <Tooltip
                contentStyle={{
                  background: "#131722",
                  border: "1px solid #262d3d",
                  borderRadius: 8,
                  color: "#e6e9ef",
                }}
              />
              <Legend />
              <Line
                dataKey="strategy"
                stroke="#22c55e"
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
                name="Strategy"
              />
              <Line
                dataKey="benchmark"
                stroke="#9aa4b8"
                dot={false}
                strokeWidth={1.5}
                strokeDasharray="5 4"
                isAnimationActive={false}
                name="Buy & Hold"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {!result && !loading && !error && (
        <div className="card muted">Configure and run a backtest to see results.</div>
      )}
    </>
  );
}
