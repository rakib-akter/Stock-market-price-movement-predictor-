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
import type { PortfolioResponse } from "../types";
import { ErrorBox, Metric, num, pct, Spinner } from "../ui";

export function Portfolio({
  universe,
  model,
  horizon,
}: {
  universe: string[];
  model: string;
  horizon: number;
}) {
  const [selected, setSelected] = useState<string[]>(universe.slice(0, 3));
  const [nSplits, setNSplits] = useState(4);
  const [result, setResult] = useState<PortfolioResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = (t: string) =>
    setSelected((s) => (s.includes(t) ? s.filter((x) => x !== t) : [...s, t]));

  const run = () => {
    if (selected.length === 0) {
      setError("Select at least one ticker.");
      return;
    }
    setLoading(true);
    setError(null);
    api
      .portfolioBacktest({ tickers: selected, model_name: model, horizon, n_splits: nSplits })
      .then(setResult)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  const m = result?.metrics;
  const series = result ? mergeCurves(result.equity_curve, result.benchmark_curve) : [];

  return (
    <>
      <div className="card">
        <h3>Equal-weight portfolio backtest</h3>
        <div className="sub">
          Long/flat across the selected names vs. an equal-weight buy-and-hold benchmark.
        </div>
        <div className="controls" style={{ gap: 8 }}>
          {universe.map((t) => (
            <label key={t} className="checkbox">
              <input
                type="checkbox"
                checked={selected.includes(t)}
                onChange={() => toggle(t)}
              />
              {t}
            </label>
          ))}
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
          <button onClick={run} disabled={loading}>
            {loading ? "Running…" : `Run (${selected.length} names)`}
          </button>
        </div>
      </div>

      {loading && <Spinner label="Backtesting portfolio…" />}
      {error && <ErrorBox message={error} />}

      {m && (
        <div className="card">
          <h3>Results · {result?.tickers.join(", ")}</h3>
          <div className="metrics">
            <Metric
              label="Total return"
              value={pct(m.total_return)}
              positive={(m.total_return ?? 0) >= 0}
              delta={`${pct(m.excess_total_return)} vs B&H`}
            />
            <Metric label="Sharpe" value={num(m.sharpe)} positive={(m.sharpe ?? 0) >= 0} />
            <Metric label="Max drawdown" value={pct(m.max_drawdown)} />
            <Metric label="Win rate" value={pct(m.win_rate)} />
            <Metric label="# Assets" value={`${m.n_assets ?? 0}`} />
          </div>
        </div>
      )}

      {result && (
        <div className="card">
          <h3>Portfolio equity curve</h3>
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
                stroke="#22d3ee"
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
                name="Portfolio"
              />
              <Line
                dataKey="benchmark"
                stroke="#9aa4b8"
                dot={false}
                strokeWidth={1.5}
                strokeDasharray="5 4"
                isAnimationActive={false}
                name="Equal-weight B&H"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  );
}
