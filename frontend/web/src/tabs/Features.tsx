import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { FeatureStabilityResponse } from "../types";
import { ErrorBox, num, Spinner } from "../ui";

export function Features({
  ticker,
  model,
  horizon,
}: {
  ticker: string;
  model: string;
  horizon: number;
}) {
  const [data, setData] = useState<FeatureStabilityResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Tree models expose importances; logistic exposes |coef|. Default to a tree.
  const effectiveModel = model === "logistic" ? "random_forest" : model;

  const run = () => {
    setLoading(true);
    setError(null);
    api
      .featureStability(ticker, effectiveModel, horizon, 5)
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  const chartData = data?.features
    .slice(0, 15)
    .map((f) => ({ feature: f.feature, importance: f.mean_importance }))
    .reverse();

  return (
    <>
      <div className="card">
        <h3>Feature-importance stability</h3>
        <div className="sub">
          Importances refit across walk-forward folds — high <b>stability</b> means a feature
          is consistently useful, not a one-fold fluke. Uses{" "}
          <span className="pill">{effectiveModel}</span>.
        </div>
        <button onClick={run} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze features"}
        </button>
      </div>

      {loading && <Spinner label="Refitting across folds…" />}
      {error && <ErrorBox message={error} />}

      {chartData && (
        <div className="card">
          <h3>Top features by mean importance</h3>
          <ResponsiveContainer width="100%" height={420}>
            <BarChart
              layout="vertical"
              data={chartData}
              margin={{ top: 6, right: 16, bottom: 0, left: 24 }}
            >
              <CartesianGrid stroke="#262d3d" horizontal={false} />
              <XAxis type="number" tick={{ fill: "#9aa4b8", fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="feature"
                tick={{ fill: "#9aa4b8", fontSize: 11 }}
                width={120}
              />
              <Tooltip
                contentStyle={{
                  background: "#131722",
                  border: "1px solid #262d3d",
                  borderRadius: 8,
                  color: "#e6e9ef",
                }}
              />
              <Bar dataKey="importance" fill="#3b82f6" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {data && (
        <div className="card">
          <h3>Stability table</h3>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Mean importance</th>
                <th>Std</th>
                <th>Mean rank</th>
                <th>Stability</th>
              </tr>
            </thead>
            <tbody>
              {data.features.map((f) => (
                <tr key={f.feature}>
                  <td>{f.feature}</td>
                  <td>{num(f.mean_importance, 4)}</td>
                  <td>{num(f.std_importance, 4)}</td>
                  <td>{num(f.mean_rank, 1)}</td>
                  <td>{num(f.stability, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!data && !loading && !error && (
        <div className="card muted">Run the analysis to rank features by stability.</div>
      )}
    </>
  );
}
