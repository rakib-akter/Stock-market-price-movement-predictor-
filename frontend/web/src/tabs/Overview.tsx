import { useEffect, useState } from "react";
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { PriceHistory } from "../types";
import { ErrorBox, Metric, pct, Spinner } from "../ui";

export function Overview({ ticker, horizon }: { ticker: string; horizon: number }) {
  const [data, setData] = useState<PriceHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    api
      .priceHistory(ticker, horizon, 250)
      .then((d) => active && setData(d))
      .catch((e) => active && setError(String(e.message ?? e)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [ticker, horizon]);

  if (loading) return <Spinner label={`Loading ${ticker}…`} />;
  if (error) return <ErrorBox message={error} />;
  if (!data) return null;

  const pred = data.prediction;
  const last = data.points[data.points.length - 1];

  return (
    <>
      <div className="card">
        <h3>{ticker} — price &amp; indicators</h3>
        <div className="sub">
          Adjusted close with 20/50-day moving averages and Bollinger Bands ·{" "}
          {data.points.length} sessions
        </div>
        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart data={data.points} margin={{ top: 6, right: 8, bottom: 0, left: 8 }}>
            <CartesianGrid stroke="#262d3d" vertical={false} />
            <XAxis dataKey="date" tick={{ fill: "#9aa4b8", fontSize: 11 }} minTickGap={48} />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fill: "#9aa4b8", fontSize: 11 }}
              width={52}
            />
            <Tooltip
              contentStyle={{
                background: "#131722",
                border: "1px solid #262d3d",
                borderRadius: 8,
                color: "#e6e9ef",
              }}
            />
            <Area
              dataKey="bb_upper"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.06}
              isAnimationActive={false}
              name="BB upper"
            />
            <Area
              dataKey="bb_lower"
              stroke="none"
              fill="#0b0e14"
              fillOpacity={1}
              isAnimationActive={false}
              name="BB lower"
            />
            <Line
              dataKey="close"
              stroke="#22d3ee"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
              name="Close"
            />
            <Line
              dataKey="sma_20"
              stroke="#f59e0b"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
              name="SMA 20"
            />
            <Line
              dataKey="sma_50"
              stroke="#a78bfa"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
              name="SMA 50"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>Snapshot</h3>
        <div className="metrics">
          <Metric label="Last close" value={last?.close?.toFixed(2) ?? "—"} />
          <Metric label="RSI (14)" value={last?.rsi_14?.toFixed(1) ?? "—"} />
          {pred && (
            <>
              <Metric
                label={`Next-${horizon}d signal`}
                value={pred.direction}
                positive={pred.direction === "UP"}
                delta={pred.direction === "UP" ? "bullish" : "bearish"}
              />
              <Metric label="P(up)" value={pct(pred.prob_up)} />
              <Metric label="Confidence" value={pct(pred.confidence)} />
            </>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Volume</h3>
        <ResponsiveContainer width="100%" height={140}>
          <ComposedChart data={data.points} margin={{ top: 6, right: 8, bottom: 0, left: 8 }}>
            <XAxis dataKey="date" hide />
            <YAxis hide />
            <Tooltip
              contentStyle={{
                background: "#131722",
                border: "1px solid #262d3d",
                borderRadius: 8,
                color: "#e6e9ef",
              }}
            />
            <Bar dataKey="volume" fill="#3b82f6" fillOpacity={0.5} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
