import { useEffect, useState } from "react";
import { api } from "../api";
import type { Prediction as Pred } from "../types";
import { ErrorBox, Metric, pct, Spinner } from "../ui";

export function Prediction({ ticker, horizon }: { ticker: string; horizon: number }) {
  const [pred, setPred] = useState<Pred | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    api
      .priceHistory(ticker, horizon, 60)
      .then((d) => active && setPred(d.prediction))
      .catch((e) => active && setError(String(e.message ?? e)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [ticker, horizon]);

  if (loading) return <Spinner label="Generating prediction…" />;
  if (error) return <ErrorBox message={error} />;
  if (!pred) return <div className="card muted">No prediction available.</div>;

  const up = pred.direction === "UP";

  return (
    <>
      <div className="card">
        <h3>
          {ticker} — next-{horizon}-day direction
        </h3>
        <div className="sub">
          As of {pred.date} · model <span className="pill">{pred.model}</span>
        </div>
        <div className="pred">
          <div className={`dir ${up ? "up" : "down"}`}>{up ? "▲ UP" : "▼ DOWN"}</div>
          <div className="gauge">
            <div className="muted" style={{ marginBottom: 6 }}>
              Probability of an up move: <b>{pct(pred.prob_up)}</b>
            </div>
            <div className="bar">
              <div className="fill" style={{ width: `${Math.round(pred.prob_up * 100)}%` }} />
            </div>
            <div className="muted" style={{ marginTop: 8 }}>
              Confidence (distance from a coin-flip): <b>{pct(pred.confidence)}</b>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="metrics">
          <Metric label="Direction" value={pred.direction} positive={up} />
          <Metric label="P(up)" value={pct(pred.prob_up)} />
          <Metric label="Confidence" value={pct(pred.confidence)} />
          <Metric label="Horizon" value={`${pred.horizon}d`} />
        </div>
      </div>

      <div className="disclaimer">
        This is a model score, not advice. A confidence number is not a probability of
        profit — even a high-confidence call can lose money after costs. See docs/warnings.md.
      </div>
    </>
  );
}
