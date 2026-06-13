// Small shared presentational components.

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="center">
      <span className="spinner" />
      {label && <span>{label}</span>}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="error">⚠️ {message}</div>;
}

export function Metric({
  label,
  value,
  delta,
  positive,
}: {
  label: string;
  value: string;
  delta?: string;
  positive?: boolean;
}) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {delta && <div className={`delta ${positive ? "pos" : "neg"}`}>{delta}</div>}
    </div>
  );
}

export const pct = (x: number | undefined, digits = 1) =>
  x === undefined || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | undefined, digits = 2) =>
  x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits);
