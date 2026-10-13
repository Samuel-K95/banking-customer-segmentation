export default function MetricStrip({ items }) {
  if (!items?.length) return null;
  return (
    <div className="metric-strip">
      {items.map((m) => (
        <div className="metric" key={m.label}>
          <span className="label">{m.label}</span>
          <span className="value">{m.value}</span>
        </div>
      ))}
    </div>
  );
}
