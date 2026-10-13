export default function StatusPill({ health, loading, error }) {
  if (loading) {
    return (
      <span className="status-pill">
        <span className="dot" /> Checking…
      </span>
    );
  }
  if (error || !health) {
    return (
      <span className="status-pill bad" title={error || "API unavailable"}>
        <span className="dot" /> Integrity fail
      </span>
    );
  }
  const freeze = health.freeze_status || "UNKNOWN";
  return (
    <span className="status-pill ok" title={`${health.run_id} · ${health.model_version}`}>
      <span className="dot" /> {freeze}
    </span>
  );
}
