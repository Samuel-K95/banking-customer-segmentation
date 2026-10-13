import { Link } from "react-router-dom";
import { getHealth, getModel } from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import { useAsync } from "../hooks/useAsync.js";

export default function HomePage() {
  const { data, error, loading } = useAsync(
    () => Promise.all([getHealth(), getModel()]).then(([health, model]) => ({ health, model })),
    [],
  );

  const health = data?.health;
  const model = data?.model;

  return (
    <section className="hero">
      <div className="hero-inner">
        <p className="hero-brand">Behavioral Banking Intelligence</p>
        <h1>Leakage-safe AE + k=6 on Berka</h1>
        <p className="hero-lead">
          Meaningful customer groups from transaction behavior — compared against traditional
          clustering, profiled for banking applications, and served as a provisional production
          candidate (as-of 1997-12-31, tenure ≥ 365).
        </p>
        <ErrorBanner error={error} />
        {loading && <p className="muted">Connecting to champion API…</p>}
        {health && (
          <p className="muted" style={{ marginBottom: "1.15rem" }}>
            {health.model_version} · {model?.n_clusters || 6} segments · {health.freeze_status}
          </p>
        )}
        <div className="cta-row">
          <Link className="btn" to="/segments">
            View segments
          </Link>
          <Link className="btn ghost" to="/compare">
            Model comparison
          </Link>
          <Link className="btn ghost" to="/score">
            Live demo
          </Link>
        </div>
      </div>
    </section>
  );
}
