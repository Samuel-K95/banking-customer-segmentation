import { Link } from "react-router-dom";
import { getMetrics, getModel } from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import MetricStrip from "../components/MetricStrip.jsx";
import PageShell from "../components/PageShell.jsx";
import { useAsync } from "../hooks/useAsync.js";

function fmt(n, digits = 3) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}

export default function ChampionPage() {
  const { data, error, loading } = useAsync(
    () => Promise.all([getModel(), getMetrics()]).then(([model, metrics]) => ({ model, metrics })),
    [],
  );

  if (loading) return <div className="page loading">Loading…</div>;

  const m = data?.metrics || {};
  const model = data?.model || {};

  return (
    <PageShell
      title="Why this champion"
      lead="Assignment requirement: compare traditional clustering with a representation-learning path, then justify the final segments on quality, stability, and banking usefulness."
    >
      <ErrorBanner error={error} />
      <MetricStrip
        items={[
          { label: "Architecture", value: "AE lat=4 → k=6" },
          { label: "Silhouette", value: fmt(m.silhouette ?? m.holdout_silhouette) },
          { label: "DB index", value: fmt(m.db_index ?? m.holdout_db_index) },
          {
            label: "Seed ARI",
            value: fmt(
              m.stability_ari_ae_latent_kmeans ??
                m.stability_ari ??
                m.seed_ari_mean,
            ),
          },
          {
            label: "Subsample ARI",
            value: fmt(
              m.stability_subsample_ari_ae_latent_kmeans ?? m.subsample_ari_mean,
            ),
          },
          { label: "Freeze", value: model.freeze_status || m.freeze_status || "—" },
        ]}
      />

      <div className="stack">
        <div className="panel">
          <h3>Selected approach</h3>
          <p className="muted" style={{ margin: 0 }}>
            Berka leakage-safe autoencoder + K-Means (<code>{model.run_id || "c_ae_lat4_k6_leakage_safe"}</code>)
            is the served product champion. Same-matrix K-Means, GMM, and hierarchical runs are
            evaluation baselines — not the live model.
          </p>
          <p style={{ margin: "0.75rem 0 0" }}>
            <Link to="/compare">Open quantitative comparison →</Link>
          </p>
        </div>

        <div className="grid-2">
          <div className="panel">
            <h3>What this system is</h3>
            <ul className="prohibitions" style={{ marginTop: 0 }}>
              <li style={{ borderColor: "var(--line)" }}>
                Unsupervised behavioral segments for CRM / service plays
              </li>
              <li>Interpretable personas with evidence-linked applications</li>
              <li>Provisional serving package with integrity contract</li>
            </ul>
          </div>
          <div className="panel">
            <h3>What it is not</h3>
            <ul className="prohibitions">
              <li>Credit scoring, PD/LGD, or auto-approval from segment_id</li>
              <li>A supervised churn / default predictor</li>
              <li>A demographic classifier (age/gender/district not clustering inputs)</li>
            </ul>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
