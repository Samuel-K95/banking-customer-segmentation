import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getComparisonBoard } from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import PageShell from "../components/PageShell.jsx";
import { useAsync } from "../hooks/useAsync.js";

function isChampion(row) {
  return String(row.role || "").toLowerCase().includes("product_champion");
}

function fmt(n, digits = 3) {
  if (n == null || n === "" || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}

function shortId(id) {
  return String(id || "")
    .replace(/_leakage_safe|_scaled/g, "")
    .replace(/^c_/, "");
}

export default function ComparePage() {
  const { data, error, loading } = useAsync(() => getComparisonBoard("champion"), []);

  if (loading) return <div className="page loading">Loading comparison…</div>;

  const rows = data?.rows || [];
  const chartRows = rows
    .filter((r) => r.silhouette != null && r.silhouette !== "")
    .map((r) => ({
      name: shortId(r.run_id || r.family),
      silhouette: Number(r.silhouette),
      champion: isChampion(r),
    }));

  return (
    <PageShell
      title="Segmentation comparison"
      lead="Traditional clustering vs representation learning on the same leakage-safe matrix — metrics that support selecting the final champion."
    >
      <ErrorBanner error={error} />

      <div className="panel" style={{ marginBottom: "1rem" }}>
        <h3>How the champion works (AE + K-Means)</h3>
        <div className="pipeline-steps">
          <div className="pipeline-step">
            <span className="pipeline-step__n">1</span>
            <div>
              <strong>Scale the 8 features</strong>
              <p>
                Interpretable Berka features (activity, balance, flows, recency, tenure, loan/card)
                are transformed with the frozen scaler so distances are comparable.
              </p>
            </div>
          </div>
          <div className="pipeline-step">
            <span className="pipeline-step__n">2</span>
            <div>
              <strong>Autoencoder → 4-D latent space</strong>
              <p>
                A neural net compresses the scaled vector into a small latent code that keeps the
                main behavioral patterns and drops noise. Clustering in this space often separates
                personas more cleanly than raw features alone.
              </p>
            </div>
          </div>
          <div className="pipeline-step">
            <span className="pipeline-step__n">3</span>
            <div>
              <strong>K-Means (k = 6)</strong>
              <p>
                Six cluster centers in latent space become the CRM personas (new / loyal basic /
                loan / card / stress / dormant). k=6 was chosen for interpretability and banking
                usefulness — not silhouette score alone.
              </p>
            </div>
          </div>
        </div>
        <p className="muted" style={{ margin: "0.75rem 0 0" }}>
          Traditional baselines below (K-Means / GMM / hierarchical on the same scaled matrix) are
          the comparison the assignment asks for: representation learning vs classical clustering.
        </p>
      </div>

      <div className="panel" style={{ marginBottom: "1rem" }}>
        <h3>Holdout silhouette</h3>
        <div className="chart-wrap chart-wrap--bar">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartRows} margin={{ top: 8, right: 8, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(232,236,232,0.08)" vertical={false} />
              <XAxis
                dataKey="name"
                angle={-20}
                textAnchor="end"
                interval={0}
                height={52}
                tick={{ fill: "#8fa3ab", fontSize: 11 }}
              />
              <YAxis tick={{ fill: "#8fa3ab", fontSize: 11 }} width={40} />
              <Tooltip contentStyle={{ background: "#122027", border: "1px solid #2a3d33" }} />
              <Bar dataKey="silhouette" animationDuration={700}>
                {chartRows.map((r, i) => (
                  <Cell key={i} fill={r.champion ? "#2db8a4" : "#5a7380"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="chart-caption muted">Teal bar = served product champion.</p>
      </div>

      <div className="panel">
        <h3>Evaluation board</h3>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: "28%" }}>Run</th>
                <th>Family</th>
                <th>k</th>
                <th>Silhouette</th>
                <th>DB</th>
                <th>CH</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.run_id} className={isChampion(r) ? "champion-row" : undefined}>
                  <td>
                    <code>{shortId(r.run_id)}</code>
                  </td>
                  <td>{r.family}</td>
                  <td>{r.k}</td>
                  <td>{fmt(r.silhouette)}</td>
                  <td>{fmt(r.db_index)}</td>
                  <td>{fmt(r.ch_index, 0)}</td>
                  <td className="muted">{String(r.role || "").replace(/_/g, " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </PageShell>
  );
}
