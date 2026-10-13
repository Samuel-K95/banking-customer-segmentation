import { Link } from "react-router-dom";
import { getMetrics, getModel, getSegments } from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import PageShell from "../components/PageShell.jsx";
import { useAsync } from "../hooks/useAsync.js";

const FEATURE_BLURBS = {
  Transaction_Count: "How often the account moves money — raw activity level.",
  Account_Balance: "Typical balance level, including overdraft / stress signals.",
  Total_Credit: "Inflows (credits) observed up to the as-of date.",
  Total_Withdrawal: "Outflows (withdrawals / debits) up to the as-of date.",
  Recency_Days: "Days since last transaction — dormancy signal.",
  Account_Age_Days: "Tenure / lifecycle stage of the account.",
  has_loan: "Whether a loan relationship exists as of the freeze date.",
  has_card: "Whether a card relationship exists as of the freeze date.",
};

const DATASETS = [
  {
    name: "Indian Bank transactions",
    track: "Track A",
    status: "Scale benchmark",
    whyNot:
      "Huge population (~838k customers), but ~83% have only one transaction and the window is short (~82 days). Weak repeated-behavior signal for CRM personas.",
  },
  {
    name: "Santander product portfolio",
    track: "Track B",
    status: "Product-flag alternative",
    whyNot:
      "24 binary product holdings at a snapshot — useful for portfolio mix, not for activity, balance, recency, or flow-based behavioral segments.",
  },
  {
    name: "Berka (PKDD’99)",
    track: "Track C · selected",
    status: "Champion dataset",
    whyNot:
      "Multi-year account histories with balances, flows, tenure, and loan/card relationships. Supports leakage-safe as-of features and interpretable banking plays.",
    selected: true,
  },
];

const INSIGHTS = [
  {
    title: "Onboarding vs established basics",
    body: "Shorter tenure and lighter product depth mark newer basic accounts, while long-tenure high-activity basics are loyal but under-monetized — a priority cross-sell lane.",
    segments: [0, 1],
  },
  {
    title: "Credit relationships already formed",
    body: "High flows combined with loan holding define a loan-centric group; high card penetration with strong balances defines solvent card-centric savers. Different owners and plays.",
    segments: [2, 3],
  },
  {
    title: "Stress and dormancy need care, not campaigns",
    body: "Negative balances and long recency gaps create thin exception-queue cells. The bank should review risk / service paths — not mass product pushes.",
    segments: [4, 5],
  },
  {
    title: "Behavior beats product flags alone",
    body: "Activity, balance, recency, and product holding travel together on Berka. That mix supports CRM decisions better than product-holding splits alone.",
    segments: [],
  },
];

export default function EdaPage() {
  const { data, error, loading } = useAsync(async () => {
    const [model, metrics, segs] = await Promise.all([
      getModel(),
      getMetrics(),
      getSegments(),
    ]);
    return { model, metrics, segs };
  }, []);

  if (loading) return <div className="page loading">Loading data…</div>;

  const m = data?.metrics || {};
  const model = data?.model || {};
  const columns =
    model.feature_columns ||
    m.feature_columns ||
    Object.keys(FEATURE_BLURBS);
  const byId = Object.fromEntries(
    (data?.segs?.segments || []).map((s) => [Number(s.segment_id), s]),
  );

  return (
    <PageShell
      title="Data & insights"
      lead="What the Berka freeze population looks like, which features drive the champion, and how those patterns map to the six segments."
    >
      <ErrorBanner error={error} />

      <div className="panel" style={{ marginBottom: "1rem" }}>
        <h3>Dataset candidates</h3>
        <p className="muted" style={{ marginTop: 0 }}>
          Three banking sources were evaluated. Berka was selected for the champion because it
          supports repeated account-level behavior over years — not because it is the largest file.
        </p>
        <div className="dataset-list">
          {DATASETS.map((d) => (
            <div
              className={`dataset-row${d.selected ? " dataset-row--selected" : ""}`}
              key={d.name}
            >
              <div className="dataset-row__head">
                <strong>{d.name}</strong>
                <span className="muted">{d.track}</span>
              </div>
              <p className="dataset-row__status">{d.status}</p>
              <p className="muted" style={{ margin: 0 }}>
                {d.whyNot}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="panel" style={{ marginBottom: "1rem" }}>
        <h3>Why Berka · freeze cohort</h3>
        <p className="muted" style={{ marginTop: 0 }}>
          Czech PKDD’99 bank data with multi-year transactions, balances, and loan/card
          relationships. Features are built with a leakage-safe as-of cut so nothing after the freeze
          date leaks into training.
        </p>
        <dl className="kv">
          <div>
            <dt>As-of date</dt>
            <dd>{m.as_of_date || model.as_of_date || "1997-12-31"}</dd>
          </div>
          <div>
            <dt>Eligibility</dt>
            <dd>Account tenure ≥ {m.minimum_tenure_days || 365} days</dd>
          </div>
          <div>
            <dt>Population</dt>
            <dd>
              Train {m.n_train ?? "2,701"} · holdout {m.n_holdout ?? "901"} · ~3,602 eligible
              accounts
            </dd>
          </div>
          <div>
            <dt>Feature set</dt>
            <dd>
              <code>
                {model.feature_version || m.feature_version || "berka_rfm_product_v1_leakage_safe"}
              </code>
            </dd>
          </div>
        </dl>
      </div>

      <div className="panel" style={{ marginBottom: "1rem" }}>
        <h3>Features used for segmentation</h3>
        <p className="muted" style={{ marginTop: 0 }}>
          Eight interpretable Berka inputs (log1p / scaled inside the frozen pipeline). Demographics
          are <strong style={{ color: "var(--ink)" }}>not</strong> clustering features.
        </p>
        <div className="feature-grid">
          {columns.map((col) => (
            <div className="feature-card" key={col}>
              <code>{col}</code>
              <p>{FEATURE_BLURBS[col] || "Model input feature."}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="stack" style={{ marginBottom: "1rem" }}>
        <h3 style={{ fontFamily: "var(--display)", margin: "0 0 0.25rem" }}>
          Insights → segments
        </h3>
        <p className="muted" style={{ margin: "0 0 0.75rem" }}>
          Each finding points to the personas it explains. Open a segment for the full profile.
        </p>
        <div className="finding-list">
          {INSIGHTS.map((f) => (
            <div className="finding" key={f.title}>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
              {f.segments.length > 0 && (
                <div className="chip-row" style={{ marginTop: "0.65rem", marginBottom: 0 }}>
                  {f.segments.map((id) => (
                    <Link key={id} className="chip" to={`/segments/${id}`}>
                      {id}: {byId[id]?.name || `Segment ${id}`}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <h3>Next</h3>
        <p className="muted" style={{ margin: 0 }}>
          See how AE + K-Means compares to traditional methods on{" "}
          <Link to="/compare">Compare</Link>, then explore the{" "}
          <Link to="/segments">segment map</Link> or{" "}
          <Link to="/decisions">banking applications</Link>.
        </p>
      </div>
    </PageShell>
  );
}
