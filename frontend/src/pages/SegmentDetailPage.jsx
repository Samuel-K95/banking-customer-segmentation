import { Link, useParams } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getProfiles,
  getSegment,
  getSegmentApplications,
  getServingContract,
} from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import PageShell from "../components/PageShell.jsx";
import { useAsync } from "../hooks/useAsync.js";

const FEAT_KEYS = [
  "Transaction_Count",
  "Account_Balance",
  "Recency_Days",
  "Account_Age_Days",
  "has_loan",
  "has_card",
];

export default function SegmentDetailPage() {
  const { id } = useParams();
  const { data, error, loading } = useAsync(async () => {
    const [segment, apps, profiles, contract] = await Promise.all([
      getSegment(id),
      getSegmentApplications(id),
      getProfiles().catch(() => null),
      getServingContract(),
    ]);
    return { segment, apps, profiles, contract };
  }, [id]);

  if (loading) return <div className="page loading">Loading segment…</div>;
  if (error) {
    return (
      <div className="page">
        <ErrorBanner error={error} />
        <Link to="/segments">← Segments</Link>
      </div>
    );
  }

  const segment = data.segment;
  const pack = (data.profiles?.packs || []).find((p) => Number(p.segment_id) === Number(id));
  const pop = data.profiles?.population_means || {};
  const centroid = segment.centroid || pack?.centroid || {};
  const chartData = FEAT_KEYS.filter((k) => centroid[k] != null).map((k) => ({
    feature: k.replace(/_/g, " "),
    segment: Number(centroid[k]),
    population: pop[k] != null ? Number(pop[k]) : 0,
  }));
  const isEx = (data.contract?.exception_segment_ids || []).map(Number).includes(Number(id));
  const apps = (data.apps?.applications || []).slice(0, 6);

  return (
    <PageShell
      title={segment.name}
      lead={segment.one_liner}
      actions={
        <Link className="btn ghost" to="/segments">
          All segments
        </Link>
      }
    >
      {isEx && (
        <div className="panel" style={{ marginBottom: "1rem", borderColor: "rgba(201,162,39,0.45)" }}>
          Exception-queue segment — prefer service / risk review over mass campaigns.
        </div>
      )}

      <div className="panel" style={{ marginBottom: "1rem" }}>
        <h3>Default play</h3>
        <p style={{ margin: 0 }}>{segment.default_play || segment.decision?.play}</p>
        {segment.decision?.guardrail && (
          <p className="muted" style={{ margin: "0.5rem 0 0" }}>
            Guardrail: {segment.decision.guardrail}
          </p>
        )}
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>Centroid vs population</h3>
          <div className="chart-wrap chart-wrap--bar">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 48 }}>
                <CartesianGrid stroke="rgba(232,236,232,0.08)" vertical={false} />
                <XAxis
                  dataKey="feature"
                  angle={-28}
                  textAnchor="end"
                  interval={0}
                  height={64}
                  tick={{ fill: "#8fa3ab", fontSize: 10 }}
                />
                <YAxis tick={{ fill: "#8fa3ab", fontSize: 10 }} width={42} />
                <Tooltip
                  contentStyle={{ background: "#122027", border: "1px solid #2a3d33" }}
                />
                <Bar dataKey="segment" fill="#2db8a4" name="Segment" />
                <Bar dataKey="population" fill="#5a7380" name="Population" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel">
          <h3>What distinguishes this group</h3>
          <dl className="kv">
            <div>
              <dt>Over-indexed</dt>
              <dd>
                {(pack?.behavioral_signature?.over_indexed || []).slice(0, 4).join(", ") || "—"}
              </dd>
            </div>
            <div>
              <dt>Under-indexed</dt>
              <dd>
                {(pack?.behavioral_signature?.under_indexed || []).slice(0, 4).join(", ") || "—"}
              </dd>
            </div>
            <div>
              <dt>Size</dt>
              <dd>
                {segment.pct != null ? `${Number(segment.pct).toFixed(1)}%` : "—"}
                {segment.n != null ? ` (n=${segment.n})` : ""}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="panel" style={{ marginTop: "1rem" }}>
        <h3>Banking applications</h3>
        <div className="finding-list" style={{ border: "none", background: "transparent", gap: "0.65rem" }}>
          {apps.map((a, i) => (
            <div className="finding" key={`${a.domain}-${i}`} style={{ border: "1px solid var(--line)" }}>
              <h3>
                {a.play}{" "}
                <span className="muted" style={{ fontSize: "0.8rem", fontFamily: "var(--sans)" }}>
                  · {String(a.domain || "").replace(/_/g, " ")}
                </span>
              </h3>
              <p>{a.evidence}</p>
              {a.guardrail && (
                <p style={{ marginTop: "0.35rem", color: "var(--amber)" }}>Guardrail: {a.guardrail}</p>
              )}
            </div>
          ))}
          {!apps.length && <p className="muted">No applications returned.</p>}
        </div>
      </div>
    </PageShell>
  );
}
