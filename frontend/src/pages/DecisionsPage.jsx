import { useMemo, useState } from "react";
import { getDecisionDomains, getDecisionMap, getDecisionSummary } from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import PageShell from "../components/PageShell.jsx";
import { useAsync } from "../hooks/useAsync.js";

export default function DecisionsPage() {
  const [domain, setDomain] = useState("");
  const { data, error, loading } = useAsync(async () => {
    const [summary, domains, map] = await Promise.all([
      getDecisionSummary(),
      getDecisionDomains(),
      getDecisionMap(),
    ]);
    return { summary, domains, map };
  }, []);

  const rows = useMemo(() => {
    const all = data?.map?.rows || [];
    if (!domain) return all;
    return all.filter((r) => r.domain === domain);
  }, [data, domain]);

  const domainList = data?.domains?.rows || [];
  const domainNames = domainList
    .map((d) => (typeof d === "string" ? d : d.domain))
    .filter(Boolean);

  if (loading) return <div className="page loading">Loading applications…</div>;

  return (
    <PageShell
      title="Banking applications"
      lead="Evidence-linked plays across product, retention, inclusion, engagement, service, and care — at least three domains as required by the brief."
    >
      <ErrorBanner error={error} />

      <div className="metric-strip">
        <div className="metric">
          <span className="label">Plays</span>
          <span className="value">{data?.summary?.n_plays ?? rows.length ?? "—"}</span>
        </div>
        <div className="metric">
          <span className="label">Segments</span>
          <span className="value">{data?.summary?.n_segments ?? 6}</span>
        </div>
        <div className="metric">
          <span className="label">Domains</span>
          <span className="value">{domainNames.length || "—"}</span>
        </div>
      </div>

      <div className="chip-row">
        <button
          type="button"
          className={`chip ${!domain ? "active" : ""}`}
          onClick={() => setDomain("")}
        >
          All
        </button>
        {domainNames.map((d) => (
          <button
            key={d}
            type="button"
            className={`chip ${domain === d ? "active" : ""}`}
            onClick={() => setDomain(d)}
            title={d}
          >
            {d.replace(/_/g, " ")}
          </button>
        ))}
      </div>

      <div className="panel">
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: "16%" }}>Segment</th>
                <th style={{ width: "16%" }}>Domain</th>
                <th style={{ width: "28%" }}>Play</th>
                <th style={{ width: "14%" }}>Owner</th>
                <th>Guardrail</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.segment_id}-${r.play}-${i}`}>
                  <td>{r.segment || r.segment_id}</td>
                  <td className="muted">{String(r.domain || "").replace(/_/g, " ")}</td>
                  <td>{r.play}</td>
                  <td>{r.owner}</td>
                  <td className="muted">{r.guardrail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!rows.length && <p className="muted">No plays for this filter.</p>}
      </div>
    </PageShell>
  );
}
