import { Link } from "react-router-dom";
import { getProfiles, getSegments, getServingContract } from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import PageShell from "../components/PageShell.jsx";
import SegmentMixChart from "../components/SegmentMixChart.jsx";
import { useAsync } from "../hooks/useAsync.js";

export default function SegmentsPage() {
  const { data, error, loading } = useAsync(
    () =>
      Promise.all([getSegments(), getServingContract(), getProfiles().catch(() => null)]).then(
        ([segs, contract, profiles]) => ({ segs, contract, profiles }),
      ),
    [],
  );

  if (loading) return <div className="page loading">Loading segments…</div>;

  const segments = data?.segs?.segments || [];
  const exceptions = data?.contract?.exception_segment_ids || [];
  const exemplars = data?.profiles?.exemplars || [];

  return (
    <PageShell
      title="Customer segments"
      lead="Six behavioral personas on a simple map: activity vs balance. Open a tile for the full profile and banking plays."
    >
      <ErrorBanner error={error} />

      <div className="panel panel--chart" style={{ marginBottom: "1.25rem" }}>
        <h3>Where the segments sit</h3>
        <SegmentMixChart
          segments={segments}
          exceptions={exceptions}
          exemplars={exemplars}
        />
      </div>

      <div className="segment-grid">
        {segments.map((s) => {
          const id = Number(s.segment_id);
          const isEx = exceptions.map(Number).includes(id);
          return (
            <Link key={id} className="segment-tile" to={`/segments/${id}`}>
              <span className="sid">
                Segment {id}
                {isEx ? <span className="badge">Exception</span> : null}
              </span>
              <h3>{s.name}</h3>
              <p className="one-liner">{s.one_liner}</p>
              <div className="meta">
                {s.pct != null ? `${Number(s.pct).toFixed(1)}%` : ""}
                {s.n != null ? ` · n=${s.n}` : ""}
              </div>
            </Link>
          );
        })}
      </div>
    </PageShell>
  );
}
