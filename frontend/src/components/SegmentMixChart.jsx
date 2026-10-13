import { useMemo } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

const COLORS = ["#2db8a4", "#5a9bb8", "#c9a227", "#8fa3ab", "#e07070", "#6b8f71"];

function colorFor(id) {
  return COLORS[Number(id) % COLORS.length];
}

function CentroidDot(props) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  const r = Math.max(10, Math.min(26, (payload.z || 12) / 8));
  return (
    <g>
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill={payload.fill}
        stroke={payload.exception ? "#c9a227" : "rgba(232,236,232,0.55)"}
        strokeWidth={payload.exception ? 3 : 1.5}
        opacity={0.95}
      />
      <text
        x={cx}
        y={cy + 4}
        textAnchor="middle"
        fill="#041210"
        fontSize={11}
        fontWeight={700}
        fontFamily="IBM Plex Sans, sans-serif"
      >
        {payload.id}
      </text>
    </g>
  );
}

function ExemplarDot(props) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={5}
      fill={payload.fill}
      opacity={0.5}
      stroke="rgba(12,20,24,0.35)"
      strokeWidth={1}
    />
  );
}

export default function SegmentMixChart({ segments, exceptions = [], exemplars = [] }) {
  const ex = useMemo(() => new Set((exceptions || []).map(Number)), [exceptions]);

  const centroids = useMemo(() => {
    return (segments || [])
      .map((s) => {
        const id = Number(s.segment_id);
        const c = s.centroid || {};
        return {
          id,
          name: s.name,
          x: Number(c.Transaction_Count),
          y: Number(c.Account_Balance),
          z: Math.max(40, Number(s.pct || 1) * 6),
          pct: s.pct,
          n: s.n,
          kind: "centroid",
          exception: ex.has(id),
          fill: colorFor(id),
        };
      })
      .filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y));
  }, [segments, ex]);

  const dots = useMemo(() => {
    return (exemplars || [])
      .map((row) => {
        const id = Number(row.segment_id);
        return {
          id,
          name: `Example · segment ${id}`,
          x: Number(row.Transaction_Count),
          y: Number(row.Account_Balance),
          z: 20,
          account_id: row.account_id,
          kind: "exemplar",
          exception: ex.has(id),
          fill: colorFor(id),
        };
      })
      .filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y));
  }, [exemplars, ex]);

  if (!centroids.length) return null;

  return (
    <div className="chart-panel">
      <div className="scatter-legend" aria-label="Segment legend">
        {centroids.map((c) => (
          <div className="scatter-legend__item" key={c.id}>
            <span className="scatter-legend__swatch" style={{ background: c.fill }} />
            <span className="scatter-legend__text">
              <strong>{c.id}</strong> {c.name}
              {c.pct != null ? ` · ${Number(c.pct).toFixed(1)}%` : ""}
              {c.exception ? " · exception" : ""}
            </span>
          </div>
        ))}
        <div className="scatter-legend__item scatter-legend__item--meta">
          <span className="scatter-legend__swatch scatter-legend__swatch--dot" />
          <span className="scatter-legend__text">Small dots = example customers</span>
        </div>
      </div>

      <div className="axis-keys" aria-hidden="true">
        <span>X: transaction count (activity)</span>
        <span>Y: account balance</span>
      </div>

      <div className="chart-wrap chart-wrap--scatter">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 20, bottom: 12, left: 8 }}>
            <CartesianGrid stroke="rgba(232,236,232,0.1)" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="x"
              tick={{ fill: "#8fa3ab", fontSize: 12 }}
              tickCount={6}
            />
            <YAxis
              type="number"
              dataKey="y"
              tick={{ fill: "#8fa3ab", fontSize: 12 }}
              width={56}
              tickFormatter={(v) =>
                Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)
              }
            />
            <ZAxis type="number" dataKey="z" range={[60, 320]} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload;
                return (
                  <div className="chart-tooltip">
                    <strong>
                      {p.kind === "centroid" ? `Segment ${p.id}: ${p.name}` : p.name}
                    </strong>
                    <div>Transactions: {Number(p.x).toLocaleString()}</div>
                    <div>Balance: {Number(p.y).toLocaleString()}</div>
                    {p.kind === "centroid" && (
                      <div>
                        Book share: {p.pct != null ? `${Number(p.pct).toFixed(1)}%` : "—"}
                        {p.n != null ? ` (n=${p.n})` : ""}
                      </div>
                    )}
                    {p.account_id != null && <div>Account: {p.account_id}</div>}
                    {p.exception && <div className="warn">Exception-queue segment</div>}
                  </div>
                );
              }}
            />
            {dots.length > 0 && (
              <Scatter
                name="examples"
                data={dots}
                shape={<ExemplarDot />}
                isAnimationActive
                animationDuration={700}
                legendType="none"
              />
            )}
            <Scatter
              name="centers"
              data={centroids}
              shape={<CentroidDot />}
              isAnimationActive
              animationDuration={800}
              legendType="none"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <p className="chart-caption">
        Numbered bubbles are segment centers (size ≈ book share). Amber ring = exception queue.
        Hover any point for exact values.
      </p>
    </div>
  );
}
