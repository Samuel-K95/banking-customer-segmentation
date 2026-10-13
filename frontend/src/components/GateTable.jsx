function verdictClass(v) {
  const s = String(v || "").toUpperCase();
  if (s.includes("FAIL") || s.includes("BLOCK")) return "fail";
  if (s.includes("LIMIT") || s.includes("PENDING") || s.includes("APPROVAL")) return "limit";
  if (s.includes("PASS")) return "pass";
  return "";
}

export default function GateTable({ gates }) {
  const entries = Object.entries(gates || {});
  if (!entries.length) return <p className="muted">No freeze gates available.</p>;
  return (
    <table className="gate-table">
      <thead>
        <tr>
          <th>Gate</th>
          <th>Verdict</th>
          <th>Detail</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([name, gate]) => {
          const verdict = gate?.verdict ?? gate;
          const detail = typeof gate === "object" ? gate.detail || "" : "";
          return (
            <tr key={name}>
              <td>
                <code>{name}</code>
              </td>
              <td>
                <span className={`verdict ${verdictClass(verdict)}`}>{String(verdict)}</span>
              </td>
              <td className="muted">{detail}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
