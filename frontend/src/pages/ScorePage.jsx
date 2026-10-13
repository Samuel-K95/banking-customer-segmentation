import { useEffect, useState } from "react";
import { getInferenceContract, postInference } from "../api/client.js";
import ErrorBanner from "../components/ErrorBanner.jsx";
import PageShell from "../components/PageShell.jsx";

const FEATURE_DEFAULTS = {
  Transaction_Count: 120,
  Account_Balance: 45000,
  Total_Credit: 80000,
  Total_Withdrawal: 35000,
  Recency_Days: 5,
  Account_Age_Days: 900,
  has_loan: 0,
  has_card: 1,
};

export default function ScorePage() {
  const [contract, setContract] = useState(null);
  const [form, setForm] = useState({ ...FEATURE_DEFAULTS });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getInferenceContract()
      .then(setContract)
      .catch((err) => setError(err));
  }, []);

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!contract) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        model_version: contract.model_version,
        as_of_date: contract.as_of_date,
        include_applications: true,
        ...Object.fromEntries(Object.entries(form).map(([k, v]) => [k, Number(v)])),
      };
      setResult(await postInference(payload));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  const assignment = result?.assignments?.[0];
  const exceptions = new Set((contract?.exception_segment_ids || []).map(Number));

  return (
    <PageShell
      title="Live demonstration"
      lead="Score a synthetic customer against the frozen serving contract. Segments guide CRM plays — not credit approval."
    >
      <ErrorBanner error={error} />

      {contract && (
        <div className="panel" style={{ marginBottom: "1rem" }}>
          Locked contract: <code>{contract.model_version}</code> · as-of{" "}
          <code>{contract.as_of_date}</code> · tenure ≥ {contract.minimum_tenure_days ?? 365} days
        </div>
      )}

      <form className="panel" onSubmit={onSubmit}>
        <div className="form-grid">
          {Object.keys(FEATURE_DEFAULTS).map((key) => (
            <div className="field" key={key}>
              <label htmlFor={key}>{key}</label>
              <input
                id={key}
                type="number"
                step={key.startsWith("has_") ? 1 : "any"}
                min={key.startsWith("has_") ? 0 : undefined}
                max={key.startsWith("has_") ? 1 : undefined}
                value={form[key]}
                onChange={(e) => setForm((prev) => ({ ...prev, [key]: e.target.value }))}
                required
              />
            </div>
          ))}
        </div>
        <div className="cta-row" style={{ marginTop: "1rem" }}>
          <button className="btn" type="submit" disabled={busy || !contract}>
            {busy ? "Scoring…" : "Score customer"}
          </button>
        </div>
      </form>

      {assignment && (
        <div className="panel" style={{ marginTop: "1rem" }}>
          <h3>
            Segment {assignment.segment_id}: {assignment.segment_name}
            {exceptions.has(Number(assignment.segment_id)) ? (
              <span className="badge">Exception</span>
            ) : null}
          </h3>
          <dl className="kv">
            <div>
              <dt>Confidence</dt>
              <dd>
                {assignment.confidence != null ? Number(assignment.confidence).toFixed(3) : "—"}
              </dd>
            </div>
            <div>
              <dt>Play</dt>
              <dd>{assignment.play}</dd>
            </div>
            {assignment.guardrail && (
              <div>
                <dt>Guardrail</dt>
                <dd>{assignment.guardrail}</dd>
              </div>
            )}
          </dl>
          {Array.isArray(assignment.applications) && assignment.applications.length > 0 && (
            <ul className="prohibitions">
              {assignment.applications.slice(0, 4).map((a, i) => (
                <li key={i}>
                  {a.play} ({String(a.domain || "").replace(/_/g, " ")})
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </PageShell>
  );
}
