import { useEffect, useState } from "react";
import { getHealth } from "../api/client.js";

export default function OverviewPage() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <section>
      <h1>Overview</h1>
      <p>API-driven scaffold. Wire comparison, fairness, and decision map pages as artifacts appear.</p>
      {error && <p className="error">{error}</p>}
      {health && (
        <pre className="panel">{JSON.stringify(health, null, 2)}</pre>
      )}
    </section>
  );
}
