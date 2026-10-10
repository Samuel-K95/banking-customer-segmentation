import { useEffect, useState } from "react";
import { getSegments } from "../api/client.js";

export default function SegmentsPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getSegments()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <section>
      <h1>Segments</h1>
      <p>Champion segment list will load from <code>GET /v1/segments</code>.</p>
      {error && <p className="error">{error}</p>}
      {data && <pre className="panel">{JSON.stringify(data, null, 2)}</pre>}
    </section>
  );
}
