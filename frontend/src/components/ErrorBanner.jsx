export default function ErrorBanner({ error }) {
  if (!error) return null;
  const msg = typeof error === "string" ? error : error.message || String(error);
  return <div className="error-banner">{msg}</div>;
}
