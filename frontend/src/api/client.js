export const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function getHealth() {
  const res = await fetch(`${apiBaseUrl}/v1/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function getSegments() {
  const res = await fetch(`${apiBaseUrl}/v1/segments`);
  if (!res.ok) throw new Error(`Segments failed: ${res.status}`);
  return res.json();
}
