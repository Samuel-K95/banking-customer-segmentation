/** API base: empty string uses Vite proxy (/v1 → :8000). Override with VITE_API_BASE_URL. */
export const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

async function request(path, options = {}) {
  const res = await fetch(`${apiBaseUrl}${path}`, {
    headers: { Accept: "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail =
      typeof body === "object" && body?.detail
        ? typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail)
        : `HTTP ${res.status}`;
    const err = new Error(detail);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

async function requestText(path) {
  const res = await fetch(`${apiBaseUrl}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
}

export const getHealth = () => request("/v1/health");
export const getModel = () => request("/v1/model");
export const getMetrics = () => request("/v1/model/metrics");
export const getFreeze = () => request("/v1/model/freeze");
export const getServingContract = () => request("/v1/model/serving-contract");
export const getSegments = () => request("/v1/segments");
export const getSegment = (id) => request(`/v1/segments/${id}`);
export const getSegmentApplications = (id) => request(`/v1/segments/${id}/applications`);
export const getDecisionSummary = () => request("/v1/decisions/summary");
export const getDecisionMap = (domain) =>
  request(domain ? `/v1/decisions/map?domain=${encodeURIComponent(domain)}` : "/v1/decisions/map");
export const getDecisionDomains = () => request("/v1/decisions/domains");
export const getProfiles = () => request("/v1/decisions/profiles");
export const getFairness = () => request("/v1/decisions/fairness");
export const getExplainability = () => request("/v1/decisions/explainability");
export const getReports = (scope = "champion") =>
  request(`/v1/reports?scope=${encodeURIComponent(scope)}`);
export const getTrackCReports = () => request("/v1/reports/tracks/berka");
export const getReportMeta = (slug) => request(`/v1/reports/${slug}`);
export const getReportRaw = (slug) => requestText(`/v1/reports/${slug}/raw`);
export const getComparisonBoard = (board = "champion") =>
  request(`/v1/comparison/board?board=${encodeURIComponent(board)}`);
export const getInferenceContract = () => request("/v1/inference/contract");
export const postInference = (payload) =>
  request("/v1/inference", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const postInferenceBatch = (payload) =>
  request("/v1/inference/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
