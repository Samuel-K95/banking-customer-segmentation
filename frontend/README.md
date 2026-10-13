# Behavioral Banking Intelligence — frontend

React + Vite analyst / demo UI for the Berka leakage-safe champion.

## Run

1. Start the API from the repo root:

```bash
uv run bcs serve --port 8000
```

2. Start the SPA:

```bash
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Dev server proxies `/v1` to `http://127.0.0.1:8000`.

Optional: set `VITE_API_BASE_URL=http://127.0.0.1:8000` to call the API directly (CORS is enabled for Vite origins).

## Pages (aligned to Week-8 deliverables)

| Route | Purpose |
|-------|---------|
| `/` | Brand hero + champion identity |
| `/champion` | Why AE+k=6; metrics; what it is / is not |
| `/compare` | Traditional vs representation-learning comparison |
| `/segments` | Segment mix chart + persona gallery |
| `/segments/:id` | Profile, centroid contrast, applications |
| `/decisions` | Banking applications (multi-domain plays) |
| `/eda` | Short behavioral insight cards |
| `/score` | Live demo / scoring playground |

## Build

```bash
npm run build && npm run preview
```

Integrity / missing `serving_contract.json` surfaces as API errors (status pill + banners) — the UI does not invent a fallback contract.
