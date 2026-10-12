# Champion API

Exposes the production-candidate champion: model bundle, decision layer, reports, and inference (single + batch).

## Clean checkout

Champion **binaries** (`encoder.pt`, `scaler.joblib`, `kmeans.joblib`) are tracked under `artifacts/segmentation/champion/`. After clone:

```bash
bcs verify-champion          # must pass (SHA-256 + load + API smoke)
bcs restore-champion         # if verify fails: promote research twin or retrain
bcs serve --port 8000
```

## Start

```bash
# from repo root (artifacts/ must be present)
bcs serve --port 8000
# or
uvicorn customer_segmentation.serve.app:app --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
Machine catalog: `GET /v1/catalog`  
Health fails (503) if champion integrity / SHA-256 / encoder is broken.

## Contract

Every scoring call must send:

| Field | Value |
|-------|-------|
| `model_version` | `production_candidate_v1` |
| `as_of_date` | `1997-12-31` |

Confirm live values with `GET /v1/inference/contract`.

Features are **interpretable-scale** (`Transaction_Count`, `Account_Balance`, `Total_Credit`, `Total_Withdrawal`, `Recency_Days`, `Account_Age_Days`, `has_loan`, `has_card`). The API applies frozen log1p + StandardScaler.

**Eligibility:** `Account_Age_Days >= minimum_tenure_days` (365). Unknown feature keys, negative recency, or non-{0,1} product flags → HTTP 400.

## Endpoint groups

### Meta
- `GET /v1/health` — liveness + champion identity + freeze status
- `GET /v1/catalog` — endpoint index

### Model
- `GET /v1/model` — overview (alias: `/v1/champion`)
- `GET /v1/model/manifest|metrics|features|freeze|serving-contract`
- `GET /v1/segments` / `GET /v1/segments/{id}`

### Decisions
- `GET /v1/decisions/summary` — run summary (alias: `/v1/decision/summary`)
- `GET /v1/decisions/map` — live expand from catalog (`?domain=`; alias: `/v1/decision-map`)
- `GET /v1/decisions/map/full` — materialized CSV
- `GET /v1/decisions/domains|profiles|fairness|explainability|domain-coverage`
- `GET /v1/segments/{id}/applications` — plays for one segment

### Reports
- `GET /v1/reports` — list markdown under `reports/champion/`
- `GET /v1/reports/{slug}` — meta + preview
- `GET /v1/reports/{slug}/raw` — full markdown (`?download=true` for file)

### Inference
- `POST /v1/inference` — single customer
- `POST /v1/inference/batch` — JSON array (≤500)
- `POST /v1/inference/batch/csv` — multipart CSV upload
- `POST /v1/customers/segment` — backward-compatible batch alias
- `GET /v1/inference/contract` — required version/as_of/features

### Monitoring
- `GET /v1/monitoring/segment-mix` — baseline mix + exception segments
- `GET /v1/monitoring/snapshot` — artifact presence checks

## Example: single inference

```bash
curl -s http://127.0.0.1:8000/v1/inference -H "Content-Type: application/json" -d "{
  \"model_version\": \"production_candidate_v1\",
  \"as_of_date\": \"1997-12-31\",
  \"Transaction_Count\": 120,
  \"Account_Balance\": 45000,
  \"Total_Credit\": 80000,
  \"Total_Withdrawal\": 35000,
  \"Recency_Days\": 5,
  \"Account_Age_Days\": 900,
  \"has_loan\": 0,
  \"has_card\": 1,
  \"include_applications\": true
}"
```

## Example: batch JSON

```bash
curl -s http://127.0.0.1:8000/v1/inference/batch -H "Content-Type: application/json" -d "{
  \"model_version\": \"production_candidate_v1\",
  \"as_of_date\": \"1997-12-31\",
  \"include_applications\": false,
  \"customers\": [
    {
      \"account_id\": 1,
      \"Transaction_Count\": 120,
      \"Account_Balance\": 45000,
      \"Total_Credit\": 80000,
      \"Total_Withdrawal\": 35000,
      \"Recency_Days\": 5,
      \"Account_Age_Days\": 900,
      \"has_loan\": 0,
      \"has_card\": 1
    }
  ]
}"
```

## Notes

- Freeze status is **PROVISIONAL** — not a full bank production freeze.
- Segments are behavioral clusters for decision plays; they are not credit scores or pricing.
- Segments 4 and 5 are exception-queue IDs in the serving contract (human review).
