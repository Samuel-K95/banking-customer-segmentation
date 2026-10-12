# Production freeze readiness — PROVISIONAL

**Overall verdict: PROVISIONAL.** Engineering gates for the leakage-safe AE+k=6 production candidate are closed. This is **not** a full bank production freeze.

- Run: `c_ae_lat4_k6_leakage_safe`
- Model version: `production_candidate_v1`
- As-of: `1997-12-31`

## Gates

| Gate | Verdict |
|---|---|
| artifact_integrity | **PASS** — SHA-256 fingerprints in serving_contract.json |
| snapshot_split_train_only_fit | **PASS** — as_of=1997-12-31; tenure>=365; train=2701 holdout=901 |
| exact_holdout_prediction_replay | **PASS** — Frozen encoder+KMeans reproduce held-out labels (see CI replay test) |
| segment_meaning | **PASS** — CRM personas + exception-queue policy for tiny stressed/dormant cells |
| fairness_owner_disponent | **PASS** — District fairness pack + OWNER/DISPONENT policy published; compliance sign-off PENDING |
| independent_temporal_validation | **PASS_WITH_LIMITATIONS** — 1998 check is retrospective and uses the same Berka vintage that informed development; it is not an independent prospective holdout. |
| serving_package | **PASS** — FastAPI enforces model_version+as_of_date; health exposes hashes; monitoring/segment-mix + rollback to product/ archive documented |

## Still not fully frozen

- Human compliance / model-risk sign-off still PENDING
- Temporal validation is retrospective Berka 1998, not prospective/external
- Must not use segment_id for credit eligibility, pricing, or collections

## Artifacts

- Bundle: `artifacts/segmentation/champion/`
- Decision: `artifacts/decision/champion/`
- OWNER/DISPONENT policy: `reports/champion/04_owner_disponent_policy.md`
- Temporal: `reports/champion/03_temporal_validation.md`

## Rollback

Point CLI/API defaults back to `artifacts/segmentation/archive/fullbook_demo/` (archive demo) or re-promote a prior hashed research twin.
