# Production freeze readiness — PROVISIONAL

**Overall verdict: PROVISIONAL.** Engineering gates for the leakage-safe AE+k=6 production candidate are evaluated from live integrity / replay / pack checks. This is **not** a full bank production freeze.

## Gates

| Gate | Verdict | Detail |
|---|---|---|
| `artifact_integrity` | **PASS** | SHA-256 fingerprints match serving_contract.json |
| `snapshot_split_train_only_fit` | **PASS** | as_of=1997-12-31; tenure>=365; train=2701 holdout=901 |
| `exact_holdout_prediction_replay` | **PASS** | Frozen encoder+KMeans reproduce holdout labels (n=901) |
| `segment_meaning` | **PASS** | CRM personas + plays for k=6; exception-queue segments=[4, 5] |
| `fairness_owner_disponent` | **PASS_WITH_LIMITATIONS** | District fairness pack + OWNER/DISPONENT policy published; compliance sign-off PENDING |
| `independent_temporal_validation` | **PASS_WITH_LIMITATIONS** | 1998 check is retrospective and uses the same Berka vintage that informed development; it is not an independent prospective holdout. |
| `serving_package` | **PASS** | FastAPI enforces model_version+as_of_date; serving_contract hashes verified |

## Rollback

Point CLI/API defaults to `artifacts/segmentation/archive/fullbook_demo/` (archive demo) or re-promote a prior hashed research twin from `artifacts/segmentation/research/leakage_safe_source/`.

