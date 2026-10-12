# Berka Champion Leakage-Safe Freeze Readiness

**Overall verdict: BLOCKED.** This is a leakage-disciplined twin of the CRM champion feature family — not an approved production freeze, and it does **not** replace `artifacts/segmentation/archive/fullbook_demo`.

## What changed vs original champion_v1

| Control | Original `champion_v1` | This twin |
|---|---|---|
| As-of date | End-of-book max txn date | **1997-12-31** |
| Tenure gate | None | **≥ 365 days** |
| Population | ~4500 full book | **3602** eligible |
| Split | None (fit on all) | **2701 / 901** (seed 8675309) |
| Scaler / AE / K-Means | Fit on all rows | **Train-only**; holdout scored frozen |
| Product k | 6 | **6** (unchanged CRM choice) |
| Research twin | — | **K-Means k=2** on same matrix |

## Holdout metrics

| Family | k | Silhouette ↑ | DB ↓ | CH ↑ | Seed ARI (train) |
|---|---:|---:|---:|---:|---:|
| ae_kmeans | 6 | 0.4591 | 0.623 | 852.0 | 0.9975 |
| kmeans | 2 | 0.2968 | 1.4088 | 208.0 | 0.998 |

AE+k=6 holdout silhouette **0.4591**; k=2 twin holdout silhouette **0.2968**.

## Why k=6 stays the product path

Under this leakage contract, **k=2** is the honest simple partition (scale/activity twin on the same 8 features). **k=6** remains the CRM segmentation because loan/card + behavior still support named personas and the decision-layer playbook. Tiny stressed/dormant cells are **exception queues**, not mass campaigns.

## Freeze gates

| Gate | Verdict |
|---|---|
| artifact_integrity | **PASS** — SHA-256 fingerprints written |
| snapshot_split_train_only_fit | **PASS** — as_of=1997-12-31; tenure>=365; train=2701 holdout=901; scaler/AE/KMeans fit on train only |
| exact_holdout_prediction_replay | **PASS** — Frozen encoder+KMeans reproduce held-out labels |
| segment_meaning | **NEEDS_APPROVAL** — CRM personas retained; tiny stressed/dormant cells = exception queues |
| fairness_owner_disponent | **NEEDS_APPROVAL** — No demographic fairness / OWNER-DISPONENT policy completed for this twin |
| independent_temporal_validation | **BLOCKED** — Requires prospective/external vintage not used in development |
| serving_package | **BLOCKED** — Artifacts exist; production serving/monitoring/rollback not validated |

## Artifacts

- Bundle: `artifacts\segmentation\champion_v1_leakage_safe`
- Parallel k=2 holdout assignments: `artifacts\segmentation\champion_v1_leakage_safe\held_out_assignments_kmeans_k2.csv`
- Fingerprints: `freeze_readiness.json` → `artifact_sha256`

## Bottom line

Leakage discipline and freeze honesty are applied to the **AE+k=6 product path**. Mahider’s original 13-feature k=2 research baseline remains separate. Neither candidate is freeze-approved.
