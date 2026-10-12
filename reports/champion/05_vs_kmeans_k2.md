# Previous AE+k=6 vs leakage-safe k=2 twin

**Important naming note:** the “current k=2” run is **K-Means k=2 on the train-scaled 8 CRM features**, not AE+k=2. We did not train an autoencoder with 2 clusters. The leakage-safe path keeps **AE+k=6** as the product model and adds this k=2 as a research twin on the same matrix.

| | **Previous AE+k=6** | **Current k=2 twin** |
|---|---|---|
| **Run ID** | `c_ae_lat4_k6` | `c_kmeans_k2_leakage_safe` |
| **Artifacts** | `artifacts/segmentation/archive/fullbook_demo/` | `artifacts/segmentation/research/leakage_safe_source/` |
| **Method** | Autoencoder (lat=4) → K-Means **k=6** | K-Means **k=2** (no AE) |
| **Features** | Same 8: txn count, balance, credit, withdrawal, recency, age, loan, card | Same 8 |
| **As-of / split** | Full history; fit on all ~4,500 | as-of **1997-12-31**, tenure ≥365, **2701/901** train/holdout |
| **Population** | 4,500 accounts | 3,602 eligible (901 holdout for metrics below) |

---

## Metrics (not a single leaderboard)

Silhouette / DB / CH live in **different spaces and evaluation contracts**, so treat them as descriptive, not a win/loss scoreboard.

| Metric | Previous AE+k=6 | Current k=2 twin |
|---|---:|---:|
| Silhouette ↑ | **0.466** (full book, latent space) | **0.297** (holdout, scaled feature space) |
| Davies–Bouldin ↓ | **0.659** | **1.409** |
| Calinski–Harabasz ↑ | **3,775** | **208** |
| Seed stability ARI ↑ | **~0.999** (latent K-Means) | **~0.998** (raw K-Means) |
| Subsample ARI ↑ (75%) | **~0.978** (latent K-Means) | — |
| Same-matrix KMeans/GMM/Hier k=6 holdout sil | AE **0.459** vs KMeans 0.374 / GMM 0.322 / Hier(train) 0.363 | — |
| Segment count | 6 | 2 |
| Smallest segment | 12 (0.3%) | ~42% of holdout (balanced) |

**Read:** AE+k=6 separates more cleanly in its latent space and supports richer personas. k=2 is extremely seed-stable and balanced, but weaker separation and only a coarse scale/activity split.

For reference, the **leakage-safe AE+k=6** twin (same contract as k=2) has holdout silhouette **0.459** — close to the original champion, under Mahider-style discipline.

---

## What each split actually means

### Previous AE+k=6 — named CRM personas

| ID | n | Role (from profiles) |
|---:|---:|---|
| 0 | 721 | Card-centric savers |
| 1 | 1,431 | New / shorter-tenure basic |
| 2 | 671 | Loan-centric high-volume |
| 3 | 1,623 | Long-tenure high-activity basic |
| 4 | 42 | Financially stressed (exception queue) |
| 5 | 12 | Dormant (exception queue) |

Supports the decision layer: product, retention, inclusion, engagement, service, risk/care.

### Current k=2 twin — holdout medians (901 accounts)

| Cluster | n (holdout) | Txn count | Balance | Total credit | Account age | Loan/card |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 522 (58%) | **274** | **43.1k** | **749k** | **1,506 d** | ~0 / ~0 |
| 1 | 379 (42%) | **98** | **27.7k** | **180k** | **584 d** | ~0 / ~0 |

**Read:** mostly **higher-activity / longer-tenure / larger flows** vs **lighter / newer / smaller flows**. Product flags do not drive the two-way cut (medians ~0). This is a scale–lifecycle split, not six banking personas.

---

## Banking use

| Question | Prefer |
|---|---|
| “Show CRM/product six actionable groups + decision plays” | **Previous AE+k=6** (or leakage-safe AE+k=6 twin) |
| “Is there a stable, honest 2-way behavioral split under train-only discipline?” | **Current k=2 twin** |
| Credit eligibility / pricing from segment alone | **Neither** |

---

## Bottom line

- **AE+k=6** remains the product segmentation: better latent separation, product-aware personas, decision-map ready.  
- **k=2** is the honest research twin: balanced, ultra-stable, leakage-safe, but too coarse for the CRM playbook.  
- They are **complementary**, not substitutes — and the k=2 run is **K-Means**, not AE+k=2.

If you want a true **AE+k=2** bake-off on the same leakage-safe matrix, that can be added as a third row.
