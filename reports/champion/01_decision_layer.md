# Analysis & Decision Layer — Berka Production Champion (`c_ae_lat4_k6_leakage_safe`)

Aligned to `04-analysis-and-decision-layer.md`.  
**THIS IS THE CURRENT CHAMPION.**  
**Model:** AE(latent=4) + K-Means k=6 (leakage-safe) · **Bundle:** `artifacts/segmentation/champion/`  
**Decision artifacts:** `artifacts/decision/champion/` · regenerate with `bcs decision-berka`  
**Freeze status:** **PROVISIONAL** (`bcs freeze-status`) — not a full bank production freeze.  
See also: [`../CURRENT_CHAMPION.md`](../../CURRENT_CHAMPION.md) · [`MASTER_CHAMPION_SEGMENTATION_AND_DECISIONS.md`](MASTER_CHAMPION_SEGMENTATION_AND_DECISIONS.md)

This layer turns `account_id → segment_id` into **profiles, explanations, fairness constraints, and multi-domain bank plays**. It is **not** a churn/default/propensity model.

---

## 1. Segment intelligence (profile packs)

| ID | Name | n | % book | One-liner |
|---:|---|---:|---:|---|
| 0 | New basic accounts | 1,176 | 32.65% | Newer, product-thin, still active |
| 1 | Long-tenure high-activity basic | 1,329 | 36.90% | Loyal, busy, almost no loan/card |
| 2 | Loan-centric high-volume | 489 | 13.58% | Credit relationship; high flows |
| 3 | Card-centric active savers | 584 | 16.21% | Card-engaged, highest balances, no loan |
| 4 | Financially stressed / overdraft risk | 18 | 0.50% | Negative mean balance |
| 5 | Dormant long-term accounts | 6 | 0.17% | Long recency gap |

Full packs: `artifacts/decision/champion/profile_packs.json`.

**Size note:** Segments 4–5 are tiny. Treat as **review / exception queues**, not mass campaigns.

---

## 2. Explainability

Surrogate tree + latent correlations: `artifacts/decision/champion/explainability.json`  
Exemplars: `artifacts/decision/champion/exemplars.csv`

---

## 3. Fairness

`artifacts/decision/champion/fairness.json` — behavior-only clustering; district concentration post-hoc; credit/pricing misuse forbidden.

---

## 4. Banking applications

**20 plays · 6 domains.** Source: `artifacts/decision/champion/decision_map_full.csv`.

---

## 5. Acceptance criteria (`04` §10)

| Criterion | Status |
|---|---|
| Full profile pack per segment | Yes — `profile_packs.json` |
| Human-usable names | Yes |
| Membership explanations | Yes — surrogate + lift + exemplars |
| Fairness analysis + risks | Yes — `fairness.json` |
| Behavior vs demographic considered | Yes |
| ≥3 banking applications | Yes — **6 domains / 20 plays** |
| Decision map published | Yes — `decision_map_full.csv` |
| Boundary vs predictive modeling | Yes |

---

## 6. Commands

```bash
bcs decision-berka
bcs verify-champion
# GET /v1/decisions/map  ·  GET /v1/decisions/summary
```
