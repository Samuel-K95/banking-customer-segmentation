# Master Description — Champion Segmentation & Decision Layer

**Audience:** product, CRM, risk, model-risk, engineering, and reviewers who need one place that explains *what the champion is, who the segments are, what we do with them, and what we must not do*.

**Authority:** numbers and names below come from the **served** bundle (`artifacts/segmentation/champion/`, `artifacts/decision/champion/`). If anything conflicted with older Track C demo reports, **this document + the live catalog win**.

| Field | Value |
|-------|--------|
| Run ID | `c_ae_lat4_k6_leakage_safe` |
| Model version | `production_candidate_v1` |
| Architecture | Autoencoder (latent **4**) → K-Means (**k = 6**) |
| Dataset | Berka (Czech PKDD’99 banking data), Track C |
| As-of / freeze cut | **1997-12-31**; tenure ≥ **365** days |
| Eligible book | **3,602** accounts (train 2,701 / holdout 901) |
| Freeze | **PROVISIONAL** — engineering candidate, not a full bank production freeze |
| One-pager | [`CURRENT_CHAMPION.md`](../../CURRENT_CHAMPION.md) |
| API | [`docs/api.md`](../../docs/api.md) · `bcs serve` |

---

## Table of contents

1. [What this system is (and is not)](#1-what-this-system-is-and-is-not)
2. [Why this champion](#2-why-this-champion)
3. [Data, features, and leakage discipline](#3-data-features-and-leakage-discipline)
4. [Model pipeline](#4-model-pipeline)
5. [Quality metrics](#5-quality-metrics)
6. [The six segments — deep profiles](#6-the-six-segments--deep-profiles)
7. [Cross-segment insights](#7-cross-segment-insights)
8. [Decision layer — domains, plays, map](#8-decision-layer--domains-plays-map)
9. [Explainability](#9-explainability)
10. [Fairness, ownership grain, and exception queues](#10-fairness-ownership-grain-and-exception-queues)
11. [Temporal check and freeze gates](#11-temporal-check-and-freeze-gates)
12. [Serving, monitoring, and how to use it](#12-serving-monitoring-and-how-to-use-it)
13. [Artifact map & related docs](#13-artifact-map--related-docs)
14. [Hard prohibitions checklist](#14-hard-prohibitions-checklist)

---

## 1. What this system is (and is not)

### It is

- An **unsupervised behavioral segmentation** of Berka accounts at a fixed as-of date.
- A **CRM / product / service decision layer**: named personas, default plays, multi-domain application portfolio (20 plays across 6 domains).
- A **production-candidate serving package**: frozen scaler + encoder + K-Means, SHA-256 contract, FastAPI single/batch inference, decision and report APIs.
- A **PROVISIONAL** freeze: engineering gates mostly pass; compliance / model-risk sign-off and true prospective vintage are still open.

### It is not

- A **credit score**, PD/LGD model, or pricing engine.
- A **supervised churn / default / propensity** predictor (no labeled outcomes drive the clusters).
- A license to **auto-approve** loans, cards, or limit increases from `segment_id` alone.
- A **demographic** classifier — age, gender, and district are **not** clustering inputs.
- A finished **bank freeze** — do not treat PROVISIONAL as “approved for automated credit decisions.”

**Rule of thumb:** segments answer *who behaves like whom and which CRM play fits*; underwriting, pricing, and collections still need their own models and human policy.

---

## 2. Why this champion

### Dataset: Berka over Santander / Indian Bank for *this* brief

The brief asks for groups that are **meaningful, stable, interpretable, and useful for banking applications**, with behaviors that distinguish them. Berka is the only track that jointly provides:

| Need | Berka | Santander | Indian Bank (as used) |
|------|-------|-----------|------------------------|
| Transaction activity / flows | Yes | No | Timing-heavy, weak lifecycle |
| Balance (incl. stress signal) | Yes | No | Partial |
| Product holding (loan / card) | Yes | Many flags, no txn | No |
| Recency / dormancy | Yes | Seniority only | Recency often collapsed |

Santander can win raw silhouette by splitting on **what people already own**. That does not answer “what **behaviors** distinguish these groups?” for a bank that wants onboarding, cross-sell, retention, and care plays. Berka does.

### Model: AE (lat=4) + K-Means k=6 (not k=2 as product)

| Path | Role |
|------|------|
| **AE + k=6 (this doc)** | **Product champion** — CRM-actionable personas (new / loyal basic / loan / card / stress / dormant) |
| **K-Means k=2 (research twin)** | Complementary leakage-safe baseline for rigor; **not** a replacement for CRM decision plays |

k=2 collapses product and lifecycle structure that the decision layer needs. k=6 keeps tiny stressed/dormant cells as **exception queues**, not mass-campaign audiences. See also [`05_vs_kmeans_k2.md`](05_vs_kmeans_k2.md).

### Leakage-safe promotion

The served bundle is the **leakage-safe twin** promoted into `artifacts/segmentation/champion/`:

- Fixed as-of `1997-12-31`
- Minimum tenure 365 days
- Train-only fit of scaler / AE / K-Means; holdout for metrics and replay tests
- Serving contract `production_candidate_v1`

Older full-book demos live under `artifacts/segmentation/archive/` and are **not** served.

---

## 3. Data, features, and leakage discipline

### Feature version

`berka_rfm_product_v1_leakage_safe` — eight **interpretable-scale** columns:

| Feature | Meaning (account grain) |
|---------|-------------------------|
| `Transaction_Count` | Number of transactions in the observation window up to as-of |
| `Account_Balance` | Balance level at / derived for as-of |
| `Total_Credit` | Sum of credit-side flows |
| `Total_Withdrawal` | Sum of withdrawal-side flows |
| `Recency_Days` | Days since last transaction relative to as-of |
| `Account_Age_Days` | Tenure to as-of |
| `has_loan` | 0/1 loan product flag |
| `has_card` | 0/1 card product flag |

**Not in the model:** age, gender, district, income, marital status. Geography may be monitored **post-hoc** for concentration risk; it does not drive cluster assignment.

### Preprocessing for scoring

1. Input arrives in **interpretable** units (API / CLI).
2. Frozen **log1p** (where applicable) + **StandardScaler** (`scaler.joblib`).
3. Frozen **encoder** → 4-D latent (`encoder.pt`).
4. Frozen **K-Means** assigns `segment_id` (`kmeans.joblib`).
5. Catalog attaches name, default play, optional application portfolio.

### Grain & joint accounts

Segmentation is at **account** grain (Berka `account_id`), with OWNER/DISPONENT policy documented separately so joint-account / multi-client realities are not silently ignored. See [`04_owner_disponent_policy.md`](04_owner_disponent_policy.md).

---

## 4. Model pipeline

```text
Berka raw tables
    → leakage-safe feature build (as_of=1997-12-31, tenure≥365)
    → train/holdout split (seed 8675309)
    → fit scaler + AE(lat=4) + KMeans(k=6) on TRAIN ONLY
    → assign all eligible accounts; profile centroids
    → name segments from centroid rules (behavior_rules_from_centroid)
    → register champion bundle + serving_contract.json
    → decision layer: profile packs, fairness, explainability, 20 plays
    → FastAPI / CLI score with contract checks
```

**Naming policy:** human names come from centroid behavior (activity, tenure, product flags, balance sign, recency) — not from demographics.

**Tiny-segment policy:** segments 4 and 5 are **exception queues**, not mass campaigns (`tiny_segments_policy = exception_queue_not_mass_campaign`).

---

## 5. Quality metrics

From `artifacts/segmentation/champion/metrics.json` (holdout-focused where noted):

| Metric | Value | Read |
|--------|-------|------|
| Holdout silhouette | **0.459** | Solid separation in usable range |
| Holdout Davies–Bouldin | **0.623** | Compact relative to separation (lower better) |
| Holdout Calinski–Harabasz | **852** | Higher better; strong between/within ratio |
| Train silhouette | 0.460 | Close to holdout → little train/holdout metric drift |
| Stability ARI (AE latent K-Means, seed pairs) | **0.997** (min 0.996) | Extremely stable under seed perturbation |
| Low-confidence fraction | ~5.3% | Soft assignments / near-boundary mass to watch |
| Parallel k=2 holdout silhouette | 0.297 | Weaker separation; research twin only |

**Size mix (full eligible book, n=3602):**

| Seg | Name | n | % book |
|-----|------|---|--------|
| 0 | New basic accounts | 1,176 | 32.65% |
| 1 | Long-tenure high-activity basic | 1,329 | 36.90% |
| 2 | Loan-centric high-volume | 489 | 13.58% |
| 3 | Card-centric active savers | 584 | 16.21% |
| 4 | Financially stressed / overdraft risk | 18 | 0.50% |
| 5 | Dormant long-term accounts | 6 | 0.17% |

Segments 0+1 ≈ **70%** of the book — the main CRM opportunity surface. Segments 4+5 ≈ **0.67%** — care / ops queues.

---

## 6. The six segments — deep profiles

Centroid values are **interpretable-scale means** from `segment_catalog.json` / `segment_profiles.csv`.

### Segment 0 — New basic accounts (32.65%, n=1,176)

| | |
|--|--|
| **One-liner** | Newer accounts with light product depth; still active — onboarding window is open. |
| **Behavioral signature** | Shortest tenure (~617 days), light txn count (~95), modest balance (~29k), almost no loan/card (~1% card), still recent (recency ≈ 1 day). |
| **Banking insight** | Acquisition worked; monetization has not. Highest risk of becoming tomorrow’s dormant book if ignored. |
| **Default play** | Onboarding + first card / digital activation; watch early dormancy. |
| **Owner** | CRM / Digital |
| **KPI examples** | Card attach within 90 days; 90-day activation; share still active at day 180 |
| **Guardrails** | Suitability only; **no aggressive unsecured credit** in first 90 days; no predatory high-fee products |

**Decision domains touched:** product recommendations, financial inclusion, targeted engagement, retention.

---

### Segment 1 — Long-tenure high-activity basic (36.90%, n=1,329) — largest cell

| | |
|--|--|
| **One-liner** | Loyal, high-activity, almost no loan/card. Trust exists; monetization does not. |
| **Behavioral signature** | Longest tenure among mass cells (~1,417 days), **highest** transaction count (~269), strong flows (credit ~687k / withdrawal ~649k), balance ~38k, **0%** loan and **0%** card on centroid, recency 0. |
| **Banking insight** | The **largest monetization gap** at lowest acquisition cost: relationship and activity already exist; product depth is near zero. Cold blasts under-convert; personalized, eligibility-aware offers fit. |
| **Default play** | Priority cross-sell: card → personal loan (sequenced), from txn history — **not** “activity = approval.” |
| **Owner** | Product / CRM (+ Service for experience protection) |
| **KPI examples** | Products per household; offer acceptance; NPS among high-activity basics |
| **Guardrails** | Consent + eligibility; explainable offer logic; service investment must not become unfair preferential pricing |

**Decision domains touched:** product recommendations, targeted engagement, service improvement.

---

### Segment 2 — Loan-centric high-volume (13.58%, n=489)

| | |
|--|--|
| **One-liner** | Credit relationship already established; high flows around loan servicing. |
| **Behavioral signature** | **~100%** loan holding, high txn (~243), **highest** credit/withdrawal flows (~1.03M / ~0.99M), balance ~45k, card penetration only ~26%, long-ish tenure (~1,060 days), recency 0. |
| **Banking insight** | Wallet is already credit-heavy. Upside is **card attach**, protection / refinance retention, and **servicing UX** — not “discover credit.” Over-leverage is the failure mode. |
| **Default play** | Card attach + payment protection / refinance retention; assisted servicing. |
| **Owner** | Lending + Cards / Lending Ops / Lending CRM |
| **KPI examples** | Card penetration among borrowers; refinance retention; missed-payment contacts resolved without collections |
| **Guardrails** | Avoid over-leverage; full PPI disclosure; not a default predictor |

**Decision domains touched:** product recommendations, targeted engagement, service improvement.

---

### Segment 3 — Card-centric active savers (16.21%, n=584)

| | |
|--|--|
| **One-liner** | High card penetration, healthy balances, no loan — solvent, engaged spenders. |
| **Behavioral signature** | **~100%** card, **highest** mean balance (~56k), strong flows, **0%** loan on centroid, active (recency 0), tenure ~1,061 days, txn ~204. |
| **Banking insight** | Best “solvent engager” profile for **premium card / wealth / careful first-loan** — but balance ≠ credit approval. Digital-first service and low-fatigue wealth content fit usage. |
| **Default play** | Premium card upgrade + careful first-loan invite with underwriting; wealth digest with suppress rules. |
| **Owner** | Cards / Wealth / Digital |
| **KPI examples** | Premium upgrade rate; responsible first-loan take-up; dispute resolution time |
| **Guardrails** | Credit assessment still required; investment suitability rules; do not infer demographics from card use |

**Decision domains touched:** product recommendations, targeted engagement, service improvement.

---

### Segment 4 — Financially stressed / overdraft risk (0.50%, n=18) — **exception queue**

| | |
|--|--|
| **One-liner** | Negative mean balance; withdrawals dominate credits. |
| **Behavioral signature** | Mean balance **≈ −4.9k**, withdrawals ≥ credits, still active (recency 0), mid loan (~44%) / low card (~6%), meaningful txn (~223). |
| **Banking insight** | Tiny but critical. Classic **care + risk** population. Sales incentives that encourage spend can worsen balances. Must never be treated as a mass “opportunity” segment. |
| **Default play** | Service & risk review; budgeting / hardship pathways; empathy-first scripts; retain via care — **not** discounts-for-spend. |
| **Owner** | Risk / Customer Care / Inclusion |
| **KPI examples** | Overdraft cure rate; hardship uptake; complaint rate; blocked new unsecured credit |
| **Guardrails** | **Do NOT** pitch investments or premium products; no predatory refinance; demographic labels forbidden in targeting copy; human review for high-risk actions |
| **Serving** | Listed in `exception_segment_ids` → API flags `guardrail` / human-review queue |

**Decision domains touched:** risk_and_care, financial_inclusion, service_improvement, retention_strategies.

---

### Segment 5 — Dormant long-term accounts (0.17%, n=6) — **exception queue**

| | |
|--|--|
| **One-liner** | Long recency gap — relationship effectively elsewhere. |
| **Behavioral signature** | Recency **≈ 391 days**, very low txn (~33), low flows, balance ~10k, no loan/card, long tenure (~1,391 days). |
| **Banking insight** | Not a sales audience. One win-back, then suppress / orderly closure. Strategy from a **dormancy proxy**, not a labeled churn model. |
| **Default play** | Win-back then closure path; suppress sales; low-cost ops workflow. |
| **Owner** | Retention / Ops / CRM Ops |
| **KPI examples** | Reactivation rate; dormant accounts closed cleanly; contact volume on dormant book |
| **Guardrails** | Hard suppress after one attempt; not a churn classifier; customer-request access must remain easy |
| **Serving** | Exception queue (same as segment 4) |

**Decision domains touched:** retention_strategies, targeted_engagement, service_improvement.

---

## 7. Cross-segment insights

1. **Product thinness is the mass story.** Segments 0 and 1 are ~70% of accounts and almost empty of loan/card — the bank’s relationship and activity are ahead of its product depth.
2. **Loan vs card splits the already-monetized minority.** Seg 2 ≈ all loan / partial card; Seg 3 ≈ all card / no loan. Plays should not mix these audiences.
3. **Activity ≠ solvency ≠ creditworthiness.** Seg 1 is hyper-active and product-thin; Seg 4 can still be active with negative balance. Never equate txn volume with lendability.
4. **Stress and dormancy are rare but strategically different.** Stress (4) needs care *now*; dormancy (5) needs one reactivation then ops. Both are exception queues.
5. **Surrogate tree importance (stakeholder explanation only)** ranks `has_card`, `has_loan`, `Transaction_Count` as top separators — consistent with the persona story. Recency/age matter structurally for dormancy/newness even when a shallow surrogate tree under-weights them.

---

## 8. Decision layer — domains, plays, map

### Design principles

- Every play cites **segment evidence** (behavior / product / balance / recency).
- Plays are **strategy / design**, not supervised predictors unless a separate model exists.
- Each play has **owner**, **KPI**, and **guardrail**.
- Portfolio covers the brief’s domains **plus** explicit `risk_and_care`.

### Domain coverage (20 plays total)

| Domain | Plays | Segments touched |
|--------|------:|-----------------:|
| product_recommendations | 4 | 0, 1, 2, 3 |
| retention_strategies | 3 | 0, 4, 5 |
| financial_inclusion | 2 | 0, 4 |
| targeted_engagement | 5 | 0, 1, 2, 3, 5 |
| service_improvement | 5 | 1, 2, 3, 4, 5 |
| risk_and_care | 1 | 4 |

Source: `artifacts/decision/champion/domain_coverage.csv` / `decision_map_full.csv`.

### Full play catalog (by segment)

#### New basic accounts (0)

| Domain | Play | Action (short) |
|--------|------|----------------|
| product_recommendations | First card / debit attach during onboarding | Day-7 / Day-30 digital card offers |
| financial_inclusion | Basic-account literacy + micro-savings nudge | In-app savings goals + fee transparency |
| targeted_engagement | Onboarding cadence (high-touch early, then taper) | 3-message journey then suppress |
| retention_strategies | Early-dormancy watchlist | If recency drifts >30d in first year → win-back |

#### Long-tenure high-activity basic (1)

| Domain | Play | Action (short) |
|--------|------|----------------|
| product_recommendations | Priority cross-sell: card then personal loan | Pre-approved *candidate* card from txn history; sequenced loan later |
| targeted_engagement | Relationship-manager / personalized data-driven outreach | Personalized “you may qualify” tied to observed flows |
| service_improvement | Protect high-activity experience | Priority digital support; fee-waiver pilots for service failures |

#### Loan-centric high-volume (2)

| Domain | Play | Action (short) |
|--------|------|----------------|
| product_recommendations | Card attach + payment-protection / refinance retention | Card for borrowers; PPI disclosure; refinance at rate triggers |
| targeted_engagement | Loan-lifecycle messaging | Stage-based comms; suppress generic investment pitches |
| service_improvement | Assisted servicing for high-flow loan accounts | Repayment support + payment-plan UX |

#### Card-centric active savers (3)

| Domain | Play | Action (short) |
|--------|------|----------------|
| product_recommendations | Premium card upgrade + careful first loan | Rewards/premium upgrade; underwritten first-loan invite |
| targeted_engagement | Wealth / savings content with low contact fatigue | Monthly digest; suppress after two non-opens |
| service_improvement | Digital-first premium service | Card controls UX + dispute speed |

#### Financially stressed / overdraft risk (4)

| Domain | Play | Action (short) |
|--------|------|----------------|
| risk_and_care | Service & risk review; budgeting tools — no investment pitch | Care outreach; budgeting / consolidation review |
| financial_inclusion | Responsible restructuring / hardship pathways | Hardship form + counseling; secured short-bridge only if eligible |
| service_improvement | Empathy-first contact scripts | Fee-transparency checklist before any sales |
| retention_strategies | Retain via care, not discounts-for-spend | Fee relief tied to repayment plans |

#### Dormant long-term accounts (5)

| Domain | Play | Action (short) |
|--------|------|----------------|
| retention_strategies | Win-back then orderly closure | One offer → closure / dormancy path |
| targeted_engagement | Suppress sales; one reactivation channel only | Remove from cross-sell; one email/SMS |
| service_improvement | Low-cost ops handling for zombie accounts | Automated dormancy workflow |

### Default play vs portfolio

Each segment also has a **catalog default** (`decision.play` in `segment_catalog.json`) for one-line dashboards. The **portfolio** expands that into multi-domain actions. Prefer the full map for operating plans; use the default for summaries.

Machine-readable map: `artifacts/decision/champion/decision_map_full.csv`  
API: `GET /v1/decisions/map`, `GET /v1/decisions/map/full`, `GET /v1/segments/{id}/applications`

---

## 9. Explainability

From `artifacts/decision/champion/explainability.json`:

- **Disclaimer:** a surrogate decision tree explains segment *labels* for stakeholders; it is **not** the clustering objective.
- **Surrogate train accuracy** ≈ **95.2%** at matching champion labels on the feature matrix.
- **Top importances:** `has_card` (~0.36), `has_loan` (~0.30), `Transaction_Count` (~0.29); flow/balance features smaller; shallow tree may put ~0 on Recency/Age even though those define dormancy/newness in centroids.
- **Latent correlations:** AE dimensions correlate with activity, product flags, balance, and recency in interpretable patterns (see JSON for z0–z3 tables).

Use explainability for **governance and storytelling**, not as a substitute for the frozen AE+KMeans scorer.

---

## 10. Fairness, ownership grain, and exception queues

### Behavior vs demographic

- Clustering features are **behavioral + product only**.
- Post-hoc **district concentration** is monitored; no district is a model input.
- Highest concentration among tiny cells is still modest (e.g. stress segment max district share ~22% on n=18) — treat as monitoring, not a demographic targeting feature.
- Compliance sign-off remains **PENDING** in freeze pack.

### Exception queues (serving contract)

```json
"exception_segment_ids": [4, 5]
```

Inference responses mark these with a human-review guardrail reason. Do **not** put them on automated mass-marketing lists.

### Owner / disponent

Joint and multi-client Berka accounts require explicit policy so account-level segments are not naïvely attributed to every client. See [`04_owner_disponent_policy.md`](04_owner_disponent_policy.md).

---

## 11. Temporal check and freeze gates

### Temporal (best-effort)

Scoring the same accounts at a 1998 as-of vs 1997:

| Metric | Value |
|--------|-------|
| Agreement rate | ~**81.8%** |
| ARI 1997 vs 1998 | ~**0.58** |
| n common | 3,602 |

**Limitation (freeze pack):** this is a **retrospective** Berka vintage check, not an independent prospective / external holdout. Gate = `PASS_WITH_LIMITATIONS`.

### Freeze readiness — PROVISIONAL

| Gate | Verdict |
|------|---------|
| Artifact integrity (SHA-256) | PASS |
| Snapshot split / train-only fit | PASS |
| Exact holdout prediction replay | PASS |
| Segment meaning / exception policy | PASS |
| Fairness + owner/disponent docs | PASS (compliance sign-off PENDING) |
| Independent temporal validation | PASS_WITH_LIMITATIONS |
| Serving package | PASS |

**Not fully frozen because:**

1. Human compliance / model-risk sign-off still PENDING  
2. Temporal validation is retrospective, not prospective/external  
3. Must not use `segment_id` for credit eligibility, pricing, or collections  

Details: [`02_freeze_readiness.md`](02_freeze_readiness.md), `artifacts/segmentation/champion/freeze_readiness.json`.

---

## 12. Serving, monitoring, and how to use it

### Contract (required on every score call)

| Field | Value |
|-------|--------|
| `model_version` | `production_candidate_v1` |
| `as_of_date` | `1997-12-31` |

Mismatch → HTTP 400. Confirm live values: `GET /v1/inference/contract`.

### CLI

```bash
bcs champion-info
bcs freeze-status
bcs decision-berka
bcs score-berka --input data/processed/... --output ...
bcs serve --port 8000
```

### API groups (prefix `/v1`)

| Group | Examples |
|-------|----------|
| Meta | `/health`, `/catalog` |
| Model | `/model`, `/segments`, `/model/serving-contract` |
| Decisions | `/decisions/summary`, `/decisions/map`, `/decisions/fairness`, `/decisions/explainability` |
| Reports | `/reports`, `/reports/{slug}/raw` |
| Inference | `POST /inference`, `/inference/batch`, `/inference/batch/csv` |
| Monitoring | `/monitoring/segment-mix`, `/monitoring/snapshot` |

Batch responses include **segment mix vs baseline** drift for ops monitoring (`baseline_segment_mix` in serving contract).

### Python

```python
from customer_segmentation.models.champion import ChampionModel
champ = ChampionModel.load()  # artifacts/segmentation/champion
scored = champ.score_interpretable(df)  # interpretable features in
```

---

## 13. Artifact map & related docs

| Path | Contents |
|------|----------|
| `artifacts/segmentation/champion/` | Served model: encoder, scaler, kmeans, catalog, metrics, freeze, serving contract, assignments |
| `artifacts/decision/champion/` | Profile packs, fairness, explainability, decision_map_full, exemplars, figures |
| `artifacts/segmentation/research/` | Leakage-safe training source / k=2 twin — **not served** |
| `artifacts/segmentation/archive/` | Old full-book demo — **archive only** |
| `reports/champion/` | Supporting write-ups (why / freeze / temporal / k2 / modularity) |
| `notebooks/champion/` | Decision-layer notebook |
| `configs/project/champion.yaml` | Project champion config |
| `configs/models/berka/champion_v1_leakage_safe.yaml` | Leakage-safe training config |
| `docs/api.md` | API reference |

**Sibling reports in this folder:**

| File | Role |
|------|------|
| [`00_why_this_champion.md`](00_why_this_champion.md) | Dataset / model selection narrative |
| [`01_decision_layer.md`](01_decision_layer.md) | Shorter decision-layer note |
| [`02_freeze_readiness.md`](02_freeze_readiness.md) | Freeze narrative |
| [`03_temporal_validation.md`](03_temporal_validation.md) | 1998 check |
| [`04_owner_disponent_policy.md`](04_owner_disponent_policy.md) | Account grain policy |
| [`05_vs_kmeans_k2.md`](05_vs_kmeans_k2.md) | Why k=6 stays product champion |
| [`06_modularity_and_readiness.md`](06_modularity_and_readiness.md) | Package / readiness checklist |

---

## 14. Hard prohibitions checklist

- [ ] Do **not** use `segment_id` as credit eligibility, pricing, or collections automation.  
- [ ] Do **not** mass-market segments **4** or **5**.  
- [ ] Do **not** pitch investments / premium products to the stressed segment.  
- [ ] Do **not** treat activity or balance as underwriting approval.  
- [ ] Do **not** put age / gender / district into the clustering feature set without a new model version and fairness review.  
- [ ] Do **not** claim a full bank freeze while status is **PROVISIONAL**.  
- [ ] Do **not** serve archive or research paths as the production champion.  
- [ ] Do **not** present the surrogate tree as the production scorer.

---

## One-page executive summary

Berka accounts as of **1997-12-31** (tenure ≥ 1 year) are compressed by an **AE (4-D)** and clustered into **six** behavioral–product segments. About **70%** of the book is product-thin (new or long-tenure basic) — the core **onboarding and cross-sell** opportunity. About **30%** already holds loan and/or card and needs **attach, upgrade, servicing, and lifecycle** plays. Under **1%** sits in **stress** or **dormancy** exception queues for care and ops. Twenty multi-domain plays turn those personas into owned actions with KPIs and guardrails. The package is **leakage-safe**, replay-checked, and API-served under `production_candidate_v1`, with freeze status **PROVISIONAL** until compliance sign-off and stronger prospective validation.

---

*Generated from live champion artifacts. Prefer catalog / decision CSVs / freeze JSON if this narrative and a file ever disagree after a retrain.*
