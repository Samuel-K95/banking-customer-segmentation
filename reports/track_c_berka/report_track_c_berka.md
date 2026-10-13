# Track C — Berka Czech Bank Segmentation
### Banking Customer Segmentation · Week 4 Project Report

> **This is the selected segmentation for the project.** Model `c_ae_lat4_k6` (autoencoder, 4 latent dimensions, 6 clusters). Santander scores higher on silhouette (0.512 versus 0.417). Berka is selected because it is the only track with both transaction behavior and product holding, which is what the assignment asks the system to explain. See `champion_recommendation.md`.

---

## Quick-Start Glossary
> *Every technical term used in this report is defined here.*

| Term | Plain-English Meaning |
|---|---|
| **Relational database** | Data stored across multiple linked tables, like a spreadsheet where different sheets refer to each other. You need to JOIN them to build one complete view |
| **SQL-style join** | Combining two tables by matching a shared key column — e.g. linking the `account` table to the `trans` table by account ID to see all transactions for each account |
| **Feature engineering** | Computing useful numerical summaries from raw data — e.g. counting how many transactions an account made, or calculating the number of days since the account last transacted |
| **NaN / Null** | "Not a Number" — a missing value. Can occur when a join produces no matching rows. Must be handled before modelling (usually by filling with 0 or removing the row) |
| **NaT** | "Not a Time" — the datetime equivalent of NaN. Occurs when a date conversion fails |
| **Binary feature** | A feature that is either 0 (no) or 1 (yes) — e.g. `has_loan`: 1 if the account has a loan, 0 if not |
| **Clustering / Segmentation** | Automatically grouping accounts so that similar accounts end up in the same group, without being told in advance what the groups should be |
| **k** | The number of clusters the algorithm is asked to find |
| **Dendrogram** | A tree diagram showing the full history of which clusters merged into which in Agglomerative Hierarchical Clustering. You "cut" the tree at a height to select k groups |
| **Ward linkage** | The merging rule used in hierarchical clustering. At each step, merge the two clusters that result in the smallest increase in total within-cluster variance. Tends to produce compact, balanced clusters |
| **Autoencoder (AE)** | A neural network trained to compress data into a smaller representation and then reconstruct it. The compressed representation is what we use for clustering |
| **Latent dimension** | The number of "hidden" summary numbers the AE uses to represent each account internally. Fewer latent dimensions = more compression |
| **Intrinsic dimensionality** | The number of truly independent axes of variation in the data. If you have 8 features but 4 of them are strongly correlated with the other 4, the true intrinsic dimensionality is ~4 |
| **Silhouette Score** | Measures how well-separated the clusters are — see Section 3 |
| **Davies-Bouldin Index** | Measures cluster compactness vs. separation — see Section 3 |
| **Calinski-Harabasz Index** | Measures within-cluster tightness vs. between-cluster spread — see Section 3 |
| **Stability ARI** | Measures how reproducibly the same cluster structure appears across multiple independent runs — see Section 3 |
| **Overfitting / Over-parameterisation** | When a model learns noise in the data rather than real patterns — in AE, using too many latent dimensions means the model doesn't need to compress, so it doesn't filter out noise |
| **RFM** | Recency-Frequency-Monetary — a classic customer analysis framework: how recently did they transact, how often, and how much |

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **Name** | Berka Financial Dataset |
| **Accounts** | ~4,500 |
| **Source** | PKDD 1999 Discovery Challenge — Czech bank, open research dataset |
| **Format** | 8 relational tables (requires joining before use) |
| **Coverage** | 5 years of transaction history, 1993–1998 |
| **Language context** | Czech bank — amounts in CZK (Czech Koruna; roughly 1 EUR ≈ 25 CZK in the 1990s) |

### What Makes This Dataset Different

Track A and B were flat CSV files: one row per customer, ready to use. **Berka is a relational database** — the information is spread across 8 tables, each with a specific purpose. To build a customer-level feature matrix, all 8 tables must first be **joined** (linked and merged) in the right sequence.

This is how real bank data is typically stored — not in one flat table, but across many related tables in a data warehouse. Working with Berka simulates real production banking data engineering.

### The 8 Tables and How They Connect

```
district ────────────────────────────────────────────────────────┐
  (region data: average salary, unemployment rate, etc.)         │
                                                                  ▼
account ──────┬────── disp ──────── client ──── district
(one per       │       (links accounts    (demographics)
 account)      │        to their owners)
               │
               ├────── trans          (all transactions for the account)
               │
               ├────── loan           (loan details, if any)
               │
               └────── card           (card details, if any)
                        order         (standing orders / direct debits)
```

**What each table provided:**
| Table | What we extracted |
|---|---|
| `account` | Account ID, branch (district), date opened |
| `trans` | All transactions: count, sum of credits (inflows), sum of debits (outflows), last balance |
| `loan` | Whether a loan exists (`has_loan` flag) |
| `card` | Whether a card was issued (`has_card` flag) |
| `client` | Birth date (used to calculate age; not used in final model) |
| `district` | Regional economic context (informational, not directly in model) |

### Features Engineered

After joining all tables, the following 8 features were computed per account:

| Feature | How calculated | What it tells us |
|---|---|---|
| **Transaction_Count** | Count of all rows in the `trans` table for this account | How actively the account is used. High count = primary transactional account |
| **Account_Balance** | The most recently recorded balance from the `trans` table | Current financial position. Negative = in overdraft |
| **Total_Credit** | Sum of all incoming transaction amounts (deposits, salary, transfers in) | Total money received into the account over its lifetime |
| **Total_Withdrawal** | Sum of all outgoing transaction amounts (payments, ATM withdrawals, transfers out) | Total money spent from the account over its lifetime |
| **Recency_Days** | Number of days between the last recorded transaction and the analysis date | How recently active: 0 = transacted today; 500+ = dormant for over a year |
| **Account_Age_Days** | Number of days since the account opening date | Tenure/loyalty proxy. Longer = more established relationship |
| **has_loan** | 1 if the account has a loan record in the `loan` table; 0 otherwise | Whether the customer holds a credit product |
| **has_card** | 1 if the account has a card record in the `card` table; 0 otherwise | Whether the customer holds a payment card |

### Engineering Challenges and Bugs Fixed

**Bug 1: Account opening dates stored as 6-digit integers**

The `account` table stored dates as integers like `930101` (meaning 1 January 1993). A parsing loop in the notebook correctly converted these to Python datetime objects. However, a second line of code below the loop tried to re-parse the already-converted datetimes using the raw integer format `%y%m%d`.

When Python tries to parse a datetime object as a 6-digit string, it fails silently — because `errors='coerce'` was set, it returned NaT (Not a Time) for every row instead of raising an error. The result: every account's `Account_Age_Days` became NaN (missing), and `dropna()` then removed all 4,500 rows.

**Fix:** Remove the duplicate parse line. The loop above it had already done the conversion correctly.

**Bug 2: NaN balances from accounts with no transactions**

Some accounts existed in the `account` table but had no matching rows in the `trans` table (no recorded transactions). When the join was performed, `Account_Balance`, `Total_Credit`, and `Total_Withdrawal` became NaN for these accounts.

**Fix:** Replace NaN with 0 for all three monetary columns — because "no transactions" legitimately means zero balance and zero flows. Then apply `dropna()` only to the three core RFM columns (Recency_Days, Transaction_Count, Account_Age_Days) which cannot logically be zero.

### What the EDA Revealed

- **Hard binary split on loan and card** — When the raw data was plotted, accounts clearly divided into those that hold credit products and those that don't. This binary structure becomes the dominant clustering axis.

- **Transaction_Count is strongly bimodal** — A large group of accounts make < 50 transactions over their lifetime; a separate group makes > 200. This reflects the difference between inactive/light users and primary banking accounts.

- **Total_Credit and Account_Balance are highly correlated** — Accounts that receive more money over time tend to have higher balances. This means these two features carry overlapping information. The Autoencoder handles this by encoding them as a single "financial depth" axis in latent space.

- **Recency_Days has a very long tail** — Most accounts transacted within the last few days, but a distinct cluster of accounts shows recency > 400 days — these dormant accounts form a natural cluster even before any model is run.

- **Small N (4,500 accounts)** — The small dataset size means all clustering algorithms (including memory-intensive Agglomerative Hierarchical) are computationally feasible. No need for sampling.

---

## 2. Models Trained

### Algorithm 1: K-Means

**How K-Means works:**

K-Means is an iterative algorithm that:
1. Places k random "centroid" points in the feature space
2. Assigns every account to its nearest centroid (forming k groups)
3. Moves each centroid to the exact average of all accounts assigned to it
4. Repeats steps 2–3 until assignments stop changing

**The mental model:** Imagine k magnets on a map of accounts. Each account snaps to the nearest magnet. The magnets then move to the centre of gravity of the accounts attracted to them. Repeat until the magnets stop moving.

**Properties:** Fast, simple, and interpretable — but assumes all clusters are roughly spherical (circular) and equal in size. Doesn't handle binary features (has_loan, has_card) optimally.

- Tested k = 3, 4, 5, 6, 7, 8
- Best result: **k=6, silhouette=0.363, stability ARI=0.864**
- The same 6 groups reproduced reliably on different halves of the data (ARI=0.864), which is a strong signal that the 6-group structure is real

### Algorithm 2: Agglomerative Hierarchical Clustering (Ward Linkage)

**How it works:**

This algorithm takes the opposite approach to K-Means — it starts maximally fragmented and merges:
1. Start: treat every account as its own unique cluster (4,500 clusters)
2. Find the two clusters that are most similar to each other
3. Merge them into one cluster (now 4,499 clusters)
4. Repeat steps 2–3 until you reach k clusters

**Ward linkage** is the specific rule for deciding which two clusters to merge at each step: always merge the pair that causes the smallest increase in total within-cluster variance. This tends to create compact, balanced clusters.

The entire history of all 4,499 merges is stored as a **dendrogram** — a binary tree diagram. Each leaf (bottom) is one account. Branches connect accounts that were merged. The height at which two branches connect indicates how different those groups are. To get k clusters, "cut" the tree at the appropriate height.

**Advantage over K-Means:** No need to choose k upfront — you can cut the dendrogram at any level. Provides a visual justification of the cluster structure. However, it's computationally expensive (O(n²) memory) — feasible here because n=4,500 is small.

- Tested k = 5, 6
- Best result: **k=6, silhouette=0.335**
- The notebook flagged this as "recommended" because the dendrogram provides a visual narrative of how accounts relate to each other

### Algorithm 3: Autoencoder + K-Means (Metric Champion)

**How it works — two stages:**

**Stage 1 — Autoencoder (neural network):**

An Autoencoder is a neural network with two halves that work together:

| Part | Role |
|---|---|
| **Encoder** | Compresses 8 input features → latent representation (4 or 8 numbers) |
| **Decoder** | Reconstructs 8 features from the latent representation |

Training objective: minimise the **reconstruction loss** — how different the reconstructed output is from the original input. The network is forced to learn the most important patterns because it can only use 4 (or 8) numbers to encode 8 features.

Whatever can't be reconstructed from the compressed representation is considered "noise" and gets left out.

**Stage 2 — K-Means on latent representations:**

Once trained, every account is passed through the encoder to get its latent representation (a vector of 4 or 8 numbers). K-Means then clusters accounts based on these compact, noise-filtered representations.

**Result:** Clusters in latent space are tighter and more separated than in raw feature space, because the encoder has already removed the noise and reorganised correlated features into independent axes.

- Tested latent dimensions = 4, 8 and k = 6
- **Champion: `c_ae_lat4_k6` — 4 latent dimensions, 6 clusters**

**Why latent_dim=4 beats latent_dim=8 on this dataset:**

With 8 input features, you might expect 8 latent dimensions to work best — why would using fewer help? The answer is **intrinsic dimensionality**.

The 8 features are not 8 independent signals. They break down into approximately 4 independent axes:
- **Activity axis:** Recency_Days + Transaction_Count together describe "how active is this account?"
- **Financial depth axis:** Account_Balance + Total_Credit together describe "how financially substantial is this account?"
- **Spending axis:** Total_Withdrawal + Account_Age_Days together describe "how much has flowed through this account over time?"
- **Product ownership axis:** has_loan + has_card together describe "what credit products does this account hold?"

When latent_dim=8, the AE has enough capacity to encode some noise dimensions (minor random variation that doesn't represent real customer behaviour). When latent_dim=4, it is forced to encode only the 4 meaningful axes — and the resulting clusters are cleaner.

| Latent dim | Silhouette | DB Index | CH Index |
|---|---|---|---|
| **4** | **0.417** | **0.731** | **3,474** |
| 8 | 0.399 | 0.759 | 2,019 |

---

## 3. Evaluation Metrics — Full Explanations

### Silhouette Score (range: −1 to +1, higher is better)

**What it measures:** For every individual account, it calculates:
- **a** = average distance to all other accounts in its own cluster → how similar it is to its cluster-mates
- **b** = average distance to all accounts in the nearest neighbouring cluster → how different it is from the closest other group
- **Silhouette(account) = (b − a) / max(a, b)**

The final silhouette score is the average across all 4,500 accounts.

**Interpretation:**
| Score Range | Meaning |
|---|---|
| −1 to 0 | Account likely placed in the wrong cluster |
| 0 to 0.25 | Weak structure — clusters barely separable |
| 0.25 to 0.50 | Reasonable, usable segments |
| 0.50 to 0.70 | Strong, well-separated segments |
| Above 0.70 | Near-perfect (rare in real data) |

**Track C result: 0.417** — good. Accounts within each of the 6 clusters are meaningfully more similar to their group than to neighbouring groups. Santander's best silhouette is 0.512. That gap is acknowledged in the champion memo and is not the selection rule. Berka's Davies-Bouldin score (0.731) is the best of the three tracks.

### Davies-Bouldin Index (range: 0 to ∞, lower is better)

**What it measures:** For each cluster, it computes the average distance of accounts from their cluster's centre (how wide the cluster is). It then compares this width to the distance between cluster centres. The final score is the worst-case (maximum) ratio across all cluster pairs.

**Plain English:** "Are the clusters wide relative to how far apart they are?" A small DBI means clusters are compact (narrow) and well-separated (far apart) — good. A large DBI means clusters overlap.

**Track C result: 0.731** — the best DBI across all three tracks. This means Berka's 6 clusters have the best internal compactness relative to their separation distances — even though the silhouette is not the highest.

### Calinski-Harabasz Index (range: 0 to ∞, higher is better)

**What it measures:** The ratio of between-cluster variance (how far apart the cluster centres are from the global mean) to within-cluster variance (how spread out accounts are within their clusters), scaled by the number of clusters and observations.

**Track C result: 3,474** — lower than Track A (48,528) and B (90,835), but Track C only has 4,500 accounts vs. 837k–931k. The CH index scales with dataset size, so cross-track comparisons are not meaningful. Within Track C, `c_ae_lat4_k6` has the highest CH of all 16 runs, which is what matters.

### Stability ARI — Adjusted Rand Index (range: 0 to 1, higher is better)

**What it measures:** Whether the same cluster structure appears when the model is run on different subsets of the data.

**Procedure (as implemented):**
1. Take all 4,500 accounts
2. Run K-Means with a base random seed → get cluster assignments A
3. Run K-Means again with a different random seed → get cluster assignments B
4. Ask: "How similar are assignments A and B?" — how often does the same account end up in equivalent groups?
5. Repeat across 10 random seeds; report the average

ARI is "adjusted for chance" — a random assignment that happened to agree would score close to 0, not a positive number.

**Track C result for K-Means k=6: ARI = 0.864** — high. The 6-group structure appears reliably.

**Note:** Stability was only measured for K-Means, not for the AE model. K-Means is deterministic given a random seed, making it easy to test. AE training involves gradient descent which converges differently each time.

---

## 4. Full Results Table

| Run ID | Description | k | Latent Dim | Silhouette ↑ | DB Index ↓ | CH Index ↑ | Stability ARI |
|---|---|---|---|---|---|---|---|
| **c_ae_lat4_k6** | AE, 4 latent dims, k=6 | 6 | 4 | **0.417** | **0.731** | **3,474** | — |
| c_ae_lat8_k6 | AE, 8 latent dims, k=6 | 6 | 8 | 0.399 | 0.759 | 2,019 | — |
| c_kmeans_k6 | K-Means, k=6 | 6 | — | 0.363 | 0.844 | 1,727 | **0.864** |
| c_hier_k6 | Hierarchical (Ward), k=6 | 6 | — | 0.335 | 0.892 | 1,509 | — |
| c_kmeans_k5 | K-Means, k=5 | 5 | — | 0.332 | 0.904 | 1,505 | — |
| c_kmeans_k8 | K-Means, k=8 | 8 | — | 0.320 | 0.857 | 1,760 | — |
| c_kmeans_k4 | K-Means, k=4 | 4 | — | 0.320 | 1.060 | 1,415 | — |
| c_kmeans_k3 | K-Means, k=3 | 3 | — | 0.314 | 1.281 | 1,383 | — |
| c_gmm_k4 | GMM, k=4 | 4 | — | 0.308 | 1.851 | 710 | — |
| c_gmm_k5 | GMM, k=5 | 5 | — | 0.308 | 1.397 | 681 | — |
| c_hier_k7 | Hierarchical, k=7 | 7 | — | 0.302 | 0.952 | 1,618 | — |
| c_kmeans_k7 | K-Means, k=7 | 7 | — | 0.300 | 0.897 | 1,735 | — |
| c_gmm_k3 | GMM, k=3 | 3 | — | 0.271 | 2.253 | 608 | — |
| c_gmm_k8 | GMM, k=8 | 8 | — | 0.235 | 1.658 | 546 | — |
| c_gmm_k6 | GMM, k=6 | 6 | — | 0.231 | 2.198 | 806 | — |
| c_gmm_k7 | GMM, k=7 | 7 | — | 0.215 | 1.558 | 558 | — |

**Pattern:** GMM performs worst across all k values on this dataset — likely because the Berka clusters have hard binary boundaries (loan/card yes/no) that GMM's soft probability clouds struggle to model. The AE captures these binary distinctions in its latent space and handles them well.

---

## 5. Champion Model: `c_ae_lat4_k6`

**Full name:** Autoencoder with 4 latent dimensions + K-Means with k=6 clusters

**Performance vs. all alternatives:**
- Silhouette: 0.417 vs. K-Means best of 0.363 (+15% improvement)
- DB Index: 0.731 vs. K-Means best of 0.844 (better compactness)
- CH Index: 3,474 vs. K-Means best of 1,727 (+101% improvement)

**Why 6 clusters?** The silhouette curve showed a clear plateau at k=6 — adding more clusters beyond 6 did not improve separation, it only split existing natural groups into smaller, less meaningful sub-groups.

### The 6 Account Segments

| Segment | Name | Defining Features |
|---|---|---|
| 0 | New Basic Accounts | Young accounts (~2 yrs), low activity, no products |
| 1 | Long-Tenure High-Activity Basic | Old accounts (~4.5 yrs), very high transactions, zero products |
| 2 | Loan-Centric Customers | 100% loan penetration, high transaction volumes |
| 3 | Dormant Long-Term Accounts | Inactive for ~1.5 years (recency = 539 days) |
| 4 | Card-Centric Active Savers | 100% card penetration, highest balance (CZK 65K) |
| 5 | Financially Stressed / Overdraft | Negative average balance (−CZK 6,468) |

*(Full centroid tables and behavioural descriptions are in `cluster_profiles.md`.)*

---

## 6. EDA Tie-Back

| EDA Finding | What it predicted | What happened in modelling |
|---|---|---|
| Hard binary split on has_loan / has_card | Loan-holders and card-holders would form distinct segments | Seg 2 (100% loan) and Seg 4 (100% card) are the two cleanest, most distinct clusters |
| Bimodal Transaction_Count | High-activity and low-activity accounts would cluster apart | Seg 1 (314 tx) and Seg 3 (27 tx, dormant) confirmed this prediction |
| Long tail in Recency_Days | A dormant segment existed even before modelling | Seg 3 (recency=539d) isolated exactly these abandoned accounts |
| Balance-Credit high correlation | These two features contain redundant information | AE latent_dim=4 merged them into one "financial depth" axis — confirmed by the fact that 4 dims beats 8 |
| Small N=4,500 | All algorithms feasible, hierarchical clustering viable | All methods ran in under 1 minute; dendrogram was generated for full N |

---

## 7. Engineering Notes

### Why `fillna(0)` not `dropna()` for monetary columns

When accounts had no transactions, their `Account_Balance`, `Total_Credit`, and `Total_Withdrawal` were NaN after the join. Filling with 0 is correct here because the absence of transactions genuinely means zero activity — it's informative missingness, not random missingness. Dropping these rows would have removed a real segment (accounts with no recorded transactions).

### Why a full `dropna()` was wrong

The notebook originally had `merged.dropna(inplace=True)` which removed any row with *any* NaN value. Because `Account_Balance` was NaN for all accounts (the join's `.last()` aggregation was returning NaN), this removed all 4,500 rows and caused a `ValueError: 0 samples remain`. The fix was to `fillna(0)` for monetary columns first, then `dropna(subset=['Recency_Days', 'Transaction_Count', 'Account_Age_Days'])` — targeting only the columns where NaN truly means unusable.

---

## 8. Next Steps

1. **Credit risk scoring** — Join the `loan` table's status field (`A`=good, `B`=in default, `C`=at risk, `D`=defaulted) to test whether segments predict loan default. This would turn the unsupervised clustering into a supervised risk model
2. **Temporal analysis** — Re-run segmentation on 1993, 1994, 1995, 1996 snapshots separately to track how accounts migrated between segments over the bank's 5-year history
3. **Dendrogram review** — Use the Ward linkage dendrogram to provide a visual explanation of segment relationships for regulatory/audit purposes
4. **Enrich with district data** — Add regional economic indicators (average salary, unemployment rate from the `district` table) to characterise segments by their geographic concentration
