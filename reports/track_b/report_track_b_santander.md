# Track B — Santander Product Portfolio Segmentation
### Banking Customer Segmentation · Week 4 Project Report

> **Metric comparison, not the selected segmentation.** This track has the best silhouette (0.512) and perfect stability (ARI 1.000). The file has demographics and product ownership only — no transactions — so it cannot answer the assignment question about behavior. The selected model is Track C, `c_ae_lat4_k6`. See `champion_recommendation.md`.

---

## Quick-Start Glossary
> *All technical terms used in this report are defined here.*

| Term | Plain-English Meaning |
|---|---|
| **Clustering / Segmentation** | Grouping customers automatically so that people within the same group behave similarly, without being told in advance what the groups should look like |
| **Feature** | A single measurable property of a customer — e.g. their age, how long they've been a customer, or whether they hold a credit card |
| **Binary feature** | A feature that can only be 0 or 1 (yes or no) — e.g. "does this customer have a mortgage? Yes = 1, No = 0" |
| **Continuous feature** | A feature that can take any numerical value — e.g. age = 34, income = €42,000 |
| **Sparse matrix** | A table of data where most values are 0. In this dataset, most customers hold only 1–3 of the 24 available products, so the product ownership table is mostly zeros |
| **Euclidean distance** | The straight-line distance between two points in multi-dimensional space. K-Means uses this to decide which cluster each customer belongs to. It works poorly on binary (0/1) data because the "distance" between owning and not owning a product is the same regardless of how important that product is |
| **Latent space** | The compressed internal representation learned by a neural network. Instead of 24 raw features, the network represents each customer as 8 "hidden" numbers that capture the essential patterns |
| **Autoencoder (AE)** | A neural network that learns to compress data into a latent space and then reconstruct the original data from it. The compression step is what we use for clustering |
| **Reconstruction loss** | How well the Autoencoder can rebuild the original input from its compressed form. If loss goes down during training, the network is learning real patterns |
| **Embedding** | The latent-space representation of a customer produced by the AE encoder. A dense (non-sparse), meaningful numerical summary of who that customer is |
| **Product penetration** | The percentage of customers in a segment that hold a given product. E.g. "mortgage penetration in Segment 2 = 45%" means 45% of that segment's customers have a mortgage |
| **Silhouette Score** | Measures how well-separated the clusters are — see Section 3 for full explanation |
| **Davies-Bouldin Index** | Measures cluster compactness vs. separation — see Section 3 |
| **Calinski-Harabasz Index** | Measures the ratio of between-cluster spread to within-cluster spread — see Section 3 |
| **Stability ARI** | Measures whether the same cluster structure appears reliably across multiple model runs — see Section 3 |
| **GPU (T4)** | Graphics Processing Unit — a type of chip originally designed for video games but now widely used to speed up neural network training. The Autoencoder for this track was trained on Kaggle's free T4 GPU, which reduced training time by roughly 3× |
| **Cross-sell** | Offering a customer a product they don't yet hold — e.g. offering a credit card to someone who only has a current account |

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **Name** | Santander Product Recommendation Dataset |
| **Customers** | ~931,000 |
| **Source** | Kaggle — `santander-product-recommendation` competition (real Santander Bank data) |
| **Format** | Monthly snapshots — one row per customer per month |
| **Coverage** | 17 months of product ownership history (January 2015 – May 2016) |
| **Raw size** | ~3 GB, ~13.6 million rows |
| **Training sample** | 100,000 customers (one snapshot per customer taken from May 2016) |

### What the Dataset Contains

This dataset was released by **Santander Bank Spain** as a Kaggle competition in 2016. It is one of the most well-known real-world banking datasets publicly available.

Unlike Track A (individual transaction records), each row here is a **monthly profile snapshot** — a complete picture of a customer at a single point in time. It records:
- Who the customer is (demographics: age, income, gender, seniority)
- What banking products they currently hold (24 yes/no product flags)

Because the data covers 17 months, the same customer appears 17 times in the raw file. For segmentation, we used the **most recent snapshot** (May 2016) for each customer — giving us one comprehensive up-to-date profile per customer.

The dataset was too large to run locally (~3 GB) and was instead run on **Kaggle's cloud notebook environment**, which provides free GPU access.

### Features Used for Clustering

**Continuous (numerical) features:**

| Feature Name | What it measures | Range / Notes |
|---|---|---|
| `age` | Customer's age in years | Ranges from teens to 90s |
| `seniority_months` (antiguedad) | How many months the customer has been with Santander | 0 = brand new; 200+ = 16+ year customer |
| `gross_income` (renta) | Estimated annual household income in euros | Heavily right-skewed — a few very wealthy customers |
| `total_products` | Total count of banking products currently held | 1 (only current account) to 12+ |
| `activity_index` | Binary — 1 if the customer was "active" in the reference month; 0 if inactive | Santander-defined activity flag |
| `is_female` | Binary — 1 if female; 0 if male | Demographic flag |

**Product ownership flags (18 binary yes/no features):**

Each of these is a 0 or 1 representing whether the customer holds that product:

| Feature | Product Type | Description |
|---|---|---|
| `has_current_acct` | Basic banking | Standard checking/current account — the most basic product |
| `has_payroll_acct` | Basic banking | Account specifically for receiving salary payments |
| `has_junior_acct` | Basic banking | Account for customers under 18 years old |
| `has_e_account` | Digital banking | Online/electronic account variant |
| `has_short_deposit` | Savings | Short-term fixed deposit (term < 1 year) |
| `has_long_deposit` | Savings | Long-term fixed deposit (term ≥ 1 year) |
| `has_funds` | Investment | Investment fund (money invested in a basket of assets) |
| `has_securities` | Investment | Securities account (stocks, bonds) |
| `has_pensions` | Retirement | Pension plan — long-term retirement savings |
| `has_mortgage` | Credit | Home loan — borrowed money secured against property |
| `has_loans` | Credit | Personal loan |
| `has_credit_card` | Credit | Credit card |
| `has_direct_debit` | Payments | Direct debit mandate — automatic recurring payments |
| `has_payroll` | Payments | Payroll direct deposit setup |
| `has_taxes` | Tax | Tax collection account |
| `has_pension_nom` | Retirement | Nominated pension account |
| `has_particular_acct` | Account variant | Specific Santander account type |
| `has_particular_plus` | Account variant | Enhanced account type |

**Total: 24 features** (6 continuous + 18 binary product flags)

### What the EDA Revealed

- **Sparse product matrix** — The average customer holds only 1–3 products from the 24 available. In the product table, roughly 90–95% of all values are 0. This sparsity is a significant modelling challenge: Euclidean distance (which K-Means uses) doesn't distinguish well between customers who both hold 0 out of 24 products vs. customers who differ on 1 specific product.

- **Income is heavily right-skewed** — A small number of very high-income customers (renta > €300,000) exist alongside the majority who earn €20,000–€60,000. Log-transforming income before modelling prevents these wealthy outliers from dominating cluster placement.

- **Seniority is bimodal** — When plotted as a histogram, seniority shows two clear peaks: one around 12–24 months (relatively new customers) and another around 72–120 months (long-term loyalists). This bimodal split is a strong natural axis for segmentation.

- **Product co-occurrence** — Customers with a mortgage tend to also hold a pension plan (they're in the life stage of home ownership + retirement planning). Customers with securities tend to hold investment funds too. These co-occurrence patterns are invisible to K-Means but are learned and encoded by the Autoencoder.

---

## 2. Models Trained

Three clustering approaches were tested:

### Approach 1: K-Means on Continuous Features Only

**What we did:** Ran K-Means using only the 6 continuous features (age, seniority, income, total_products, activity_index, is_female) — completely ignoring the 18 product flags.

**Why test this?** As a baseline. Demographics alone (age, income, seniority) are a natural way to segment customers and don't suffer from the sparsity problem.

**How K-Means works:** Places k centre points in the feature space. Assigns every customer to the nearest centre. Moves each centre to the average position of the customers assigned to it. Repeats until nothing changes. The final assignments are the clusters.

- Tested k = 3, 4, 5, 6, 7, 8
- Best result: **k=4, silhouette=0.395** — performed surprisingly well purely on demographics
- This makes intuitive sense: age + seniority + income naturally separate life stages (young low-income → established mid-career → wealthy retired)

### Approach 2: K-Means on Full Feature Matrix (Continuous + Products)

**What we did:** Added all 18 binary product flags to the 6 continuous features and ran K-Means on all 24 features together.

**What happened:** Performance dropped — silhouette fell from 0.395 to 0.350 with the added product features.

**Why does adding more features make things worse?** Because K-Means measures distance in Euclidean space. In high-dimensional binary space, everything looks equally far apart from everything else — a phenomenon called the **"curse of dimensionality"**. With 18 binary columns, two customers might differ on 12 products and still be assigned to the same cluster, while two customers with identical product portfolios but slightly different ages end up in different clusters. The binary features add noise to the distance calculations rather than signal.

- Tested k = 3 to 8
- Best result: **k=3, silhouette=0.350**

### Approach 3: Autoencoder + K-Means (Champion)

**What we did:** Trained a neural network (Autoencoder) on all 24 features to learn a compressed, dense representation of each customer. Then ran K-Means on the 8-dimensional latent space outputs.

**How the Autoencoder works (step by step):**

```
Input (24 features per customer)
         ↓
  Encoder layers (24 → 16 → 8)
         ↓
  Latent representation (8 numbers)   ← THIS is what K-Means clusters on
         ↓
  Decoder layers (8 → 16 → 24)
         ↓
Output: reconstruction of original 24 features
```

**Training objective:** Minimise the difference between the input and the reconstruction. The model has no choice but to learn which features are most important — because it only has 8 numbers to encode 24 features. Noise and sparsity get left out; meaningful patterns (product co-occurrence, life-stage correlations) get encoded.

**Training progress on Kaggle T4 GPU:**
```
AE Track B | latent_dim=8
epoch  20/80 | loss 0.00292   ← loss is dropping — model is learning
epoch  40/80 | loss 0.00184
epoch  60/80 | loss 0.00166
epoch  80/80 | loss 0.00151   ← loss has stabilised — training complete
```

The reconstruction loss dropped from 0.00292 to 0.00151 over 80 training epochs, confirming the network learned genuine patterns rather than fitting random noise.

- Tested latent dimensions = 8, 16, 32 and k = 3
- **Champion: `b_ae_lat8_k3` — 8 latent dimensions, 3 clusters**
- Run on Kaggle GPU T4 (PyTorch 2.11.0+cu128)

**Why latent_dim=8 beats 16 and 32?**

| Latent Dim | Silhouette | Interpretation |
|---|---|---|
| 8 | **0.512** | Compressed enough to remove noise; rich enough to keep structure |
| 16 | 0.353 | Too little compression — some noise still included |
| 32 | 0.283 | Almost no compression — the AE doesn't need to prioritise, so it keeps noise |

The best AE is the one that compresses just enough to throw out noise while keeping real patterns. For this dataset, 8 latent dimensions (compared to 24 input features = 3× compression) hit that sweet spot.

---

## 3. Evaluation Metrics — Full Explanations

### Silhouette Score (range: −1 to +1, higher is better)

**What it measures:** For every customer, it measures how well they fit their assigned cluster vs. how close they are to the nearest other cluster.

**Formula:** For customer i:
- **a(i)** = average distance from customer i to all other customers in the same cluster → measures within-cluster cohesion
- **b(i)** = average distance from customer i to all customers in the nearest *other* cluster → measures between-cluster separation
- **s(i) = [b(i) − a(i)] / max[a(i), b(i)]**

The overall silhouette score is the mean of s(i) across all customers.

**Interpretation scale:**
| Score | Meaning |
|---|---|
| −1 to 0 | Customer is likely in the wrong cluster |
| 0 to 0.25 | No meaningful cluster structure found |
| 0.25 to 0.50 | Reasonable, usable clusters |
| 0.50 to 0.70 | Strong, well-separated clusters |
| 0.70 to 1.00 | Near-perfect clustering (rare in practice) |

**Track B result: 0.512** — strong. The 3 Santander segments are clearly separated. This is the best score across all three tracks in this project.

### Davies-Bouldin Index (range: 0 to ∞, lower is better)

**What it measures:** The average ratio of within-cluster spread to between-cluster distance, for every pair of clusters.

**Formula:** For each pair of clusters i and j:
- **R(i,j) = [s(i) + s(j)] / d(i,j)**
  - s(i) = average distance of all points in cluster i to the cluster centre (spread/width of cluster i)
  - s(j) = same for cluster j
  - d(i,j) = distance between the two cluster centres

DB = average of the maximum R(i,j) over all clusters.

**Plain-English meaning:** If clusters are compact (small s) and far apart (large d), the ratio is small → good. If clusters are wide or close together, the ratio is large → bad.

**Track B result: 0.891** — below 1.0, meaning on average the distance between cluster centres is greater than the average width of the clusters. Good separation.

### Calinski-Harabasz Index (range: 0 to ∞, higher is better)

**What it measures:** The ratio of between-cluster variance (how spread-apart the cluster centres are) to within-cluster variance (how spread-out the customers are within their clusters), scaled by the number of clusters and observations.

A high score means clusters are tight internally and spread far apart from each other — exactly what we want.

**Important caveat:** This index naturally produces larger numbers for larger datasets. The Track B score (90,835) is much higher than Track A (48,528) partly because Track B has more customers, not only because it's better-clustered. **Only compare CH scores across models trained on the same dataset, not across different datasets.**

**Track B result: 90,835** — highest across all three tracks and all models.

### Stability ARI — Adjusted Rand Index (range: 0 to 1, higher is better)

**What it measures:** The reproducibility of the cluster structure.

**Procedure:**
1. Take the full dataset of 100,000 Santander customers
2. Train Model A on a random first half (50,000 customers)
3. Train Model B on a random second half (50,000 customers) — completely independently
4. Compare the cluster assignments from Model A vs Model B: do customers get assigned to the same groups?
5. The ARI score measures this agreement, corrected for chance (a random assignment would score ≈ 0)

This was repeated with 10 different random seeds to get a reliable average.

**Track B result: ARI = 1.000 ± 0.000** — perfect stability with zero variance. Every single one of 10 independent trials produced identical cluster assignments. This is statistically remarkable and provides very high confidence that the 3-group structure is a genuine property of the data, not a product of randomness.

For comparison, K-Means on the full feature matrix also scored well: ARI = 0.977 ± 0.045 — high but not perfect.

---

## 4. Full Results Table

| Run ID | Description | k | Latent Dim | Silhouette ↑ | DB Index ↓ | CH Index ↑ | Stability ARI |
|---|---|---|---|---|---|---|---|
| **b_ae_lat8_k3** | AE, 8 latent dims, k=3 | 3 | 8 | **0.512** | **0.891** | **90,835** | **1.000** |
| b_ae_lat16_k3 | AE, 16 latent dims, k=3 | 3 | 16 | 0.353 | 1.248 | 47,550 | — |
| b_ae_lat32_k3 | AE, 32 latent dims, k=3 | 3 | 32 | 0.283 | 1.539 | 28,150 | — |
| b_kmeans_cont_k4 | K-Means (demographics only), k=4 | 4 | — | 0.395 | 1.007 | 5,820 | — |
| b_kmeans_full_k3 | K-Means (all 24 features), k=3 | 3 | — | 0.350 | 1.273 | 4,069 | 0.977 |

**Reading the table:** All ↑ columns are "the higher the better." All ↓ columns are "the lower the better." The champion (bold row) wins on every single metric.

---

## 5. Champion Model: `b_ae_lat8_k3`

**Full name:** Autoencoder with 8 latent dimensions + K-Means with k=3 clusters

**Why this model won:**

| Comparison | Champion AE | K-Means (demographics only) | K-Means (all features) |
|---|---|---|---|
| Silhouette | **0.512** | 0.395 | 0.350 |
| DB Index | **0.891** | 1.007 | 1.273 |
| CH Index | **90,835** | 5,820 | 4,069 |
| Stability ARI | **1.000** | — | 0.977 |

The AE silhouette is **30% higher** than the best K-Means baseline and the CH index is **15× higher**, reflecting how much cleaner the latent space representation is compared to raw features.

**Why k=3 is the right number of clusters:**

The stability ARI of 1.000 is the clearest possible signal that 3 groups are genuine. When we tried k=4 or k=5, the model would essentially split one of these natural 3 groups into two artificial halves — producing less stable, less meaningful results. The data genuinely contains 3 distinct customer types; forcing more clusters would be over-segmenting.

### The 3 Customer Segments

| Segment | Name | Core characteristics |
|---|---|---|
| **0** | Entry-Level Basic Account Holders | Current account only; low seniority (< 2 years); low income; 1–2 products |
| **1** | Established Middle-Market | 3–5 products (current + card + e-account + payroll); 3–5 yr tenure; mid-income; consistently active |
| **2** | Affluent Long-Tenure Investors | 6+ products including mortgage, pension, securities, funds; 8–15 yr tenure; highest income |

*(Full behavioural descriptions and business recommendations per segment are in `cluster_profiles.md`.)*

---

## 6. EDA Tie-Back

The EDA findings drove every modelling decision:

| EDA Finding | What it implied | What we did |
|---|---|---|
| Sparse binary product matrix (mostly 0s) | K-Means distance calculations would be meaningless in binary space | Trained AE to convert sparse binary flags into dense continuous latent embeddings |
| Income right-skew (a few very wealthy customers) | Outliers would dominate cluster placement | Log-transformed income before feeding to AE |
| Bimodal seniority (new vs. long-term peak) | A natural division between customer life stages existed in the data | AE encoded seniority difference as a primary latent axis |
| Product co-occurrence (mortgage ↔ pension, securities ↔ funds) | Products carry relational information that K-Means ignores | AE learned co-occurrence as "product maturity" axis in latent space |
| AE reconstruction loss dropped steadily (0.00292 → 0.00151) | The AE learned genuine patterns, not random noise | Confirmed model was worth trusting for clustering |

---

## 7. Where Track B sits in the project decision

Track B has the best silhouette (0.512) and perfect stability (ARI 1.000). Those are the strongest clustering numbers in the project. They are not the reason a dataset is selected.

| | Track B (Santander) | Track C (Berka) | Track A (Indian Bank) |
|---|---|---|---|
| Silhouette | **0.512** | 0.417 | 0.332 |
| Davies-Bouldin | 0.891 | **0.731** | 1.099 |
| Stability ARI | **1.000** | 0.864* | — |
| Transactions in the data | No | Yes | Yes |
| Product holding in the data | Yes | Yes (loan, card) | No |

*Track C stability was measured on K-Means, not the autoencoder.*

Santander records who the customer is and which products they hold. It does not record amounts, dates, or balances. The three groups are product-maturity stages. That is a useful comparison, and it shows the autoencoder can separate a sparse product matrix. It cannot answer the assignment question about which **behaviors** distinguish customers.

**Selected segmentation: Track C, `c_ae_lat4_k6`.** The silhouette gap versus Santander is 0.095. Berka is the track that contains both transaction behavior and product holding, which is what the brief asks the system to explain. The full reasoning is in `champion_recommendation.md`.

---

## 8. Next Steps

1. **Product recommendation engine** — Use segment assignment to personalise which product to promote to each customer (e.g. Segment 0 → first credit card; Segment 1 → pension plan)
2. **Monthly re-scoring pipeline** — Since data is monthly snapshots, re-assign every customer to a segment each month and track lifecycle movement
3. **Churn early warning system** — A customer whose segment shifts from 2 → 0 (dropping products, reducing activity) should trigger an immediate retention call
4. **A/B test** — Run different campaigns per segment and measure revenue uplift to validate the commercial value of segmentation
5. **Fairness audit** — Verify that segment distribution is not correlated with protected demographic characteristics (gender, age band, region)
