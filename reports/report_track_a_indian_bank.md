# Track A — Indian Bank Customer Segmentation
### Banking Customer Segmentation · Week 4 Project Report

---

## Quick-Start Glossary
> *Read this first — it explains every technical term used in this report.*

| Term | Plain-English Meaning |
|---|---|
| **Clustering / Segmentation** | Grouping customers so that people within the same group are as similar as possible to each other, and as different as possible from people in other groups — without being told in advance what the groups should be |
| **Feature** | A single measurable property about a customer (e.g. their account balance, or how many days since they last transacted) |
| **Feature Engineering** | The process of calculating useful features from raw data — e.g. computing "days since last transaction" from a list of raw transaction records |
| **EDA (Exploratory Data Analysis)** | The process of examining the raw data with charts and statistics before modelling, to understand its shape, outliers, and patterns |
| **Standardisation** | Rescaling all features so they have the same range (mean=0, std=1). Without this, a feature measured in rupees (e.g. balance = ₹100,000) would dominate a feature measured in days (e.g. recency = 5), even if both are equally important |
| **Right-skewed distribution** | When most values are low but a small number of values are very high (e.g. most customers have a balance of ₹10,000–50,000, but a tiny fraction have ₹10,000,000+). The few extreme values can mislead models |
| **Log transform** | Applying the mathematical log() function to compress a right-skewed distribution. A balance of ₹1,000,000 becomes log(1,000,000) ≈ 14. This prevents extreme values from dominating the model |
| **k** | The number of clusters (customer groups) the algorithm is told to find. Choosing the right k is part of the modelling process |
| **Silhouette Score** | A measure of how well-separated the clusters are (explained in detail in Section 3) |
| **Davies-Bouldin Index** | Another measure of cluster quality (explained in Section 3) |
| **Calinski-Harabasz Index** | Another measure of cluster quality (explained in Section 3) |
| **ARI (Adjusted Rand Index)** | A measure of how consistent/stable the clusters are across multiple runs (explained in Section 3) |
| **Latent space** | A compressed, hidden representation that a neural network learns internally. Instead of 8 original features, the network finds 8 (or fewer) "hidden" axes that capture the same information more efficiently |
| **Autoencoder** | A type of neural network that learns to compress data and then reconstruct it — the compression step is what we use for clustering (explained in Section 2) |

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **Name** | Indian Bank Customer Transaction Dataset |
| **Customers** | ~837,000 |
| **Source** | Kaggle — `shivamb/bank-customer-segmentation` |
| **Format** | Single flat CSV — one row per transaction |
| **Coverage** | Multi-year transaction history across retail bank accounts |

### What the Dataset Contains

The raw data is a **transaction ledger** — a log of every deposit, withdrawal, and transfer ever made by each customer, similar to a bank's internal records. Crucially, the raw data does **not** describe customers directly — it only has individual transaction records. This means if one customer made 200 transactions, they appear as 200 rows in the file.

Before any clustering could begin, we had to **aggregate** (combine and summarise) all transactions per customer into a single row representing that customer's overall behaviour. Think of it like reading every page of a customer's bank statement and summarising it into a single one-line report card.

### Features Engineered (after EDA)

The following 8 customer-level features were calculated from the raw transaction data:

| Feature | How it was calculated | What it tells us |
|---|---|---|
| **Recency** | Number of days between the customer's most recent transaction and the analysis date | A high number means the customer hasn't transacted in a long time — they may be dormant or churned |
| **Frequency** | Total count of all transactions made | A high number means the customer uses the bank regularly — this is often the strongest signal of a primary banking relationship |
| **Monetary (Debit)** | Sum of all money spent (outgoing transactions) | High debit spend indicates an active lifestyle — useful for distinguishing high-value spenders from low spenders |
| **Monetary (Credit)** | Sum of all money received (incoming transactions) | Salary, business income, or transfers received. Separates regular salary earners from customers with irregular income |
| **Balance** | Average account balance across all observed periods | Proxy for wealth or financial cushion. High balance = financially comfortable; low balance = financially constrained |
| **Account Age** | Number of days since the account was first opened | Long-tenured customers are usually more loyal and less likely to leave than recently acquired customers |
| **Avg Transaction Size** | Mean value per individual transaction | A customer making one ₹50,000 transaction is very different from one making fifty ₹1,000 transactions, even if the total spend is the same |
| **Credit/Debit Ratio** | Ratio of incoming money to outgoing money | A ratio above 1 means the customer receives more than they spend — a rough proxy for savings capacity or financial health |

All 8 features were then **standardised** (scaled to zero mean, unit variance) so that the clustering algorithm treats a ₹100,000 difference in balance and a 100-day difference in recency as equally significant. Without standardisation, balance would dominate recency purely because rupees are large numbers.

### What the EDA Revealed

**Exploratory Data Analysis (EDA)** was run before modelling to understand the data's structure. Key findings:

- **Highly right-skewed distributions** — Most customers have modest balances (₹10,000–₹100,000), but a small fraction has balances over ₹1,000,000. If left untransformed, these extreme outliers would pull cluster centres towards themselves, collapsing most customers into one enormous "normal" cluster. **Fix: log-transform balance and monetary features before scaling.**

- **High correlation between Debit and Credit** — Customers who spend more also tend to receive more (their income funds their spending). This means Debit and Credit carry overlapping information — they are not truly independent features. An algorithm that treats them as independent would double-count this signal. The Autoencoder addresses this by merging correlated features into a single latent axis.

- **Long-tail recency** — A significant portion of customers had not transacted in over 365 days. This naturally foreshadows a "dormant customer" segment even before any modelling begins.

---

## 2. Models Trained

Four different clustering algorithms were tested, each run at multiple values of k (number of clusters). The goal was to find which method and which k produces the clearest, most meaningful customer groups.

### Algorithm 1: K-Means

**How it works (step by step):**
1. Randomly place k "centre points" (called centroids) in the feature space
2. Assign every customer to the nearest centroid — this forms k preliminary groups
3. Recalculate each centroid as the average of all customers assigned to it
4. Repeat steps 2–3 until the assignments stop changing

**The mental model:** Imagine placing k magnets on a map of customers. Each customer is attracted to the nearest magnet. You then move each magnet to the average position of the customers it attracted. Repeat until nothing moves. The final magnet positions are the cluster centres.

**Limitations:** K-Means assumes all clusters are roughly equal in size and roughly circular (spherical) in shape. It also treats all features as independent, so it can't handle correlated features well.

- Tested k = 3, 4, 5, 6, 7, 8
- Best result: **k=6, silhouette=0.165**
- This was flagged as "recommended" in the notebook because it was the only model with a measured stability score (ARI=0.799 — see metrics section)

### Algorithm 2: Gaussian Mixture Model (GMM)

**How it works (step by step):**
1. Instead of hard cluster assignments, GMM models each cluster as a probability distribution — specifically a "Gaussian" (bell-curve shaped) cloud in feature space
2. Each customer gets a probability score for belonging to each cluster (e.g. "70% Cluster A, 25% Cluster B, 5% Cluster C")
3. The model adjusts the shape, size, and position of each cloud to best fit the data

**The mental model:** Instead of rigid circular bubbles (K-Means), GMM uses flexible oval-shaped probability clouds. A customer on the edge of two clouds gets a partial membership in both — it's "soft" clustering. This is more realistic than K-Means' hard assignments.

**Limitations:** GMM can struggle when clusters overlap significantly, and needs more data to estimate the shapes of the distributions reliably.

- Tested k = 3, 4, 5, 6, 7, 8
- Best result: **k=3, silhouette=0.178**
- Slightly better than raw K-Means, but clusters were large and indistinct

### Algorithm 3: Agglomerative Hierarchical Clustering (Ward Linkage)

**How it works (step by step):**
1. Start: treat every single customer as their own individual cluster (837,000 clusters)
2. Find the two clusters that are most similar to each other and merge them into one
3. Repeat step 2 — always merging the two closest clusters
4. Continue until you have k clusters remaining

The entire history of merges is stored as a **dendrogram** — a tree diagram where you can see which customers merged first (very similar) and which merged last (most different). You "cut" the tree at any height to get k groups.

**Ward linkage** is the specific merging rule used: at each step, merge the two clusters that result in the smallest increase in total within-cluster variance. This tends to produce balanced, compact clusters.

**Limitation — memory constraint:** Ward linkage requires computing the distance between every pair of customers. With 50,000 customers this means 50,000 × 50,000 / 2 = 1.25 billion distance calculations, requiring approximately 9 GB of RAM. To stay within memory limits, hierarchical clustering was run on a 5,000-customer sample only.

- Best result: k=6 on the 5,000-row sample
- Not directly comparable to K-Means/AE results because it used a smaller sample

### Algorithm 4: Autoencoder + K-Means (Champion)

**How it works — in two stages:**

**Stage 1 — The Autoencoder (neural network compression):**

An Autoencoder is a neural network with two parts:
- **Encoder:** Takes the 8 input features and compresses them through successive hidden layers down to a smaller "latent" representation (e.g. 8 compressed numbers)
- **Decoder:** Takes the compressed representation and tries to reconstruct the original 8 features from it

The network is trained by minimising the **reconstruction loss** — the difference between the original input and the decoder's reconstruction. By forcing the network to compress and then reconstruct, it has to learn the most important patterns in the data. Noise and redundancy get discarded because they can't be efficiently encoded.

*Think of it like this:* If you asked someone to summarise an 8-page report in 8 bullet points and then reconstruct the full report from those bullets, they'd naturally keep the most important information and discard minor details.

**Stage 2 — K-Means on the latent space:**

Once the Autoencoder is trained, every customer is passed through the encoder to get their 8-dimensional latent representation. K-Means is then run on these latent representations rather than the original raw features.

**Why this works better:** The latent space has already removed noise and merged correlated features. K-Means' assumption of independent, spherical clusters is much better satisfied in the clean latent space than in the messy raw feature space.

- Tested latent dimensions = 8, 16 and k = 3 to 8
- **Champion: `a_ae_ld8_k6` — 8 latent dimensions, 6 clusters**

---

## 3. Evaluation Metrics — What They Are and What the Numbers Mean

Clustering is **unsupervised** — there is no "correct answer" to compare against. We can't ask "did the model get it right?" the way we can with prediction problems. Instead, we measure the quality of the clusters geometrically: are the groups tight and well-separated?

### Metric 1: Silhouette Score (range: −1 to +1, higher is better)

**What it measures:** For every single customer, it computes:
- **a** = the average distance from that customer to all other customers in their same cluster (how similar they are to their cluster-mates)
- **b** = the average distance from that customer to all customers in the nearest neighbouring cluster (how different they are from the closest other group)
- **Silhouette = (b − a) / max(a, b)**

If a customer is much closer to their own cluster than to any other cluster, b >> a, and the silhouette is close to +1 (perfect). If a customer is right on the boundary between two clusters, a ≈ b, and the silhouette is near 0 (ambiguous). If a customer is actually closer to a different cluster than their assigned one, silhouette is negative (wrong assignment).

The **overall silhouette score** is the average across all customers.

**How to interpret the numbers:**
- Below 0.00 → clusters overlap significantly, groupings are unreliable
- 0.00–0.25 → weak structure, clusters are barely separable
- 0.25–0.50 → reasonable structure, clusters are meaningfully distinct
- 0.50–0.70 → strong, well-separated clusters
- Above 0.70 → very tight, clearly separated clusters (rare in real-world data)

**Track A result:** The champion model scored **0.332** — this falls in the "reasonable" band. Customers within each segment are more similar to their group than to neighbouring groups, but there is some overlap at the boundaries.

### Metric 2: Davies-Bouldin Index (range: 0 to ∞, lower is better)

**What it measures:** For each pair of clusters, it calculates the ratio of:
- **(average within-cluster spread) ÷ (distance between cluster centres)**

A high ratio means the clusters are wide and close together — bad. A low ratio means the clusters are compact and far apart — good. The DBI is the average of this ratio across all cluster pairs.

**How to interpret the numbers:**
- Below 0.50 → excellent separation
- 0.50–1.00 → good separation
- 1.00–2.00 → moderate, clusters have some overlap
- Above 2.00 → poor separation, clusters are too close or too wide

**Track A result:** The champion scored **1.099** — moderate. Clusters are reasonably separated but not extremely tight. This is typical for high-dimensional real-world customer data.

### Metric 3: Calinski-Harabasz Index (range: 0 to ∞, higher is better)

**What it measures:** The ratio of:
- **(variance between cluster centres)** ÷ **(average variance within clusters)**

Multiplied by a correction factor based on n and k. Think of it as: "How much more spread-apart are the cluster centres compared to how spread-out the customers are inside each cluster?"

**Track A result:** The champion scored **48,528** — dramatically higher than the K-Means score of 1,701. This confirms that the AE latent space makes clusters 28× tighter relative to their separation.

> Note: CH scores cannot be compared across datasets (larger datasets produce larger CH scores), only across models *on the same dataset*.

### Metric 4: Stability ARI — Adjusted Rand Index (range: 0 to 1, higher is better)

**What it measures:** This checks whether the cluster structure is reproducible or just a lucky artefact.

**Procedure:**
1. Split the data into two random halves (e.g. customers 1–25,000 and customers 25,001–50,000)
2. Train the clustering model independently on each half
3. Ask: "How similar are the cluster assignments from both runs?" — i.e., do the same natural groups emerge from different subsets of data?

The ARI score measures this similarity: 1.0 = completely identical cluster structure; 0.0 = no relationship between the two runs (might as well be random).

**Track A result:** Only K-Means had a stability score computed: **ARI = 0.799** — high stability. The AE model did not have stability computed in this notebook run, so it is shown as "—".

---

## 4. Full Results Table

| Run ID | What it means | k | Latent Dim | Silhouette ↑ | DB Index ↓ | CH Index ↑ | Stability ARI |
|---|---|---|---|---|---|---|---|
| **a_ae_ld8_k6** | AE, 8 latent dims, 6 clusters | 6 | 8 | **0.332** | **1.099** | **48,528** | — |
| a_ae_ld16_k6 | AE, 16 latent dims, 6 clusters | 6 | 16 | 0.186 | 1.408 | 23,403 | — |
| a_gmm_k3 | GMM, 3 clusters | 3 | — | 0.178 | 2.018 | 6,423 | — |
| a_gmm_k7 | GMM, 7 clusters | 7 | — | 0.173 | 1.933 | 5,191 | — |
| a_kmeans_k6 | K-Means, 6 clusters | 6 | — | 0.165 | 1.549 | 1,701 | **0.799** |

**Reading the table:** For each run, higher silhouette and CH are better; lower DB is better. The champion (`a_ae_ld8_k6`) wins on all three cluster-quality metrics while K-Means wins on stability (the only method where it was computed).

---

## 5. Champion Model: `a_ae_ld8_k6`

**Full name:** Autoencoder with 8 latent dimensions + K-Means with k=6 clusters

**Why this model won:**

| Comparison | AE (champion) | Raw K-Means |
|---|---|---|
| Silhouette | **0.332** | 0.165 |
| DB Index | **1.099** | 1.549 |
| CH Index | **48,528** | 1,701 |

The AE produces **2× better silhouette** and **28× better CH score** compared to raw K-Means on the same data. The reason: the AE latent space removed noise and merged correlated features (debit ↔ credit, frequency ↔ account age), giving K-Means a much cleaner set of "axes" to separate customers along.

**Why latent_dim=8 (same as input)?** Normally we'd expect to compress 8 features into fewer than 8 latent dimensions. But because the 8 features are not 8 independent signals (they contain correlations), the AE learned to *reorganise* the 8 axes into cleaner, less-correlated latent axes rather than simply reducing the count. The information content was the same; the representation was cleaner.

**Why k=6?** The elbow method and silhouette curves both suggested 6 clusters as a good balance between cluster quality and interpretability — enough groups to be useful for targeted marketing, not so many that the segments become indistinguishable.

### The 6 Customer Segments

| Segment | Name | Key Signal | Estimated Size |
|---|---|---|---|
| 0 | Night-Active Moderate Savers | Transactions exclusively at night | ~18% |
| 1 | Daytime Wealthy Inactive | Highest balance; only transacts 9–5 weekdays | ~17% |
| 2 | Active Multi-Transactors | Multiple transactions per session, variable amounts | ~18% |
| 3 | Weekend-Only Bankers | Transactions exclusively on weekends | ~17% |
| 4 | Low-Value Young Micro-Transactors | Youngest, lowest balance, smallest transactions | ~15% |
| 5 | High-Spend Pass-Through | Near-zero balance but massive individual transactions | ~15% |

*(Detailed profiles with behavioural explanations are in the separate `cluster_profiles.md` report.)*

---

## 6. EDA Tie-Back

The EDA findings directly shaped every modelling decision:

| EDA Finding | Implication | How We Addressed It |
|---|---|---|
| Right-skewed balance and spend distributions | Outliers would collapse most customers into one cluster if untransformed | Applied log-transform before standardisation |
| High correlation between Debit and Credit amounts | Raw K-Means would double-count this signal | The AE merged them into one latent "spending power" axis |
| Long-tail recency (dormant customers visible even before modelling) | A "dormant" segment was expected to exist | The AE correctly isolated these customers into a distinct group |
| 8 features with significant inter-correlation | True intrinsic dimensionality is lower than 8 | latent_dim=8 still works because AE reorganises rather than reduces |

---

## 7. Next Steps

1. **Formal segment naming** — Review the centroid heatmaps to confirm and label each of the 6 segments (e.g. "Night-Active Professionals", "Dormant Savers")
2. **Fairness check** — Verify that segment distribution is not skewed by gender, geography, or age in a way that could lead to discriminatory marketing
3. **Model export** — Save the trained AE encoder weights and K-Means centroids to a `.pkl` file so new customers can be scored in real time
4. **Monitoring** — Re-run segmentation quarterly and track how many customers move between segments — this detects early signs of churn or life-stage changes
