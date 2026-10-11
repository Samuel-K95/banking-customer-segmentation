# Champion Selection — Which Dataset & Model to Use
### Banking Customer Segmentation · Week 4 Project Report

---

## Quick-Start Glossary

| Term | Plain-English Meaning |
|---|---|
| **Champion model** | The segmentation we would hand a bank. Chosen for meaningful behavior, stability, interpretability, and banking use — not for the highest clustering score alone |
| **Silhouette Score** | A number between −1 and +1 that measures how well-separated the clusters are. Customers closer to their own cluster than to any other cluster score near +1. Scores above 0.25 are acceptable; above 0.50 is strong |
| **Davies-Bouldin Index** | Measures the ratio of cluster width (how spread-out accounts are within a cluster) to cluster separation (how far apart cluster centres are). Lower = better. Below 1.0 = clusters are wider apart than they are wide |
| **Stability ARI** | Measures whether the exact same cluster structure emerges when two independent models are trained on different halves of the data. 1.0 = perfect agreement; 0.0 = no relationship (essentially random) |
| **Autoencoder (AE)** | A neural network that compresses input features into a compact latent representation, removing noise and correlated redundancy in the process. We cluster on the compressed output |
| **K-Means** | A clustering algorithm that places k centre points in feature space and assigns each customer to the nearest one |
| **Latent dimension** | The number of compressed "summary numbers" the Autoencoder uses to represent each customer internally |
| **Unsupervised learning** | A type of machine learning where the algorithm finds patterns without any labelled examples or pre-defined correct answers |
| **Cross-sell** | Offering a customer an additional product they don't yet hold |
| **Fairness audit** | Checking whether model outputs are systematically biased against protected groups (age, gender, region, etc.) |

---

## Bottom Line

> **Use: Track C (Berka) + Autoencoder + K-Means, k=6, latent_dim=4**
> **Model ID: `c_ae_lat4_k6`**
>
> The assignment asks which customer groups exist, what **behaviors** distinguish them, and how a bank would use that. Berka is the only track that contains both transaction behavior and product holding. Santander has the higher silhouette, and that gap is real, but it is not large enough to override a dataset with no transactions.

---

## Cross-Track Comparison Table

| Track | Role | What the data actually contains | Champion model | Silhouette ↑ | DB Index ↓ | Stability ARI ↑ | k |
|---|---|---|---|---|---|---|---|
| **C — Berka** | **Selected segmentation** | Transactions, balances, recency, tenure, loan, card | AE + K-Means (lat=4) | 0.417 | **0.731** | 0.864* | 6 |
| B — Santander | Metric comparison only | Demographics and product ownership. No transactions | AE + K-Means (lat=8) | **0.512** | 0.891 | **1.000** | 3 |
| A — Indian Bank | Required starting dataset | Transaction ledger, engineered into timing and spend features | AE + K-Means (lat=8) | 0.332 | 1.099 | — | 6 |

↑ = higher is better. ↓ = lower is better. *Track C stability was measured for K-Means, not the autoencoder.

The silhouette gap between Santander (0.512) and Berka (0.417) is 0.095. Both sit in the usable range. Berka has the best Davies-Bouldin score of the three (0.731), which means its clusters are the most compact relative to how far apart they sit.

---

## Why the highest score is not the selection

The assignment says the comparison exists so we can decide which approach produces groups that are meaningful, stable, interpretable, and useful for banking applications. The assessment criteria put more weight on reasoning and practical relevance than on a high clustering metric.

Santander wins silhouette and stability. Those numbers describe a clean split of customers by **what they already own** and how long they have been with the bank. The file has no payment amounts, no transaction dates, and no balance history. A bank cannot answer "what behaviors distinguish these groups?" from that file.

Indian Bank is the dataset the assignment names as the starting point, and it does contain transactions. After feature engineering, the six groups separated mainly by **when** people transact (night, weekend, weekday) rather than by a full picture of activity, balance, and products. Recency came out as 0 days in every segment. That is a real finding, and it is not enough for the behavioral applications the brief asks for. The assignment allows a second public dataset when the starting variables cannot support that analysis.

Berka is that second dataset. Each account has transaction counts, credits, withdrawals, balance, recency, account age, and whether a loan or a card exists.

---

## Why Track C (Berka) is the segmentation we use

### 1. It is the only track with both behavior and products

| What the brief asks for | Indian Bank | Santander | Berka |
|---|---|---|---|
| Transaction behavior | Timing of spend, weak recency signal | Not in the data | Count, credit, withdrawal, recency |
| Balance patterns | Yes | Not in the data | Yes, including a negative-balance group |
| Product usage | Not in the data | 18 product flags | Loan and card |
| Engagement and lifecycle | Account age, but recency did not separate groups | Seniority in months | Recency and account age, including a group inactive for about 1.5 years |

### 2. The groups are explainable in banking language

| Segment | What makes them similar | Banking use |
|---|---|---|
| 0 — New basic accounts | About 2 years old, light activity, almost no loan or card | Onboarding and a first card offer |
| 1 — Long-tenure high-activity basic | About 4.5 years, 314 transactions, no loan and no card | Highest-value cross-sell: the relationship already exists |
| 2 — Loan-centric | Every account has a loan, high flows, recently active | Card and protection offers tied to the loan |
| 3 — Dormant | Last transaction about 539 days ago, only 27 transactions | Retention or closure, not a sales push |
| 4 — Card-centric savers | Every account has a card, highest balance, no loan | Premium card and a first loan to solvent customers |
| 5 — Financially stressed | Average balance about −6,500 CZK, withdrawals exceed credits | Service and risk review, not investment products |

### 3. The metric case is close, and Berka wins one of the three quality scores

| | Santander `b_ae_lat8_k3` | Berka `c_ae_lat4_k6` |
|---|---|---|
| Silhouette | **0.512** | 0.417 |
| Davies-Bouldin | 0.891 | **0.731** |
| Stability | **ARI 1.000** on the autoencoder | ARI 0.864 on K-Means |

A silhouette of 0.417 means an account is, on average, clearly closer to its own group than to the next group. It is below Santander's 0.512, and that should be stated. It is not a weak or arbitrary clustering. K-Means on the same Berka data reproduced the six-group structure with ARI 0.864 across random restarts, so the groups are not a one-off initialisation.

### What we give up by choosing Berka

- **Size.** 4,500 accounts versus about 931,000 Santander customers. The patterns are interpretable, and they still need revalidation on a larger, current book before anyone treats them as production policy.
- **Era and place.** Czech bank, 1993–1998, amounts in koruna. The behaviors (dormant, overdrawn, loan-only, card-only) transfer as types of customers. The currency levels do not.
- **Stability was not measured on the autoencoder itself.** The 0.864 ARI is the K-Means run. The selected model is the autoencoder. That gap should be closed before deployment by repeating the stability test on the latent representations.

---

## Why K-Means Alone Is Not Enough — The AE Advantage

In every track, the Autoencoder + K-Means combination outperformed raw K-Means:

| Track | AE Silhouette | K-Means Silhouette | Improvement |
|---|---|---|---|
| A — Indian Bank | 0.332 | 0.165 | **+101%** |
| B — Santander | 0.512 | 0.395 | **+30%** |
| C — Berka | 0.417 | 0.363 | **+15%** |

**Why K-Means underperforms — three reasons explained:**

**1. K-Means assumes all features are independent.** In reality, features are correlated. In Track A, "Total_Debit" and "Total_Credit" are highly correlated — customers who spend more also earn more. Raw K-Means treats these as two separate axes and double-counts this signal. The AE encoder compresses both into a single "spending power" latent axis.

**2. K-Means uses Euclidean distance, which breaks down on binary data.** In Track B, the 18 product flags are 0 or 1. Euclidean distance treats the difference between holding/not holding a mortgage identically to the difference between holding/not holding any other product — they're all just a distance of 1. In reality, some products are more "distant" than others in financial meaning. The AE learns the relative importance of each product through the reconstruction task.

**3. K-Means cannot handle noise dimensions.** Raw features include some random variation that has nothing to do with real customer behaviour (measurement noise, data entry errors, edge cases). The AE, by being forced to compress and reconstruct, automatically discards features it cannot efficiently encode — leaving only real patterns for K-Means to cluster on.

---

## Why the Autoencoder with Fewer Latent Dimensions Wins (Track C)

In Track C, latent_dim=4 beat latent_dim=8 (silhouette 0.417 vs 0.399). This is counter-intuitive — why does *less* capacity produce *better* results?

**Intrinsic dimensionality:** The 8 input features are not 8 independent signals. They contain approximately 4 truly independent axes of variation:
- Activity: Recency + Transaction_Count → one axis
- Financial depth: Balance + Total_Credit → one axis
- Spending: Total_Withdrawal + Account_Age → one axis
- Product: has_loan + has_card → one axis

When the AE has exactly 4 latent dimensions, it is forced to find and encode exactly these 4 real axes. It has no spare dimensions to waste on noise.

When the AE has 8 latent dimensions, it has capacity to also encode random noise (small random fluctuations in the data). This noise ends up in the latent space and slightly degrades the clusters K-Means finds.

The general principle: **latent dimensions should match the intrinsic dimensionality of the data, not the number of raw features.**

---

## Recommended Configuration

```yaml
# configs/champion.yaml

# ─── Selection rule ──────────────────────────────────────────────
# Highest silhouette is recorded, not used as the selection rule.
# Selected for behavioral coverage, interpretability, and banking use.
metric_leader: b_ae_lat8_k3     # Santander, silhouette 0.512 — comparison only

# ─── Dataset ─────────────────────────────────────────────────────
track: C
dataset: berka
accounts: 4500
role: selected_segmentation
starting_dataset: indian_bank   # assignment primary source; supplemented

# ─── Model architecture ──────────────────────────────────────────
model: ae_kmeans                # Autoencoder + K-Means
latent_dim: 4                   # 4 compressed dimensions (from 8 input features)
n_clusters: 6

# ─── Performance ─────────────────────────────────────────────────
silhouette: 0.417
db_index: 0.731                 # best Davies-Bouldin of the three tracks
ch_index: 3474
stability_ari_kmeans: 0.864     # measured on K-Means, not yet on the AE
run_id: c_ae_lat4_k6

# ─── Segment names ───────────────────────────────────────────────
segments:
  0: New basic accounts
  1: Long-tenure high-activity basic
  2: Loan-centric
  3: Dormant
  4: Card-centric savers
  5: Financially stressed
```

---

## What to do with this result

1. **Treat Berka as the segmentation the report defends.** Indian Bank stays as the required starting analysis. Santander stays as the metric and representation-learning comparison.

2. **Export the trained model** — Save the Berka encoder weights and the six K-Means centroids so a new account can be scored from the same eight features: transaction count, balance, total credit, total withdrawal, recency, account age, loan flag, card flag.

3. **Connect each segment to one banking action, with the evidence named.** Dormant accounts (segment 3, recency about 539 days) are a retention case. High-activity accounts with no loan and no card (segment 1) are a product case. Negative-balance accounts (segment 5) are a service and risk case. Do not present the clusters themselves as a churn or default predictor unless a separate supervised model is trained and evaluated.

4. **Repeat the stability test on the autoencoder.** The published ARI of 0.864 is the K-Means run. The selected model is `c_ae_lat4_k6`.

5. **Fairness check on behavior versus demographics.** Berka has district and client fields. Check whether segment membership is driven by region or age rather than by activity, balance, and product holding, and write down the limitation if it is.
