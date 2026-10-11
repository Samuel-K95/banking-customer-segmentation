# Champion Selection — Which Dataset & Model to Use
### Banking Customer Segmentation · Week 4 Project Report

---

## Quick-Start Glossary

| Term | Plain-English Meaning |
|---|---|
| **Champion model** | The single best-performing model selected after comparing all candidates across all evaluation criteria |
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

> **Use: Track B (Santander) + Autoencoder + K-Means, k=3, latent_dim=8**
> **Model ID: `b_ae_lat8_k3`**
>
> It has the best cluster quality, the largest real-world dataset, perfect reproducibility, and segments that map directly onto actionable banking strategies.

---

## Cross-Track Comparison Table

| Track | Dataset | Customers | Champion Model | Silhouette ↑ | DB Index ↓ | Stability ARI ↑ | k |
|---|---|---|---|---|---|---|---|
| **B — Santander** | Product portfolio | 931,000 | AE + K-Means (lat=8) | **0.512** | 0.891 | **1.000** | 3 |
| C — Berka | Czech bank relational | 4,500 | AE + K-Means (lat=4) | 0.417 | **0.731** | 0.864* | 6 |
| A — Indian Bank | Transaction ledger | 837,000 | AE + K-Means (lat=8) | 0.332 | 1.099 | — | 6 |

↑ = higher is better. ↓ = lower is better. *Track C stability was measured for K-Means, not AE.

---

## Why Track B (Santander) Is the Champion

### Reason 1: Best Silhouette Score — 0.512

**What silhouette means, precisely:** For every individual customer, the silhouette score computes how much closer that customer is to their own cluster than to the nearest other cluster. The overall score is the average across all customers.

- A score of 0.512 means that, on average across 931,000 customers, each customer is **51% more similar to their own cluster-mates than to customers in the next-closest cluster**
- Scores above 0.50 indicate strong, commercially meaningful separation
- Track A's 0.332 means customers are 33% more similar to their group — usable but weaker
- Track C's 0.417 is good but still below the 0.50 threshold for "strong" clustering

### Reason 2: Perfect Stability — ARI = 1.000

**What stability testing involves:**
1. Take all 931,000 Santander customers
2. Randomly split them into two groups of ~465,000 each
3. Train the Autoencoder + K-Means model on each group completely independently
4. Compare the cluster assignments: does each customer end up in the same type of group in both models?
5. Repeat 10 times with different random splits

**What ARI = 1.000 means:** Every single one of 10 independent trials produced identical cluster assignments (with zero variance — ± 0.000). The 3-group structure is a genuine, stable property of the Santander customer base, not a result of random initialisation. This is the gold standard for reliability in unsupervised learning.

For context:
- ARI = 1.000 → perfect match between two independent models
- ARI = 0.800 → strong but not perfect consistency
- ARI = 0.000 → cluster assignments are essentially random

Track B's ARI = 1.000 is the highest possible score — an extremely rare result in real-world clustering.

### Reason 3: Largest Validated Dataset — 931,000 Customers

Statistical patterns are more reliable when discovered on more data. Finding the same 3 customer types consistently across 931,000 real Santander customers is much stronger evidence than finding 6 clusters in 4,500 accounts (Track C) or 837,000 transaction records where individual customers may appear only once (Track A).

The Track B model was trained on a random sample of 100,000 customers (for computational efficiency), but the full dataset of 931,000 was used for validation. The fact that ARI = 1.000 means the model generalises across the entire population.

### Reason 4: Most Directly Actionable Segments

The 3 Santander segments each have an immediate, obvious business use because the features used in clustering (product ownership) directly correspond to what the bank sells:

| Segment | Who they are | What to do — immediately |
|---|---|---|
| **0 — Mass Market** | Young, new customers, current account only, below-median income | Offer: savings account → first credit card |
| **1 — Established Middle** | 3–5 year customers, 3–5 products, active and engaged | Offer: mortgage, pension plan, personal loan |
| **2 — Affluent Senior** | 8–15 year customers, 6+ products, high income, investments | Retain: premium advisory, wealth management, estate planning |

There is zero ambiguity about what to do with each segment — the product ownership features *are* the bank's product catalogue.

---

## Why Track C Is Second-Best (and When to Use It Instead)

Track C's Autoencoder (latent=4, k=6) achieved the **best Davies-Bouldin Index (0.731)** across all three tracks. DB measures cluster compactness relative to cluster separation — Track C's clusters are the most internally tight relative to the distances between them.

**When to use Track C instead of B:**

Track C is the only dataset that includes **loan repayment history with default status**. The `loan` table records whether each loan was repaid on time (`A`) or defaulted (`D`). This means Track C's segments can be **labelled with credit risk** — turning the unsupervised clustering into a supervised credit risk prediction tool.

If your goal is building a **credit risk model** rather than a product recommendation engine, Track C is the right starting point.

**Limitation:** 4,500 accounts is very small. Results may not generalise to a full bank's customer base without revalidation on larger data.

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

# ─── Dataset ─────────────────────────────────────────────────────
track: B
dataset: santander_product_recommendation
raw_file: train_ver2.csv
training_sample_size: 100000    # rows sampled from 931,453 total

# ─── Model architecture ──────────────────────────────────────────
model: ae_kmeans                # Autoencoder + K-Means
latent_dim: 8                   # 8 compressed dimensions (from 24 input features)
encoder_layers: [24, 16, 8]     # progressive compression
decoder_layers: [8, 16, 24]     # symmetric reconstruction path

# ─── Clustering ──────────────────────────────────────────────────
n_clusters: 3                   # k=3 (validated by ARI=1.000 stability test)
kmeans_n_init: 20               # number of random restarts for K-Means

# ─── Performance ─────────────────────────────────────────────────
silhouette: 0.512               # strong (>0.50 threshold)
db_index: 0.891                 # good (<1.00 threshold)
ch_index: 90835
stability_ari: 1.000            # perfect stability
run_id: b_ae_lat8_k3

# ─── Segment names ───────────────────────────────────────────────
segments:
  0: Mass Market                # young, basic, low-seniority
  1: Established Middle         # multi-product, active, mid-tenure
  2: Affluent Senior            # high-income, investments, 10+ yr tenure
```

---

## What to Do With This Result

1. **Export the trained model** — Save the AE encoder weights (`.pt` PyTorch file) and K-Means centroids (`.pkl` file) so new customers can be scored without retraining

2. **Build a scoring pipeline** — When a new customer record arrives:
   - Extract the 24 features (age, seniority, product flags)
   - Pass through the saved encoder → 8 latent numbers
   - Assign to the nearest K-Means centroid → segment 0, 1, or 2
   - Route to the corresponding marketing campaign

3. **Monthly refresh** — Santander's data was monthly snapshots. Re-run segment assignment each month. Track customers who move between segments (e.g. Segment 0 → 1 = progression; Segment 2 → 0 = churn risk)

4. **A/B test the strategy** — Split each segment into a "treatment" group (receives the targeted campaign) and a "control" group (receives no specific campaign). Measure revenue uplift to quantify the business value of segmentation

5. **Fairness audit** — Before deploying, check that segment membership is not strongly correlated with protected characteristics (gender, age band, region). A model that systematically assigns certain demographic groups to the lowest-value segment could create regulatory or ethical problems
