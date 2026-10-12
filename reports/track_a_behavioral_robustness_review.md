# Track A Behavioral Robustness and Evaluation Review

**Run date:** 2026-10-11
**Population:** 837,951 validated customers
**Purpose:** review cluster stability, transaction-history association, ratio sensitivity, and limitations before downstream use. No champion is selected.

## Executive summary

The historical full-population k=2 K-Means partition is primarily a one-transaction versus repeat-customer split. On its 3,000-person holdout, singleton-status ARI is 0.9865; one cluster is entirely repeat customers and the other is 99.68% single-transaction customers. At k=3, K-Means further separates singleton balance/spending patterns while retaining a repeat-heavy segment.

Ratio clipping in the initial globally standardized feature space changed 0/3,000 k=2 assignments and 12/3,000 (0.40%) and 2/3,000 (0.07%) k=3 assignments after label alignment. This describes a post-global-scaling sensitivity and inherits its transductive limitation. A separate repeat-only K-Means analysis produced k=2 cross-seed ARI 0.9434 on 489 repeat evaluation customers; its population and transform differ from the later raw-aggregate, train-fitted comparison. Neither result establishes a general behavioral segmentation or a banking action rule.

## Population history and behavioral structure

| Transactions per customer | Customers | Population percent |
|---:|---:|---:|
| 1 | 696,449 | 83.1133% |
| 2 or more | 141,502 | 16.8867% |

In the 3,000-person evaluation sample, 2,511 customers are single-transaction and 489 have repeat history. The 50,000-row training samples contain 8,390 repeat customers for seed 42 and 8,424 for seed 2024. Across the validated population, exactly two transactions account for 123,194 customers (14.7018% of the population and approximately 87.1% of the 2+ cohort).

| Transactions | Customers | Median balance | Median average transaction | Median transaction SD | Median total spend | Ratio P50 / P90 | Median recency | Weekend P50 / P90 | Night P50 / P90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 696,449 | 16,032.18 | 431.54 | 0.00 | 431.54 | 0.02561 / 0.94988 | 57 | 0 / 1 | 0 / 1 |
| 2 | 123,194 | 28,032.18 | 646.50 | 458.65 | 1,293.00 | 0.05266 / 2.29614 | 47 | 0.5 / 0.5 | 0 / 0.5 |
| 3 | 16,492 | 36,131.76 | 754.88 | 673.16 | 2,264.65 | 0.08020 / 3.61086 | 43 | 0.333 / 0.667 | 0.333 / 0.667 |
| 4 | 1,661 | 42,343.19 | 799.25 | 779.47 | 3,197.00 | 0.10872 / 4.61798 | 41 | 0.25 / 0.5 | 0.25 / 0.5 |
| 5 | 141 | 40,265.99 | 1,004.80 | 1,034.36 | 5,024.00 | 0.27816 / 6.22898 | 40 | 0.2 / 0.6 | 0.2 / 0.6 |
| 6 | 14 | 67,866.94 | 1,406.56 | 2,002.62 | 8,439.37 | 0.15513 / 1.30553 | 39.5 | 0.333 / 0.5 | 0.25 / 0.5 |

For single-transaction customers, the sample standard deviation is undefined and is filled with zero; average transaction amount equals total spend. This is a mathematical consequence of one observation, not a low-value or inactivity indicator. With only two observations, sample standard deviation has one degree of freedom, and weekend/night ratios are quantized. Segment interpretation must account for observation depth.

### Representative original-matrix profiles

These selected profiles come from the earlier globally transformed full-population comparison. They are retained here as historical context, not as matched comparators for the later raw-aggregate train-fitted evaluation.

| Model / k / seed | Cluster | Count / % | Singleton % | Median transactions | Median balance | Median average transaction | Median total spend | Median ratio | Median recency | Median weekend ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K-Means-9 / 2 / 42 | 0 | 481 / 16.0% | 0.0% | 2 | 31,008.52 | 649.96 | 1,372.50 | 0.04771 | 45 | 0.333 |
| K-Means-9 / 2 / 42 | 1 | 2,519 / 84.0% | 99.7% | 1 | 16,169.86 | 411.00 | 414.00 | 0.02540 | 58 | 0.000 |
| K-Means-9 / 3 / 42 | 0 | 475 / 15.8% | 0.0% | 2 | 31,483.28 | 679.00 | 1,390.00 | 0.04918 | 45 | 0.333 |
| K-Means-9 / 3 / 42 | 1 | 1,346 / 44.9% | 99.9% | 1 | 30,005.97 | 1,000.00 | 1,000.00 | 0.03876 | 57 | 0.000 |
| K-Means-9 / 3 / 42 | 2 | 1,179 / 39.3% | 98.9% | 1 | 7,881.86 | 128.00 | 130.00 | 0.01486 | 58 | 0.000 |
| GMM-9 / 3 / 42 | 0 | 489 / 16.3% | 0.0% | 2 | 29,266.61 | 633.50 | 1,353.86 | 0.04729 | 45 | 0.333 |
| GMM-9 / 3 / 42 | 1 | 2,325 / 77.5% | 100.0% | 1 | 19,312.84 | 403.40 | 403.40 | 0.02131 | 57 | 0.000 |
| GMM-9 / 3 / 42 | 2 | 186 / 6.2% | 100.0% | 1 | 52.07 | 541.93 | 541.93 | 7.99610 | 59 | 0.000 |
| AE-8 + K-Means / 2 / 42 | 0 | 489 / 16.3% | 0.0% | 2 | 29,266.61 | 633.50 | 1,353.86 | 0.04729 | 45 | 0.333 |
| AE-8 + K-Means / 2 / 42 | 1 | 2,511 / 83.7% | 100.0% | 1 | 16,184.16 | 415.00 | 415.00 | 0.02540 | 58 | 0.000 |

The k=2 groups differ in balance/spend while being strongly separated by transaction history. At k=3, two K-Means segments remain almost entirely single-transaction customers. The small GMM component has a very high ratio median and warrants low-balance/outlier review.

### Association with singleton status

| Model family | k | Evaluation ARI vs. singleton status | Evaluation NMI vs. singleton status |
|---|---:|---:|---:|
| K-Means-9 | 2 | 0.9865 | 0.9643 |
| K-Means-9 | 3 | 0.3617–0.3630 | 0.5715–0.5753 |
| GMM-9 | 2 | 0.8129–1.0000 | 0.7201–1.0000 |
| GMM-9 | 3 | 0.7799–0.7818 | 0.7997–0.8009 |
| AE-9 | 2 | 0.1813–0.6805 | 0.1545–0.5946 |
| AE-9 | 3 | 0.4606–0.4674 | 0.6412–0.6440 |
| AE-8 | 2 | 0.2022–1.0000 | 0.2782–1.0000 |
| AE-8 | 3 | 0.4277 | 0.6279 |
| K-Means-8 | 2 | 0.9865 | 0.9643 |
| K-Means-8 | 3 | 0.3618–0.3631 | 0.5715–0.5754 |

These values characterize partitions on the historical globally transformed evaluation set. Strong association with singleton status is not evidence of spending personas or causal behavior.

### AE and traditional K-Means in a common original space

| Representation comparison | Seed | k | AE silhouette | K-Means silhouette in same original space | Assignment ARI |
|---|---:|---:|---:|---:|---:|
| AE-9 vs K-Means-9 | 42 | 2 | 0.1701 | 0.3327 | 0.1761 |
| AE-9 vs K-Means-9 | 2024 | 2 | 0.2545 | 0.3327 | 0.6695 |
| AE-9 vs K-Means-9 | 42 | 3 | 0.2539 | 0.2215 | 0.2692 |
| AE-9 vs K-Means-9 | 2024 | 3 | 0.2537 | 0.2216 | 0.2724 |
| AE-8 vs K-Means-8 | 42 | 2 | 0.3309 | 0.3328 | 0.9865 |
| AE-8 vs K-Means-8 | 2024 | 2 | 0.1949 | 0.3328 | 0.1972 |
| AE-8 vs K-Means-8 | 42 | 3 | 0.2551 | 0.2228 | 0.2678 |
| AE-8 vs K-Means-8 | 2024 | 3 | 0.2551 | 0.2229 | 0.2735 |

The AE-8 seed-42 k=2 match is the singleton/repeat split (ARI 0.9865), while seed 2024 does not reproduce it (ARI 0.1972). This is not stable evidence of added AE value. Latent metrics are excluded from this same-space comparison.

## Historical preprocessing exposure and evaluation reuse

The original StandardScaler was fit on all 837,951 customers before its model samples were drawn. Consequently, the initial K-Means/GMM metrics and the ratio-clipping sensitivity on that matrix are transductive with respect to scaler statistics. This is feature-distribution leakage, not use of target labels. The global preprocessing artifact is retained for reproducibility; later leakage-controlled work rebuilds raw aggregates and fits its transformations on training IDs only.

The 3,000-person evaluation set was reused to compare many candidates and therefore is a development/model-selection set, not independent final validation. The relevant original fit/evaluation manifests, membership files, metrics, and hashes remain in `artifacts/experiments/` and `artifacts/segments/candidates/`. The candidate inventory and corrected ratio audit are linked from [`track_a_model_readiness.md`](track_a_model_readiness.md).

## Initial ratio-clipping sensitivity

This sensitivity clips a copy of the initial globally standardized ratio feature at each training sample's 0.5th/99.5th percentiles. It does not overwrite the original matrix. The following values are in standardized units.

| Seed | q0.5 lower | q99.5 upper | Train below / above / changed | Evaluation below / above / changed |
|---:|---:|---:|---:|---:|
| 42 | -0.027234 | 0.611756 | 250 / 250 / 500 (1.000%) | 13 / 7 / 20 (0.667%) |
| 2024 | -0.027234 | 0.535400 | 250 / 250 / 500 (1.000%) | 13 / 9 / 22 (0.733%) |

| Seed | k | Original K-Means S / DB / CH | Clipped labels in original space S / DB / CH | Clipped labels in clipped space S / DB / CH | Evaluation membership changes | ARI vs nine-feature K-Means | ARI vs eight-feature K-Means |
|---:|---:|---|---|---|---:|---:|---:|
| 42 | 2 | 0.3327 / 1.1637 / 922.8 | 0.3327 / 1.1637 / 922.8 | 0.3328 / 1.1541 / 961.0 | 0 / 3,000 | 1.0000 | 1.0000 |
| 2024 | 2 | 0.3327 / 1.1637 / 922.8 | 0.3327 / 1.1637 / 922.8 | 0.3328 / 1.1541 / 961.0 | 0 / 3,000 | 1.0000 | 1.0000 |
| 42 | 3 | 0.2215 / 1.6064 / 895.8 | 0.2216 / 1.6057 / 895.8 | 0.2230 / 1.5961 / 941.5 | 12 / 3,000 (0.400%) | 0.9858 | 0.9881 |
| 2024 | 3 | 0.2216 / 1.6072 / 895.8 | 0.2216 / 1.6070 / 895.8 | 0.2229 / 1.5975 / 941.5 | 2 / 3,000 (0.067%) | 0.9976 | 1.0000 |

Matched-label training changes were 1/50,000 and 0/50,000 at k=2, and 169/50,000 (0.338%) and 27/50,000 (0.054%) at k=3. These small k=2/3 assignment changes describe this particular post-scaling experiment; they do not settle raw-ratio clipping in a training-fitted pipeline. Exact cutoff and changed-customer artifacts remain with the original comparison run.

## Repeat-only K-Means comparison

The repeat-only analysis filters the original seed-specific training samples and the common 3,000-person evaluation sample to customers with at least two transactions. Training has 8,390 (seed 42) and 8,424 (seed 2024) customers; evaluation has 489 customers. It starts from the globally transformed feature matrix, applies the configured logarithms, and fits a new scaler on repeat-only training IDs. Therefore it inherits the earlier global-scaling exposure and is not the same as the later raw-aggregate train-fitted nine-feature analysis.

| Seed | k | Repeat training customers | Shared evaluation customers | Silhouette in repeat-only space | DB | CH | Evaluation sizes | Cross-seed ARI |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 42 | 2 | 8,390 | 489 | 0.1780 | 1.8716 | 116.93 | 238 / 251 | 0.9434 |
| 2024 | 2 | 8,424 | 489 | 0.1730 | 1.9672 | 39.46 | 231 / 258 | 0.9434 |
| 42 | 3 | 8,390 | 489 | 0.1769 | 1.8695 | 116.86 | 246 / 243 | 0.7586 |
| 2024 | 3 | 8,424 | 489 | 0.1917 | 1.7850 | 28.80 | 233 / 199 / 57 | 0.7586 |

These scores describe a different 489-customer population and repeat-only geometry, not the 10,000-customer raw-aggregate train-fitted comparison. The k=2 seed-42 medians show balance 20,875 vs. 43,850.95 and total spend 700 vs. 2,906.50, while both clusters have median transaction count 2. Profiles are descriptive, not causal.

## Cluster interpretation and limitations

- Initial full-population k=2 K-Means and some GMM/AE candidates are strongly associated with singleton status. This largely reflects observation depth, not a validated behavioral persona.
- Initial k=3 K-Means separates singleton spending/balance levels while retaining a repeat-heavy segment; most members of two clusters remain single-transaction customers.
- Diagonal GMM k=3 contains a small singleton-only, high-ratio component that merits denominator/low-balance review.
- Representation-learning latent metrics are not directly comparable to original-space metrics. The matched autoencoder results show unstable k=2 memberships and no consistent added value over the train-fitted traditional baselines; see [`track_a_matched_model_comparison.md`](track_a_matched_model_comparison.md).
- The validated 2+ population is 16.8867% of all validated customers, with 87.1% exactly-two transaction customers. The 3+ population is only 2.1849%; it is a separate sensitivity population, not a substitute for the 2+ analysis.

No cluster names or customer-facing descriptions are assigned. Any future use requires a new independent/prospective evaluation, sufficient transaction history, fairness review, and an explicit non-adverse-use policy. DEC/IDEC remains deferred.

## Reproducibility and records

The reusable implementation and saved run artifacts remain in `src/customer_segmentation/models/` and the versioned subdirectories under `artifacts/`. The corrected ratio-clipping ledger interpretation is summarized in [`track_a_train_fitted_segmentation.md`](track_a_train_fitted_segmentation.md); the detailed machine-readable check remains under the versioned diagnostics directory. The comparison board is append-only and retains the original model and metric records.
