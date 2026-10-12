# Berka Behavioral Clustering Baselines

**Snapshot:** account-level Berka snapshot as of 1997-12-31; **population:** accounts with at least 365 elapsed days of tenure; **methods:** K-Means and diagonal-covariance Gaussian Mixture Models (GMM).

**Scope:** Initial unsupervised baselines. No outcomes or 1998 transactions were used.

## Summary

The primary population contains 3,602 eligible accounts. A deterministic account split produced 2,701 training accounts and 901 held-out accounts, shared by every algorithm, cluster count, and initialization seed. Imputation, transformations, clipping bounds, and scaling were fitted using training accounts only.

The most defensible initial solutions are the two-cluster K-Means and GMM candidates. Both are seed-stable, have no tiny held-out segment, and distinguish accounts mainly by transaction scale and balance behavior. Held-out silhouette scores are modest (0.210 and 0.216, respectively), so these are useful baselines rather than evidence of strong natural segmentation. The higher-silhouette three-component GMM creates a 15–28-account segment (1.66%–3.11%); all tested settings with three or more clusters create at least one segment below 5%.

No business personas are assigned. The available evidence does **not** justify an autoencoder yet: the stable two-cluster solutions are simple, while finer partitions either produce tiny segments or, for K-Means at k=4, show poor seed stability. Review feature sensitivity and the reserved 1998 history before considering a learned representation.

## Data and feature review

The runner reads only the validated `berka_account_snapshot_v1` CSV and metadata. It verifies the 1997-12-31 snapshot date, feature definitions, row count, and account-key uniqueness. The eligible cohort is defined by `account_tenure_days >= 365`; no `disp.csv` join is performed. `account_id` remains an alignment key and is never a model input. The 1998 transaction tail was not read or used.

The fixed split uses seed **8675309** and sorted account IDs before seeded permutation: **2,701 train / 901 held out**. All 42 source numeric features were profiled on the training accounts only. There were **3 missing values** in the reviewed numeric inputs, all in `activity_change_ratio`, and **4 constant features** (`has_90d_history`, `has_365d_history`, `unclassified_transaction_count`, and `recent_window_days_observed`). None of the 13 selected model inputs was missing in this cohort’s training split.

Skew and outlier diagnostics support explicit transforms. In the training data, `recency_days` had skew 31.17 (median 0, maximum 480); `negative_balance_frequency` had skew 14.39 (median 0, maximum 0.3902); gross money movement had skew 1.64 (median 737,503, maximum 6,851,198); and median transaction amount had skew 1.50 (median 2,302, maximum 17,500). Tukey 1.5-IQR counts included 131 outliers for gross money movement, 142 for negative-balance frequency, 94 for median transaction amount, and 174 for activity-change count. Outlier counts are descriptive only; rows were not removed.

Strong source-feature redundancies included credit/debit amount totals (Spearman 0.999), activity-change ratio/count (0.982), transaction count/account tenure (0.912), credit share/debit frequency (absolute 0.854), and balance median/volatility (0.816). Within the selected 13-feature set, the largest absolute Spearman correlations were 0.759 (debit frequency/recent transaction frequency) and 0.752 (gross movement/balance volatility).

### Exact clustering inputs

| Feature | Use and preprocessing |
|---|---|
| `account_tenure_days` | Account age at snapshot; `log1p` |
| `recency_days` | Days since last observed transaction; `log1p` |
| `credit_frequency_per_day` | Credit activity normalized by observation days; `log1p` |
| `debit_frequency_per_day` | Debit activity normalized by observation days; `log1p` |
| `gross_money_movement` | Derived as `credit_amount_total + debit_amount_total`; `log1p` |
| `net_cash_flow` | Signed credit minus debit total; `asinh` |
| `transaction_amount_median` | Typical transaction size; `log1p` |
| `balance_std` | Account balance volatility; `log1p` |
| `negative_balance_frequency` | Fraction of negative-balance transactions; unchanged, bounded in [0, 1] |
| `interarrival_days_median` | Typical gap between transactions; `log1p` |
| `interarrival_days_std` | Variability of inter-arrival gaps; `log1p` |
| `recent_90d_transaction_frequency_per_day` | Recent activity normalized by observed days; `log1p` |
| `activity_change_count` | Recent versus prior 90-day count difference; signed `asinh` |

The exact exclusion list and rationale are in `configs/models/berka_behavioral_baselines_v1_1.yaml`. It excludes identifiers; tenure-constant flags and zero-valued unclassified counts; count features redundant with tenure-normalized frequencies; complementary shares; separate credit/debit totals (combined into gross movement); undefined/redundant activity-change ratio; redundant amount and balance summaries; duplicate recent/prior activity summaries; and observation-window fields constant in this cohort. No demographic, client, or future-outcome features are used. `balance_median`, credit/debit totals, and other descriptive fields remain available for held-out segment profiling, not model fitting.

### Preprocessing

The saved preprocessor applies training-median imputation after feature transforms, then feature-specific transforms, training-fitted 1st/99th percentile clipping, and `StandardScaler` fitted on training accounts. Positive-skewed inputs use `log1p`; signed cash flow and activity change use `asinh`; bounded negative-balance frequency retains its original scale before clipping and standardization. The resulting fixed feature order is stored with each artifact. The persisted preprocessor is a portable state payload and restores deterministic held-out transformations.

## Baseline evaluation

Each candidate was fitted on the same 2,701 training accounts using seeds **42, 2024, and 123**. Metrics below are means across those three initializations; held-out cluster-size range and pairwise adjusted Rand index (ARI) are across the same 901 held-out accounts. Silhouette, Davies–Bouldin, and Calinski–Harabasz scores use all 901 held-out accounts in the preprocessed feature space. `Tiny` means at least one segment is below 5%, `severe tiny` is below 1%, and `unstable` means mean pairwise seed ARI is below 0.75.

| Algorithm | k | Silhouette | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Smallest segment | Seed ARI mean / min | Median fit time (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K-Means | 2 | 0.2097 | 1.7299 | 231.12 | 43.29%–43.73% | 0.9882 / 0.9823 | 0.034 | Stable, no tiny segment |
| K-Means | 3 | 0.2179 | 1.2943 | 253.55 | 1.66% | 1.0000 / 1.0000 | 0.063 | Tiny segment |
| K-Means | 4 | 0.2152 | 1.3031 | 231.61 | 1.66% | 0.6270 / 0.4432 | 0.099 | Tiny and unstable |
| K-Means | 5 | 0.1959 | 1.3569 | 229.54 | 1.66% | 0.9674 / 0.9511 | 0.141 | Tiny segment |
| K-Means | 6 | 0.1700 | 1.4487 | 211.56 | 1.66% | 0.9925 / 0.9916 | 0.090 | Tiny segment |
| Diagonal GMM | 2 | 0.2162 | 1.6789 | 219.72 | 32.85% | 1.0000 / 1.0000 | 0.080 | Stable, no tiny segment |
| Diagonal GMM | 3 | 0.3223 | 1.5158 | 138.78 | 1.66%–3.11% | 0.8915 / 0.8452 | 0.096 | Highest silhouette; tiny segment |
| Diagonal GMM | 4 | 0.2061 | 1.4420 | 198.58 | 1.66%–3.11% | 0.7866 / 0.6811 | 0.116 | Tiny; less stable |
| Diagonal GMM | 5 | 0.1364 | 1.6527 | 187.20 | 1.66%–3.11% | 0.9252 / 0.8879 | 0.253 | Tiny segment |
| Diagonal GMM | 6 | 0.1261 | 1.6764 | 175.48 | 1.66% | 0.9949 / 0.9923 | 0.188 | Tiny segment |

No candidate collapsed to fewer clusters than requested. No minimum segment fell below 1%, and all GMM fits converged (14–34 iterations) without captured warnings. Mean fit time across all 15 runs was 0.111 seconds for K-Means and 0.145 seconds for GMM in the thread-limited local run; timing excludes held-out scoring and is environment-specific. K-Means has no BIC/AIC.

| GMM k | Mean held-out BIC ↓ | Mean held-out AIC ↓ | Mean training BIC ↓ | Mean training AIC ↓ |
|---:|---:|---:|---:|---:|
| 2 | 21,494.27 | 21,239.68 | 61,978.17 | 61,665.40 |
| 3 | 11,712.47 | 11,328.19 | 32,887.14 | 32,415.03 |
| 4 | 9,756.47 | 9,242.49 | 27,276.02 | 26,644.57 |
| 5 | 8,778.60 | 8,134.93 | 24,621.42 | 23,830.64 |
| 6 | 8,361.66 | 7,588.30 | 23,126.24 | 22,176.12 |

BIC/AIC decrease as components are added over this tested range, but the additional components produce small segments. They do not override held-out separation, size, and stability considerations.

## Held-out profiles and cluster drivers

The following profiles use seed 42 for the stable two-cluster candidates. Cluster numbers are arbitrary; values are medians on the original, untransformed held-out features.

| Candidate / segment | Accounts | Tenure days | Gross movement | Credit total | Debit total | Net flow | Credit / debit tx per day | Median balance | Balance SD | Activity change count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K-Means k=2 / 0 | 394 (43.7%) | 798 | 334,655 | 179,130 | 154,928 | 23,123 | 0.0666 / 0.0900 | 22,910 | 5,843 | 1 |
| K-Means k=2 / 1 | 507 (56.3%) | 1,214 | 1,476,816 | 753,813 | 705,628 | 49,102 | 0.0704 / 0.1230 | 41,860 | 13,825 | 1 |
| GMM k=2 / 0 | 605 (67.1%) | 1,091 | 1,273,202 | 662,939 | 613,723 | 45,753 | 0.0704 / 0.1195 | 39,797 | 13,121 | 1 |
| GMM k=2 / 1 | 296 (32.9%) | 786 | 263,935 | 142,857 | 120,990 | 22,064 | 0.0637 / 0.0896 | 21,244 | 5,351 | 1 |

Cluster association eta-squared (share of raw held-out feature variance associated with cluster membership) confirms that these partitions are more strongly related to financial scale and balance behavior than to account age or recent activity change:

| Driver | K-Means k=2 | GMM k=2 |
|---|---:|---:|
| Account tenure | 0.045 | 0.028 |
| Gross money movement | 0.363 | 0.292 |
| Net cash flow | 0.329 | 0.258 |
| Median balance | 0.433 | 0.384 |
| Balance volatility | 0.336 | 0.349 |
| Debit frequency | 0.306 | 0.242 |
| Activity-change count | 0.002 | <0.001 |

The two-cluster results therefore appear to separate relatively lower- and higher-scale financial behavior, including balance variability and debit frequency. The activity-change feature barely differentiates these segments; account tenure explains little of their raw variance. This is evidence of financial-scale structure, not by itself evidence of distinct actionable customer behavior.

## Artifacts and reproducibility

Reusable experiment code: `src/customer_segmentation/models/berka_behavioral_baselines.py`; configuration: `configs/models/berka_behavioral_baselines_v1_1.yaml`; versioned outputs: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/`.

The output directory contains the copied configuration, run metadata and split-ID hashes, training-only diagnostics and Spearman matrix, account split manifest, train-fitted preprocessing state (`preprocessor_train_fitted.joblib`), 30 candidate model files, all held-out predictions with GMM maximum-component probabilities, the complete per-candidate metric table, and per-segment held-out profiles. The 27,030 assignments preserve `account_id` and model/seed/cluster alignment. Reloading all 30 model files reproduced the saved held-out assignments exactly.

The implementation tests pass (6 tests), Ruff passes, and serialization/prediction replay was verified. Model and dataset files are excluded by the repository’s existing `.gitignore`; no comparison ledger was read for writing or changed.
