# Berka Behavioral Feature Sensitivity

**Snapshot:** 1997-12-31; accounts with at least 365 days of tenure; **population:** 3,602 accounts; **split:** 2,701 training and 901 held out.

**Models:** k=2 K-Means and diagonal-covariance GMM; seeds 42, 2024, and 123

## Findings

The results indicate **partial, method-dependent sensitivity** to gross financial movement and balance volatility. K-Means retains a reproducible broad low-/high-movement partition when either or both features are removed, although agreement with the original assignment declines from moderate (ARI 0.756–0.773 for single ablations) to 0.603 when both are removed. Its remaining clusters continue to differ in credit/debit frequency, cash flow, transaction size, and balance volatility, showing those patterns are also reflected by correlated features.

GMM is more sensitive to the feature set. Its original two-cluster assignment is almost unrelated to the reduced-feature assignments after removing gross movement (ARI 0.073) and both target features (ARI −0.011). GMM nevertheless reproduces each reduced-feature partition perfectly across initialization seeds. Some reduced-feature partitions reflect behavioral differences: without gross movement, GMM isolates 30 accounts with very low activity and long inter-arrival times; without both variables, its 8.1% segment differs in balance volatility, tenure, and activity-change count, but frequency and regularity contrasts are modest. Thus, seed stability within a feature space does not imply stability to feature choice.

The 30-account GMM segment after removing gross movement is below the 5% small-segment threshold. Do not interpret its high silhouette as proof of superiority. Silhouette, Davies–Bouldin, and Calinski–Harabasz values below are reported **within each feature space**; they are not compared across feature configurations as evidence that one configuration is better.

## Design and reproducibility

The baseline config, saved preprocessing state, snapshot metadata/feature definitions, feature dictionary, and preceding baseline report were inspected before implementation. The experiment reads only the validated 1997-12-31 snapshot and saved baseline split. It uses exactly the same 2,701 training IDs and 901 held-out IDs, verified against the baseline run’s ordered ID hashes. The original 13-feature configuration exactly reproduced all six existing baseline k=2 assignment vectors (two algorithms × three seeds). No 1998 transactions were accessed.

Four preprocessing states were fit independently, each using only the same 2,701 training accounts. For each configuration, training medians, feature-specific transforms, 1st/99th percentile clipping bounds, and `StandardScaler` were refit for the remaining inputs. Models were then fit with the same algorithms, k=2, and seeds 42, 2024, and 123. Held-out profiles use seed 42 and report medians in original feature units. For comparability only, profile labels are oriented by held-out gross movement (lower/higher); model labels and cross-configuration ARI remain unmodified and permutation-invariant. Gross movement used for this profile orientation was never used to fit an ablated configuration that excludes it.

### Feature configurations

| Configuration | Clustering inputs |
|---|---|
| Original 13 | `account_tenure_days`, `recency_days`, `credit_frequency_per_day`, `debit_frequency_per_day`, `gross_money_movement`, `net_cash_flow`, `transaction_amount_median`, `balance_std`, `negative_balance_frequency`, `interarrival_days_median`, `interarrival_days_std`, `recent_90d_transaction_frequency_per_day`, `activity_change_count` |
| Without gross movement | Original inputs except `gross_money_movement` (12 features) |
| Without balance volatility | Original inputs except `balance_std` (12 features) |
| Without both | Original inputs except `gross_money_movement` and `balance_std` (11 features) |

`gross_money_movement` is the derived sum of credit and debit totals. All other features, transformations, and fitting policies follow the inspected Berka baseline config.

The balance-volatility ablation removes `balance_std` only; `negative_balance_frequency` remains as a separate behavior feature. Excluding gross movement likewise retains related monetary inputs `net_cash_flow` and `transaction_amount_median`. The ablations test the named variables, not every correlated financial or balance signal.

## Held-out clustering metrics

Scores and segment percentages are means/ranges across the three seeds; seed ARI is the mean/minimum pairwise ARI among those seeds on identical held-out account IDs. A small segment is below 5%.

### Original 13 features

| Algorithm | Silhouette | Davies–Bouldin | Calinski–Harabasz | Smallest segment | Seed ARI mean / min |
|---|---:|---:|---:|---:|---:|
| K-Means | 0.2097 | 1.7299 | 231.12 | 43.29%–43.73% | 0.9882 / 0.9823 |
| Diagonal GMM | 0.2162 | 1.6789 | 219.72 | 32.85% | 1.0000 / 1.0000 |

### Excluding gross financial movement

| Algorithm | Silhouette | Davies–Bouldin | Calinski–Harabasz | Smallest segment | Seed ARI mean / min |
|---|---:|---:|---:|---:|---:|
| K-Means | 0.1890 | 1.8475 | 201.31 | 43.73%–44.40% | 0.9823 / 0.9735 |
| Diagonal GMM | 0.5662 | 1.0457 | 182.57 | 3.33% | 1.0000 / 1.0000 |

### Excluding balance volatility

| Algorithm | Silhouette | Davies–Bouldin | Calinski–Harabasz | Smallest segment | Seed ARI mean / min |
|---|---:|---:|---:|---:|---:|
| K-Means | 0.1952 | 1.7874 | 215.35 | 44.73% | 1.0000 / 1.0000 |
| Diagonal GMM | 0.3163 | 1.5170 | 174.20 | 12.32% | 1.0000 / 1.0000 |

### Excluding both gross movement and balance volatility

| Algorithm | Silhouette | Davies–Bouldin | Calinski–Harabasz | Smallest segment | Seed ARI mean / min |
|---|---:|---:|---:|---:|---:|
| K-Means | 0.1756 | 1.8933 | 189.70 | 46.73%–46.84% | 0.9941 / 0.9911 |
| Diagonal GMM | 0.4433 | 2.1883 | 99.20 | 8.10% | 1.0000 / 1.0000 |

All 24 candidates produced two nonempty clusters, and all 12 GMM fits converged. GMM without gross movement has the only segment below 5%: **30 of 901 held-out accounts**. The other reduced GMM partitions have minority segments of 111 (12.32%) and 73 (8.10%) accounts.

## Assignment sensitivity

ARI compares predictions on the same ordered 901 held-out account IDs. Entries pair the same algorithm across the two listed feature configurations; `paired seed` compares identical seeds, while the all-pairs statistic compares all nine seed combinations.

| Ablation versus original | Algorithm | Paired-seed ARI mean / min | All-seed-pair ARI mean / min |
|---|---|---:|---:|
| Exclude gross movement | K-Means | 0.7562 / 0.7434 | 0.7571 / 0.7434 |
| Exclude gross movement | Diagonal GMM | 0.0726 / 0.0726 | 0.0726 / 0.0726 |
| Exclude balance volatility | K-Means | 0.7731 / 0.7704 | 0.7731 / 0.7704 |
| Exclude balance volatility | Diagonal GMM | 0.3003 / 0.3003 | 0.3003 / 0.3003 |
| Exclude both | K-Means | 0.6026 / 0.5945 | 0.6026 / 0.5945 |
| Exclude both | Diagonal GMM | −0.0111 / −0.0111 | −0.0111 / −0.0111 |

Complete pairwise comparisons for every configuration pair and seed combination are saved in the artifact directory. The original assignments match the baseline exactly, despite arbitrary cluster-number permutations being possible in general.

## Held-out financial and behavioral profiles

Each value is the median for seed 42, in low-/high-gross-movement profile order. Amounts are source units. The profile orientation is descriptive only and does not affect model fitting or metrics.

| Features / algorithm | Segment size % L/H | Tenure days L/H | Gross movement L/H | Credit total L/H | Debit total L/H | Net flow L/H | Median balance L/H | Balance SD L/H | Credit frequency/day L/H | Debit frequency/day L/H | Median transaction amount L/H | Median inter-arrival days L/H | Recent frequency/day L/H | Activity change L/H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original / K-Means | 43.7 / 56.3 | 798 / 1,214 | 334,655 / 1,476,816 | 179,130 / 753,813 | 154,928 / 705,628 | 23,123 / 49,102 | 22,910 / 41,860 | 5,843 / 13,825 | .0666 / .0704 | .0900 / .1230 | 1,548 / 3,600 | 4 / 3 | .178 / .211 | 1 / 1 |
| Original / GMM | 32.9 / 67.1 | 786 / 1,091 | 263,935 / 1,273,202 | 142,857 / 662,939 | 120,990 / 613,723 | 22,064 / 45,753 | 21,244 / 39,797 | 5,351 / 13,121 | .0637 / .0704 | .0896 / .1195 | 1,345 / 3,250 | 4 / 3 | .167 / .211 | 1 / 1 |
| No gross / K-Means | 44.4 / 55.6 | 854 / 1,092 | 352,414 / 1,410,384 | 191,219 / 725,954 | 163,524 / 680,003 | 23,123 / 48,658 | 23,232 / 41,203 | 6,124 / 13,606 | .0665 / .0712 | .0897 / .1239 | 1,680 / 3,440 | 4 / 3 | .167 / .222 | 1 / 1 |
| No gross / GMM | 3.3 / 96.7 | 673 / 1,016 | 77,824 / 774,672 | 56,776 / 410,344 | 30,600 / 368,303 | 22,107 / 36,553 | 22,822 / 33,201 | 7,701 / 10,553 | .0263 / .0686 | .0143 / .1079 | 1,250 / 2,400 | 30 / 3.5 | .033 / .200 | 0 / 1 |
| No balance SD / K-Means | 44.7 / 55.3 | 753 / 1,272 | 348,682 / 1,466,339 | 188,925 / 753,450 | 162,274 / 702,459 | 23,526 / 48,205 | 23,456 / 40,923 | 6,184 / 13,318 | .0666 / .0709 | .0894 / .1249 | 1,680 / 3,380 | 4 / 3 | .178 / .222 | 1 / 1 |
| No balance SD / GMM | 12.3 / 87.7 | 554 / 1,097 | 130,137 / 900,257 | 76,626 / 466,409 | 53,276 / 432,043 | 19,018 / 39,044 | 19,343 / 35,203 | 5,333 / 11,184 | .0590 / .0691 | .0669 / .1110 | 1,200 / 2,570 | 6 / 3 | .156 / .200 | 1 / 1 |
| Neither / K-Means | 46.7 / 53.3 | 798 / 1,180 | 385,273 / 1,396,205 | 204,628 / 721,382 | 176,135 / 673,324 | 23,641 / 48,040 | 23,960 / 40,106 | 6,445 / 12,993 | .0667 / .0719 | .0897 / .1273 | 1,800 / 3,213 | 4 / 3 | .167 / .222 | 1 / 1 |
| Neither / GMM | 91.9 / 8.1 | 998 / 1,303 | 724,967 / 1,015,659 | 382,073 / 513,723 | 347,304 / 501,936 | 36,471 / 29,400 | 33,239 / 29,865 | 10,225 / 17,113 | .0683 / .0704 | .1083 / .0925 | 2,300 / 2,940 | 3.75 / 4 | .200 / .167 | 1 / 0 |

The no-gross GMM minority is distinctly low-activity: median credit/debit frequencies are 0.0263/0.0143 per day, median inter-arrival time is 30 days, and median recent activity is 0.033 transactions/day. Its 3.3% size makes it a small niche segment rather than a broad partition. With both dominant inputs excluded, GMM separates 73 accounts with higher observed balance volatility and tenure, but only modest differences in credit/debit frequency and inter-arrival regularity; its activity-change median is 0 versus 1 in the larger segment. K-Means retains larger, more balanced groups and similar low/high gross-movement profiles in all four configurations, although assignment agreement with its original result falls to 0.603 when both features are removed.

## Interpretation and recommendation

Balance volatility is not necessary for K-Means to retain its broad two-group structure, while removing gross movement changes it moderately. Removing both changes more of the K-Means assignment but leaves large, seed-stable groups associated with transaction frequency, net flow, and regularity. Gross movement and volatility therefore influence the baseline, but do not solely determine the K-Means split.

GMM’s original split is more dependent on the selected feature space. Without gross movement, it shifts to a highly stable but small low-activity segment; without both variables, it produces a different, stable partition with weaker and less coherent frequency/regularity contrasts. A stable split exists within each reduced space, but it should not be treated as the same underlying segment structure as the original GMM solution.

These sensitivity checks support retaining simple behavioral baselines for further review, not choosing a champion. Examine feature/transform choices and later 1998 retrospective stability before making claims about persistent customer behavior. No personas or business interpretations are assigned.

## Artifacts and checks

Reusable runner: `src/customer_segmentation/models/berka_feature_sensitivity.py`; configuration: `configs/models/berka_feature_sensitivity_v1.yaml`; separate artifacts: `artifacts/berka_feature_sensitivity/v1/asof_1997-12-31/`.

Artifacts include the copied configuration, four train-fitted preprocessor states, 24 fitted candidate models, held-out assignments (21,624 rows), per-seed metrics, segment profiles, full pairwise cross-configuration ARIs, the shared split, and run metadata. Reloaded preprocessing and model artifacts reproduced all 21,624 saved assignments. The original 13-feature assignments exactly match the six baseline predictions. No baseline artifact or experiment ledger was modified.
