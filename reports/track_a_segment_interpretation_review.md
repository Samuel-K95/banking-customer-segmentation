# Track A Segment Interpretation Review

**Decision status:** three candidate families retained for review; no champion or segment names assigned.
**Population:** 2+ transactions, 141,502 of 837,951 validated customers (16.8867%); 87.1% of this cohort has exactly two transactions.
**Evaluation cohort:** shared 10,000 CustomerIDs, ordered SHA-256 `1430463f2074d93caea16c49f07cc3209d399bad51bc8fba410ceff33112ffd4`.

## Candidate performance and segment coverage

All candidates use 50,000 training rows per seed and the same 10,000-customer holdout. K-Means candidates use unclipped nine-feature train-fitted preprocessing. GMM candidates clip `Avg_Spend_Ratio` at each training sample's q0.5/q99.5 before fitting their scaler; the GMM geometry is distinct. S/DB/CH denote silhouette / Davies–Bouldin / Calinski–Harabasz. Cross-seed stability is the mean/min pairwise ARI across seeds 42, 2024, and 123, always on the same ordered evaluation IDs.

| Candidate family | Seed | k | S / DB / CH | Evaluation segment sizes | Minimum segment | Cross-seed ARI mean / minimum |
|---|---:|---:|---|---|---:|---:|
| K-Means, nine features | 42 | 2 | 0.19271 / 1.78793 / 2,695.55 | 5,094 (50.94%) / 4,906 (49.06%) | 49.06% | 0.9788 / 0.9698 |
| K-Means, nine features | 2024 | 2 | 0.19237 / 1.79085 / 2,593.15 | 4,921 (49.21%) / 5,079 (50.79%) | 49.21% | 0.9788 / 0.9698 |
| K-Means, nine features | 123 | 2 | 0.19278 / 1.78396 / 2,600.04 | 5,018 (50.18%) / 4,982 (49.82%) | 49.82% | 0.9788 / 0.9698 |
| K-Means, nine features | 42 | 3 | 0.20920 / 1.61278 / 2,165.77 | 4,260 (42.60%) / 4,507 (45.07%) / 1,233 (12.33%) | 12.33% | 0.9748 / 0.9634 |
| K-Means, nine features | 2024 | 3 | 0.21013 / 1.61189 / 2,084.44 | 4,254 (42.54%) / 4,506 (45.06%) / 1,240 (12.40%) | 12.40% | 0.9748 / 0.9634 |
| K-Means, nine features | 123 | 3 | 0.20875 / 1.61727 / 2,062.13 | 1,231 (12.31%) / 4,424 (44.24%) / 4,345 (43.45%) | 12.31% | 0.9748 / 0.9634 |
| Diagonal GMM, ratio-clipped | 42 | 2 | 0.16028 / 3.10454 / 707.93 | 2,287 (22.87%) / 7,713 (77.13%) | 22.87% | 0.9904 / 0.9856 |
| Diagonal GMM, ratio-clipped | 2024 | 2 | 0.16177 / 3.09810 / 714.84 | 2,302 (23.02%) / 7,698 (76.98%) | 23.02% | 0.9904 / 0.9856 |
| Diagonal GMM, ratio-clipped | 123 | 2 | 0.15804 / 3.14563 / 695.50 | 2,320 (23.20%) / 7,680 (76.80%) | 23.20% | 0.9904 / 0.9856 |

Exact candidate IDs and the corresponding models, assignments, metadata, preprocessors, and profile paths are in the verified `candidate_inventory.csv` under `artifacts/diagnostics/track_a/model_readiness_20261011/`.

## Side-by-side customer profiles

Profiles use raw interpretable evaluation features joined in exact CustomerID order. Values show P10 / P50 / P90; balance and spend are INR, recency is days before the fixed 2016-11-01 observation date. The machine-readable profile file also contains P25/P75 and every feature quantile for all 21 candidate-seed-cluster combinations.

| Candidate family | Seed | k | Cluster | Customers / % | Transactions P10/P50/P90 | Balance P10/P50/P90 | Median avg transaction | Total spend P10/P50/P90 | Ratio P50/P90 | Recency P10/P50/P90 |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|
| K-Means, nine features | 42 | 2 | 0 | 5,094 / 50.94% | 2/2/2 | 3,487/19,738/130,329 | 325 | 220/680/1,357 | 0.038/2.093 | 44/58/77 |
| K-Means, nine features | 42 | 2 | 1 | 4,906 / 49.06% | 2/2/3 | 8,278/48,291/356,749 | 1,475 | 1,629/3,179/11,852 | 0.083/2.731 | 44/57/77 |
| K-Means, nine features | 42 | 3 | 0 | 4,260 / 42.60% | 2/2/2 | 7,531/47,489/353,347 | 1,463 | 1,499/2,926/11,195 | 0.078/2.675 | 44/58/78 |
| K-Means, nine features | 42 | 3 | 1 | 4,507 / 45.07% | 2/2/2 | 3,231/18,752/123,147 | 310 | 205/620/1,263 | 0.036/1.901 | 44/58/77 |
| K-Means, nine features | 42 | 3 | 2 | 1,233 / 12.33% | 3/3/4 | 8,763/38,189/246,938 | 807 | 795/2,563/9,638 | 0.086/3.174 | 40/54/70 |
| Diagonal GMM, ratio-clipped | 42 | 2 | 0 | 2,287 / 22.87% | 2/3/3 | 1,884/22,011/177,104 | 800 | 541/2,198/9,825 | 1.751/30.204 | 44/55/75 |
| Diagonal GMM, ratio-clipped | 42 | 2 | 1 | 7,713 / 77.13% | 2/2/2 | 6,059/32,676/256,519 | 650 | 296/1,299/5,981 | 0.041/0.426 | 44/58/78 |
| K-Means, nine features | 2024 | 2 | 0 | 4,921 / 49.21% | 2/2/3 | 8,325/48,384/356,597 | 1,470 | 1,624/3,168/11,846 | 0.082/2.748 | 44/57/77 |
| K-Means, nine features | 2024 | 2 | 1 | 5,079 / 50.79% | 2/2/2 | 3,476/19,608/127,988 | 324 | 220/680/1,351 | 0.038/2.089 | 44/58/77 |
| K-Means, nine features | 2024 | 3 | 0 | 4,254 / 42.54% | 2/2/2 | 7,534/47,590/353,977 | 1,468 | 1,500/2,936/11,232 | 0.078/2.693 | 44/58/78 |
| K-Means, nine features | 2024 | 3 | 1 | 4,506 / 45.06% | 2/2/2 | 3,231/18,759/123,410 | 310 | 205/621/1,266 | 0.036/1.878 | 44/58/78 |
| K-Means, nine features | 2024 | 3 | 2 | 1,240 / 12.40% | 3/3/4 | 8,758/37,790/247,910 | 797 | 772/2,538/9,427 | 0.085/3.186 | 40/54/70 |
| Diagonal GMM, ratio-clipped | 2024 | 2 | 0 | 2,302 / 23.02% | 2/3/3 | 1,875/21,927/176,818 | 803 | 539/2,200/9,833 | 1.728/29.717 | 44/55/75 |
| Diagonal GMM, ratio-clipped | 2024 | 2 | 1 | 7,698 / 76.98% | 2/2/2 | 6,065/32,739/256,624 | 649 | 296/1,298/5,956 | 0.041/0.421 | 44/58/78 |
| K-Means, nine features | 123 | 2 | 0 | 5,018 / 50.18% | 2/2/2 | 3,473/19,454/125,282 | 321 | 219/670/1,333 | 0.038/2.059 | 44/58/77 |
| K-Means, nine features | 123 | 2 | 1 | 4,982 / 49.82% | 2/2/3 | 8,217/48,372/355,043 | 1,452 | 1,601/3,133/11,698 | 0.082/2.770 | 44/57/77 |
| K-Means, nine features | 123 | 3 | 0 | 1,231 / 12.31% | 3/3/4 | 8,760/38,405/247,570 | 808 | 807/2,565/9,650 | 0.086/3.186 | 40/54/70 |
| K-Means, nine features | 123 | 3 | 1 | 4,424 / 44.24% | 2/2/2 | 3,170/18,588/122,738 | 304 | 203/608/1,229 | 0.036/1.960 | 44/58/77 |
| K-Means, nine features | 123 | 3 | 2 | 4,345 / 43.45% | 2/2/2 | 7,580/47,099/343,709 | 1,439 | 1,461/2,878/11,009 | 0.077/2.645 | 44/58/78 |
| Diagonal GMM, ratio-clipped | 123 | 2 | 0 | 2,320 / 23.20% | 2/3/3 | 1,853/21,750/176,197 | 802 | 531/2,182/9,829 | 1.689/29.578 | 44/55/75 |
| Diagonal GMM, ratio-clipped | 123 | 2 | 1 | 7,680 / 76.80% | 2/2/2 | 6,115/32,858/256,722 | 649 | 296/1,298/5,953 | 0.041/0.410 | 44/58/78 |

The unclipped k=2 K-Means clusters consistently have lower vs. higher balance/spend medians, while their transaction-count medians remain 2 and recency medians are similar. At k=3, one cluster has transaction-count median 3 and a 12.3% share; the other two remain mostly exactly-two-transaction customers. The clipped GMM’s ~23% high-ratio cluster has median `Avg_Spend_Ratio` around 1.7 and P90 near 30, compared with median about 0.04 and P90 below 0.43 for its other component. Review low-balance denominator sensitivity before describing that component.

## Membership overlap

ARI compares candidate labelings on the exact same ordered 10,000 evaluation IDs; the underlying contingency tables are in `candidate_segment_overlap.csv`.

| Candidate pair | Seed 42 | Seed 2024 | Seed 123 |
|---|---:|---:|---:|
| K-Means k=2 vs. K-Means k=3 | 0.6891 | 0.6955 | 0.6834 |
| K-Means k=2 vs. clipped GMM k=2 | 0.0170 | 0.0164 | 0.0150 |
| K-Means k=3 vs. clipped GMM k=2 | 0.2259 | 0.2263 | 0.2224 |

The GMM alternative is not a relabeling of K-Means. Labels are candidate-specific integers; do not carry cluster IDs across seeds or model versions without a separately validated mapping.

## Demographic representation and fairness review

The 837,951-row validated population has 72.41% `Gender_M=1`, 27.59% `Gender_M=0`, and 35.18% Tier 1 city. Age uses the 2016 convention; 55.18% of the population is age 25–34. Candidate-segment representation differs from the shared evaluation base by at most 3.47 percentage points for age bands, 2.59 points for `Gender_M`, and 0.86 points for Tier 1 city. Nine age-band/segment cells have fewer than 30 customers.

There are no missing values in these fields among validated customer rows because the source filtering/age-validity policy excludes affected records. At transaction-row grain, raw source counts include 3,397 missing DOB, 1,100 missing gender, and 151 missing location; 62,373 rows have missing, unparsable, or out-of-range age derivation, overlapping DOB missingness. The full tables are `demographic_population_distribution.csv`, `demographic_missingness.csv`, `raw_source_demographic_missingness.csv`, and `finalist_demographic_distribution.csv` in the versioned audit directory.

`Gender_M=0` is a non-M indicator and should not automatically be treated as a verified female category. City tier is a geographic proxy; transaction depth, balance, spend, and recency may also proxy access or socioeconomic factors. These representation comparisons are descriptive, not proof of discrimination. Avoid interpreting small cells, and do not use these exploratory clusters alone for adverse banking actions.

## Segment coverage and use limits

The 2+ eligible population is 16.8867% of the validated population; 696,449 customers (83.1133%) have only one transaction and are not modeled by these candidates. Among eligible customers, 87.1% have exactly two transactions. A two-row sample standard deviation has one degree of freedom; behavioral ratios are quantized. Interpret labels as exploratory groupings in this population, not stable customer personas or a complete customer segmentation.

The raw ratio-clipped GMM used training q0.5/q99.5 bounds and actually clipped 101/106/101 held-out values for seeds 42/2024/123. Its metric geometry differs from the unclipped K-Means candidates. GMM maximum-component posterior values are uncalibrated; K-Means provides no confidence score.

## Artifacts and serving handoff

The canonical candidate inventory and hashes are in `artifacts/diagnostics/track_a/model_readiness_20261011/candidate_inventory.csv`. Full profiles, candidate overlap, cross-seed stability, and demographic tables are alongside it. Saved model/assignment/metadata files are identified exactly there.

Downstream consumers should use the saved candidate bundle, carry its model version and candidate-specific cluster-profile map, preserve CustomerID order, route fewer-than-two-transaction rows as `insufficient_observed_history`, and keep demographics profile-only. The saved inference pipelines use a fixed 2016-11-01 as-of date; they reject transactions later than that date. A current-date deployment needs a newly defined as-of policy and compatible train-fitted artifacts. The serving interface and invalid-input behavior are detailed in `track_a_model_readiness.md`.
