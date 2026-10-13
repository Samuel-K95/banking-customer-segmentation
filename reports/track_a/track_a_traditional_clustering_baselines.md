# Track A Traditional Clustering Baselines

**Run date:** 2026-10-11
**Population:** 837,951 validated Track A customers
**Methods:** K-Means and diagonal-covariance Gaussian Mixture Models (GMM)

## Executive summary

Twenty real-data comparison candidates completed successfully across two training samples, five values of k, and two algorithms; all ten GMM fits converged. On the common 3,000-customer holdout, nine-feature K-Means k=2 is exactly stable across the two seeds (ARI 1.0000) and has a balanced 481/2,519 split. That split is predominantly transaction history: one cluster has no single-transaction customers and the other is 99.7% single-transaction customers. The full-population transform was fitted before sampling, so these metrics are transductive rather than independent inductive validation.

The `Avg_Spend_Ratio` tail is a material sensitivity risk. A complementary feature diagnostic is in [`track_a_feature_diagnostics.md`](track_a_feature_diagnostics.md). No champion or customer-facing segment names were selected.

## Objective and feature contract

The experiment compares conventional unsupervised baselines on exactly these nine inputs, in contract order:

```text
AccountBalance, Transaction_Count, Avg_Transaction_Amt, Std_Transaction_Amt,
Total_Spend, Avg_Spend_Ratio, Recency_Days, Weekend_Spend_Ratio,
Night_Spend_Ratio
```

`CustomerID` is an alignment key and is never passed to an estimator. `Age`, `Gender_M`, and `Is_Tier_1_City` remain profile-only. The fitted preprocessing artifact transforms an unscaled interpretable matrix using four `log1p` operations and a StandardScaler; the scaler was fit on all 837,951 customers before the baseline samples were created. No additional scaler or clipping was applied by these estimators.

## Experimental design

| Phase | Training rows | Shared evaluation rows | Seeds | k | Algorithms |
|---|---:|---:|---|---|---|
| Smoke | 10,000 | 3,000 | 42 | 3 | K-Means, GMM |
| Comparison | 50,000 per seed | 3,000 | 42, 2024 | 2, 3, 4, 5, 6 | K-Means, GMM |

Training IDs were sampled deterministically by row position and are disjoint from the shared evaluation sample (evaluation seed 8675309). The same training IDs were used by both algorithms for each seed. The ordered evaluation ID hash is `5496dbb5ee72aed5b66bea49410f00624c724173413606edf4ce23085c5a16d3`.

| Algorithm | Configuration |
|---|---|
| K-Means | `n_init=10`, `max_iter=300`, Lloyd algorithm |
| GMM | Diagonal covariance, `reg_covar=1e-6`, `n_init=1`, `max_iter=200`, `tol=0.001` |

Silhouette uses a deterministic sample of up to 3,000 holdout rows (`silhouette_seed=13579`); Davies–Bouldin and Calinski–Harabasz use all 3,000 rows. GMM AIC/BIC use the same 3,000 evaluation rows. No score is computed over the full 837,951-customer population.

## Results

### Smoke

Membership values are cluster count (evaluation percentage).

| Algorithm | Silhouette | Davies–Bouldin | Calinski–Harabasz | Evaluation memberships | Minimum segment | Inertia (train) | GMM BIC / AIC (eval) | Iterations / convergence | Fit time | Peak working set |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| K-Means, k=3, seed 42 | 0.2212 | 1.6084 | 895.54 | 0: 1,319 (43.97%); 1: 476 (15.87%); 2: 1,205 (40.17%) | 15.87% | 51,777.38 | — | 13 / — | 0.410 s | 162.6 MiB |
| GMM, k=3, seed 42 | 0.2630 | 1.6323 | 593.46 | 0: 489 (16.30%); 1: 2,319 (77.30%); 2: 192 (6.40%) | 6.40% | — | -28,873.84 / -29,210.20 | 12 / converged | 0.200 s | 164.9 MiB |

### 50,000-row comparison

Membership counts are on the common 3,000-row evaluation set. Inertia is a training metric; GMM AIC/BIC are evaluation metrics. A missing cluster label means that component received no evaluation customer, so observed label coverage is shown explicitly.

| Algorithm | k | Seed | Silhouette | DB | CH | Evaluation segment counts (n; %) | Minimum segment | K-Means inertia (train) | GMM BIC / AIC (eval) | GMM converged | Fit seconds | Peak WS MiB | Cross-seed ARI |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|---:|---:|---:|
| K-Means | 2 | 42 | 0.3327 | 1.1637 | 922.77 | 0: 481 (16.03%); 1: 2,519 (83.97%) | 16.03% | 360,800.83 | — | — | 1.001 | 180.3 | 1.0000 |
| GMM | 2 | 42 | 0.2946 | 1.6708 | 661.49 | 0: 612 (20.40%); 1: 2,388 (79.60%) | 20.40% | — | -21,644.32 / -21,866.56 | Yes | 1.348 | 176.8 | 0.8129 |
| K-Means | 3 | 42 | 0.2215 | 1.6064 | 895.82 | 0: 475 (15.83%); 1: 1,346 (44.87%); 2: 1,179 (39.30%) | 15.83% | 304,614.95 | — | — | 0.767 | 187.7 | 0.9667 |
| GMM | 3 | 42 | 0.2599 | 1.6333 | 587.87 | 0: 489 (16.30%); 1: 2,325 (77.50%); 2: 186 (6.20%) | 6.20% | — | -29,086.10 / -29,422.45 | Yes | 1.161 | 188.2 | 0.9976 |
| K-Means | 4 | 42 | 0.2216 | 1.6059 | 895.71 | 0: 1,167 (38.90%); 1: 1,357 (45.23%); 2: 476 (15.87%) | 15.87% | 258,248.08 | — | — | 0.811 | 191.1 | 0.5833 |
| GMM | 4 | 42 | 0.2474 | 1.6562 | 589.41 | 0: 493 (16.43%); 1: 2,297 (76.57%); 2: 210 (7.00%) | 7.00% | — | -29,555.62 / -30,006.10 | Yes | 2.146 | 194.0 | 0.5187 |
| K-Means | 5 | 42 | 0.2391 | 1.4595 | 819.91 | 0: 1,078 (35.93%); 1: 839 (27.97%); 2: 475 (15.83%); 3: 608 (20.27%) | 15.83% | 226,186.09 | — | — | 2.088 | 193.9 | 0.6347 |
| GMM | 5 | 42 | 0.2340 | 1.6109 | 575.30 | 0: 521 (17.37%); 1: 1,553 (51.77%); 2: 785 (26.17%); 4: 141 (4.70%) | 4.70% | — | -60,493.93 / -61,058.53 | Yes | 2.901 | 198.1 | 0.5832 |
| K-Means | 6 | 42 | 0.2554 | 1.3052 | 788.11 | 0: 594 (19.80%); 1: 611 (20.37%); 2: 476 (15.87%); 3: 812 (27.07%); 4: 507 (16.90%) | 15.87% | 202,186.82 | — | — | 4.213 | 197.2 | 0.8420 |
| GMM | 6 | 42 | 0.2369 | 1.6613 | 499.92 | 0: 493 (16.43%); 1: 1,553 (51.77%); 2: 69 (2.30%); 4: 141 (4.70%); 5: 744 (24.80%) | 2.30% | — | -63,519.46 / -64,198.18 | Yes | 1.848 | 203.9 | 0.2725 |
| K-Means | 2 | 2024 | 0.3327 | 1.1637 | 922.77 | 0: 481 (16.03%); 1: 2,519 (83.97%) | 16.03% | 338,417.52 | — | — | 1.041 | 196.1 | 1.0000 |
| GMM | 2 | 2024 | 0.3308 | 1.1771 | 920.47 | 0: 489 (16.30%); 1: 2,511 (83.70%) | 16.30% | — | 6,268.92 / 6,046.68 | Yes | 0.408 | 192.7 | 0.8129 |
| K-Means | 3 | 2024 | 0.2216 | 1.6072 | 895.75 | 0: 1,180 (39.33%); 1: 1,343 (44.77%); 2: 477 (15.90%) | 15.90% | 282,390.40 | — | — | 1.114 | 196.5 | 0.9667 |
| GMM | 3 | 2024 | 0.2614 | 1.6331 | 589.91 | 0: 489 (16.30%); 1: 2,323 (77.43%); 2: 188 (6.27%) | 6.27% | — | -23,749.89 / -24,086.25 | Yes | 1.363 | 197.3 | 0.9976 |
| K-Means | 4 | 2024 | 0.2392 | 1.4582 | 820.19 | 0: 851 (28.37%); 1: 475 (15.83%); 2: 1,060 (35.33%); 3: 614 (20.47%) | 15.83% | 250,131.39 | — | — | 1.130 | 198.3 | 0.5833 |
| GMM | 4 | 2024 | 0.2394 | 1.6167 | 557.92 | 0: 516 (17.20%); 1: 1,698 (56.60%); 2: 661 (22.03%); 3: 125 (4.17%) | 4.17% | — | -59,840.84 / -60,291.31 | Yes | 2.538 | 199.6 | 0.5187 |
| K-Means | 5 | 2024 | 0.2563 | 1.3066 | 788.30 | 0: 593 (19.77%); 1: 828 (27.60%); 2: 595 (19.83%); 3: 477 (15.90%); 4: 507 (16.90%) | 15.90% | 226,394.77 | — | — | 2.399 | 197.1 | 0.6347 |
| GMM | 5 | 2024 | 0.2654 | 1.4637 | 605.34 | 0: 489 (16.30%); 1: 1,237 (41.23%); 2: 65 (2.17%); 3: 586 (19.53%); 4: 623 (20.77%) | 2.17% | — | -60,315.99 / -60,880.58 | Yes | 1.518 | 202.6 | 0.5832 |
| K-Means | 6 | 2024 | 0.2585 | 1.3137 | 778.64 | 0: 699 (23.30%); 1: 476 (15.87%); 2: 586 (19.53%); 3: 406 (13.53%); 5: 833 (27.77%) | 13.53% | 201,889.52 | — | — | 2.278 | 197.8 | 0.8420 |
| GMM | 6 | 2024 | 0.2280 | 1.5729 | 564.64 | 0: 489 (16.30%); 1: 750 (25.00%); 2: 65 (2.17%); 3: 123 (4.10%); 4: 623 (20.77%); 5: 950 (31.67%) | 2.17% | — | -59,222.82 / -59,901.54 | Yes | 2.115 | 204.7 | 0.2725 |

The k=2 K-Means split is largely single-transaction versus repeat history: the smaller cluster has 0% single-transaction customers while the larger is 99.7–100% singletons, depending on seed. K=3 separates singleton spend/balance patterns while preserving a repeat-heavy component. GMM k=3 contains a small singleton-only, high-ratio component. These are associations, not customer personas.

## Runtime, artifacts, and limitations

K-Means required 16.842 seconds total across ten fits (0.767–4.213 seconds each); GMM required 17.348 seconds (0.408–2.901 seconds each). Sampled peak process working set ranged 176.8–204.7 MiB. These are absolute process measurements, not estimator-only allocations.

Artifacts include the baseline configuration, feature contract, append-only comparison board, model/assignment files, and run metadata under `artifacts/`. The board retains its original schema; each candidate records input features, seed, metrics, sample hashes, and artifact paths.

The historical full-population scaler was fit before the train/evaluation split, so these metrics are transductive and may be optimistic relative to future unseen customers. The data contains 696,449 single-transaction customers (83.1133% of the validated population); the k=2 result predominantly partitions observation depth rather than established behavioral types. `Avg_Spend_Ratio` can dominate distances and GMM likelihoods. These results are exploratory baselines, not a champion or an approved banking decision rule.
