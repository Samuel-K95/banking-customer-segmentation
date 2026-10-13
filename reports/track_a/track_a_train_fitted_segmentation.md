# Track A Train-Fitted Repeat-Customer Segmentation

**Run date:** 2026-10-11
**Population base:** 837,951 validated customers
**Decision:** no champion selected; no API or neural-clustering integration.

## Executive summary

The raw transaction aggregates were rebuilt for a train-fitted experiment with a fixed observation date of 2016-11-01. It trained 96 candidates: K-Means and diagonal GMM, k=2/3/4/5, three seeds, two preprocessing variants, and 2+ and 3+ transaction populations. All candidates completed and all 48 GMM fits converged.

The 2+ population contains 141,502 customers (16.8867% of validated customers); the 3+ population contains 18,308 (2.1849%). The 2+ k=2 K-Means result is balanced and stable across its three seeds; k=3 is also stable and has a minimum segment near 12.3%. Higher k can produce very small clusters. These partitions remain exploratory: 87.1% of eligible 2+ customers have exactly two transactions, and a segment can reflect observation depth rather than a stable behavior persona.

The ratio-clipped variant changes feature geometry and can substantially alter higher-k membership. A later audit corrected the held-out clipping counts: held-out ratio values were clipped in every seed. The models and historical metrics are preserved; only the old zero-count ledger/report field is superseded by the corrected audit.

## Population and observation depth

| Transactions per customer | Customers | Percent of validated population |
|---:|---:|---:|
| 1 | 696,449 | 83.1133% |
| 2 | 123,194 | 14.7018% |
| 3 | 16,492 | 1.9681% |
| 4 | 1,661 | 0.1982% |
| 5 | 141 | 0.0168% |
| 6 | 14 | 0.0017% |

All single-transaction customers have `Std_Transaction_Amt == 0` because sample standard deviation is undefined for one observation and the feature builder fills it with zero. Their mean transaction amount equals total spend. This is a mathematical consequence of history length, not low financial value or inactivity.

| Eligibility | Customers | Coverage | Treatment |
|---|---:|---:|---|
| 1+ transactions | 837,951 | 100.0000% | Validated population |
| 2+ transactions | 141,502 | 16.8867% | Primary repeat-customer experiment |
| 3+ transactions | 18,308 | 2.1849% | Reduced-size sensitivity experiment |
| Exactly one | 696,449 | 83.1133% | `insufficient_observed_history` for rich behavior segments |
| 1–2 transactions | 819,643 | 97.8151% | Ineligible for the 3+ population |

## Raw aggregation and temporal semantics

The raw source has 1,048,567 rows; its complete-case rule retains 1,041,614. Aggregates include mean account balance, transaction count, mean/sample SD/total transaction amount, mean transaction-level `amount / (balance + 1)`, weekend/night ratios, and last transaction date. A one-row sample SD is filled with zero. The initial feature matrix used the maximum transaction date over the full input to create recency, so its scaler and recency statistics contain full-population information.

Train-fitted experiments rebuild aggregate features from the raw source for sampled IDs. The source maximum date is 2016-10-21; the fixed observation date is 2016-11-01. Thus `Recency_Days = 2016-11-01 - Last_Transaction`, with no future transaction dates. For the 113,948 unique sampled IDs, transaction counts exactly match the existing interpretable data. Maximum absolute aggregate differences were `7.5e-9` for balance, `2.9e-11` for total spend, `1.5e-11` for mean/SD transaction amounts, `9.1e-13` for the ratio, and zero for recency; weekend/night ratios differed by at most `5.6e-17`.

Historical baseline and representation-learning results used a StandardScaler fitted on all 837,951 customers before sampling. Those results are transductive with respect to preprocessing. The train-fitted experiments instead created population manifests before learned transforms and fit each scaler/clipping threshold only on its corresponding training IDs. These are different evaluation designs; their scores are not interchangeable.

## Sampling design and configuration

| Population | Training rows per seed | Shared evaluation rows | Evaluation ID SHA-256 | Seeds | k values |
|---|---:|---:|---|---|---|
| 2+ | 50,000 | 10,000 | `1430463f2074d93caea16c49f07cc3209d399bad51bc8fba410ceff33112ffd4` | 42, 2024, 123 | 2, 3, 4, 5 |
| 3+ | 8,000 | 3,000 | `ee63b6d0abcb5d47ed46b35474a6e0b488d17b0a9262bb0a1c78e053d755b67e` | 42, 2024, 123 | 2, 3, 4, 5 |

The 3+ population cannot support the 50,000/10,000 design; it is a separate reduced-size sensitivity analysis. Training and evaluation IDs are disjoint. The shared evaluation set was repeatedly used for candidate comparison and is therefore an exploratory model-selection set, not an independent champion validation set.

The two preprocessing variants are:

| Variant | Deterministic transform | Learned transform |
|---|---|---|
| `nine_feature_trainfit` | `log1p` for balance, average transaction amount, transaction SD, and total spend; recency from the fixed date | StandardScaler fitted on training IDs only |
| `ratio_clip_trainfit` | Same transforms; clip raw `Avg_Spend_Ratio` at training q0.5/q99.5 | StandardScaler fitted after clipping on training IDs only |

Both variants use the nine approved features in contract order. CustomerID is an alignment key; age, gender, and city tier are not inputs.

K-Means uses `n_init=10`, `max_iter=300`, and Lloyd's algorithm. Diagonal GMM uses `reg_covar=1e-6`, `n_init=1`, `max_iter=200`, and `tol=0.001`. Silhouette uses a deterministic sample of at most 3,000 evaluation rows; DB/CH and GMM AIC/BIC use all evaluation rows.

## Ratio-clipping audit correction

Cutoffs are in raw ratio units, learned on training rows. Training changes are exactly 1% of the ratio feature (500/50,000 in 2+; 80/8,000 in 3+).

| Population | Seed | q0.5 lower | q99.5 upper | Train below / above | Evaluation below / above | Evaluation values changed |
|---|---:|---:|---:|---:|---:|---:|
| 2+ | 42 | 0.000877 | 341.743548 | 250 / 250 | 53 / 48 | 101 |
| 2+ | 2024 | 0.000898 | 325.151509 | 250 / 250 | 57 / 49 | 106 |
| 2+ | 123 | 0.000856 | 328.054935 | 250 / 250 | 52 / 49 | 101 |
| 3+ | 42 | 0.002071 | 308.649955 | 40 / 40 | 8 / 19 | 27 |
| 3+ | 2024 | 0.002315 | 333.341243 | 40 / 40 | 11 / 19 | 30 |
| 3+ | 123 | 0.002055 | 354.729830 | 40 / 40 | 8 / 17 | 25 |

The previous report and comparison-board parameter fields stated zero evaluation values changed. The per-candidate JSON metadata already contains the correct counts; the discrepancy was a ledger handoff bug. The corrected `candidate_inventory.csv` and neighboring ratio audit files record both the historical ledger field and the verified JSON/transform count. The original ledger rows, models, assignments, and metrics have not been rewritten. Model evaluation applied the saved clipping transforms, so the reported metrics remain the original fitted-model results; their old interpretation that no held-out feature values were clipped is superseded.

Clipping changed 0.13%–45.23% of evaluation memberships in the cross-geometry K-Means comparison depending on population, seed, and k; GMM differences reached 44.60%. These compare models refit under different geometries, not a causal effect of held-out clipping alone. Exact counts, ARIs, and confusion/contingency matrices are in the ratio-clipping audit package.

## Candidate results

### 2+ customers; 10,000 evaluation rows

| Preprocessing | Model | k | Silhouette mean [range] | DB mean | CH mean | Minimum segment % range | Cross-seed ARI mean / min | GMM converged |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| Nine-feature train-fit | K-Means | 2 | 0.1926 [0.1924–0.1928] | 1.7876 | 2,629.6 | 49.06–49.82 | 0.9788 / 0.9698 | — |
| Nine-feature train-fit | K-Means | 3 | 0.2094 [0.2088–0.2101] | 1.6140 | 2,104.1 | 12.31–12.40 | 0.9748 / 0.9634 | — |
| Nine-feature train-fit | K-Means | 4 | 0.1741 [0.1562–0.2093] | 1.6179 | 1,742.9 | 0.01–11.85 | 0.5781 / 0.3850 | — |
| Nine-feature train-fit | K-Means | 5 | 0.1566 [0.1565–0.1568] | 1.5189 | 1,639.3 | 0.01–0.05 | 0.9480 / 0.9259 | — |
| Nine-feature train-fit | Diagonal GMM | 2 | 0.1381 [0.1367–0.1389] | 5.0621 | 179.4 | 12.14–13.53 | 0.9473 / 0.9211 | Yes |
| Nine-feature train-fit | Diagonal GMM | 3 | 0.1409 [0.1353–0.1464] | 3.6035 | 534.4 | 4.55–9.42 | 0.9645 / 0.9502 | Yes |
| Nine-feature train-fit | Diagonal GMM | 4 | 0.1278 [0.1246–0.1340] | 3.9698 | 494.5 | 2.73–3.03 | 0.9649 / 0.9520 | Yes |
| Nine-feature train-fit | Diagonal GMM | 5 | 0.1224 [0.1130–0.1290] | 4.0363 | 792.1 | 2.60–2.84 | 0.5119 / 0.2891 | Yes |
| Ratio clip train-fit | K-Means | 2 | 0.1890 [0.1888–0.1892] | 1.8270 | 2,521.1 | 48.95–49.77 | 0.9662 / 0.9495 | — |
| Ratio clip train-fit | K-Means | 3 | 0.2050 [0.2045–0.2054] | 1.6590 | 2,005.8 | 12.31–12.39 | 0.9721 / 0.9608 | — |
| Ratio clip train-fit | K-Means | 4 | 0.1837 [0.1286–0.2113] | 1.4799 | 1,899.5 | 0.75–0.82 | 0.4915 / 0.2488 | — |
| Ratio clip train-fit | K-Means | 5 | 0.1617 [0.1581–0.1682] | 1.5530 | 1,853.2 | 0.75–0.81 | 0.6023 / 0.4198 | — |
| Ratio clip train-fit | Diagonal GMM | 2 | 0.1600 [0.1580–0.1618] | 3.1161 | 706.1 | 22.87–23.20 | 0.9904 / 0.9856 | Yes |
| Ratio clip train-fit | Diagonal GMM | 3 | 0.1457 [0.1355–0.1649] | 3.4729 | 715.3 | 10.01–12.74 | 0.8097 / 0.7152 | Yes |
| Ratio clip train-fit | Diagonal GMM | 4 | 0.1015 [0.0918–0.1180] | 3.4618 | 605.1 | 4.33–14.28 | 0.4893 / 0.3687 | Yes |
| Ratio clip train-fit | Diagonal GMM | 5 | 0.0965 [0.0925–0.0991] | 3.2723 | 729.8 | 4.17–4.86 | 0.5370 / 0.3086 | Yes |

### 3+ customers; reduced 8,000/3,000 design

| Preprocessing | Model | k | Silhouette mean [range] | DB mean | CH mean | Minimum segment % range | Cross-seed ARI mean / min | GMM converged |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| Nine-feature train-fit | K-Means | 2 | 0.2034 [0.2028–0.2039] | 1.7400 | 817.5 | 42.93–44.47 | 0.9596 / 0.9396 | — |
| Nine-feature train-fit | K-Means | 3 | 0.1920 [0.1366–0.2202] | 1.7038 | 631.7 | 10.07–17.67 | 0.5005 / 0.2612 | — |
| Nine-feature train-fit | K-Means | 4 | 0.2005 [0.1628–0.2200] | 1.5170 | 602.6 | 0.13–10.07 | 0.5804 / 0.3861 | — |
| Nine-feature train-fit | K-Means | 5 | 0.1617 [0.1605–0.1627] | 1.5877 | 550.9 | 0.07–9.63 | 0.8937 / 0.8577 | — |
| Nine-feature train-fit | Diagonal GMM | 2 | 0.0975 [0.0935–0.1017] | 6.4652 | 39.0 | 14.50–18.50 | 0.8712 / 0.8080 | Yes |
| Nine-feature train-fit | Diagonal GMM | 3 | 0.1362 [0.1348–0.1389] | 3.1698 | 119.7 | 3.60–5.13 | 0.9608 / 0.9418 | Yes |
| Nine-feature train-fit | Diagonal GMM | 4 | 0.0952 [0.0809–0.1150] | 4.5069 | 153.2 | 1.63–3.33 | 0.8114 / 0.7344 | Yes |
| Nine-feature train-fit | Diagonal GMM | 5 | 0.1137 [0.0962–0.1296] | 3.5365 | 218.1 | 1.73–3.40 | 0.5091 / 0.4494 | Yes |
| Ratio clip train-fit | K-Means | 2 | 0.2006 [0.1993–0.2013] | 1.7831 | 769.4 | 42.93–43.97 | 0.9691 / 0.9591 | — |
| Ratio clip train-fit | K-Means | 3 | 0.2121 [0.2076–0.2158] | 1.4972 | 633.8 | 1.13–10.10 | 0.8263 / 0.7652 | — |
| Ratio clip train-fit | K-Means | 4 | 0.1738 [0.1406–0.2222] | 1.5806 | 586.4 | 1.07–9.53 | 0.4421 / 0.2549 | — |
| Ratio clip train-fit | K-Means | 5 | 0.1664 [0.1659–0.1667] | 1.4986 | 612.3 | 1.07–1.13 | 0.9046 / 0.8680 | — |
| Ratio clip train-fit | Diagonal GMM | 2 | 0.1821 [0.1793–0.1853] | 3.2297 | 192.5 | 20.20–21.03 | 0.9750 / 0.9625 | Yes |
| Ratio clip train-fit | Diagonal GMM | 3 | 0.1625 [0.1117–0.1899] | 3.4802 | 193.1 | 8.17–10.37 | 0.7612 / 0.6439 | Yes |
| Ratio clip train-fit | Diagonal GMM | 4 | 0.1157 [0.0754–0.1672] | 3.1181 | 228.9 | 3.17–10.37 | 0.4428 / 0.3318 | Yes |
| Ratio clip train-fit | Diagonal GMM | 5 | 0.0759 [0.0721–0.0782] | 3.4366 | 213.4 | 2.23–4.53 | 0.4552 / 0.2624 | Yes |

The 3+ design is a separate, reduced-size sensitivity population and is not a substitute for the 2+ evaluation. Detailed candidate-level metrics, segment profiles, posterior summaries, and model/assignment paths remain in the comparison ledger and per-run metadata.

## Selected segment profiles

| Population / candidate | Cluster | Count / % | Median transactions | Median balance | Median average transaction | Median SD | Median total spend | Median ratio | Median recency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2+, nine-feature K-Means k=2, seed 42 | 0 | 5,094 / 50.94% | 2 | 19,738 | 324.60 | 183.85 | 680.26 | 0.0381 | 58 |
| 2+, nine-feature K-Means k=2, seed 42 | 1 | 4,906 / 49.06% | 2 | 48,291 | 1,475.25 | 1,348.81 | 3,178.50 | 0.0829 | 57 |
| 3+, nine-feature K-Means k=2, seed 42 | 0 | 1,334 / 44.47% | 3 | 58,034 | 1,656.33 | 1,774.57 | 5,130.38 | 0.1118 | 54 |
| 3+, nine-feature K-Means k=2, seed 42 | 1 | 1,666 / 55.53% | 3 | 25,408 | 433.97 | 351.48 | 1,351.02 | 0.0563 | 53 |
| 2+, ratio-clipped GMM k=2, seed 42 | 0 | 2,287 / 22.87% | 3 | 22,011 | 800.00 | 694.30 | 2,198.00 | 1.7512 | 55 |
| 2+, ratio-clipped GMM k=2, seed 42 | 1 | 7,713 / 77.13% | 2 | 32,676 | 649.50 | 459.62 | 1,299.00 | 0.0413 | 58 |

The k=2 K-Means partition separates spending/balance levels while remaining close to a transaction-history split. The 3+ sample is too small to generalize to the wider 2+ population. The clipped GMM's high-ratio component may be affected by low balances and requires denominator sensitivity review. No segment names are assigned. GMM posterior maxima are uncalibrated.

## Repeat-only sensitivity and comparison context

| Seed | k | Training customers | Evaluation customers | Silhouette, repeat-only space | DB | CH | Evaluation sizes | Cross-seed ARI |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 42 | 2 | 8,390 | 489 | 0.1780 | 1.8716 | 116.93 | 238 / 251 | 0.9434 |
| 2024 | 2 | 8,424 | 489 | 0.1730 | 1.9672 | 39.46 | 231 / 258 | 0.9434 |
| 42 | 3 | 8,390 | 489 | 0.1769 | 1.8695 | 116.86 | 246 / 243 | 0.7586 |
| 2024 | 3 | 8,424 | 489 | 0.1917 | 1.7850 | 28.80 | 233 / 199 / 57 | 0.7586 |

This repeat-only sensitivity uses the earlier globally transformed customer matrix filtered to repeat customers, with a new scaler fit on repeat-only training rows. It is not directly comparable with the later raw-aggregate train-fitted 10,000-customer evaluation. The historical autoencoder evaluation used a different all-customer population and global transform and is also not a matched comparator; see [`track_a_autoencoder_evaluation.md`](track_a_autoencoder_evaluation.md).

## Robustness interpretation and limitations

- The initial full-population feature scaler was fit before sampling, so its clustering scores are transductive with respect to scaler statistics.
- The 3,000-customer holdout was reused for many comparisons and is not an independent final validation set.
- Single-transaction customers represent 83.1133% of the validated population. Their zero transaction SD and equality of average transaction amount to total spend are consequences of one observed row.
- Ratio clipping in the initial standardized matrix changed at most 12/3,000 k=3 evaluation assignments. This is a different sensitivity from the raw-ratio clipping in the train-fitted experiment.
- Later raw-ratio clipping actually changed held-out feature values (101/106/101 in the 2+ seeds); the corrected count audit is in the train-fitted segmentation report and the machine-readable readiness package.
- None of the metrics establishes a stable, general behavioral segmentation or the utility of a banking decision rule.

The diagnostic run used CPU scikit-learn with a one-thread limit. Clipping fits took 0.685–1.109 seconds; repeat-only fits took 0.163–0.367 seconds; sampled peak working set was 424.1–454.0 MiB. A preliminary incomplete invocation and its preserved files are recorded in the run metadata; the successful comparison used fresh candidate IDs and appended completed rows.
