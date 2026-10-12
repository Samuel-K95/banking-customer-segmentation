# Track A Feature Distribution Diagnostics

## Executive summary

The validated customer matrix contains 837,951 customers and nine contract-defined clustering inputs. `Avg_Spend_Ratio` has an extreme right tail: its maximum is 358.7799 standardized units, and 348 customers exceed 10 absolute standardized units. This can dominate Euclidean distances and affect Gaussian-mixture likelihoods. The diagnostics retain all customers and do not alter the saved preprocessing or feature values.

## Scope and feature contract

This analysis reads `data/processed/track_a_features_interpretable.csv` and the aligned `track_a_features_model.csv`. The model feature order is:

```text
AccountBalance, Transaction_Count, Avg_Transaction_Amt, Std_Transaction_Amt,
Total_Spend, Avg_Spend_Ratio, Recency_Days, Weekend_Spend_Ratio,
Night_Spend_Ratio
```

`CustomerID` is an alignment key. `Age`, `Gender_M`, and `Is_Tier_1_City` are profile-only and are excluded from the nine clustering inputs. The model matrix was already transformed by the historical full-population preprocessing artifact; these diagnostics did not refit or modify it.

## Transaction-history prevalence

| Transactions per customer | Customers | Percent |
|---:|---:|---:|
| 1 | 696,449 | **83.1133%** |
| 2 | 123,194 | 14.7018% |
| 3 | 16,492 | 1.9681% |
| 4 | 1,661 | 0.1982% |
| 5 | 141 | 0.0168% |
| 6 | 14 | 0.0017% |

For every one-transaction customer, sample standard deviation is undefined and the historical feature builder filled it with zero. Their average transaction amount equals total spend; their spend ratio represents one transaction rather than a multi-transaction average. This is a data-depth limitation, not evidence that these customers are inactive or low value.

## Distribution and outlier diagnostics

Variance and quantiles below are computed on the existing transformed model matrix (after the configured logarithms and standardization). “Max absolute z” is a descriptive maximum, not a proposed clipping threshold.

| Feature | Skew | P50 | P95 | P99 | Max absolute z | Count above 3 std units | Count above 10 std units |
|---|---:|---:|---:|---:|---:|---:|---:|
| `AccountBalance` | -1.0379 | 0.1069 | 1.4006 | 1.9374 | 4.0191 | 13,713 | 0 |
| `Transaction_Count` | 2.5289 | -0.4219 | 1.7633 | 3.9485 | 10.5043 | 18,308 | 14 |
| `Avg_Transaction_Amt` | -0.1869 | 0.0603 | 1.5759 | 2.4032 | 5.3097 | 6,789 | 0 |
| `Std_Transaction_Amt` | 2.0562 | -0.4320 | 2.4831 | 3.1514 | 5.0685 | 12,345 | 0 |
| `Total_Spend` | -0.2015 | 0.0152 | 1.5527 | 2.3471 | 5.0250 | 6,160 | 0 |
| `Avg_Spend_Ratio` | **190.0051** | -0.0271 | -0.0120 | 0.1990 | **358.7799** | 1,172 | 348 |
| `Recency_Days` | -0.2221 | -0.0171 | 1.4936 | 1.6907 | 3.6297 | 3,455 | 0 |
| `Weekend_Spend_Ratio` | 0.8333 | -0.6979 | 1.5706 | 1.5706 | 1.5706 | 0 | 0 |
| `Night_Spend_Ratio` | 0.9351 | -0.6660 | 1.6452 | 1.6452 | 1.6452 | 0 | 0 |

No feature has zero or near-zero variance under the stated `1e-6` population-variance threshold. There are 51,980 customers with at least one feature above three absolute standardized units and 362 above ten. The maximum `Avg_Spend_Ratio` value alone contributes approximately 128,723 to squared Euclidean distance, compared with an order-of-nine contribution from nine typical unit-variance features. Extreme values may therefore dominate affected K-Means distances and GMM likelihood/covariance estimates.

## Small-balance sensitivity

The ratio is the mean of transaction-level `TransactionAmount (INR) / (CustAccountBalance + 1)` values. The table groups by interpreted mean customer balance; ratio calculations still use transaction-row balances.

| Mean account-balance group | Customers | Ratio median | Ratio P95 | Ratio max |
|---|---:|---:|---:|---:|
| `AccountBalance = 0` | 1,463 | 600.00 | 8,290.10 | 125,000.00 |
| `0 < AccountBalance <= 1` | 2,711 | 137.25 | 2,010.49 | 41,978.22 |
| `1 < AccountBalance <= 10` | 9,178 | 22.04 | 400.71 | 41,322.31 |
| `10 < AccountBalance <= 100` | 29,658 | 5.27 | 73.34 | 8,834.29 |
| `100 < AccountBalance <= 1,000` | 48,178 | 0.8647 | 13.85 | 15,150.41 |
| `AccountBalance > 1,000` | 746,763 | 0.02269 | 0.6997 | 41,225.08 |

The sharp rise at low mean balances and large maxima among customers with higher mean balances warrant denominator and aggregation-semantics review. The baseline feature definition remains unchanged in these diagnostics. Any transform or cap must be a separately identified, train-fitted sensitivity evaluated on matched customer IDs.

## Interpretation and follow-up

The evidence identifies ratio-tail sensitivity, not grounds for deleting customers or silently clipping values. Compare the unchanged baseline with a separately defined train-fitted robust scaling and a documented ratio transformation. Report membership changes, cluster sizes, held-out metrics, and seed stability in their respective feature geometries. Detailed model results are in [`track_a_traditional_clustering_baselines.md`](track_a_traditional_clustering_baselines.md) and [`track_a_train_fitted_segmentation.md`](track_a_train_fitted_segmentation.md).
