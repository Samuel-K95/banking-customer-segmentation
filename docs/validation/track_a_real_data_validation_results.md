# Track A Real-Data Validation Results

**Validation date:** 2026-10-11
**Source:** local `data/raw/bank_transactions.csv`
**Scope:** real Track A feature generation, quality/contract checks, and saved-preprocessor replay. No clustering model was fit in this data-layer validation.

## Executive summary

The Track A builder processed 1,048,567 transaction rows and produced a validated population of 837,951 customers. The output matrices contain 13 columns each, preserve unique aligned CustomerIDs, and pass the feature-contract preflight. A versioned `StandardScaler` artifact was saved and replayed byte-identically. The raw retention and age-validity filters are material: the final population is 94.7624% of distinct raw CustomerIDs. The current global preprocessing fit used all validated rows and is therefore unsuitable for independent held-out evaluation; subsequent train-fitted experiments rebuilt raw aggregates and fit their transforms only on their training IDs.

## Raw input and retention

The raw CSV is 67,564,190 bytes and has these observed fields:

| Source column | Dtype | Null transaction rows |
|---|---|---:|
| `TransactionID` | string | 0 |
| `CustomerID` | string | 0 |
| `CustomerDOB` | string | 3,397 |
| `CustGender` | string | 1,100 |
| `CustLocation` | string | 151 |
| `CustAccountBalance` | float64 | 2,369 |
| `TransactionDate` | string | 0 |
| `TransactionTime` | int64 | 0 |
| `TransactionAmount (INR)` | float64 | 0 |

`CustomerID` is present on all 1,048,567 raw transaction rows, representing 884,265 distinct raw customers. The numeric source columns had no non-numeric or infinite values.

The builder drops any raw transaction row containing a null, then drops aggregated customers without a valid age in the 18–100 range. This policy was not changed.

| Retention measure | Observed value |
|---|---:|
| Raw transaction rows | 1,048,567 |
| Rows with at least one null | 6,953 (0.663095%) |
| Complete-case rows retained | 1,041,614 (99.336905%) |
| Distinct customers with an incomplete transaction | 6,934 |
| Customers with no complete transaction remaining | 4,907 |
| Customers with both complete and incomplete transactions | 2,027 |
| Distinct customers with complete-case transactions | 879,358 |
| Complete-case customers with a valid age | 837,951 |
| Complete-case customers removed for invalid age | 41,407 (4.708776% of complete-case customers) |
| Final retention vs. distinct raw customer IDs | 837,951 / 884,265 (94.762430%) |

The retained IDs match complete-case customers with a valid derived age and transaction date. The 2,027 customers with both complete and incomplete transactions have aggregates based on retained rows only; their transaction counts and spend may be understated. This retention selection must be included in population coverage and fairness reviews.

## Generated files and schemas

| File | Dimensions | Bytes |
|---|---:|---:|
| `data/processed/track_a_features_interpretable.csv` | 837,951 × 13 | 67,875,835 |
| `data/processed/track_a_features_model.csv` | 837,951 × 13 | 178,360,579 |
| `artifacts/preprocessors/track_a.joblib` | Versioned fitted preprocessor | 1,864 |

The two CSVs share this physical column order:

```text
CustomerID, Age, AccountBalance, Transaction_Count, Avg_Transaction_Amt,
Std_Transaction_Amt, Total_Spend, Avg_Spend_Ratio, Weekend_Spend_Ratio,
Night_Spend_Ratio, Is_Tier_1_City, Gender_M, Recency_Days
```

The nine clustering inputs are `AccountBalance`, `Transaction_Count`, `Avg_Transaction_Amt`, `Std_Transaction_Amt`, `Total_Spend`, `Avg_Spend_Ratio`, `Recency_Days`, `Weekend_Spend_Ratio`, and `Night_Spend_Ratio`. CustomerID, age, gender, and city tier are retained as key/profile fields, not baseline inputs. The saved global scaler has ten inputs because it also includes Age for matrix compatibility; model consumers must select only the nine contract fields.

## Feature quality and population distribution

The contract preflight passed for all 837,951 rows. It found unique non-null IDs, exact row/ID alignment between matrices, all nine model fields, no required numeric null/NaN/infinite values, and no constant or duplicate baseline features.

| Feature | Min | Mean | Max | Population SD |
|---|---:|---:|---:|---:|
| `AccountBalance` | 0.00 | 107,217.45 | 115,035,495.10 | 774,781.57 |
| `Transaction_Count` | 1 | 1.1931 | 6 | 0.4576 |
| `Avg_Transaction_Amt` | 0.00 | 1,471.87 | 1,560,034.99 | 6,060.13 |
| `Std_Transaction_Amt` | 0.00 | 284.22 | 511,784.09 | 2,399.06 |
| `Total_Spend` | 0.00 | 1,772.13 | 1,560,034.99 | 6,880.56 |
| `Avg_Spend_Ratio` | 0.00 | 9.4879 | 125,000.00 | 348.38 |
| `Recency_Days` | 0 | 55.2602 | 81 | 15.2244 |
| `Weekend_Spend_Ratio` | 0.00 | 0.3076 | 1.00 | 0.4408 |
| `Night_Spend_Ratio` | 0.00 | 0.2882 | 1.00 | 0.4327 |

The initial matrix scaler was fitted over the full validated population. Its transformed model inputs have means near zero and standard deviations near one. The unscaled profile indicators show `Gender_M=1` for 606,722 customers (72.4054%) and `Is_Tier_1_City=1` for 294,804 (35.1815%). The ratio maximum is 125,000 (about 358.78 standard deviations in the transformed matrix); see [`track_a_feature_diagnostics.md`](../reports/track_a/track_a_feature_diagnostics.md) for the ratio/low-balance diagnostics.

## Preprocessing replay and verification

The saved artifact records version, feature order, identifier, scaler inputs, and transformations; CustomerID is absent from scaler inputs. Replaying the builder with preprocessing reuse preserved SHA-256 hashes of both CSVs and the joblib artifact.

| Check | Result |
|---|---|
| Feature-contract preflight | PASS; 837,951 interpretable and model rows |
| CustomerID uniqueness and matrix alignment | PASS; 837,951 unique IDs, zero nulls/duplicates |
| Baseline feature presence and numeric validity | PASS; all nine contract fields, no required nonfinite values |
| Saved preprocessor replay | PASS; CSV and joblib SHA-256 values unchanged |
| Synthetic/project tests after integration changes | 27 passed |
| Compileall, Ruff, and `git diff --check` | Passed |

The replay hashes are `FEEB96E3B9791DBE8566184FEC2CACF5CA58E44674B9B6F4D343F6A7C9B9C487` for the interpretable matrix, `50534E3EA333678FD5779F7B6D75A24BD10BF7E32E84424874D7DE1BB2E68EE2` for the model matrix, and `CBEB7EF69862C5D17B8CE111827935E7E3BDD4FAFB319BF5F8741B6FF6C6E3A4` for the preprocessing artifact. These historical hashes identify the generated files; the global scaler's population exposure is documented in [`track_a_model_readiness.md`](../reports/track_a/track_a_model_readiness.md).

## Track and modeling readiness

Track A has real data and leakage-controlled repeat-customer experiments. The other tracks were not processed in this validation. This document's initial preflight and test counts describe the data-layer build stage; the current repository-wide tests and model-selection status are in the model-readiness report. No selected model or live prediction contract is approved by these data-layer checks alone.
