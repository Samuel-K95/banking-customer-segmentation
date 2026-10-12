# Berka Snapshot Feature Preparation — v1 Readiness

**Snapshot date:** 1997-12-31
**Source:** `data/raw/berka/{account,trans}.csv`
**Output:** `data/processed/berka_snapshot/v1/asof_1997-12-31/`
**Scope:** Account-level, interpretable behavioral feature preparation. No model was trained.

## Cutoff assessment and data coverage

The cutoff is suitable for an initial retrospective snapshot. All 4,500 accounts were open by December 31, 1997 (latest opening date: December 29, 1997). Of these, 4,303 had at least 90 elapsed days of tenure and 3,602 had at least 365. The 1997 snapshot includes 734,043 transactions; 322,277 transactions (30.51% of all 1,056,320 rows) fall in 1998 and remain available for retrospective checks. Those 1998 rows cover 4,492 accounts; eight snapshot accounts have no 1998 transaction row.

Transaction-count distributions below use transactions dated on or before the cutoff and include every account meeting the tenure condition.

| Eligible account population | Accounts | Min | P25 | Median | Mean | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| At least 90 days tenure | 4,303 | 4 | 73 | 141 | 170.39 | 266 | 335 | 559 |
| At least 365 days tenure | 3,602 | 8 | 104 | 178 | 197.64 | 283 | 345 | 559 |

The 1998 tail supports retrospective behavioral stability checks, but it is not a label or evidence of future outcomes. Use only if a later evaluation defines a meaningful, time-bounded comparison.

## Implemented features and temporal contract

`src/customer_segmentation/features/build_berka_snapshot_features.py` is separate from the established Track C builder. Run it with `--as-of-date YYYY-MM-DD`; by default the output directory is derived from that date. Existing outputs are never overwritten.

The output has **4,500 rows and 43 columns**: one `account_id` key plus 42 numeric features. Every feature’s exact definition is recorded in the output JSON metadata. Feature families include:

- tenure, available observation days, 90/365-day history flags, and recency;
- credit/debit counts, frequencies, transaction shares, totals, and net cash flow;
- transaction-amount mean, median, sample standard deviation, 90th percentile, and maximum;
- balance mean, median, sample standard deviation, negative-balance count and frequency;
- mean, median, and standard deviation of inter-arrival days;
- recent 90-day activity and totals; the preceding 90-day activity and activity change when the prior window is fully observable.

Transactions after the cutoff and accounts opened after it are excluded. Recent dates are `(as_of_date - 90 days, as_of_date]`; prior dates are `(as_of_date - 180 days, as_of_date - 90 days]`. Prior-period features are missing unless the complete prior period is observable. Account count/total features use zero for an empty set; undefined distribution, balance, inter-arrival, and recency values remain missing. Frequency/share ratios with zero denominators are zero; activity-change ratio is missing when prior count is zero. Source amount magnitude is used with direction from transaction type; balance signs, including negatives, are preserved. Optional operation/counterparty attributes are ignored.

The pipeline reads account and transaction tables only. It does not join `disp.csv`: the source has exactly one OWNER per account and 869 additional DISPONENT relationships, and expanding those relationships would duplicate account behavior. The output remains account-grain; no customer identity or joint-holder attribution is inferred.

## Observed feature distributions and missingness

Selected untransformed distributions across the 4,500 account snapshots:

| Feature | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| Transactions through cutoff | 1 | 133 | 163.12 | 559 |
| Credit transactions | 1 | 53 | 63.61 | 228 |
| Debit transactions | 0 | 79 | 99.51 | 425 |
| Credit amount total | 200 | 299,776.20 | 503,668.48 | 3,464,042.50 |
| Debit amount total | 0 | 262,690.45 | 465,889.35 | 3,387,155.60 |
| Net cash flow | −14,673.30 | 33,118.50 | 37,779.13 | 116,961.00 |
| Transaction amount median (per account) | 46.30 | 2,400 | 3,315.09 | 34,010 |
| Balance median (per account) | 200 | 32,112.83 | 33,908.77 | 84,713.60 |
| Recent 90-day transaction count | 0 | 17 | 16.88 | 41 |
| Recency days | 0 | 0 | 1.23 | 493 |

There are no duplicate account rows. Missing feature values are limited to mathematically unavailable or unobservable statistics/windows:

| Features | Missing rows per feature |
|---|---:|
| Amount standard deviation, balance standard deviation, inter-arrival mean and median | 42 each |
| Inter-arrival standard deviation | 59 |
| Prior 90-day count/credit/debit totals and activity-change count | 435 each |
| Activity-change ratio | 442 |

The `feature_missingness` and complete min/mean/median/max summaries are included in the generated metadata JSON. Negative balances occur in 196 accounts in this cutoff snapshot (maximum 70 negative-balance transaction rows for one account); they are retained rather than clipped.

## Modeling readiness

**Ready for controlled unsupervised-clustering preparation; not ready for direct unprocessed clustering.** The snapshot is leakage-safe by construction, account-unique, and has broad behavioral coverage. However, feature magnitudes differ substantially, distributions are skewed, and 42–442 rows are missing for selected features. A subsequent modeling task should define a documented missing-value policy and train/fitting scope for scaling or transforms, exclude `account_id` from inputs, check sensitivity to high-volume and low-tenure accounts, and compare clusters on a consistent eligible population. Keep the 1998 rows out of snapshot feature computation; use them only for a separately specified retrospective stability analysis. No model, scaler, experiment ledger, or existing pipeline output was changed by this preparation.

## Verification

- Focused synthetic tests: 6 passed (cutoff isolation, account uniqueness/cutoff handling, credit/debit direction, negative balances, deterministic output, empty/sparse windows).
- Ruff: passed for the new module and focused tests.
- Generated files: `berka_account_snapshot_features.csv` and `berka_account_snapshot_metadata.json` in the versioned output directory above.
