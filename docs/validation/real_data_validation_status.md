# Real-Data Availability and Validation Status

## Purpose and chronology

This document records the initial source-availability/preflight check performed before datasets were staged locally. At that point, `data/raw/`, `data/processed/`, and `artifacts/` contained only `.gitkeep` placeholders, so no real rows, processed dimensions, alignment, or preprocessing artifacts could be measured. Track A was later generated and validated; its observed results are documented in [`track_a_real_data_validation_results.md`](track_a_real_data_validation_results.md). This initial status is historical, not the current Track A result.

No raw banking data was downloaded during the initial availability check. Kaggle credential checks tested only for expected files/environment-variable presence; no credential values were read or printed. The observed state at that time was:

- `%USERPROFILE%\.kaggle\kaggle.json`: absent
- `%USERPROFILE%\.kaggle\access_token`: absent
- `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_API_TOKEN`: unset
- Kaggle CLI: 2.2.4

## Source availability observed at the initial check

| Track | Source | Expected local inputs | Metadata/access observation at that time |
|---|---|---|---|
| A — Indian Bank Transactions | [Kaggle dataset](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation), `shivamb/bank-customer-segmentation` | `data/raw/bank_transactions.csv` | Metadata listed the CSV (67,564,190 bytes); it was not then present locally |
| B — Santander Product Portfolio | [Kaggle competition](https://www.kaggle.com/competitions/santander-product-recommendation), `santander-product-recommendation` | `data/raw/santander/train_ver2.csv` | Metadata command required authentication; accepted competition access was unavailable |
| C — Berka Financial Dataset | [Kaggle dataset](https://www.kaggle.com/datasets/marceloventura/the-berka-dataset) | `trans`, `account`, and `client` files under `data/raw/berka/` | Metadata listed the required tables and `disp.csv`; none were then present locally |

The Berka `disp` table relates accounts to clients, but the existing Track C feature builder remains account-grain. Customer-grain aggregation requires a separate policy for multiple accounts and joint dispositions. The Santander loader expects 24 product columns; the actual competition CSV header could not be checked during the initial availability review.

## Initial environment and preflight

At the initial check, available free disk space was approximately 228.1 GiB and free physical memory approximately 4.0 GiB. No processing occurred and these resources were not consumed by a data build. The project environment was repaired locally using Python 3.12.10 and the project lock; no global packages were installed.

The feature-contract preflight was run against the then-empty processed directory and returned **BLOCKED** (exit code 1) for six absent files:

- Track A: `track_a_features_interpretable.csv`, `track_a_features_model.csv`
- Track B: `track_b_features_interpretable.csv`, `track_b_features_model.csv`
- Track C: `track_c_berka_interpretable.csv`, `track_c_berka_model.csv`

No data-row scans or real-data quality checks were possible at that time. The recorded initial test result was 25 passed; compileall, Ruff, and `git diff --check` passed. These are historical integration checks; current repository-wide results are reported in the latest model-readiness report.

## Current follow-up

Track A source data was later staged and processed. The 837,951-customer validated population, missing-data retention, feature distributions, saved preprocessing replay, and current limitations are documented in [`track_a_real_data_validation_results.md`](track_a_real_data_validation_results.md). Tracks B and C still require their own source-access, output-generation, schema/alignment, and preflight verification before model comparisons.
