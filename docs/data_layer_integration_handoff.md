# Data Layer Integration Handoff

## Summary

The three feature-building paths are available through their package module names. The runner supports staged local data, optional Kaggle retrieval, reuse of saved preprocessing artifacts, and a final feature-contract preflight. Track A customer IDs are retained as alignment keys and excluded from model transformations.

This handoff records the integration implementation. Its initial validation occurred before raw data was staged; subsequent Track A processing results are documented in [`track_a_real_data_validation_results.md`](track_a_real_data_validation_results.md). Track B and Track C have separate data-availability limitations described in [`real_data_validation_status.md`](real_data_validation_status.md).

## Dataset grains and contracts

### Track A — Indian Bank Transactions

Transactions are aggregated to customer grain. `CustomerID` remains an explicit column in the interpretable and model matrices, and is excluded from scaling and clustering inputs. The current builder drops raw transactions containing any null before aggregation; the real-data analysis records the resulting retention and age-validity effects.

### Track B — Santander Product Portfolio

The builder uses the 2016-05-28 customer snapshot and maps the 24 contracted product indicators. Product values must be 0, 1, or missing; missing values become 0, and other values fail with field/value details.

The source `sexo` interpretation is H=female and V=male. `is_female` maps H→1 and V→0. Unknown or missing values remain null; the normalized source category is retained for profile review.

### Track C — Berka Financial Dataset

The current builder aggregates at account grain and preserves `account_id`. The source `disp` relationship table, which links accounts and clients, is not used. Customer-grain aggregation requires a separate decision about multiple accounts and joint dispositions; no such grain change is made here.

## Outputs and preprocessing

Each builder returns an unscaled interpretable matrix and a transformed model matrix. The feature contract explicitly selects model fields and excludes identifiers, demographics, fairness-profile fields, and source-defined customer tiers.

Versioned joblib artifacts record track, ordered features, identifier columns, scaler inputs, and transformation rules. Normal execution fits and saves the configured transformer. `fit_preprocessor=False`, `--reuse-preprocessor`, or the runner's `--reuse-preprocessors` loads compatible saved artifacts without fitting. Reuse requires the same feature schema and order; transformed model matrices must not be sent through the preprocessing path a second time.

## Ingestion and execution

The downloader raises on Kaggle failures, corrupt or incomplete archives, and missing expected files. Existing non-empty files are reused. The runner stops on the first failed input or builder and only reports completion after all requested builders and the feature-contract preflight succeed.

When downloads are authorized:

```bash
uv sync --extra data
bash scripts/run_data_layer.sh
```

For raw files already staged locally:

```bash
bash scripts/run_data_layer.sh --skip-downloads
```

The runner uses `uv run --project . --extra data` when available, or the project-local environment otherwise. Kaggle credentials are required for download operations; credential values must not be stored in the repository.

## Current validation status and next actions

Track A has since been generated and validated against the real transaction source; see [`track_a_real_data_validation_results.md`](track_a_real_data_validation_results.md). The source missingness policy, retained population, saved preprocessing, and real feature distributions are documented there and in the feature dictionary.

Track B and Track C status is recorded in [`real_data_validation_status.md`](real_data_validation_status.md). Once their authorized raw sources are staged, run the appropriate builder and feature-contract preflight, inspect actual row retention/schema/alignment, and verify the saved preprocessing artifacts before modeling. Track C remains account-level until a customer-linkage policy is approved.
