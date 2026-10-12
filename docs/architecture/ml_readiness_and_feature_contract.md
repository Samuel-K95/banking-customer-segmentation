# Machine-Learning Readiness and Feature Contract

## Current status

Track A real data has been processed and passed the configured feature-contract preflight. Leakage-controlled repeat-customer experiments and a matched autoencoder comparison are complete; no champion is approved. Tracks B and C remain blocked on their source/data-generation requirements. See [`real_data_validation_status.md`](real_data_validation_status.md) for track-specific availability and [`track_a_real_data_validation_results.md`](track_a_real_data_validation_results.md) for measured Track A results.

Estimators must select only the contract's `model_features`. Do not pass every numeric column in a model CSV to an estimator. Interpretable outputs retain customer/account keys and profile fields for reporting, not as clustering inputs.

## Track contracts

| Track | Grain / key | Baseline input | Current status |
|---|---|---|---|
| A — Indian Bank Transactions | Customer / `CustomerID` | Nine behavioral fields listed below | 837,951 validated customers; model inputs and saved preprocessing verified |
| B — Santander Product Portfolio | Customer at 2016-05-28 / `customer_id` | 24 contracted product flags | Source competition access and local outputs require validation |
| C — Berka Financial Dataset | Account / `account_id` | Six account-level fields | Source files/outputs and account-to-customer policy require validation |

Track A baseline features, in fixed order: `AccountBalance`, `Transaction_Count`, `Avg_Transaction_Amt`, `Std_Transaction_Amt`, `Total_Spend`, `Avg_Spend_Ratio`, `Recency_Days`, `Weekend_Spend_Ratio`, `Night_Spend_Ratio`. Age, gender, city tier, and `CustomerID` are profile/alignment fields, not model inputs.

Track B uses all 24 explicitly mapped 0/1 product fields. Missing product values are filled with 0; other non-binary values are rejected. Gender, age, income, source-defined customer tiers, and other review attributes remain profile-only. `sexo` is H=female and V=male; unknown values remain null.

Track C remains account-level. The Berka `disp` relationship table is not currently used; account-to-client aggregation needs a separate disposition policy.

## Preprocessing and artifact contract

Interpretable values remain unscaled; model matrices are produced separately. Saved joblib artifacts include schema version, track, ordered feature names, identifiers, scaler inputs, and transformation rules. Customer/account keys are excluded from all scalers.

Fit the transformer only on the declared training data for a new inductive experiment; persist it with the model. For inference, load the saved transformer and apply it without fitting. Incoming feature names must satisfy the contract; do not rescale an already transformed model matrix.

Track A's initial full-population scaler is retained for historical baseline reproducibility and is not a suitable scaler for independent evaluation. The leakage-controlled Track A experiments instead split IDs before fitting learned preprocessing and use a fixed observation date. Their validation and selection limitations are documented in the model-readiness and final-validation reports.

## Preflight and execution

Run the feature-contract preflight from the repository root:

```bash
uv run python -m customer_segmentation.models.preflight --config configs/models/feature_contracts.yaml
```

It checks files/columns, key uniqueness and alignment, row counts, numeric finiteness, constants/duplicates, and binary product values. Categorical profile fields are validated separately and are not silently converted into model inputs. Exit code 0 means every configured track passed; exit code 1 means at least one track is blocked.

For ingestion and feature generation, use [`data_layer_integration_handoff.md`](data_layer_integration_handoff.md). Track A has current real-data outputs; later Track B/C files must be generated and checked before those tracks are modeled.
