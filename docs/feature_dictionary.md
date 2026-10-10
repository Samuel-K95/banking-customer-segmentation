# Feature Dictionary

This document serves as the formal schema and feature dictionary for the Data Layer, as required by `02-data-layer.md`.

## Track A: Indian Bank Features (RFM + Behavioral)

| Feature Name | Definition | Source Columns | Grain | Dtype | Transform (Model Matrix) | Missing Policy | PII / Sensitive | Used in Clustering |
|---|---|---|---|---|---|---|---|---|
| `Age` | Age of customer as of 2016 | `CustomerDOB` | Customer | Float | Robust Scaler | Impute via Century Pivot / NaN | Yes | Yes |
| `Gender_M` | Binary flag for Male | `CustGender` | Customer | Int | None (Binary) | 0 if unknown | Yes | Yes |
| `Is_Tier_1_City` | 1 if Mumbai, Delhi, Bangalore, etc. | `CustLocation` | Customer | Int | None (Binary) | 0 if unknown | No | Yes |
| `Transaction_Count` | Number of transactions | `TransactionID` | Customer | Int | Scaler | None (Count) | No | Yes |
| `AccountBalance` | Mean account balance | `CustAccountBalance` | Customer | Float | Log1p + Scaler | Drop NA | No | Yes |
| `Avg_Transaction_Amt` | Mean transaction amount | `TransactionAmount` | Customer | Float | Log1p + Scaler | Drop NA | No | Yes |
| `Std_Transaction_Amt` | Volatility of spend | `TransactionAmount` | Customer | Float | Log1p + Scaler | 0 if single txn | No | Yes |
| `Total_Spend` | Sum of all transactions | `TransactionAmount` | Customer | Float | Log1p + Scaler | None | No | Yes |
| `Avg_Spend_Ratio` | Spend divided by balance | Derived | Customer | Float | Scaler | None | No | Yes |
| `Weekend_Spend_Ratio` | % of txns on weekends | `TransactionDate` | Customer | Float | Scaler | None | No | Yes |
| `Night_Spend_Ratio` | % of txns between 8PM and 4AM | `TransactionTime` | Customer | Float | Scaler | None | No | Yes |
| `Recency_Days` | Days since last txn (relative to max date) | `TransactionDate` | Customer | Int | Scaler | None | No | Yes |

## Track B: Santander Features (Portfolio Autoencoder)

| Feature Name | Definition | Source Columns | Grain | Dtype | Transform (Model Matrix) | Missing Policy | PII / Sensitive | Used in Clustering |
|---|---|---|---|---|---|---|---|---|
| `age` | Age in years | `age` | Customer | Float | Scaler | Median Impute | Yes | Yes |
| `gross_income` | Gross customer income | `renta` | Customer | Float | Scaler | Province Median Impute | Yes | Yes |
| `seniority_months` | Months as customer | `antiguedad` | Customer | Float | Scaler | Median Impute | No | Yes |
| `total_products_owned` | Sum of all products held | `ind_*_ult1` | Customer | Int | Scaler | Fill 0 | No | Yes |
| `is_female` | Binary flag for Female | `sexo` | Customer | Int | None (Binary) | 0 if unknown | Yes | Yes |
| `foreigner_index` | 1 if foreigner | `indext` | Customer | Int | Map S/N to 1/0 | 0 if unknown | No | Yes |
| `activity_index` | 1 if active customer | `ind_actividad_cliente` | Customer | Int | None (Binary) | 0 if unknown | No | Yes |
| `has_[product]` | 24 individual product binary flags | `ind_*_ult1` | Customer | Int | None (Binary) | 0 if unknown | No | Yes |

*(Note: Data quality is enforced via Pandera schema gates during pipeline execution. Both `_interpretable` and `_model` matrices are output to `data/processed/`.)*

## Track C: Berka Features (Financial Transactions)

| Feature Name | Definition | Source Columns | Grain | Dtype | Transform (Model Matrix) | Missing Policy | PII / Sensitive | Used in Clustering |
|---|---|---|---|---|---|---|---|---|
| `Account_Age_Days` | Days since account creation | `date` (account) | Account | Int | Log1p + Scaler | None | No | Yes |
| `Account_Balance` | Final balance snapshot | `balance` | Account | Float | Log1p + Scaler | Drop NA | No | Yes |
| `Transaction_Count` | Number of transactions | `trans_id` | Account | Int | Scaler | None | No | Yes |
| `Total_Credit` | Sum of all credit amounts | `amount`, `type` | Account | Float | Log1p + Scaler | None | No | Yes |
| `Total_Withdrawal` | Sum of all withdrawal amounts | `amount`, `type` | Account | Float | Log1p + Scaler | None | No | Yes |
| `Recency_Days` | Days since last txn | `date` (trans) | Account | Int | Scaler | None | No | Yes |
