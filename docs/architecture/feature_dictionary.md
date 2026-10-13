# Feature Dictionary and Model-Input Contract

This dictionary describes the customer/account feature outputs and distinguishes identifiers, profile attributes, and clustering inputs. The interpretable matrix contains unscaled values for profiling; the model matrix applies its configured transformations. Keys remain in both files for alignment but are never scaled or selected as clustering inputs.

## Track A — Indian Bank Transactions

**Grain:** customer. `CustomerID` is the unique key. The baseline feature contract is in `configs/models/feature_contracts.yaml`.

| Field | Meaning | Historical matrix treatment | Clustering input |
|---|---|---|---|
| `CustomerID` | Source customer key | Untouched, retained for alignment | No |
| `Age` | Age derived from DOB using the 2016 convention | Standardized in the initial global matrix for compatibility | No; profile only |
| `Gender_M` | Indicator for source gender `M` | Binary, unchanged | No; profile/fairness review only |
| `Is_Tier_1_City` | Source location maps to a major-city indicator | Binary, unchanged | No; geographic proxy/profile only |
| `AccountBalance` | Mean customer account balance | `log1p`, then StandardScaler | Yes |
| `Transaction_Count` | Count of customer transactions | StandardScaler | Yes |
| `Avg_Transaction_Amt` | Mean transaction amount | `log1p`, then StandardScaler | Yes |
| `Std_Transaction_Amt` | Transaction-amount standard deviation; zero for one retained transaction | `log1p`, then StandardScaler | Yes |
| `Total_Spend` | Sum of retained transaction amounts | `log1p`, then StandardScaler | Yes |
| `Avg_Spend_Ratio` | Mean transaction amount divided by transaction-row balance plus one | StandardScaler | Yes; low-balance sensitivity review required |
| `Weekend_Spend_Ratio` | Share of retained transactions on weekends | StandardScaler | Yes |
| `Night_Spend_Ratio` | Share of retained transactions from 20:00 through 03:59 | StandardScaler | Yes |
| `Recency_Days` | Days since last transaction, relative to the feature-building maximum date in the initial matrix | StandardScaler | Yes in the initial contract; leakage-controlled experiments rebuild it at a fixed as-of date |

The initial model CSV includes `Age` as a compatibility/scaler field, but the clustering contract selects only the nine rows marked “Yes.” Leakage-controlled experiments use the nine unscaled interpreted aggregates as input and fit a new scaler only on training customers. Their recency reference date is recorded with each candidate.

## Track B — Santander Product Portfolio

**Grain:** customer at the `2016-05-28` snapshot. `customer_id` is retained in both matrices.

| Source product | Output feature | Meaning |
|---|---|---|
| `ind_ahor_fin_ult1` | `has_savings_account` | Savings account |
| `ind_aval_fin_ult1` | `has_bank_guarantees` | Bank guarantees |
| `ind_cco_fin_ult1` | `has_current_account` | Current account |
| `ind_cder_fin_ult1` | `has_derivada_account` | Derivative account |
| `ind_cno_fin_ult1` | `has_payroll_account` | Payroll account |
| `ind_ctju_fin_ult1` | `has_junior_account` | Junior account |
| `ind_ctma_fin_ult1` | `has_mas_particular_account` | Más particular account |
| `ind_ctop_fin_ult1` | `has_particular_account` | Particular account |
| `ind_ctpp_fin_ult1` | `has_particular_plus_account` | Particular plus account |
| `ind_deco_fin_ult1` | `has_short_term_deposits` | Short-term deposits |
| `ind_deme_fin_ult1` | `has_medium_term_deposits` | Medium-term deposits |
| `ind_dela_fin_ult1` | `has_long_term_deposits` | Long-term deposits |
| `ind_ecue_fin_ult1` | `has_e_account` | e-account |
| `ind_fond_fin_ult1` | `has_funds` | Funds |
| `ind_hip_fin_ult1` | `has_mortgage` | Mortgage |
| `ind_plan_fin_ult1` | `has_pensions` | Pensions |
| `ind_pres_fin_ult1` | `has_loans` | Loans |
| `ind_reca_fin_ult1` | `has_taxes` | Taxes |
| `ind_tjcr_fin_ult1` | `has_credit_card` | Credit card |
| `ind_valo_fin_ult1` | `has_securities` | Securities |
| `ind_viv_fin_ult1` | `has_home_account` | Home account |
| `ind_nomina_ult1` | `has_payroll` | Payroll |
| `ind_nom_pens_ult1` | `has_pensions_nom` | Payroll pensions |
| `ind_recibo_ult1` | `has_direct_debit` | Direct debit |

All 24 product flags must be 0, 1, or missing in the input. Missing values become 0; other values cause a builder error. Product flags are the contracted model inputs and remain unchanged in the model matrix.

| Profile attribute | Definition / policy | Clustering input |
|---|---|---|
| `age` | Age; median imputed in the historical matrix | No |
| `gross_income` | Province-median imputation followed by fallback median | No; socioeconomic profile only |
| `seniority_months` | Relationship tenure | No; review before any later use |
| `total_products_owned` | Sum of the 24 product flags | No; redundant profile value |
| `gender` | Normalized source category H/V | No; profile/fairness review only |
| `is_female` | H→1, V→0; unknown/missing remains null | No; profile/fairness review only |
| `customer_segment_*` | Source-defined customer tier | No; source-tier profile only |
| `foreigner_index`, `deceased_index`, `spouse_index`, `new_customer_index`, `activity_index` | Source indicators retained for profile/audit | No |

The historical Track B scaler was fitted only on `age`, `gross_income`, `seniority_months`, and `total_products_owned`; it excludes the key, product indicators, and gender fields. The model contract selects only the 24 product flags.

## Track C — Berka Financial Dataset

**Grain:** account, not customer. `account_id` is the account key. The source `disp` relationship table is not currently used; customer aggregation would require a policy for multiple accounts and joint dispositions.

| Field | Meaning | Historical matrix treatment | Clustering input |
|---|---|---|---|
| `account_id` | Source account key | Untouched, retained for alignment | No |
| `Account_Age_Days` | Account age relative to the source's maximum transaction date | Clip below zero, `log1p`, then StandardScaler | Yes |
| `Account_Balance` | Last available transaction balance per account | Clip below zero, `log1p`, then StandardScaler | Yes |
| `Transaction_Count` | Number of account transactions | StandardScaler | Yes |
| `Total_Credit` | Sum of credited transaction amounts | Clip below zero, `log1p`, then StandardScaler | Yes |
| `Total_Withdrawal` | Sum of withdrawal transaction amounts | Clip below zero, `log1p`, then StandardScaler | Yes |
| `Recency_Days` | Days since last transaction, relative to the source maximum date | StandardScaler | Yes |

## Transformation persistence and inference

Builders produce separate interpretable/model matrices and persist versioned joblib preprocessing artifacts under `artifacts/preprocessors/`. Artifacts record track, ordered features, identifiers, scaler inputs, and transform recipes. A load-only transform reuses its saved artifact without fitting; input schema and feature names must match. Never pass an already transformed model matrix back through the raw-feature transformation path.
