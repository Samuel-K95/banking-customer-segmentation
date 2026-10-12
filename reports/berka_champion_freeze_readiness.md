# Berka Champion Freeze Readiness

**Overall verdict: BLOCKED. No champion is approved or frozen.** This review covers the original 13-feature Berka K-Means k=2, seed-42 candidate and its saved preprocessing. The seed-42 diagonal GMM k=2 artifact is retained as a comparison candidate, not a second freeze recommendation.

## Candidate and baseline evidence

The candidate was trained on the 2,701 training accounts from the 3,602-account population eligible at 365 days’ tenure and evaluated on the shared 901-account holdout. The observation date is **1997-12-31**. Scores below are means across seeds 42, 2024, and 123 in the same original 13-feature, train-fitted feature space; the seed-42 artifacts are the reference under review.

| Algorithm | k | Silhouette | Davies–Bouldin | Calinski–Harabasz | Smallest segment | Cross-seed ARI, mean / minimum | Median fit time |
|---|---:|---:|---:|---:|---:|---:|---:|
| K-Means | 2 | 0.2097 | 1.7299 | 231.12 | 43.29%–43.73% | 0.9882 / 0.9823 | 0.034 s |
| Diagonal GMM | 2 | 0.2162 | 1.6789 | 219.72 | 32.85% | 1.0000 / 1.0000 | 0.080 s |

K-Means is the **provisional freeze candidate**, not an approved champion: it has slightly higher Calinski–Harabasz and more balanced segments and is less assignment-sensitive than GMM in the feature ablations. Counterevidence remains: GMM has slightly better silhouette and Davies–Bouldin scores and perfect cross-seed ARI. Both algorithms’ assignments are associated primarily with financial scale and activity intensity.

The selected seed-42 artifacts were replayed against their stored assignments during temporal validation and matched exactly. These held-out results remain exploratory, not an independent champion validation.

## Sensitivity, temporal evidence, and interpretation

Feature sensitivity supports the provisional K-Means lead, while setting limits on that conclusion. Relative to the original assignments, K-Means ARI was 0.756 when gross movement was excluded, 0.773 when balance volatility was excluded, and 0.603 when both were excluded. The comparable GMM ARIs were 0.073, 0.300, and −0.011. Each reduced-space model was reproducible across seeds, but GMM’s segment membership changed substantially with feature choice. In the joint ablation, K-Means retained two large groups; GMM’s two groups were stable but had weaker frequency/regularity contrast. Silhouette scores across these different feature spaces are not treated as comparable evidence of superiority. The balance-volatility ablation removed `balance_std` only; other related inputs remained.

In the retrospective 1998 outcomes, K-Means cluster 1 had medians of 265,060 credit and 254,205 debit amount, 49,446 balance, and 77 transactions, versus 69,510 credit, 65,006 debit, 23,144 balance, and 62 transactions in cluster 0. GMM cluster 0 had the higher-scale profile: 246,292 credit, 239,268 debit, 46,553 balance, and 75 transactions, versus 60,198 credit, 57,545 debit, 22,004 balance, and 61 transactions in cluster 1. Median active months were 12 in every segment. Typical inter-arrival time was about one day shorter in each higher-activity profile. The groups therefore remain distinguishable, primarily by financial scale and activity intensity, with a secondary regularity difference—not by monthly presence.

Median account balances rose by roughly 5–8% from 1997 to 1998 in all four algorithm/segment groups. Paired balance-volatility changes were uncertain: every 95% bootstrap interval included zero. The comparison also found 29 accounts with negative-balance observations in 1997 and 36 in 1998, while median negative-balance frequency remained zero. Detailed definitions, distributions, bootstrap intervals, and segment profiles are in the [Berka temporal validation](berka_temporal_behavior_validation.md); the sensitivity and baseline details are in the [feature sensitivity](berka_feature_sensitivity.md) and [baseline](berka_behavioral_clustering_baselines.md) reports.

**The 1998 analysis is retrospective and has already informed development.** It is not an untouched final holdout and cannot support an independent confirmatory claim or a current-date forecast. No supervised outcome labels exist; all findings describe associations and persistence, not causation.

## Segment interpretation, grain, and fairness

The seed-42 K-Means segments are 394/507 accounts (43.7%/56.3%); the GMM segments are 605/296 (67.1%/32.9%). In the saved K-Means model, cluster 0 is the lower-scale profile and cluster 1 the higher-scale profile; for GMM, cluster 1 is lower-scale and cluster 0 higher-scale. This descriptive mapping is supported by the reported account-level median flow, transaction, and balance profiles and is tied to these exact artifacts. These are not customer personas, and cluster IDs must not be transferred between algorithms or model versions without a validated mapping.

Berka is **account-grain**, not customer-grain. The `disp` relation has one `OWNER` per account; 869 accounts also have a `DISPONENT`. The behavioral pipeline does not join `disp.csv`, so it neither duplicates accounts nor attributes joint-account transactions to a specific person. A later customer-level application needs an explicit owner/disponent policy.

No demographic fairness analysis has been completed for these Berka candidates. Behavior, balances, geography, and account tenure may proxy for access or socioeconomic circumstances. Client/district attributes are not included in this account-level feature pipeline, so their representation and proxy effects have not been measured. These exploratory clusters should not be used alone for eligibility, pricing, credit, or other adverse decisions. Berka’s 1990s Czech banking context, small account population, and historical source units further limit generalization.

## Candidate artifacts and serving contract

The frozen Berka reference models and preprocessing are:

- K-Means k=2 seed 42: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/models/kmeans_k2_seed42.joblib`
- Diagonal GMM k=2 seed 42: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/models/gaussian_mixture_k2_seed42.joblib`
- Shared fitted preprocessing state: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/preprocessor_train_fitted.joblib`
- Baseline configuration: `configs/models/berka_behavioral_baselines_v1_1.yaml`
- Artifact hashes and saved split provenance: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/run_metadata.json`; temporal replay fingerprints: `artifacts/berka_temporal_behavior_validation/v1/asof_1997-12-31/frozen_model_integrity.json`

| Artifact | SHA-256 |
|---|---|
| K-Means k=2 seed 42 | `9c1c32f9c3009deb6cb22325df09eacd456f208012dfdf9fe205369d3b0f0046` |
| Diagonal GMM k=2 seed 42 | `1c27842c08535f0751859ff893f43d9555253513700187cf726efff4adaa8be4` |
| Shared preprocessor state | `030fecf56ded589ebd4400a6ff28eadcdb69ab9ca9429f4a236908a904c0d1bb` |

The loaded artifacts match the fingerprints recorded during temporal validation, deserialize in the current runtime, and reproduce all 901 saved predictions per algorithm. The baseline artifact versions are `berka_account_snapshot_v1` and `berka_behavioral_baselines_v1_1`; the model metadata identifies K-Means k=2 seed 42 and diagonal GMM k=2 seed 42. Current runtime versions used for this audit are Python 3.12.10, NumPy 2.2.6, pandas 3.0.5, scikit-learn 1.9.1, and joblib 1.6.0. The original baseline run metadata does not record training-runtime package versions or a lockfile, so compatibility with a different environment still needs to be pinned and tested.

Both models use this ordered 13-feature contract: `account_tenure_days`, `recency_days`, `credit_frequency_per_day`, `debit_frequency_per_day`, `gross_money_movement`, `net_cash_flow`, `transaction_amount_median`, `balance_std`, `negative_balance_frequency`, `interarrival_days_median`, `interarrival_days_std`, `recent_90d_transaction_frequency_per_day`, `activity_change_count`. Gross movement is credit total plus debit total; source amount direction comes from transaction type. Account IDs are alignment keys, never model inputs.

| Feature group | Units and interpretation |
|---|---|
| `account_tenure_days`, `recency_days`, `interarrival_days_median`, `interarrival_days_std` | Calendar days relative to the fixed observation date / transaction dates |
| `credit_frequency_per_day`, `debit_frequency_per_day`, `recent_90d_transaction_frequency_per_day` | Transactions per observed day |
| `gross_money_movement`, `net_cash_flow`, `transaction_amount_median` | Original Berka amount units; no FX or inflation conversion |
| `balance_std` | Standard deviation of transaction balance in source balance units; negative balances are preserved in source outcomes |
| `negative_balance_frequency` | Unitless share in [0, 1] |
| `activity_change_count` | Recent minus prior 90-day transaction count |

Preprocessing depends on `joblib`, NumPy, pandas, and scikit-learn (`StandardScaler`) and was fit on the 2,701 training accounts. After transforms, missing values use training medians; the positive `log1p` features are `account_tenure_days`, `recency_days`, `credit_frequency_per_day`, `debit_frequency_per_day`, `gross_money_movement`, `transaction_amount_median`, `balance_std`, `interarrival_days_median`, `interarrival_days_std`, and `recent_90d_transaction_frequency_per_day`. Signed `net_cash_flow` and `activity_change_count` use `asinh`; bounded `negative_balance_frequency` is unchanged before clipping. Training-fitted 1st/99th percentile clipping is followed by a training-fitted `StandardScaler`. The selected training population had no missing values in these 13 features, but the transform retains median-imputation rules for undefined future aggregates. The fitted state is restored by `BerkaTrainFittedPreprocessor.from_state_dict()` in `src/customer_segmentation/models/berka_behavioral_baselines.py`.

The snapshot version is `berka_account_snapshot_v1`, as-of **1997-12-31**. Tenure and recency use this cutoff; recent activity uses `(1997-10-02, 1997-12-31]` and the prior window `(1997-07-04, 1997-10-02]`. Only account histories with at least 365 days’ tenure were clustered. The saved split has 2,701 training and 901 held-out accounts, generated with seed 8675309. The preprocessor records the ordered training-ID digest and a 2,701-row fit; model bundles record the same training and held-out ID digests.

A deterministic serving feature job must reject any transaction dated after its declared as-of date, apply type-based credit/debit direction to positive source amounts, derive the gross/net features consistently, use the exact feature order and saved transforms without fitting, retain account-ID alignment, and enforce the 365-day eligibility rule. The current artifacts are fixed-date research candidates, not a deployed inference API; serving at another as-of date requires a new validated temporal contract, leakage review, and drift assessment. GMM component probabilities are uncalibrated, and K-Means has no confidence output.

### Proposed FastAPI contract (not implemented)

`POST /v1/berka/segments:predict` would accept a versioned snapshot request with `model_version`, `snapshot_version`, `as_of_date`, and an ordered `accounts` array. Each item contains an opaque `account_id` and a finite numeric `features` object with exactly the 13 names above. The current artifact must reject `as_of_date` values other than `1997-12-31`; an upstream feature service must prove no post-cutoff records entered its aggregates. The response preserves input order and returns `account_id`, `model_version`, `as_of_date`, `eligibility_status`, and the model-specific integer `cluster_id` (null with `insufficient_history` when tenure is under 365 days). A versioned descriptive mapping may accompany the integer; it must not imply a persona. The API should reject duplicate IDs, missing/extra features, non-finite values, invalid ranges, and unknown model versions. A future-date request requires a newly approved model/snapshot contract, not silent reuse of these research artifacts.

## Remaining gates before champion freeze

| Gate | Verdict | Evidence or remaining requirement |
|---|---|---|
| Artifact integrity and loadability | **PASS** | Saved K-Means, GMM, and preprocessing files match the recorded SHA-256 fingerprints and load in the audited runtime. |
| Snapshot, eligibility, split, and train-only fit | **PASS** | Snapshot date, 365-day population, 2,701/901 split hashes, feature order, fit row count, and training-ID digest match saved metadata. |
| Exact frozen prediction replay | **PASS** | Both seed-42 models reproduce their saved 901 held-out cluster assignments exactly without fitting. |
| Feature units and operational semantics | **NEEDS APPROVAL** | Source amount units are not explicitly currency-normalized; approve definitions, missing handling, cutoff, and unit presentation for the intended use. |
| Segment meaning and mapping | **NEEDS APPROVAL** | Profiles support lower-/higher-scale descriptions only. Approve a model-version-specific mapping; labels are arbitrary and do not persist across retraining. |
| Fairness and disposition policy | **NEEDS APPROVAL** | Complete proxy/fairness review and decide how an account-level use handles OWNER/DISPONENT relationships; no Berka demographic fairness audit exists. |
| Independent temporal validation | **BLOCKED** | The 1998 retrospective analysis has already informed development. A new prospective/external vintage and predeclared evaluation are required before confirmatory use. |
| Serving package and operational controls | **BLOCKED** | No composed inference pipeline or FastAPI service is implemented; dependency versions are not pinned in the baseline metadata. Serving, monitoring, access control, and rollback remain to be validated. |

**Overall champion-freeze readiness: BLOCKED.** Freeze requires independent/prospective validation with a predeclared cutoff and outcomes; model-risk and product-use approval; fairness/proxy and owner/disponent review; minimum segment and inactive-account policy; pinned runtime dependencies; deterministic end-to-end serving tests; and monitoring/rollback procedures. The existing 1998 retrospective is not an untouched final holdout. No champion is approved in this review.
