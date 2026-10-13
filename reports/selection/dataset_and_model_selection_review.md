# Dataset and Model Selection Review

## Review conclusion

**Berka is provisionally preferred for the primary behavioral-intelligence demonstration; Indian Bank remains the large-scale benchmark.** Berka supports repeated, account-level histories and a retrospective year-over-year view. Indian Bank supplies a much larger validated population for scale, but 83.1133% of its 837,951 customers have only one transaction; only 141,502 customers have at least two transactions, and raw coverage is about 82 days. Neither dataset establishes a production-ready or causally meaningful segmentation.

For Berka, **K-Means k=2 is the provisional lead for review, not a frozen champion**. It gives similarly strong seed stability to GMM, a more balanced split, and is less sensitive to removal of the gross-movement and balance-volatility inputs. GMM has slightly better held-out silhouette and Davies–Bouldin scores, higher seed stability, and remains a useful counterpoint. Both principally separate financial scale and activity intensity.

## Dataset choice

The validated Berka snapshot contains 4,500 accounts, with 3,602 accounts meeting the 365-day tenure eligibility used for clustering. The fixed observation date is **1997-12-31**. The held-out set contains 901 accounts, each assigned once at account grain. The one-year follow-up includes 1998 transactions for 898 of these accounts; three have no 1998 transactions and are retained as inactive with undefined statistics left missing.

Indian Bank is retained as a scale benchmark: 1,048,567 transaction rows and 837,951 validated customers. Its short 2016 transaction window and high singleton share limit the number of customers for whom repeated-event behavior can be characterized. Berka is the better fit for demonstrating account-level frequency, regularity, flow, balance, and temporal change. This is a choice of demonstration suitability, not a direct quality ranking across different datasets, grains, time periods, or feature geometries.

## Berka k=2 baseline comparison

The table reports means across seeds 42, 2024, and 123 on the same 901 held-out accounts, with the minimum segment range across seeds. Metrics are computed in the original 13-feature, train-fitted feature space. Lower Davies–Bouldin is preferable; higher silhouette and Calinski–Harabasz are preferable within this comparison.

| Algorithm | Silhouette | Davies–Bouldin | Calinski–Harabasz | Smallest segment | Cross-seed ARI, mean / minimum | Median fit time |
|---|---:|---:|---:|---:|---:|---:|
| K-Means, k=2 | 0.2097 | 1.7299 | 231.12 | 43.29%–43.73% | 0.9882 / 0.9823 | 0.034 s |
| Diagonal GMM, k=2 | 0.2162 | 1.6789 | 219.72 | 32.85% | 1.0000 / 1.0000 | 0.080 s |

GMM’s mean silhouette and Davies–Bouldin score are marginally better; K-Means has higher Calinski–Harabasz and a more balanced held-out split. The metrics do not identify an unambiguous winner. The selected seed-42 artifacts were replayed against their stored assignments during temporal validation and matched exactly.

## Sensitivity, temporal evidence, and interpretation

Feature sensitivity supports the provisional K-Means lead, while setting limits on that conclusion. Relative to the original assignments, K-Means ARI was 0.756 when gross movement was excluded, 0.773 when balance volatility was excluded, and 0.603 when both were excluded. The comparable GMM ARIs were 0.073, 0.300, and −0.011. Each reduced-space model was reproducible across seeds, but GMM’s segment membership changed substantially with feature choice. In the joint ablation, K-Means retained two large groups; GMM’s two groups were stable but had weaker frequency/regularity contrast. Silhouette scores across these different feature spaces are not treated as comparable evidence of superiority. The balance-volatility ablation removed `balance_std` only; other related inputs remained.

In the retrospective 1998 outcomes, K-Means cluster 1 had medians of 265,060 credit and 254,205 debit amount, 49,446 balance, and 77 transactions, versus 69,510 credit, 65,006 debit, 23,144 balance, and 62 transactions in cluster 0. GMM cluster 0 had the higher-scale profile: 246,292 credit, 239,268 debit, 46,553 balance, and 75 transactions, versus 60,198 credit, 57,545 debit, 22,004 balance, and 61 transactions in cluster 1. Median active months were 12 in every segment. Typical inter-arrival time was about one day shorter in each higher-activity profile. The groups therefore remain distinguishable, primarily by financial scale and activity intensity, with a secondary regularity difference—not by monthly presence.

Median account balances rose by roughly 5–8% from 1997 to 1998 in all four algorithm/segment groups. Paired balance-volatility changes were uncertain: every 95% bootstrap interval included zero. The comparison also found 29 accounts with negative-balance observations in 1997 and 36 in 1998, while median negative-balance frequency remained zero. Detailed definitions, distributions, bootstrap intervals, and segment profiles are in the [Berka temporal validation](berka_temporal_behavior_validation.md); the sensitivity and baseline details are in the [feature sensitivity](berka_feature_sensitivity.md) and [baseline](berka_behavioral_clustering_baselines.md) reports.

**The 1998 analysis is retrospective and has already informed development.** It is not an untouched final holdout and cannot support an independent confirmatory claim or a current-date forecast. No supervised outcome labels exist; all findings describe associations and persistence, not causation.

## Segment interpretation, grain, and fairness

The seed-42 K-Means segments are 394/507 accounts (43.7%/56.3%); the GMM segments are 605/296 (67.1%/32.9%). The supported description is lower- versus higher-scale financial activity, with cluster numbering specific to each model. These are not customer personas, and cluster IDs must not be transferred between algorithms or model versions without a validated mapping.

Berka is **account-grain**, not customer-grain. The `disp` relation has one `OWNER` per account; 869 accounts also have a `DISPONENT`. The behavioral pipeline does not join `disp.csv`, so it neither duplicates accounts nor attributes joint-account transactions to a specific person. A later customer-level application needs an explicit owner/disponent policy.

No demographic fairness analysis has been completed for these Berka candidates. Behavior, balances, geography, and account tenure may proxy for access or socioeconomic circumstances. The Indian Bank reports also flag representation and proxy limitations; its `Age`, `Gender_M`, and `Is_Tier_1_City` are profile-only, not clustering inputs. Neither set of exploratory clusters should be used alone for eligibility, pricing, credit, or other adverse decisions. Berka’s 1990s Czech banking context, small account population, and historical source units further limit generalization.

## Candidate artifacts and serving contract

The frozen Berka reference models and preprocessing are:

- K-Means k=2 seed 42: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/models/kmeans_k2_seed42.joblib`
- Diagonal GMM k=2 seed 42: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/models/gaussian_mixture_k2_seed42.joblib`
- Shared fitted preprocessing state: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/preprocessor_train_fitted.joblib`
- Baseline configuration: `configs/models/berka_behavioral_baselines_v1_1.yaml`
- Artifact hashes and saved split provenance: `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/run_metadata.json`; temporal replay fingerprints: `artifacts/berka_temporal_behavior_validation/v1/asof_1997-12-31/frozen_model_integrity.json`

Both models use this ordered 13-feature contract: `account_tenure_days`, `recency_days`, `credit_frequency_per_day`, `debit_frequency_per_day`, `gross_money_movement`, `net_cash_flow`, `transaction_amount_median`, `balance_std`, `negative_balance_frequency`, `interarrival_days_median`, `interarrival_days_std`, `recent_90d_transaction_frequency_per_day`, `activity_change_count`. Gross movement is credit total plus debit total; source amount direction comes from transaction type. Account IDs are alignment keys, never model inputs.

Preprocessing depends on `joblib`, NumPy, pandas, and scikit-learn (`StandardScaler`) and was fit on the 2,701 training accounts: median imputation after transforms, `log1p` for positive features, `asinh` for signed net flow and activity change, training-fitted 1st/99th percentile clipping, then `StandardScaler`. Negative-balance frequency is bounded and left untransformed before clipping/scaling. The fitted state is stored in the joblib payload and restored by `BerkaTrainFittedPreprocessor.from_state_dict()` in `src/customer_segmentation/models/berka_behavioral_baselines.py`.

The feature contract is fixed to the **1997-12-31 as-of snapshot** and the 365-day eligible population. A serving implementation must aggregate each account using only data available through its declared as-of date, preserve the ordered feature schema and fitted transform state, enforce account-key alignment and history eligibility, and keep model-specific cluster mappings/version identifiers. The current artifacts are fixed-date research candidates, not a deployed inference API; serving at a new as-of date requires a validated temporal contract and drift review. GMM component probabilities are uncalibrated, and K-Means has no confidence output.

## Remaining gates before champion freeze

Champion freeze requires a new independent or prospective data vintage with a predeclared cutoff, population, outcomes, and minimum segment-size criteria; repeated-period stability checks; product/use-case approval; model-risk and fairness/proxy review including disposition policy; small-cell and inactive-account handling; and an end-to-end serving test for aggregation, schema, hashes, cluster mapping, and monitoring. The existing 1998 retrospective evidence does not satisfy independent validation. No champion is approved in this review.
