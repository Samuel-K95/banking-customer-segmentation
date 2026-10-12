# Berka Out-of-Period Behavioral Validation

**Snapshot and segment date:** 1997-12-31; **prior comparison window:** 1997-01-01 through 1997-12-31; **out-of-period window:** 1998-01-01 through 1998-12-31.

**Population:** the frozen 901-account held-out population

## Executive findings

The original seed-42, k=2 segments remained distinguishable during 1998, but mostly along the same **financial-scale and transaction-activity gradient** visible in 1997. Credit/debit totals, median balances, and balance volatility remained substantially different between segments. Transaction frequency and transaction regularity also differed, though monthly presence was similar: each segment’s median and 25th percentile was 12 active months. Median transaction amounts and typical inter-arrival times also retained their relative segment patterns.

The year-to-year changes were generally modest: transaction-count medians moved by only one or two transactions, balance medians rose, and median paired balance-volatility changes had confidence intervals spanning zero. The 1998 results support descriptive longitudinal differentiation, not causal effects or proven prediction of a labeled outcome.

K-Means and GMM both retained meaningful financial-scale contrasts. GMM had perfect seed-42 assignment reproducibility by construction and perfect baseline cross-seed ARI in the previous experiment; K-Means had cross-seed ARI 0.988. Their held-out partitions were only moderately aligned with each other (ARI **0.611**) and use different raw cluster numbers. Neither is selected as a champion.

## Frozen models and integrity checks

The models were loaded from `artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31/`: original-13-feature K-Means k=2 seed 42 and diagonal GMM k=2 seed 42. Saved held-out assignment labels were used as-is. Loaded models were only used to replay predictions as an integrity check; no fitting, refitting, relabeling, or 1998-based adjustment occurred.

The split matched the saved training and held-out SHA-256 ID digests, with **2,701 training and 901 held-out accounts**. Preprocessor metadata matched the baseline run metadata, including feature order, training fit-row count, training account digest, and train-only fit scope. Both seed-42 models matched their saved held-out cluster assignments exactly on replay. SHA-256 fingerprints for the preprocessing artifact, both model files, assignment file, snapshot files, configuration, and transaction source are recorded in `frozen_model_integrity.json` under the validation artifacts.

The original baseline run metadata contains no predeclared model-file checksums. The validation therefore records fresh file SHA-256 digests and corroborates them with the embedded estimator metadata, split hashes, preprocessing metadata, and exact prediction replay rather than claiming comparison to a prior artifact digest.

The baseline run metadata did not contain previously declared file checksums; therefore, the recorded file digests are integrity fingerprints for this validation run, alongside internal metadata and exact assignment-replay verification.

## Outcomes, coverage, and definitions

Outcomes use only held-out account transactions from the two fixed windows. The 1997 window is the 365 calendar days immediately preceding the cutoff; 1998 is the following 365-day year. The source contained **65,792** transactions for the held-out accounts in 1997 and **65,380** in 1998. All 901 held-out accounts remain in the outcome table.

Three accounts had no 1998 transactions (and three had none in 1997); one account was inactive in the prior window but active in 1998, and one was active in 1997 but inactive in 1998. Inactive accounts have zero transaction counts, active months, and credit/debit totals. Amount, balance, negative-balance frequency, and inter-arrival statistics remain missing when undefined. No balance or ratio was imputed.

Credit and debit directions follow normalized transaction type (`PRIJEM` versus `VYDAJ`/`VYBER`); source amount magnitudes are used. Net cash flow is credits minus debits. Average monthly transaction frequency is total transactions divided by 12; frequency per active month is missing for inactive accounts. Inter-arrival medians require at least two transactions; sample inter-arrival standard deviation requires at least two gaps. Negative balance frequency is calculated only when valid balances exist.

Confidence intervals are deterministic 95% percentile bootstrap intervals using 2,000 account resamples and seed 20261012. Segment contrasts are cluster 0 minus cluster 1 using the original saved cluster numbers. Interval estimates are descriptive and unadjusted for multiple comparisons.

## 1998 segment outcomes

All values below are held-out account medians in original source units. Cluster IDs are the frozen labels, not relabeled from 1998 behavior.

| Algorithm / frozen cluster | Accounts | 1998 active months | Transactions | Credit total | Debit total | Net cash flow | Median amount | Median balance | Balance volatility | Median inter-arrival days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K-Means / 0 | 394 | 12 | 62 | 69,510 | 65,006 | 2,896 | 1,560 | 23,144 | 4,587 | 4 |
| K-Means / 1 | 507 | 12 | 77 | 265,060 | 254,205 | 8,058 | 3,528 | 49,446 | 12,097 | 3 |
| GMM / 0 | 605 | 12 | 75 | 246,292 | 239,268 | 7,818 | 3,300 | 46,553 | 11,441 | 3 |
| GMM / 1 | 296 | 12 | 61 | 60,198 | 57,545 | 2,602 | 1,300 | 22,004 | 3,747 | 4 |

The K-Means high-scale segment (cluster 1) continued to show higher transaction activity, flows, balances, volatility, and transaction amounts than cluster 0. GMM cluster 0 carried the higher-scale profile and GMM cluster 1 the lower-scale profile. For both methods, median transaction count and balance level remained separated, while the 1998 median gap between transactions was about one day shorter in the higher-scale segment. Most accounts were active in all 12 months, so the distinction is primarily activity intensity and scale rather than presence versus absence of a banking relationship.

### 1998 cluster contrasts with bootstrap intervals

| Outcome, cluster 0 minus cluster 1 | K-Means difference (95% CI) | GMM difference (95% CI) |
|---|---:|---:|
| Transaction count | −15 (−18, −13) | +14 (+12, +16) |
| Credit amount total | −195,551 (−211,063, −177,579) | +186,094 (+171,949, +200,487) |
| Debit amount total | −189,199 (−203,136, −175,151) | +181,723 (+167,075, +192,823) |
| Net cash flow | −5,162 (−7,075, −3,634) | +5,216 (+3,751, +7,145) |
| Median transaction amount | −1,968 (−2,286, −1,596) | +2,000 (+1,613, +2,295) |
| Median balance | −26,302 (−29,098, −23,316) | +24,549 (+22,250, +26,972) |
| Balance volatility | −7,510 (−8,336, −6,792) | +7,694 (+7,089, +8,171) |
| Median inter-arrival days | +1 (+1, +1) | −1 (−1, −1) |

Median negative-balance frequency was zero in every segment in both periods. However, accounts with at least one negative-balance observation rose from **29/901** in the prior year to **36/901** in 1998. The account-level table preserves the nonzero observations; the zero median should not be read as no negative balances in the population.

## Year-over-year behavior

The trailing 365-day period already showed the same raw-cluster ordering. In 1997, K-Means cluster 1 had median credit/debit totals of 270,824/268,547 versus 70,455/66,623 in cluster 0. GMM cluster 0 had 247,778/245,307 versus 60,189/57,682 in cluster 1. Median 1997 active months were 12 in every segment.

Median paired account changes (1998 minus prior 365 days) were modest for transaction volume and positive for balance level:

| Algorithm / cluster | Transaction-count change (95% CI) | Credit-total change | Debit-total change | Net-cash-flow change | Median-balance change (95% CI) | Balance-volatility change (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| K-Means / 0 | −1 (−1, 0) | +60 | −1,617 | +2,007 (+148, +3,504) | +1,234 (+599, +1,746) | +43 (−99, +207) |
| K-Means / 1 | −2 (−2, −1) | +91 | −4,055 | +4,268 (+1,425, +7,467) | +3,180 (+2,125, +4,155) | +236 (−83, +497) |
| GMM / 0 | −1 (−2, −1) | +60 | −3,874 | +4,268 (+1,521, +7,027) | +3,041 (+2,034, +3,806) | +189 (−83, +481) |
| GMM / 1 | −1 (−1, 0) | +72 | −1,617 | +1,562 (0, +3,221) | +1,269 (+622, +1,803) | +53 (−96, +223) |

Active-month medians remained 12; median inter-arrival time was unchanged within each segment. Median balance increases were positive across all four segments. Balance-volatility changes were uncertain: every paired-delta 95% interval included zero. Credit totals were broadly steady in paired account changes; debit totals tended to decline, contributing to increased net cash flow. These are account-level associations and period changes, not causal effects or predictions of a labeled event.

## Interpretation

1. **Do segments remain distinguishable?** Yes, most clearly by flow totals, balances, transaction size, and activity frequency. The original assignments retain strong 1998 contrasts.
2. **Which characteristics persist?** Financial scale and balance level/volatility persist. Transaction counts, credit/debit frequencies, and typical transaction gaps also remain different. Median active months do not distinguish the segments.
3. **Scale or broader behavior?** Primarily financial scale, with a secondary activity-intensity/regularity component. Both algorithms separate amount and balance levels by roughly twofold or more; higher-scale groups also have more transactions and shorter typical gaps. No clear difference in median active months or median negative-balance frequency appears.
4. **K-Means or GMM?** Both produce interpretable, persistent scale-related profiles on the identical accounts. GMM was more stable across baseline initializations (seed ARI 1.000 versus 0.988 for K-Means); K-Means produced somewhat more balanced segment sizes. Their mutual held-out ARI was 0.611, so their exact partitions are not interchangeable. This evidence does not establish a superior model.
5. **Berka versus Indian Bank for a behavioral demonstration?** Berka is better supported for a demonstration centered on longitudinal behavior: the held-out accounts provide adjacent full-year activity windows, while the validated Indian Bank population had 83.11% of customers with only one transaction. This supports choosing Berka for a *temporal behavior* demonstration, not claiming better predictive performance or general superiority across populations.
6. **Limitations:** The full Berka source has only 4,500 accounts and this analysis uses 901 held-out accounts; segments are account-grain, not customer-grain. The data are from a historical Czech banking system in the 1990s, with source-specific amount units and banking context. The validation spans one subsequent year and has no supervised outcomes. Bootstrap intervals describe account-sample uncertainty, not causal uncertainty or transportability to modern customers.

## Reproducibility and artifacts

Runner: `src/customer_segmentation/models/berka_temporal_validation.py`; configuration: `configs/models/berka_temporal_validation_v1.yaml`; outputs: `artifacts/berka_temporal_behavior_validation/v1/asof_1997-12-31/`.

Outputs include the 901-row account outcome table, the unmodified seed-42 assignments, period-by-segment distribution summaries, bootstrap segment contrasts, paired year-over-year changes, frozen model/preprocessor integrity fingerprints, and run metadata. The outcome table preserves all accounts and missing values for undefined statistics. No model was fitted, no candidate was changed, and no historical result or Indian Bank pipeline was modified.
