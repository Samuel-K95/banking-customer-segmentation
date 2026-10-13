# Track A Model Readiness and Release Review

**Review date:** 2026-10-11
**Decision:** retain a finalist shortlist for joint modeling and segment-interpretation review; no champion is approved.
**Machine-readable provenance:** `artifacts/diagnostics/track_a/model_readiness_20261011/`

## Executive assessment

Three candidate families merit further review: train-fitted nine-feature K-Means k=2 and k=3, and ratio-clipped diagonal GMM k=2 as a separate sensitivity alternative. The autoencoder comparison did not show consistent benefit in the common feature space or assignment stability. None of the current candidates has an independent final evaluation set or an approved business use; no winner is selected.

The 2+ cohort contains 141,502 of the 837,951 validated customers (16.8867%); 87.1% of eligible customers have exactly two transactions. The evidence supports only repeat-customer exploration, not general population personas or automatic banking actions.

## Verified candidate families and artifact traceability

The completed run is identified in the machine-readable comparison ledger and candidate inventory. The inventory lists all three seeds for each family, training/evaluation ID hashes, exact model/assignment/profile/metadata paths, hashes, metrics, and verified prediction status.

| Family | Training sample per seed | Evaluation sample | Geometry | Evaluation stability |
|---|---:|---:|---|---:|
| K-Means k=2 | 50,000 | Shared 10,000 2+ customers | Nine raw aggregate features, train-fitted transform | Mean three-seed ARI 0.9788; minimum 0.9698 |
| K-Means k=3 | 50,000 | Shared 10,000 2+ customers | Same unclipped nine-feature geometry | Mean three-seed ARI 0.9748; minimum 0.9634 |
| Diagonal GMM k=2 | 50,000 | Shared 10,000 2+ customers | Ratio clipped at training q0.5/q99.5, then train-fitted scaling | Mean three-seed ARI 0.9904; minimum 0.9856 |

The comparison ledger is `artifacts/experiments/comparison_board.csv`; the independent audit index is `candidate_inventory.csv` in the provenance directory above. Each model bundle is paired with a CustomerID-aligned assignment file, seed-specific fitted preprocessing, per-run metadata, and evaluation profile. The GMM model uses a different clipped feature space and its results are not a direct same-geometry ranking against unclipped K-Means.

### Stability metric context

The repeat-only K-Means k=2 result has ARI **0.9434** across seeds on 489 repeat customers, using a repeat-only scaler and an earlier globally transformed feature matrix. The matched raw-aggregate K-Means k=2 and k=3 baseline pair has ARIs **0.9908** and **0.9923** on 10,000 2+ customers. Those scores differ in training population/sample size, evaluation population, and feature transform. The matched autoencoder k=2/k=3 ARIs are 0.2097/0.2566; the 0.9908/0.9923 values are traditional K-Means baselines, not AE stability.

## Historical preprocessing and manifest exposure

Thirty-six saved sample/internal-validation manifests were audited. Their union contains 208,141 unique IDs: 197,048 appear in completed fit manifests, 35,029 in evaluation/model-selection/internal-validation manifests, and 23,936 are in both role unions. There are 629,810 IDs absent from all saved manifests, including 23,883 2+ customers.

There are **zero strictly unexposed customer IDs** in the validated population: the initial global StandardScaler was fitted on all 837,951 validated customers before the original sample splits. Manifest absence therefore does not establish independence from every earlier learned preprocessing step. The manifests overlap substantially: representation-learning internal splits derive from original training samples, repeat-only samples are filtered subsets, and the partial autoencoder attempt reused an existing training manifest.

## Independent validation feasibility

Raw-source coverage is 1,048,567 transaction rows, 1,041,614 complete-case rows, no invalid complete-case dates, and a date range of **2016-08-01 through 2016-10-21**. The later fixed date of 2016-11-01 is an observation reference, not a future-data window.

| Snapshot | 2+ history customers | Manifest-unseen 2+ customers | Future window through 2016-10-21 | New repeaters with 2+ future transactions | Manifest-unseen within that new cohort |
|---|---:|---:|---:|---:|---:|
| 2016-09-01 | 64,744 | 10,057 | 50 days | 16,418 | 2,966 |
| 2016-09-15 | 128,065 | 21,319 | 36 days | 385 | 83 |
| 2016-09-30 | 140,620 | 23,700 | 21 days | 5 | 0 |

The earliest snapshot could nominally support 50,000 training plus 10,000 holdout rows, but only 4,744 2+ customers exceed that total and the 10,057 manifest-unseen holdout IDs provide only 57 IDs of headroom. It remains exposed to the historical global scaler. The future-only repeat cohort absent from previous manifests is just 2,966 customers. Later snapshots have too little future activity. A raw-only historical snapshot backtest can be rehearsed without future features, but it is not a defensible independent final test. A new external or prospective data period is required; details are in [`track_a_independent_validation_protocol.md`](track_a_independent_validation_protocol.md).

## Demographic and behavioral guardrails

Demographics are profile-only, never clustering inputs. The validated population is 72.41% `Gender_M=1` (male indicator), 27.59% `Gender_M=0` (non-M indicator), and 35.18% Tier 1 city. Age uses the 2016 convention; age 25–34 is 55.18% of validated customers. Validated customer rows have no missing values for these profile fields because upstream complete-case and age-validity rules filtered rows. Raw transaction data had 1,100 gender-missing rows, 151 location-missing rows, 3,397 DOB-missing rows, and 62,373 rows with invalid/unavailable 2016 age derivation; the age count overlaps DOB missingness and is transaction-row based.

Across the candidate segments, the largest descriptive group-share differences from the 10,000-customer evaluation base are 3.47 percentage points by age band, 2.59 points by gender indicator, and 0.86 points by city tier. Nine age-band/segment cells contain fewer than 30 customers. These distribution differences are not evidence of discrimination. Balance, spending, transaction depth, recency, and the ratio may act as socioeconomic or access proxies. GMM posterior maxima are uncalibrated; K-Means emits no confidence. Do not make adverse banking decisions from these exploratory segments alone.

## Candidate-selection rubric and joint review memo

Use a 0–4 rating per criterion (0=fails/no evidence, 1=weak, 2=mixed, 3=good, 4=strong) with the following discussion weights. It is not an automatic score-maximization rule.

| Criterion | Weight | Evidence to review |
|---|---:|---|
| Behavioral interpretability | 20% | Distinct profiles stable across seeds and periods; no unsupported personas |
| Stability | 20% | ARI on exact shared IDs across seeds and future snapshots |
| Comparable-space quality | 15% | Metrics in the same original train-fitted feature geometry; clipped/latent scores kept separate |
| Segment size and coverage | 10% | Predeclared minimum segment size and explicit eligibility coverage |
| Application usefulness | 15% | Valid exploratory use and measurable value, approved by product review |
| Fairness and guardrails | 10% | Representation, proxy analysis, small-cell handling, prohibited uses |
| Operational simplicity | 5% | Aggregation/as-of reliability, latency, monitoring cost |
| Reproducibility and inference reliability | 5% | Hashed manifests/transforms, reloads, versioned segment mapping |

**Review memo:** keep K-Means k=2 and k=3 as the unclipped reference candidates; retain clipped GMM k=2 only as a separate ratio-sensitivity challenger. The autoencoder does not advance on current evidence. Final selection requires a new independent/prospective validation period, segment-level interpretation, and joint modeling/product review. No score is assigned and no champion is frozen.

## Inference contract for eventual serving

The saved `BehavioralSegmentationPipeline.predict_features()` accepts unscaled named aggregates: `CustomerID`, `Transaction_Count`, `Last_Transaction`, `AccountBalance`, `Avg_Transaction_Amt`, `Std_Transaction_Amt`, `Total_Spend`, `Avg_Spend_Ratio`, `Weekend_Spend_Ratio`, and `Night_Spend_Ratio`. `Recency_Days` is recomputed from the saved as-of date; caller-provided recency is not used. The preprocessor selects its fixed nine-feature order by column name and reuses persisted clipping/scaler parameters.

CustomerIDs must be unique and non-null; transaction counts positive; numeric features finite; dates parseable and no later than the saved as-of date. Missing columns, invalid dates, non-finite values, and invalid log1p inputs raise a batch `ValueError`; malformed rows are not imputed. Customers below the configured threshold route to `insufficient_observed_history` with a null cluster. Output preserves CustomerID order and includes the metric-space ID.

All current candidate bundles are fixed to 2016-11-01 and reject later `Last_Transaction` values. A future serving version must freeze a new snapshot policy and fit compatible transforms/models. It must return an explicit candidate/model version, preserve model-specific cluster-ID mapping, and monitor eligibility and drift. The current output schema does not add `candidate_id`; a serving wrapper must supply it. No API integration has been performed.

## Reproducibility files

- Reports: this document, [`track_a_segment_interpretation_review.md`](track_a_segment_interpretation_review.md), and [`track_a_independent_validation_protocol.md`](track_a_independent_validation_protocol.md)
- Machine-readable candidate/artifact inventory and fairness/manifest tables: `artifacts/diagnostics/track_a/model_readiness_20261011/`
- Read-only provenance, manifest, profile, and demographic tables were produced by a repository audit utility. The audit used saved candidate bundles and did not fit models.
- Audit helper regression tests are included in the project test suite.
