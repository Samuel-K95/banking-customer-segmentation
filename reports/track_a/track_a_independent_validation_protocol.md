# Track A Independent Validation Protocol and Feasibility

**Status:** no strictly independent final holdout exists in the current data vintage. No new evaluation, model fit, or champion selection was performed.
**Feasibility audit:** `artifacts/diagnostics/track_a/model_readiness_20261011/`

## Customer-ID exposure audit

The audit enumerated 36 saved sampling and internal-validation manifests created across the historical baseline, representation, robustness, train-fitted clustering, and matched autoencoder experiments. Pairwise overlaps and row-level usage flags are in `manifest_pairwise_overlaps.csv` and `manifest_customer_usage.csv`.

| Exposure definition | Distinct IDs in validated population |
|---|---:|
| Validated customer population | 837,951 |
| Appearing in any saved manifest | 208,141 |
| Appearing in a completed training/fit manifest | 197,048 |
| Appearing in a completed evaluation/model-selection/internal-validation manifest | 35,029 |
| Absent from all saved manifests | 629,810 |
| Manifest-absent within the 2+ population | 23,883 of 141,502 |
| Strictly untouched by any previous learned preprocessing | **0** |

The zero strictly untouched count follows from the saved initial StandardScaler metadata: `n_samples_seen_ = 837951`, fit before the original baseline manifests were made. The 629,810 manifest-absent customers are unexposed to the sampled candidate-level train/evaluation sets, but they are not untouched under a definition that includes learned preprocessing. The sampled-ID union is useful for sensitivity/design work, not a final-test claim.

The inventory includes a first incomplete sampling attempt and a partial AE fit. The incomplete sampling attempt has the same manifest hashes as the completed train-fitted run; it adds no distinct IDs and did not fit a candidate. The partial AE fit used 45,000 internal-train IDs plus 5,000 internal-validation IDs from an existing 50,000-row training manifest. These exposures are recorded separately from completed runs.

## Source coverage and temporal feasibility

The raw source contains 1,048,567 rows, 1,041,614 rows retained by the complete-case rule, and no invalid retained transaction dates. The date range is **2016-08-01 to 2016-10-21**, only 82 calendar days. The saved inference pipelines use a historical `2016-11-01` as-of date; no transactions exist after the latest observed date.

| Snapshot cutoff | 2+ customers by cutoff | 2+ IDs absent from saved manifests | Future window | Snapshot-eligible with any future transaction | With 2+ later events | New post-cutoff customers with 2+ events | Of those, manifest-absent |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2016-09-01 | 64,744 | 10,057 | 50 days | 8,519 | 638 | 16,418 | 2,966 |
| 2016-09-15 | 128,065 | 21,319 | 36 days | 2,810 | 45 | 385 | 83 |
| 2016-09-30 | 140,620 | 23,700 | 21 days | 205 | 0 | 5 | 0 |

The Sep 1 cohort's 2+ history is shallow: 59,283 of 64,744 (91.57%) have exactly two transactions. It can nominally supply a 50,000-row train plus 10,000-row evaluation sample, with only 4,744 extra eligible IDs. A manifest-absent snapshot holdout of 10,057 is only 57 rows above 10,000 and remains exposed to the initial full-population scaler. The future-only repeat cohort has 16,418 customers, but only 2,966 are absent from all saved manifests. Later snapshots have very little follow-up volume.

### Historical snapshot backtest feasibility

A raw-only historical snapshot backtest is mechanically constructible without future leakage: freeze `2016-09-01` before selecting candidates, rebuild every input using complete-case transactions dated on/before that date, set recency relative to the cutoff, and use post-cutoff events only for follow-up outcomes. It could rehearse a raw train-fitted pipeline, but it is **not** a defensible independent final validation because the same vintage is short, the future window is limited, most new repeaters were previously manifested, and all validated IDs contributed to historical global preprocessing. Changing a random seed or only changing the as-of date would not create an out-of-time experiment.

This retrospective backtest was **not executed**. The date-by-date coverage, history sufficiency, and demographic distributions are in `oot_snapshot_feasibility.csv`, `oot_snapshot_history_sufficiency.csv`, `raw_source_daily_coverage.csv`, and `oot_demographic_population_shift.csv` under the feasibility audit directory.

## Prospective validation protocol

A final decision requires a new future data vintage or external data source, collected after candidate families and evaluation rules are frozen. Before opening any results, agree on and version the following protocol:

1. **Observation policy:** specify a fixed snapshot cutoff and lookback window. Construct every aggregate and recency value solely from complete-case events at or before the cutoff; confirm the cutoff is independent of evaluation data.
2. **Population:** define customer-level history eligibility using the snapshot only. Record source coverage, excluded customers, incomplete transactions, and the number routed to insufficient history.
3. **Split:** separate CustomerIDs before fitting any learned preprocessing. Keep all rows/events for each customer in one split. Persist ordered IDs/hashes and ensure the final evaluation set is disjoint from all candidate training and model-selection IDs.
4. **Fit discipline:** fit clipping (if preregistered) and scalers on training customers only. Save the exact transformer, feature order, fitted-ID hash, model version, and segment mapping.
5. **Candidate set:** preregister unclipped nine-feature K-Means k=2/k=3 and ratio-clipped diagonal GMM k=2 as separate feature geometries. Do not add candidates after inspecting final holdout metrics.
6. **Evaluation:** compare original-space metrics only within the same transformed feature geometry. Use identical ordered holdout IDs for all candidates and ARI stability. Report counts, proportions, minimum segment size, quantile profiles, and demographic representation.
7. **Future outcomes:** hold future-window transactions out of every feature, transform, assignment, and model-selection step. Use those events only for predeclared behavior follow-up (e.g., subsequent transaction count/spend). Explicitly treat lack of follow-up as censored/unknown rather than inactivity.
8. **Fairness and action limits:** predefine age/gender/city-tier descriptions and minimum group sizes; suppress small cells; review proxy pathways. Segment membership alone must not drive adverse banking decisions.
9. **Sample size and decision:** Product and analytics reviewers must specify the use case and minimum segment/subgroup precision before selecting evaluation size. If the new period cannot satisfy the preregistered thresholds, defer and collect more data rather than reuse the development holdout.

The current historic source can be used for implementation rehearsal, but only new external or prospective data can resolve the independent-validation limitation. No model training or evaluation is part of this protocol document.
