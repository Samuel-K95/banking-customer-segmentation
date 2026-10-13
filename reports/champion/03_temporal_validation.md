# Production temporal validation (best-effort)

**Verdict: PASS_WITH_LIMITATIONS**

1998 check is retrospective and uses the same Berka vintage that informed development; it is not an independent prospective holdout.

## Setup

- Train / serving as-of: `1997-12-31`
- Future as-of: `1998-12-31`
- Common accounts: **3602**
- Model version: `production_candidate_v1`

## Results

- Replay ARI (1997 features vs stored production labels): **1.0000**
- Temporal ARI (1997 vs 1998 assignments, frozen model): **0.5817**
- Label agreement rate: **0.8179**

## Segment size persistence

| segment_id | n_1997 | n_1998 | n_stayed | med_balance_97 | med_balance_98 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1176 | 567 | 566 | 26425.6 | 23256.7 |
| 1 | 1329 | 1860 | 1314 | 33119.8 | 37874.2 |
| 2 | 489 | 551 | 483 | 39814.5 | 49966.4 |
| 3 | 584 | 576 | 570 | 55404.3 | 64477.8 |
| 4 | 18 | 37 | 8 | -3133.0 | -5284.5 |
| 5 | 6 | 11 | 5 | 10074.1 | 10000.0 |

## Freeze implication

This gate is **PASS_WITH_LIMITATIONS**. Overall production freeze remains
**PROVISIONAL** until a true prospective/external vintage is evaluated.
