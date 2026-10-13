# Berka new-period scoring: 1998-12-31 (frozen production_candidate_v1)

**Status: REVIEW** (PSI review threshold 0.25). Retrospective Berka data that informed development; not a prospective holdout.

- 3602 returning accounts: 81.8% stayed in their 1997-12-31 segment (ARI 0.5817).
- 898 accounts are new at 1998-12-31 and were never seen in training; median fit confidence 0.7181 vs 0.7076 for returning accounts.
- Segment-mix PSI 0.0262; features needing review: Account_Age_Days.

## Where returning accounts moved (row = earlier segment, share of that row)

| segment_before | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 0.481 | 0.461 | 0.038 | 0.004 | 0.01 | 0.005 |
| 1 | 0.0 | 0.989 | 0.003 | 0.0 | 0.008 | 0.0 |
| 2 | 0.0 | 0.0 | 0.988 | 0.0 | 0.012 | 0.0 |
| 3 | 0.0 | 0.0 | 0.024 | 0.976 | 0.0 | 0.0 |
| 4 | 0.0 | 0.222 | 0.278 | 0.056 | 0.444 | 0.0 |
| 5 | 0.167 | 0.0 | 0.0 | 0.0 | 0.0 | 0.833 |

## Segment mix

| segment_id | train_share | new_period_share | new_accounts_share |
|---|---|---|---|
| 0 | 0.326 | 0.26 | 0.673 |
| 1 | 0.369 | 0.42 | 0.031 |
| 2 | 0.136 | 0.149 | 0.134 |
| 3 | 0.162 | 0.159 | 0.155 |
| 4 | 0.005 | 0.009 | 0.004 |
| 5 | 0.002 | 0.003 | 0.003 |

## Feature drift (PSI vs training snapshot)

| feature | psi | review |
|---|---|---|
| Transaction_Count | 0.195 | False |
| Account_Balance | 0.064 | False |
| Total_Credit | 0.043 | False |
| Total_Withdrawal | 0.045 | False |
| Recency_Days | 0.0 | False |
| Account_Age_Days | 0.477 | True |
| has_loan | 0.001 | False |
| has_card | 0.0 | False |
