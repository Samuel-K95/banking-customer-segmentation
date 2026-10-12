# Production OWNER / DISPONENT policy

**Status:** published for engineering freeze · **Compliance sign-off:** PENDING

## Grain

The production champion scores **accounts**, not clients. Berka `disp` can attach
OWNER and DISPONENT rows to the same `account_id`. Segment membership must not be
interpreted as a person-level label without an approved mapping policy.

## Rules

1. Features and `segment_id` are account-grain only.
2. Do not use `segment_id` for credit eligibility, pricing, or collections aggression.
3. Joint accounts (`n_accounts_with_multiple_disp` > 0 possible): suppress person-targeted
   offers derived solely from the account segment until an OWNER-primary policy is approved.
4. Exception-queue segments (stressed / dormant / tiny cells) require human review.
5. Geography: monitor district concentration; do not target by district from segment alone.

## Snapshot from fairness pack

- District join available: `True`
- DISP type counts: `{'OWNER': 4500, 'DISPONENT': 869}`
- Accounts with multiple disp rows: `869`

## Rollback / misuse

If segment outputs are used for adverse decisions, revoke serving access and roll back
to the previous hashed production bundle.
