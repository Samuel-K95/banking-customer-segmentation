# Segment Count Justification — Leakage-Safe Berka Champion

**Selected k = 6** for `c_ae_lat4_k6_leakage_safe` (Autoencoder latent_dim=4 → K-Means).  
**Contract:** as_of `1997-12-31`, tenure ≥ 365 days, train-only scaler/AE/K-Means (n_train=2,701 / n_holdout=901).

Numbers below are from the **served** leakage-safe champion after evaluation enrichment (`bcs enrich-champion-eval`), using shared `evaluate_clustering` on holdout latents / same-matrix baselines.

---

## Candidate k = 6

### Why not 4
Merges distinct banking plays that need different treatments:
- Loan holders and card-only savers collapse into a single “has a product” blob.
- Dormant / stressed tails mix with mass basic accounts, killing retention vs care actions.

### Why not 5
Still blends the financially stressed (negative balance) group with active basic accounts. Overdraft risk needs a separate **exception queue**, not a mass CRM cell.

### Why not 2 (research twin on the same matrix)
Holdout silhouette for K-Means k=2 ≈ **0.297** vs AE+k=6 ≈ **0.459**. k=2 is an honest scale/activity split for rigor, **not** a CRM replacement — personas (loan / card / stress / dormant) disappear.

### Why not 7–8
Extra splits on this ~3.6k eligible book create tiny unstable cells and do not improve operable CRM coverage beyond the six named personas.

### Metrics at k = 6 (leakage-safe holdout / train protocols)

| Metric | Value |
|--------|------:|
| Holdout silhouette (AE latent) | **0.459** |
| Holdout Davies–Bouldin | **0.623** |
| Holdout Calinski–Harabasz | **852** |
| Train silhouette (AE latent) | 0.460 |
| Multi-seed ARI (AE latent K-Means) | **0.999** (min 0.997) |
| Subsample ARI (75%, 5 rounds) | **0.978** (min 0.950) |
| Raw K-Means k=6 holdout silhouette | 0.374 |
| GMM k=6 holdout silhouette | 0.322 |
| Hierarchical k=6 train silhouette | 0.363 |
| K-Means k=2 holdout silhouette | 0.297 |

### Business constraints
- Four mass segments map to onboarding / cross-sell / loan servicing / card-centric plays.
- Two tiny segments (stressed ≈0.5%, dormant ≈0.2%) are kept as **exception queues**, not mass campaigns (`size_sanity` fails the 3% floor by design).

### Decision
**k = 6** balances holdout separation, dual stability protocols (seed + subsample), same-matrix traditional baselines (AE wins silhouette), and distinct banking applications required by brief 04.
