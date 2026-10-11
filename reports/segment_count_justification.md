# Segment Count Justification — Track C (Berka Champion)

**Selected k = 6** for `c_ae_lat4_k6` (Autoencoder latent_dim=4 → K-Means).

---

## Candidate k = 6

### Why not 4
Merges distinct banking plays that need different treatments:
- Loan holders (100% `has_loan`) and card-only savers (100% `has_card`) collapse into a single “has a product” blob.
- Dormant accounts (recency ≈ 539 days) get mixed with low-activity new accounts, killing retention vs onboarding actions.

### Why not 5
Still blends the financially stressed (negative balance) group with active basic accounts. Overdraft risk needs a separate play from “loyal, active, no products.”

### Why not 7–8
Silhouette and DB stop improving meaningfully past k=6 on this matrix. Extra splits are unstable under multi-seed K-Means (segments dissolve / swap) and are too small for operable CRM campaigns on a 4,500-account book.

### Metrics at k = 6 (champion AE lat=4)
From the modular pipeline (`artifacts/segmentation/champion_v1/metrics.json`):

| Metric | Value |
|--------|------:|
| Silhouette | 0.466 |
| Davies–Bouldin | 0.659 |
| Calinski–Harabasz | 3,775 |
| K-Means multi-seed ARI | 0.964 |
| AE-latent K-Means multi-seed ARI | 0.996 |

### Business constraints
- All six segments map to distinct treatments: onboard, cross-sell first product, loan servicing, win-back/close, premium card / first loan, risk/service.
- Two segments are intentionally small: negative-balance / stressed (~39 accounts in the book) and deep-dormant (recency > 365 days is rare — only ~8–16 accounts). The automated `size_sanity` flag fails the 3% floor for those groups; they are kept because they are real behavioral tails with distinct banking actions, not metric noise. See `figures/size_bars.png`.

### Decision
**k = 6** balances metric plateaus, dendrogram structure (hierarchical prior), stability, and distinct banking applications required by the Week 8 brief.
