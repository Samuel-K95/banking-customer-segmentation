# track_d_segmentation_v1 (track_d)

**Selected:** `gmm_k3` (runner-up `gmm_k4`). All gates met.

Every candidate was fit on the same training customers and scored on the same holdout in the same standardized feature space.


## Candidates

| candidate | holdout_silhouette | holdout_davies_bouldin | seed_ari_mean | subsample_ari_mean | min_segment_fraction | max_segment_fraction |
|---|---|---|---|---|---|---|
| kmeans_k2 | 0.4882 | 1.35 | 0.9946 | 0.991 | 0.1984 | 0.8016 |
| kmeans_k3 | 0.4274 | 1.3883 | 0.996 | 0.9645 | 0.034 | 0.7529 |
| kmeans_k4 | 0.3503 | 1.403 | 1.0 | 0.956 | 0.0263 | 0.6644 |
| kmeans_k5 | 0.1948 | 1.5665 | 0.9802 | 0.8831 | 0.0253 | 0.4728 |
| kmeans_k6 | 0.2172 | 1.5886 | 0.9626 | 0.8045 | 0.0214 | 0.4844 |
| kmeans_k7 | 0.1411 | 1.6037 | 0.9864 | 0.7479 | 0.0185 | 0.3356 |
| kmeans_k8 | 0.147 | 1.5062 | 0.9792 | 0.7688 | 0.0126 | 0.3356 |
| gmm_k2 | 0.3424 | 1.7678 | 0.9992 | 0.9616 | 0.3161 | 0.6839 |
| gmm_k3 | 0.2728 | 1.7636 | 1.0 | 0.9052 | 0.0584 | 0.5992 |
| gmm_k4 | 0.2591 | 1.602 | 0.9667 | 0.9781 | 0.0447 | 0.5788 |
| gmm_k5 | 0.1525 | 2.4422 | 0.7494 | 0.7586 | 0.0331 | 0.4757 |
| gmm_k6 | 0.1393 | 2.8931 | 0.7768 | 0.7009 | 0.0175 | 0.4533 |
| gmm_k7 | 0.0892 | 2.6687 | 0.8032 | 0.7831 | 0.0136 | 0.4533 |
| gmm_k8 | 0.018 | 3.1574 | 0.8718 | 0.7415 | 0.0097 | 0.4212 |
| ae_kmeans_k2 | 0.5454 | 1.302 | 0.7735 | 0.9925 | 0.1352 | 0.8648 |
| ae_kmeans_k3 | 0.4677 | 1.4058 | 0.7779 | 0.9881 | 0.0311 | 0.8084 |
| ae_kmeans_k4 | 0.4169 | 1.4505 | 0.5735 | 0.9449 | 0.0282 | 0.7802 |
| ae_kmeans_k5 | 0.3319 | 1.5366 | 0.6277 | 0.9194 | 0.0185 | 0.6654 |
| ae_kmeans_k6 | 0.3238 | 1.6543 | 0.4894 | 0.7188 | 0.0156 | 0.6615 |
| ae_kmeans_k7 | 0.1144 | 1.7578 | 0.3444 | 0.7511 | 0.0117 | 0.4387 |
| ae_kmeans_k8 | 0.1152 | 1.8264 | 0.3409 | 0.9077 | 0.0068 | 0.4407 |

## Segments

- **0 — high n_merchants / high max_amount**: 1362 customers (33.1%); high n_merchants (+0.60 sd), high max_amount (+0.57 sd), high std_amount (+0.55 sd)
- **1 — low active_days / low txn_count**: 272 customers (6.6%); low active_days (-2.65 sd), low txn_count (-2.65 sd), low share_transport (-2.36 sd)
- **2 — high active_days / high txn_count**: 2478 customers (60.3%); high active_days (+0.59 sd), high txn_count (+0.58 sd), high share_transport (+0.51 sd)

## Explainability

A depth-3 decision tree on readable features reproduces the segment for 96.5% of holdout customers (rules in `surrogate_tree_rules.txt`).

## Fairness

- `age_band`: Cramér's V 0.0317, representation ratio 0.748–1.125 (groups with ≥30 customers)
- `gender`: Cramér's V 0.0281, representation ratio 0.91–1.072 (groups with ≥30 customers)
- Demographics alone predict the segment for 59.9% of holdout customers vs 59.9% by always guessing the largest segment.

## Outcome validation (not used in fitting)

- `any_fraud` overall 36.1%; by segment 0: 65.6% [63.0%–68.0%], 1: 86.4% [81.8%–90.0%], 2: 14.3% [13.0%–15.8%]; chi-square p=1.6e-287, Cramér's V 0.5667.

Outcome rates describe the segments; they are not predictions.

**Caveat:** Fraudulent payments are part of the spend features (amounts and category shares), so part of any association between segments and fraud is mechanical, not independent.
