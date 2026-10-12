# Berka Exploration Findings — AE + Extra Clustering Models

Experiment run: `artifacts/experiments/berka_explore/`  
Script: `python -m customer_segmentation.pipelines.run_berka_explore`

This does **not** automatically replace the registered champion (`c_ae_lat4_k6`). It is a bake-off to see what else is worth promoting.

---

## What was tried

### Classical / simple
- K-Means, MiniBatch K-Means, GMM (k=6 + BIC sweep k=3..8)
- Birch, Hierarchical (Ward + Average), Spectral

### Density-based
- HDBSCAN (two settings), DBSCAN (eps from kNN heuristic)

### Linear representation baseline
- PCA(2/4/8) → K-Means k=6

### Autoencoder variants (then multiple heads)
| AE config | Idea |
|-----------|------|
| lat=2 / 4 / 8 / 16 | compression sweep |
| hidden=64 | more capacity |
| denoising noise 0.10 / 0.20 | DAE |
| longer train (80 epochs) | training hygiene |
| Heads | K-Means, GMM, MiniBatch, HDBSCAN |

---

## Top signals (by silhouette)

| Rank | Run | Silhouette | Caveat |
|-----:|-----|----------:|--------|
| 1 | `hier_average_k6` | 0.656 | **Reject for ops** — one segment holds ~99% of accounts (`max_frac≈0.99`). Metric illusion. |
| 2 | `ae_lat4_h64_hdbscan` | 0.575 | AE + density; may label noise |
| 3 | `ae_lat4_long_hdbscan` | 0.569 | Longer AE train helps |
| 4–5 | `ae_lat4_h64_gmm` / `ae_lat4_long_gmm` | ~0.52–0.53 | Soft membership, weaker DB than K-Means |
| … | `ae_lat4_h32_kmeans` (champion-like) | ~0.47 | Still strong, stable, interpretable |

**Takeaway:** Higher silhouette alone is not enough. Prefer runs that also have sane sizes, good DB, and profiles you can name.

---

## What improved the AE

1. **Latent dim = 4** still looks right — lat=2 is OK but limited; lat=8/16 reconstruct better but cluster less cleanly (identity / noise dims).
2. **Wider encoder (hidden=64)** and **longer training** both beat the default AE+HDBSCAN and lift GMM heads.
3. **Denoising AE** is competitive but did not beat a well-trained plain AE here.
4. **HDBSCAN on AE latent** often scores high — useful as an experimental density head; watch noise fraction and whether segments stay bankable.
5. **PCA + K-Means** ≈ raw K-Means — the AE is doing real non-linear work beyond PCA.

---

## Recommendation

| Keep as champion (for now) | Why |
|----------------------------|-----|
| **AE(lat=4) + K-Means k=6** | Best balance of metrics, stability history, fixed k for CRM, clear profiles |

| Promote next if you want a v2 candidate | Why |
|----------------------------------------|-----|
| **AE(lat=4, hidden=64, longer train) + K-Means** | Same head as champion, better encoder |
| **AE + HDBSCAN** | Only if you accept a noise/review bucket and variable k |

| Interesting but secondary | Why |
|---------------------------|-----|
| Hierarchical average | Metric spike — validate sizes/profiles before any promotion |
| AE + GMM | Soft membership for borderline customers |

---

## Re-run

```bash
PYTHONPATH=src python -m customer_segmentation.pipelines.run_berka_explore
```

Outputs:
- `artifacts/experiments/berka_explore/comparison_board_explore.csv`
- `artifacts/experiments/berka_explore/family_summary.csv`
- `artifacts/experiments/berka_explore/summary.json`
- `artifacts/experiments/berka_explore/figures/`
