# Segmentation artifacts

| Path | Contents |
|------|----------|
| `champion_v1/` | Registered Berka champion (`c_ae_lat4_k6`) |
| `champion_v1/assignments.csv` | `account_id → segment_id + confidence` |
| `champion_v1/figures/` | Sweep, PCA, heatmap, sizes, AE loss, stability |
| `champion_v1/manifest.json` | Version tag + file index |
| `../experiments/comparison_board_track_c.csv` | Full bake-off table |

Reproduce:

```bash
PYTHONPATH=src python -m customer_segmentation.pipelines.run_berka_segmentation
# or
bcs segment-berka
```
