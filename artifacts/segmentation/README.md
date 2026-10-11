# Segmentation artifacts

| Path | Contents |
|------|----------|
| `champion_v1/` | Registered Berka champion (`c_ae_lat4_k6`) |
| `champion_v1/assignments.csv` | `account_id → segment_id + confidence` |
| `champion_v1/segment_catalog.json` | Frozen names, personas, decision plays |
| `champion_v1/decision_map.csv` | Segment → play / owner / KPI / guardrail |
| `champion_v1/encoder.pt` + `kmeans.joblib` + `scaler.joblib` | Scorable model bundle |
| `champion_v1/figures/` | Sweep, PCA, heatmap, sizes, AE loss, stability |
| `champion_v1/manifest.json` | Version tag + file index |
| `../experiments/comparison_board_track_c.csv` | Full bake-off table |

## Modular load / score

```python
from customer_segmentation.models.champion import ChampionModel

champ = ChampionModel.load("artifacts/segmentation/champion_v1")
print(champ.summary())
scored = champ.score_interpretable(interpretable_df)  # segment_id, name, play, confidence
```

CLI:

```bash
PYTHONPATH=src bcs champion-info
PYTHONPATH=src bcs score-berka --limit 100
PYTHONPATH=src bcs segment-berka   # full bake-off + re-register
```
