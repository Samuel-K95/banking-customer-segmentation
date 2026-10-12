# Data

| Path | Role |
|------|------|
| [`raw/`](raw/) | Source extracts (`berka-dataset/`, `santander/`, Indian Bank) |
| [`processed/track_c/`](processed/track_c/) | Berka interpretable + model matrices |
| [`processed/track_a/`](processed/track_a/) | (reserved) Indian Bank matrices |
| [`processed/track_b/`](processed/track_b/) | (reserved) Santander matrices |
| [`processed/berka_snapshot/`](processed/berka_snapshot/) | Leakage-safe as-of snapshots (when generated) |
| [`external/`](external/) | Optional external lookups |

Feature builders write to `processed/<track>/` via `customer_segmentation.features.io.track_outputs`.
