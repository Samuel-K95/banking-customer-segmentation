# Segmentation Layer — Build Plan

Aligned to `03-segmentation-layer.md` and current project decision (champion = Berka `c_ae_lat4_k6`).

## Done already (notebooks / reports)

- Traditional methods on A/B/C (K-Means, GMM, Hierarchical)
- Representation learning (AE → K-Means) on all three tracks
- Comparison boards for A and C
- Champion selection narrative (Berka over Santander metrics)
- Cluster profiles (technical + behavioural)

## Gaps this build closes

| Gap | Deliverable |
|-----|-------------|
| Logic trapped in notebooks | Modular `src/customer_segmentation/models/*` + pipeline |
| No segment-count memo | `reports/selection/segment_count_justification.md` |
| Champion not registered | `configs/project/champion.yaml` + `artifacts/segmentation/champion/` |
| No assignment export | `account_id → segment_id (+ confidence)` CSV |
| Soft membership unused | Distance-to-centroid confidence on champion |
| Feature config | `configs/project/features.yaml` + leakage-safe model config |
| Production CLI | `bcs segment-berka-leakage-safe` → `promote-production` → `verify-champion` |

## Package layout

```text
src/customer_segmentation/
  models/
    metrics.py          # sil / DB / CH
    stability.py        # multi-seed + subsample ARI
    traditional.py      # K-Means, GMM, Hierarchical sweeps
    autoencoder.py      # AE train + cluster latent
    champion.py         # ChampionModel.load / encode / score
    segment_intel.py    # naming rules, personas, decision map
    viz.py              # save PNG figures
    registry.py         # write champion artifacts + manifest + catalog
  pipelines/
    run_berka_segmentation_leakage_safe.py
    promote_production.py
    verify_champion.py / restore_champion.py
  features/build_berka_segmentation_features.py
  data/berka_dataset_loader.py
  serve/app.py + routers/            # FastAPI → ChampionModel
  cli.py                             # bcs restore/verify/serve/promote/...
```

## Artifact layout

```text
artifacts/
  experiments/
    comparison_board_track_c.csv          # full bake-off table
  segmentation/
    champion/
      manifest.json
      metrics.json
      assignments.csv                     # account_id, segment_id, confidence
      segment_profiles.csv
      encoder.pt
      kmeans.joblib
      scaler.joblib
      feature_columns.json
      figures/
        kmeans_sweep.png
        pca_segments.png
        centroid_heatmap.png
        size_bars.png
        ae_recon_loss.png
        stability_ari.png
```

## Incremental phases

1. **Modules + configs** — done
2. **Berka pipeline CLI run** → populate artifacts — done
3. **Docs** — segment count justification + champion.yaml freeze — done
4. **Docker** — `scripts/run_segmentation_layer.sh` + `docker compose` service `segmentation-layer` (same image as data-layer)
5. Later (optional): Indian Bank / Santander pipelines using the same modules

## Docker

```bash
docker compose build
docker compose run --rm data-layer            # if Berka CSVs missing
docker compose run --rm segmentation-layer    # leakage-safe train → promote → decision → verify
```

Image installs CPU PyTorch for the AE track. Volumes mount `data/`, `artifacts/`, `configs/`, `reports/`.

## Selection policy (declared)

```text
score = 0.25*norm(silhouette)
      + 0.25*norm(stability_ARI)
      + 0.20*interpretability_rubric
      + 0.15*size_sanity
      + 0.15*application_distinctness
```

Champion for the product is **not** max(silhouette). Behavioral coverage + banking applications decide after metrics are recorded. See `reports/champion/00_why_this_champion.md`.
