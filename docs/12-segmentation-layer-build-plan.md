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
| No segment-count memo | `reports/segment_count_justification.md` |
| Champion not registered | `configs/champion.yaml` + `artifacts/segmentation/champion_v1/manifest.json` |
| No assignment export | `customer_id → segment_id (+ confidence)` CSV |
| Figures not persisted | `artifacts/segmentation/champion_v1/figures/*.png` |
| Soft membership unused | Distance-to-centroid confidence on champion |
| Feature config empty | `configs/features.yaml` for Berka model/profile columns |
| CLI stub | `bcs segment-berka` |

## Package layout

```text
src/customer_segmentation/
  models/
    metrics.py          # sil / DB / CH
    stability.py        # multi-seed + subsample ARI
    traditional.py      # K-Means, GMM, Hierarchical sweeps
    autoencoder.py      # AE train + cluster latent
    viz.py              # save PNG figures
    registry.py         # write champion artifacts + manifest
  pipelines/
    run_berka_segmentation.py
  features/build_berka_features.py   # + has_loan / has_card
  data/berka_dataset_loader.py       # berka-dataset path + loan/card
  cli.py                             # bcs segment-berka
```

## Artifact layout

```text
artifacts/
  experiments/
    comparison_board_track_c.csv          # full bake-off table
  segmentation/
    champion_v1/
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
docker compose run --rm segmentation-layer    # bake-off + champion_v1
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

Champion for the product is **not** max(silhouette). Behavioral coverage + banking applications decide after metrics are recorded. See `reports/champion_recommendation.md`.
