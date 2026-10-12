# CURRENT CHAMPION (read this first)

## The winning model

| Field | Value |
|-------|--------|
| **Name** | Leakage-safe Autoencoder + K-Means |
| **Run ID** | `c_ae_lat4_k6_leakage_safe` |
| **Architecture** | AE latent dim = **4**, K-Means **k = 6** |
| **Dataset** | Berka (Track C) |
| **As-of date** | `1997-12-31` (train/holdout, tenure ≥ 365 days) |
| **Freeze status** | **PROVISIONAL** (engineering-ready candidate; not a full bank freeze) |
| **Model version** | `production_candidate_v1` |

## Where it lives (one place each)

| What | Path |
|------|------|
| **Model bundle** | [`artifacts/segmentation/champion/`](artifacts/segmentation/champion/) |
| **Decision plays** | [`artifacts/decision/champion/`](artifacts/decision/champion/) |
| **Reports** | [`reports/champion/`](reports/champion/) |
| **Master description** | [`reports/champion/MASTER_CHAMPION_SEGMENTATION_AND_DECISIONS.md`](reports/champion/MASTER_CHAMPION_SEGMENTATION_AND_DECISIONS.md) |
| **Notebook** | [`notebooks/champion/`](notebooks/champion/) |
| **Config** | [`configs/project/champion.yaml`](configs/project/champion.yaml) |

## What is NOT the champion

| Path | Role |
|------|------|
| `artifacts/segmentation/archive/fullbook_demo/` | Old full-book demo — **archive only** |
| `artifacts/segmentation/research/leakage_safe_source/` | Training source / k=2 twin — **not served** |
| `artifacts/decision/archive/` | Old decision packs — **archive only** |

## Quick commands

```bash
PYTHONPATH=src bcs champion-info
PYTHONPATH=src bcs freeze-status
PYTHONPATH=src bcs verify-champion
PYTHONPATH=src bcs restore-champion
PYTHONPATH=src bcs decision-berka
```

```python
from customer_segmentation.models.champion import ChampionModel
champ = ChampionModel.load()  # always loads artifacts/segmentation/champion
```
