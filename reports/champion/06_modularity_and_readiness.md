# Champion modularity & readiness checklist

**Verdict:** modular and thoroughly prepared for the project production-candidate path.  
**Freeze:** still **PROVISIONAL** (not a full bank go-live stamp).

## Modular layers

| Layer | Package / path | Role |
|-------|----------------|------|
| Features | `features/build_berka_segmentation_features.py` | 8 CRM cols; as-of leakage-safe builder |
| Model | `models/champion.py` + `tabular_ae.py` | Load/score AE+k=6 bundle |
| Registry | `models/registry.py` | Persist champion artifacts |
| Pipelines | `pipelines/` | Train leakage-safe → promote → fairness/temporal/freeze |
| Decision | `decision/` | Profiles, explain, fairness, multi-domain plays |
| Serve | `serve/app.py` | API with `model_version` + `as_of_date` contract |
| CLI | `cli.py` (`bcs`) | `champion-info`, `decision-berka`, `freeze-status`, `promote-production` |
| Config | `configs/project/champion.yaml` | Single source of truth for IDs/paths |
| Docs | `CURRENT_CHAMPION.md` + `reports/champion/` | Human-facing clarity |

## Artifact contract (one folder each)

- Model: `artifacts/segmentation/champion/`
- Decisions: `artifacts/decision/champion/`
- Reports: `reports/champion/`
- Notebook: `notebooks/champion/`

## Preparedness checks (engineering)

| Check | Status |
|-------|--------|
| Default `ChampionModel.load()` → champion folder | Pass |
| Encoder + scaler + kmeans + catalog present | Pass |
| Serving contract + freeze JSON + fairness pack | Pass |
| Decision layer regenerated for champion | Pass |
| Holdout replay + decision unit tests | Pass |
| Notebook / READMEs point at champion (not archive) | Pass |
| API rejects wrong `model_version` / `as_of_date` | Pass |

## Still not “full bank frozen”

- Compliance / model-risk human sign-off pending  
- Temporal validation is retrospective Berka 1998 only  
- Segment must not drive credit eligibility / pricing  

## How to re-verify

```bash
PYTHONPATH=src bcs champion-info
PYTHONPATH=src bcs freeze-status
PYTHONPATH=src pytest tests/test_champion.py tests/test_production_champion_replay.py tests/test_decision_layer.py -q
```
