# CHAMPION MODEL BUNDLE

**This directory is the winning / current champion.**

- Run ID: `c_ae_lat4_k6_leakage_safe`
- Method: Autoencoder (latent=4) → K-Means (k=6)
- Leakage-safe: fixed as-of `1997-12-31`, tenure ≥365, train-only scaler/AE/K-Means
- Freeze: **PROVISIONAL** (`freeze_readiness.json`)
- Model version: `production_candidate_v1`

## Required executables (must be present on clean checkout)

| File | Role |
|------|------|
| `encoder.pt` | Frozen AE encoder |
| `scaler.joblib` | Frozen StandardScaler |
| `kmeans.joblib` | Frozen K-Means on latent space |
| `serving_contract.json` | `model_version`, `as_of_date`, SHA-256 fingerprints |

```bash
bcs verify-champion     # integrity + SHA + load + API smoke
bcs restore-champion    # repair via promote / retrain if incomplete
```

Sibling decision packs: `artifacts/decision/champion/`  
Human reports: `reports/champion/`
