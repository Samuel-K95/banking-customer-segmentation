# CHAMPION MODEL BUNDLE

**This directory is the winning / current champion.**

- Run ID: `c_ae_lat4_k6_leakage_safe`
- Method: Autoencoder (latent=4) → K-Means (k=6)
- Leakage-safe: fixed as-of `1997-12-31`, tenure ≥365, train-only scaler/AE/K-Means
- Freeze: **PROVISIONAL** (`freeze_readiness.json`)

Key files: `encoder.pt`, `kmeans.joblib`, `scaler.joblib`, `assignments.csv`,
`segment_catalog.json`, `serving_contract.json`, `manifest.json`.

Sibling decision packs: `artifacts/decision/champion/`  
Human reports: `reports/champion/`
