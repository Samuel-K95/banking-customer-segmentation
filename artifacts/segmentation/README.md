# Segmentation artifacts

## THIS IS THE CHAMPION

**Folder:** [`champion/`](champion/)

| | |
|--|--|
| Model | AE (latent=4) + K-Means **k=6** |
| Run | `c_ae_lat4_k6_leakage_safe` |
| Contract | as-of `1997-12-31`, leakage-safe train/holdout |
| Status | freeze **PROVISIONAL** |

```bash
PYTHONPATH=src bcs champion-info
PYTHONPATH=src bcs freeze-status
```

## Not the champion

| Folder | Role |
|--------|------|
| [`research/leakage_safe_source/`](research/leakage_safe_source/) | Source twin used to promote into `champion/`; includes k=2 research split |
| [`archive/fullbook_demo/`](archive/fullbook_demo/) | Pre-leakage full-book demo — **archived** |
