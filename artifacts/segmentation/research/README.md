# Research (not the served champion)

`leakage_safe_source/` is the leakage-safe training run that gets **promoted** into:

→ `artifacts/segmentation/champion/`

Also contains the parallel **k=2** research twin. For CRM/decision use the champion folder, not this one.

```bash
PYTHONPATH=src bcs promote-production   # copy source → champion + refresh freeze
```
