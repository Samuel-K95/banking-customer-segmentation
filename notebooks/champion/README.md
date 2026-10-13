# Champion notebook

Notebook in this folder documents the **winning champion**:

**Leakage-safe AE (lat=4) + K-Means k=6**

| Resource | Path |
|----------|------|
| Model | `artifacts/segmentation/champion/` |
| Decisions | `artifacts/decision/champion/` |
| Reports | `reports/champion/` |
| One-pager | `CURRENT_CHAMPION.md` (repo root) |

```python
from customer_segmentation.models.champion import ChampionModel
champ = ChampionModel.load()  # artifacts/segmentation/champion
```

If the notebook still mentions older `product/` or `production/` paths, treat those as outdated — the live default is `champion/`.
