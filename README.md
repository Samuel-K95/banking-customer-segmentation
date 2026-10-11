# Banking Customer Segmentation and Behavioral Intelligence

Week 8 project scaffold for banking **customer segmentation** and **behavioral intelligence**.

## Branches

| Branch | Use |
|--------|-----|
| `main` | Final only — **do not push directly** |
| `staging` | Integration |
| `dev-tsegaye` | Tsegaye |
| `dev-mahider` | Mahider |
| `dev-samuel` | Samuel |
| `dev-biruk` | Biruk |

Flow: `dev-<name>` → PR → `staging` → PR → `main`

## Setup

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
```

```bash
uvicorn customer_segmentation.serve.app:app --reload --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

## Layout

```text
banking-customer-segmentation/
├── artifacts/       # pipeline outputs (gitignored content)
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── experiments/     # model runs, comparison notes, tracking exports
├── frontend/        # React + Vite (API client)
├── notebooks/       # EDA / exploration (logic still belongs in src/)
├── pipelines/       # orchestration (e.g. Airflow DAGs)
├── reports/         # writeups, figures for submission
├── src/customer_segmentation/
└── tests/
```

## Docker Setup

One image, two layers (see `docker-compose.yml`).

### Data layer (ingest + features)

```bash
docker compose build
docker compose run --rm data-layer
```

Equivalent one-off (legacy):

```bash
docker build -t bcs-platform .
docker run --rm -v "%CD%/data:/app/data" -v "%USERPROFILE%/.kaggle:/root/.kaggle" bcs-platform
```

### Segmentation layer (Track C bake-off + champion_v1)

Requires Berka CSVs already under `data/raw/berka-dataset/` (run the data layer first, or mount existing data).

```bash
docker compose run --rm segmentation-layer
```

Outputs land on the host via volume mounts:

- `artifacts/experiments/comparison_board_track_c.csv`
- `artifacts/segmentation/champion_v1/` (assignments, metrics, figures, manifest)
- `data/processed/track_c_berka_*.csv`

Without Compose:

```bash
docker run --rm ^
  -v "%CD%/data:/app/data" ^
  -v "%CD%/artifacts:/app/artifacts" ^
  -v "%CD%/configs:/app/configs" ^
  -e MPLBACKEND=Agg ^
  bcs-platform ./scripts/run_segmentation_layer.sh
```
