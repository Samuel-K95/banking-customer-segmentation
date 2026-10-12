# Banking Customer Segmentation and Behavioral Intelligence

Repository for banking **customer segmentation** and **behavioral intelligence** experiments, data processing, and evaluation.

## Documentation

- [Reports index](reports/README.md)
- [Engineering documentation index](docs/README.md)
- [Track A feature dictionary](docs/feature_dictionary.md)

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

Equivalent one-off:

```bash
docker build -t bcs-platform .
# Windows
docker run --rm -v "%CD%/data:/app/data" -v "%USERPROFILE%/.kaggle:/root/.kaggle" bcs-platform
# Unix
docker run --rm -v ~/.kaggle:/root/.kaggle -v "$PWD/data:/app/data" bcs-platform
```

### Segmentation layer (Track C bake-off + champion_v1)

Requires Berka CSVs already under `data/raw/berka/` or `data/raw/berka-dataset/` (run the data layer first).

```bash
docker compose run --rm segmentation-layer
```

Outputs land on the host via volume mounts:

- `artifacts/experiments/comparison_board_track_c.csv`
- `artifacts/segmentation/champion_v1/` (assignments, metrics, figures, manifest, `segment_catalog.json`)
- `data/processed/track_c_berka_*.csv`

**Modular champion (load / score — not notebook-only):**

```python
from customer_segmentation.models.champion import ChampionModel

champ = ChampionModel.load("artifacts/segmentation/champion_v1")
scored = champ.score_interpretable(interpretable_df)  # segment_id, name, play, confidence
```

```bash
PYTHONPATH=src python -m customer_segmentation.cli champion-info
PYTHONPATH=src python -m customer_segmentation.cli score-berka --limit 100
```

Without Compose (segmentation):

```bash
docker run --rm ^
  -v "%CD%/data:/app/data" ^
  -v "%CD%/artifacts:/app/artifacts" ^
  -v "%CD%/configs:/app/configs" ^
  -e MPLBACKEND=Agg ^
  bcs-platform ./scripts/run_segmentation_layer.sh
```

Without Docker (needs the `kaggle` CLI on your PATH):

```bash
uv sync --extra dev
bash scripts/run_data_layer.sh
uv run pytest
```

The script stops at the first failing step. Santander is downloaded from the
competition when your Kaggle account has accepted its rules, otherwise from a public
mirror of the same `train_ver2.csv`.

## Airflow (data-layer DAG)

`pipelines/airflow/` runs the same steps as an Airflow 3 DAG, `bcip_data_layer`:

```text
download_<source> → check_raw_<source> → build_track_<x>   (one chain per dataset, in parallel)
                                                    └──► dq_report
```

- Each task calls the project's own functions, and the repo is mounted into the
  containers, so code changes need no rebuild.
- `check_raw_*` fails on missing files or columns and records a SHA-256 for each raw file.
- `dq_report` writes `artifacts/quality/dq_report_<run_id>.json` (plus `dq_report_latest.json`)
  and fails the run if any track breaks a gate in `configs/data.yaml` (`quality:`).

```bash
cd pipelines/airflow
cp .env.example .env               # set AIRFLOW_UID to `id -u`
docker compose up -d --build       # UI: http://localhost:8080 (airflow / airflow)
docker compose exec airflow-scheduler airflow dags unpause bcip_data_layer
docker compose exec airflow-scheduler airflow dags trigger bcip_data_layer
docker compose down                # stop; add -v to also drop Airflow's database
```

Downloads are skipped when the files are already in `data/raw/`.
