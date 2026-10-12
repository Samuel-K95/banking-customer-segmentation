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

## Docker Setup (Data Layer)

To run the full data ingestion and feature engineering pipeline in an isolated container:

```bash
# Build the image
docker build -t data-layer .

# Run the pipeline (make sure your Kaggle credentials are in ~/.kaggle on the host).
# Mount data/ so the downloads and the processed matrices survive the container.
docker run --rm -v ~/.kaggle:/root/.kaggle -v "$PWD/data:/app/data" data-layer
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

`pipelines/airflow/` runs `scripts/run_data_layer.sh` step for step as an Airflow 3 DAG,
`bcip_data_layer`, then segments Berka:

```text
download_<source> → check_raw_<source> → build_<track>     (one chain per selected track, in parallel)
    → dq_report → feature_contract
    → berka_snapshot → berka_baselines → berka_feature_sensitivity
                                       → berka_temporal_validation
```

- Each task calls the project's own code from the repo root (mounted into the containers),
  so code changes need no rebuild and repo-relative paths resolve as in the shell script.
- `check_raw_*` fails on missing files or columns and records a SHA-256 for each raw file.
- `dq_report` writes `artifacts/quality/dq_report_<run_id>.json` (plus `dq_report_latest.json`)
  and fails the run if any track breaks a gate in `configs/data.yaml` (`quality:`).
  Columns listed under a track's `nullable_columns` are reported but do not fail the gate.
- `feature_contract` is shell step 4 (`models.preflight` against `configs/models/feature_contracts.yaml`).
- The Berka segmentation stage runs only when `track_c` is selected.

Run parameters (Trigger → "Trigger DAG w/ config" in the UI) mirror the shell flags:

| Parameter | Shell flag | Default |
|---|---|---|
| `tracks` | `--tracks` | `["track_a", "track_b", "track_c"]` |
| `skip_downloads` | `--skip-downloads` | `false` |
| `reuse_preprocessors` | `--reuse-preprocessors` | `false` |

```bash
cd pipelines/airflow
cp .env.example .env               # set AIRFLOW_UID to `id -u`; change AIRFLOW_PORT if 8080 is taken
docker compose up -d --build       # UI: http://localhost:8080 (airflow / airflow)
docker compose exec airflow-scheduler airflow dags unpause bcip_data_layer
docker compose exec airflow-scheduler airflow dags trigger bcip_data_layer
docker compose exec airflow-scheduler airflow dags trigger bcip_data_layer \
  --conf '{"tracks": ["track_c"], "skip_downloads": true}'
docker compose down                # stop; add -v to also drop Airflow's database
```

Downloads are skipped when the files are already in `data/raw/`.
