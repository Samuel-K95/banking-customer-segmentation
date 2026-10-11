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
