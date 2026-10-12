#!/bin/bash
set -e

echo "Starting Segmentation Layer Pipeline (Track C / Berka champion)..."

# Ensure output dirs exist (host mounts may be empty on first run)
mkdir -p data/raw data/processed artifacts/experiments artifacts/segmentation

if [ ! -f data/raw/berka-dataset/trans.csv ] && [ ! -f data/raw/berka/trans.csv ]; then
  echo "ERROR: Berka raw data not found under data/raw/berka-dataset/ (or data/raw/berka/)."
  echo "Run the data-layer container first, or mount a host volume that already has the CSVs:"
  echo "  docker compose run --rm data-layer"
  echo "  docker compose run --rm segmentation-layer"
  exit 1
fi

echo "1. Building Berka features (interpretable + model matrices)..."
uv run python -c "
from customer_segmentation.features.build_berka_features import build_and_save_berka_features
bundle = build_and_save_berka_features()
print(f'  saved {len(bundle.interpretable)} accounts, cols={bundle.feature_columns}')
"

echo "2. Running bake-off + registering champion_v1..."
uv run python -m customer_segmentation.pipelines.run_berka_segmentation

echo "Segmentation Layer completed."
echo "  Comparison board : artifacts/experiments/bakeoffs/comparison_board_track_c.csv"
echo "  Champion bundle  : artifacts/segmentation/archive/fullbook_demo/"
echo "  Figures          : artifacts/segmentation/archive/fullbook_demo/figures/"
