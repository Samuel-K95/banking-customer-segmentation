#!/bin/bash
set -e

echo "Starting Segmentation Layer (leakage-safe production candidate)..."

# Ensure output dirs exist (host mounts may be empty on first run)
mkdir -p data/raw data/processed artifacts/experiments artifacts/segmentation reports/champion

if [ ! -f data/raw/berka-dataset/trans.csv ] && [ ! -f data/raw/berka/trans.csv ]; then
  echo "ERROR: Berka raw data not found under data/raw/berka-dataset/ (or data/raw/berka/)."
  echo "Run the data-layer container first, or mount a host volume that already has the CSVs:"
  echo "  docker compose run --rm data-layer"
  echo "  docker compose run --rm segmentation-layer"
  exit 1
fi

echo "1. Train leakage-safe AE+k=6 twin (+ k=2 research twin)..."
uv run bcs segment-berka-leakage-safe

echo "2. Promote research twin → artifacts/segmentation/champion (gated)..."
uv run bcs promote-production

echo "3. Regenerate decision layer on production champion..."
uv run bcs decision-berka

echo "4. Verify champion integrity (binaries, SHA-256, load, stale paths)..."
uv run bcs verify-champion

echo "Segmentation Layer completed."
echo "  Champion bundle  : artifacts/segmentation/champion/"
echo "  Decision artifacts: artifacts/decision/champion/"
echo "  (Legacy full-book bake-off remains available via: bcs segment-berka)"
