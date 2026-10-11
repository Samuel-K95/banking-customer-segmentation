#!/bin/bash
set -euo pipefail

echo "Starting Data Layer Pipeline..."

echo "1. Downloading and extracting datasets..."
uv run python -m customer_segmentation.ingest.download_datasets

echo "2. Building features for Track A (Indian Bank)..."
uv run python -m customer_segmentation.features.build_indian_bank_features

echo "3. Building features for Track C (Berka)..."
uv run python -m customer_segmentation.features.build_berka_features

echo "4. Building features for Track B (Santander)..."
uv run python -m customer_segmentation.features.build_santander_features

echo "Data Layer Pipeline completed successfully! Outputs are in data/processed/"
