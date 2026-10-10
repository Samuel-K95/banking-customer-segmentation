#!/bin/bash
set -e

echo "Starting Data Layer Pipeline..."

echo "1. Downloading and extracting datasets..."
uv run python -m src.customer_segmentation.ingest.download_datasets

echo "2. Building features for Track A (Indian Bank)..."
uv run python -m src.customer_segmentation.features.build_track_a_features

echo "3. Building features for Track C (Berka)..."
uv run python -m src.customer_segmentation.features.build_berka_features

echo "3. Building features for Track B (Santander)..."
uv run python -m src.customer_segmentation.features.build_track_b_features

echo "Data Layer Pipeline completed successfully! Outputs are in data/processed/"
