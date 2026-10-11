#!/bin/bash
set -euo pipefail

echo "Starting Data Layer Pipeline..."

echo "1. Downloading and extracting datasets..."
uv run python -m customer_segmentation.ingest.download_datasets

echo "2. Checking raw files (presence, required columns, checksums)..."
uv run python -m customer_segmentation.validate.raw_checks

echo "3. Building features for Track A (Indian Bank)..."
uv run python -m customer_segmentation.features.build_indian_bank_features

echo "4. Building features for Track C (Berka)..."
uv run python -m customer_segmentation.features.build_berka_features

echo "5. Building features for Track B (Santander)..."
uv run python -m customer_segmentation.features.build_santander_features

echo "6. Writing the DQ report (fails on a critical gate)..."
uv run python -m customer_segmentation.validate.report

echo "Data Layer Pipeline completed successfully! Outputs are in data/processed/, DQ report in artifacts/quality/"
