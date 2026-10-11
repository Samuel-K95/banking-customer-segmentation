#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_DOWNLOADS="${SKIP_DOWNLOADS:-0}"
REUSE_PREPROCESSORS="${REUSE_PREPROCESSORS:-0}"
TRACKS=(track_a track_b track_c)

usage() {
  cat <<'EOF'
Usage: scripts/run_data_layer.sh [--skip-downloads] [--reuse-preprocessors] [--tracks TRACK [TRACK ...]]

  --skip-downloads       Use existing data/raw inputs; do not contact Kaggle.
  --reuse-preprocessors  Load saved joblib transformations rather than fitting.
  --tracks               Process only selected keys: track_a, track_b, track_c (default: all).

Environment equivalents: SKIP_DOWNLOADS=1, REUSE_PREPROCESSORS=1.
EOF
}

while (($#)); do
  case "$1" in
    --skip-downloads) SKIP_DOWNLOADS=1 ;;
    --reuse-preprocessors) REUSE_PREPROCESSORS=1 ;;
    -h|--help) usage; exit 0 ;;
    --tracks)
      shift
      TRACKS=()
      while (($#)) && [[ "$1" != -* ]]; do
        TRACKS+=("$1")
        shift
      done
      if ((${#TRACKS[@]} == 0)); then
        echo "--tracks requires at least one track key." >&2
        usage >&2
        exit 2
      fi
      continue
      ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for track in "${TRACKS[@]}"; do
  case "$track" in
    track_a|track_b|track_c) ;;
    *) echo "Unknown track key: $track" >&2; exit 2 ;;
  esac
done

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER="uv"
  run_module() {
    local module="$1"
    shift
    uv run --project "$ROOT_DIR" --extra data python -m "$module" "$@"
  }
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_RUNNER="$ROOT_DIR/.venv/bin/python"
  run_module() {
    local module="$1"
    shift
    "$PYTHON_RUNNER" -m "$module" "$@"
  }
elif [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON_RUNNER="$ROOT_DIR/.venv/Scripts/python.exe"
  run_module() {
    local module="$1"
    shift
    "$PYTHON_RUNNER" -m "$module" "$@"
  }
else
  echo "No project environment found. Install uv or create .venv from pyproject.toml first." >&2
  exit 1
fi

echo "Using project Python runner: $PYTHON_RUNNER"

if [[ "$SKIP_DOWNLOADS" != "1" ]]; then
  echo "1. Downloading/extracting required raw datasets for: ${TRACKS[*]} (already-present files are reused)..."
  run_module customer_segmentation.ingest.download_datasets --tracks "${TRACKS[@]}"
else
  echo "1. Skipping Kaggle downloads; selected builders will fail if required raw files are absent."
fi

REUSE_ARGS=()
if [[ "$REUSE_PREPROCESSORS" == "1" ]]; then
  REUSE_ARGS=(--reuse-preprocessor)
fi

for track in "${TRACKS[@]}"; do
  case "$track" in
    track_a)
      echo "Building Track A Indian Bank customer features..."
      run_module customer_segmentation.features.build_indian_bank_features "${REUSE_ARGS[@]}"
      ;;
    track_b)
      echo "Building Track B Santander customer portfolio features..."
      run_module customer_segmentation.features.build_santander_features "${REUSE_ARGS[@]}"
      ;;
    track_c)
      echo "Building Track C Berka account features..."
      run_module customer_segmentation.features.build_berka_features "${REUSE_ARGS[@]}"
      ;;
  esac
done

echo "Validating generated outputs against the P2 feature contract for: ${TRACKS[*]}..."
run_module customer_segmentation.models.preflight --config configs/models/feature_contracts.yaml --tracks "${TRACKS[@]}"

echo "Data Layer pipeline completed; all required outputs passed preflight."
