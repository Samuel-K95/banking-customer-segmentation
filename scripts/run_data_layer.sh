#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_DOWNLOADS="${SKIP_DOWNLOADS:-0}"
REUSE_PREPROCESSORS="${REUSE_PREPROCESSORS:-0}"
TRACKS=(track_a track_b track_c track_d)

usage() {
  cat <<'EOF'
Usage: scripts/run_data_layer.sh [--skip-downloads] [--reuse-preprocessors] [--tracks TRACK [TRACK ...]]

  --skip-downloads       Use existing configured raw inputs; do not contact Kaggle.
  --reuse-preprocessors  Load saved joblib transformations rather than fitting.
  --tracks               Process selected config track keys (default: all configured tracks).

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

if ((${#TRACKS[@]} == 0)); then
  echo "At least one track must be selected." >&2
  exit 2
fi

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER="uv"
  run_python() {
    uv run --project "$ROOT_DIR" --extra data python "$@"
  }
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_RUNNER="$ROOT_DIR/.venv/bin/python"
  run_python() {
    "$PYTHON_RUNNER" "$@"
  }
elif [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON_RUNNER="$ROOT_DIR/.venv/Scripts/python.exe"
  run_python() {
    "$PYTHON_RUNNER" "$@"
  }
else
  echo "No project environment found. Install uv or create .venv from pyproject.toml first." >&2
  exit 1
fi

run_module() {
  local module="$1"
  shift
  run_python -m "$module" "$@"
}

echo "Using project Python runner: $PYTHON_RUNNER"

# Resolve selected source keys from the track mapping in configs/data.yaml.
SOURCE_LIST="$(run_python -c '
import sys
from customer_segmentation.config import load_data_config

tracks = load_data_config()["tracks"]
unknown = [track for track in sys.argv[1:] if track not in tracks]
if unknown:
    raise SystemExit(f"Unknown track key(s): {unknown}")
print("\n".join(dict.fromkeys(tracks[track]["source"] for track in sys.argv[1:])))
' "${TRACKS[@]}")"
SOURCES=()
while IFS= read -r source; do
  [[ -n "$source" ]] && SOURCES+=("$source")
done <<< "$SOURCE_LIST"
if ((${#SOURCES[@]} == 0)); then
  echo "Selected tracks did not resolve to configured data sources." >&2
  exit 2
fi

if [[ "$SKIP_DOWNLOADS" != "1" ]]; then
  echo "1. Downloading/extracting configured raw sources: ${SOURCES[*]} (verified existing files are reused)..."
  run_module customer_segmentation.ingest.download_datasets --tracks "${TRACKS[@]}"
else
  echo "1. Skipping downloads; selected sources will still be validated before feature generation."
fi

RAW_CHECKS_FILE="$(mktemp "${TMPDIR:-/tmp}/bcip-raw-checks.XXXXXX")"
trap 'rm -f -- "$RAW_CHECKS_FILE"' EXIT
echo "2. Checking selected raw files, headers, required columns, and checksums..."
run_python -c '
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from customer_segmentation.config import data_path
from customer_segmentation.validate.raw_checks import check_raw_source

raw_dir = data_path("raw_dir")
with redirect_stdout(sys.stderr):
    notes = [check_raw_source(source, raw_dir) for source in sys.argv[2:]]
Path(sys.argv[1]).write_text(json.dumps(notes), encoding="utf-8")
' "$RAW_CHECKS_FILE" "${SOURCES[@]}"

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
    track_d)
      echo "Building Track D BankSim customer features..."
      run_module customer_segmentation.features.build_banksim_features "${REUSE_ARGS[@]}"
      ;;
  esac
done

echo "3. Publishing the selected-track data-quality report..."
run_python -c '
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from customer_segmentation.features.io import read_track_summary
from customer_segmentation.validate.report import build_report, publish_report

raw_notes = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
track_keys = sys.argv[2:]
summaries = [read_track_summary(track) for track in track_keys]
run_id = "shell_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
publish_report(build_report(run_id, summaries, raw_notes))
' "$RAW_CHECKS_FILE" "${TRACKS[@]}"

echo "4. Validating selected outputs against the P2 feature contract..."
run_module customer_segmentation.models.preflight \
  --config configs/models/feature_contracts.yaml \
  --tracks "${TRACKS[@]}"

echo "Data Layer pipeline completed; selected sources and outputs passed validation."
