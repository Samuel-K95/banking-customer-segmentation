"""Checks on the landed raw files, run before any feature code touches them."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

from customer_segmentation.config import data_path, load_data_config


class DataQualityError(RuntimeError):
    """A critical data problem; the pipeline must stop here."""


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_raw_source(source: str, raw_dir: Path | None = None) -> dict:
    """Checks every file of one source and returns its source note (sizes + checksums).

    Raises DataQualityError listing every problem found, not just the first.
    """
    raw_dir = raw_dir or data_path("raw_dir")
    config = load_data_config()["sources"][source]
    problems = []
    files = []
    for name in config["files"]:
        path = raw_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{name}: missing or empty at {path}")
            continue
        header = pd.read_csv(path, sep=config["sep"], nrows=0).columns
        missing = [c for c in config["required_columns"].get(name, []) if c not in header]
        if missing:
            problems.append(f"{name}: missing required columns {missing}")
        files.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256(path)})

    if problems:
        raise DataQualityError(f"Raw check failed for {source}:\n  " + "\n  ".join(problems))
    kaggle_ref = config.get("kaggle") or config.get("kaggle_competition")
    print(f"Raw check passed for {source}: {len(files)} file(s)")
    return {"source": source, "kaggle": kaggle_ref, "files": files}


def main() -> int:
    failures = []
    for source in load_data_config()["sources"]:
        try:
            check_raw_source(source)
        except DataQualityError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            failures.append(source)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
