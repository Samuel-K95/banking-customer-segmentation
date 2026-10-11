"""Project paths and settings read from configs/*.yaml."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml


def project_dir() -> Path:
    """Repo root, or BCIP_PROJECT_DIR when the code runs from somewhere else (e.g. Airflow)."""
    override = os.environ.get("BCIP_PROJECT_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


@cache
def load_data_config() -> dict:
    with open(project_dir() / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def data_path(key: str) -> Path:
    """Absolute path for one of the `paths:` entries in configs/data.yaml."""
    return project_dir() / load_data_config()["paths"][key]


def source_files(source: str) -> list[Path]:
    """Absolute paths of the raw files a source in configs/data.yaml must provide."""
    raw_dir = data_path("raw_dir")
    return [raw_dir / name for name in load_data_config()["sources"][source]["files"]]
