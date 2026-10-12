"""Writing feature matrices, shared by the three track builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from customer_segmentation.config import data_path, load_data_config


def track_outputs(track: str, out_dir: Path | None = None) -> tuple[Path, Path]:
    """(interpretable_csv, model_csv) paths for a track from configs/data.yaml.

    Matrices land under ``processed_dir/<track>/`` so tracks stay separated.
    """
    root = out_dir or data_path("processed_dir")
    track_dir = Path(root) / track
    prefix = load_data_config()["tracks"][track]["output_prefix"]
    return track_dir / f"{prefix}_interpretable.csv", track_dir / f"{prefix}_model.csv"


def save_matrices(
    track: str, interpretable_df: pd.DataFrame, model_df: pd.DataFrame, out_dir: Path | None = None
) -> dict:
    """Saves both matrices of a track and returns a small summary for logs/XCom.

    The summary carries the counts the DQ report gates on (rows, nulls, duplicate ids),
    so under Airflow the report never has to re-read the CSVs.
    """
    interpretable_path, model_path = track_outputs(track, out_dir)
    interpretable_path.parent.mkdir(parents=True, exist_ok=True)
    interpretable_df.to_csv(interpretable_path, index=False)
    model_df.to_csv(model_path, index=False)
    print(f"Saved {len(interpretable_df):,} rows to {interpretable_path} and {model_path}")
    return summarize(track, interpretable_df, model_df, interpretable_path, model_path)


def summarize(
    track: str, interpretable_df: pd.DataFrame, model_df: pd.DataFrame, interpretable_path: Path, model_path: Path
) -> dict:
    id_column = load_data_config()["tracks"][track]["id_column"]
    has_id = id_column in interpretable_df.columns and id_column in model_df.columns
    return {
        "track": track,
        "rows": len(interpretable_df),
        "columns": len(interpretable_df.columns),
        "id_column": id_column,
        "has_id_column": has_id,
        "duplicate_ids": int(interpretable_df[id_column].duplicated().sum()) if has_id else None,
        "null_cells": int(interpretable_df.isna().sum().sum() + model_df.isna().sum().sum()),
        "interpretable_path": str(interpretable_path),
        "model_path": str(model_path),
    }


def read_track_summary(track: str, out_dir: Path | None = None) -> dict:
    """The same summary as save_matrices, rebuilt from the CSVs already on disk."""
    interpretable_path, model_path = track_outputs(track, out_dir)
    for path in (interpretable_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"{track}: expected output {path} is missing")
    return summarize(track, pd.read_csv(interpretable_path), pd.read_csv(model_path), interpretable_path, model_path)
