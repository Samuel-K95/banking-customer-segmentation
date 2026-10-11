"""Download (when needed), extract, and verify the raw P1 dataset inputs."""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _run_kaggle(command: list[str], dataset_name: str) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI is unavailable. Install the project's data extra and configure Kaggle credentials."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Kaggle download failed for {dataset_name} (exit {exc.returncode}).") from exc


def _require_files(paths: list[Path], dataset_name: str) -> None:
    missing = [str(path) for path in paths if not _is_nonempty_file(path)]
    if missing:
        raise FileNotFoundError(
            f"{dataset_name} extraction is incomplete; required non-empty file(s) missing: {missing}"
        )


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> list[Path]:
    """Extract a valid archive, leaving cleanup to the caller after verification."""
    archive = Path(zip_path)
    if not archive.is_file():
        raise FileNotFoundError(f"Expected download archive does not exist: {archive}")
    destination = Path(extract_to)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive, "r") as zip_file:
            corrupt_member = zip_file.testzip()
            if corrupt_member is not None:
                raise zipfile.BadZipFile(f"Corrupt member {corrupt_member!r} in {archive}")
            zip_file.extractall(destination)
            return [destination / name for name in zip_file.namelist()]
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid or corrupt dataset archive {archive}: {exc}") from exc


def _remove_archive(path: Path) -> None:
    if path.is_file():
        os.remove(path)


def download_bank_transactions(raw_dir: str | Path = "data/raw") -> Path:
    """Ensure the Indian Bank transaction CSV exists and is non-empty."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_csv = raw_dir / "bank_transactions.csv"
    if _is_nonempty_file(target_csv):
        print("Indian Bank dataset already exists; skipping download.")
        return target_csv

    archive = raw_dir / "bank-customer-segmentation.zip"
    print("Downloading Indian Bank dataset from Kaggle...")
    _run_kaggle(
        ["kaggle", "datasets", "download", "-d", "shivamb/bank-customer-segmentation", "-p", str(raw_dir)],
        "Indian Bank",
    )
    extract_zip(archive, raw_dir)
    _require_files([target_csv], "Indian Bank")
    _remove_archive(archive)
    return target_csv


def download_santander(raw_dir: str | Path = "data/raw/santander") -> Path:
    """Ensure the Santander training CSV exists, including its nested archive."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_csv = raw_dir / "train_ver2.csv"
    if _is_nonempty_file(target_csv):
        print("Santander dataset already exists; skipping download.")
        return target_csv

    outer_archive = raw_dir / "santander-product-recommendation.zip"
    inner_archive = raw_dir / "train_ver2.csv.zip"
    print("Downloading Santander dataset from Kaggle...")
    _run_kaggle(
        ["kaggle", "competitions", "download", "-c", "santander-product-recommendation", "-p", str(raw_dir)],
        "Santander",
    )
    extract_zip(outer_archive, raw_dir)
    if inner_archive.is_file():
        extract_zip(inner_archive, raw_dir)
    _require_files([target_csv], "Santander")
    _remove_archive(inner_archive)
    _remove_archive(outer_archive)
    return target_csv


def _berka_table_path(raw_dir: Path, table_name: str) -> Path | None:
    for suffix in (".csv", ".asc"):
        candidate = raw_dir / f"{table_name}{suffix}"
        if _is_nonempty_file(candidate):
            return candidate
    return None


def download_berka(raw_dir: str | Path = "data/raw/berka") -> list[Path]:
    """Ensure the transaction/account/client tables from Berka are available."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_names = ("trans", "account", "client")
    available = [_berka_table_path(raw_dir, table) for table in table_names]
    if all(available):
        print("Berka tables already exist; skipping download.")
        return [path for path in available if path is not None]

    archive = raw_dir / "the-berka-dataset.zip"
    print("Downloading Berka dataset from Kaggle...")
    _run_kaggle(
        ["kaggle", "datasets", "download", "-d", "marceloventura/the-berka-dataset", "-p", str(raw_dir)],
        "Berka",
    )
    extract_zip(archive, raw_dir)
    resolved = [_berka_table_path(raw_dir, table) for table in table_names]
    missing = [table for table, path in zip(table_names, resolved) if path is None]
    if missing:
        raise FileNotFoundError(
            f"Berka extraction is incomplete; expected trans/account/client as .csv or .asc; missing {missing}."
        )
    _remove_archive(archive)
    return [path for path in resolved if path is not None]


def download_datasets(tracks: list[str] | None = None) -> None:
    """Ensure raw source files for selected tracks; default preserves all-track behavior."""
    downloaders = {
        "track_a": download_bank_transactions,
        "track_b": download_santander,
        "track_c": download_berka,
    }
    selected = list(downloaders) if tracks is None else list(dict.fromkeys(tracks))
    unknown = [track for track in selected if track not in downloaders]
    if unknown:
        raise ValueError(f"Unknown track key(s) for ingestion: {unknown}")
    if not selected:
        raise ValueError("At least one track must be selected for ingestion.")

    print("Starting raw dataset ingestion...")
    for track in selected:
        downloaders[track]()
    print(f"Raw dataset ingestion complete; required files verified for: {', '.join(selected)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        metavar="TRACK",
        help="Ingest selected track keys only (default: all tracks).",
    )
    download_datasets(parser.parse_args().tracks)
