"""Download, extract, and verify raw dataset inputs using the project config."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path

from customer_segmentation.config import load_data_config, source_files


class DownloadError(RuntimeError):
    """Raised when a dataset could not be landed and verified in the raw layer."""


class KaggleUnavailableError(DownloadError):
    """Raised when the Kaggle command-line client is unavailable."""


class KaggleCommandError(DownloadError):
    """Raised when a Kaggle download command exits unsuccessfully."""


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _run_kaggle(command: list[str], dataset_name: str) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise KaggleUnavailableError(
            "Kaggle CLI is unavailable. Install the project's data extra and configure Kaggle credentials."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise KaggleCommandError(
            f"Kaggle download failed for {dataset_name} (exit {exc.returncode})."
        ) from exc


def _require_files(paths: list[Path], dataset_name: str) -> None:
    missing = [str(path) for path in paths if not _is_nonempty_file(path)]
    if missing:
        raise FileNotFoundError(
            f"{dataset_name} extraction is incomplete; required non-empty file(s) missing: {missing}"
        )


def require_file(path: Path, dataset: str) -> None:
    """Compatibility wrapper for callers that validate one landed file."""
    _require_files([Path(path)], dataset)


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> list[Path]:
    """Extract a valid archive; callers remove it only after output verification."""
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


def _configured_source_paths(
    source: str,
    raw_dir: str | Path,
    *,
    default_argument: str | Path,
) -> list[Path]:
    """Resolve configured source files while supporting legacy dataset-dir args.

    The default argument resolves through ``source_files`` (and therefore
    ``BCIP_PROJECT_DIR``/configs/data.yaml). Airflow passes the configured raw
    root. Historical direct callers may pass a source subdirectory instead.
    """
    if Path(raw_dir) == Path(default_argument):
        return source_files(source)
    configured = [Path(name) for name in load_data_config()["sources"][source]["files"]]
    base = Path(raw_dir)
    first_directory = configured[0].parent
    if first_directory != Path(".") and base.name == first_directory.name:
        base = base.parent
    return [base / relative_path for relative_path in configured]


def _nonempty_table_variant(configured_path: Path) -> Path | None:
    for candidate in (configured_path, configured_path.with_suffix(".asc")):
        if _is_nonempty_file(candidate):
            return candidate
    return None


def download_bank_transactions(raw_dir: str | Path = "data/raw") -> Path:
    """Ensure the configured Indian Bank transaction CSV exists and is non-empty."""
    (target_csv,) = _configured_source_paths(
        "indian_bank", raw_dir, default_argument="data/raw"
    )
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    if _is_nonempty_file(target_csv):
        print("Indian Bank dataset already exists; skipping download.")
        return target_csv

    source = load_data_config()["sources"]["indian_bank"]
    dataset_slug = source["kaggle"].rsplit("/", 1)[-1]
    archive = target_csv.parent / f"{dataset_slug}.zip"
    print("Downloading Indian Bank dataset from Kaggle...")
    _run_kaggle(
        ["kaggle", "datasets", "download", "-d", source["kaggle"], "-p", str(target_csv.parent)],
        "Indian Bank",
    )
    extract_zip(archive, target_csv.parent)
    _require_files([target_csv], "Indian Bank")
    _remove_archive(archive)
    return target_csv


def download_santander(raw_dir: str | Path = "data/raw/santander") -> Path:
    """Ensure the configured Santander snapshot is present and non-empty.

    On competition-access failure, retry the configured public mirror. Archive
    cleanup occurs only after the expected CSV has been verified.
    """
    (target_csv,) = _configured_source_paths(
        "santander", raw_dir, default_argument="data/raw/santander"
    )
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    if _is_nonempty_file(target_csv):
        print("Santander dataset already exists; skipping download.")
        return target_csv

    source = load_data_config()["sources"]["santander"]
    competition = source["kaggle_competition"]
    outer_archive = target_csv.parent / f"{competition}.zip"
    inner_archive = target_csv.parent / f"{target_csv.name}.zip"
    print("Downloading Santander dataset from Kaggle...")
    try:
        _run_kaggle(
            ["kaggle", "competitions", "download", "-c", competition, "-p", str(target_csv.parent)],
            "Santander competition",
        )
    except KaggleCommandError:
        mirror = source.get("kaggle_mirror")
        if not mirror:
            raise
        print(f"Competition download failed; trying configured mirror {mirror}...")
        _run_kaggle(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                mirror,
                "-f",
                target_csv.name,
                "-p",
                str(target_csv.parent),
            ],
            "Santander mirror",
        )

    if outer_archive.is_file():
        extract_zip(outer_archive, target_csv.parent)
    if inner_archive.is_file():
        extract_zip(inner_archive, target_csv.parent)
    _require_files([target_csv], "Santander")
    _remove_archive(inner_archive)
    _remove_archive(outer_archive)
    return target_csv


def _berka_table_path(configured_path: Path) -> Path | None:
    return _nonempty_table_variant(configured_path)


def _resolve_berka_configured_tables(configured_paths: list[Path]) -> list[Path | None]:
    """Accept legacy ASC archives while materializing configured CSV paths.

    The DAG's raw-check task validates the resulting configured CSV headers and
    required columns before any feature builder runs.
    """
    resolved: list[Path | None] = []
    for configured_path in configured_paths:
        found = _berka_table_path(configured_path)
        if found is not None and found.suffix.lower() == ".asc" and configured_path.suffix.lower() == ".csv":
            configured_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, configured_path)
            if not _is_nonempty_file(configured_path):
                raise FileNotFoundError(f"Berka table copy is missing or empty: {configured_path}")
            found = configured_path
        resolved.append(found)
    return resolved


def download_berka(raw_dir: str | Path = "data/raw/berka") -> list[Path]:
    """Ensure all configured Berka transaction/account/client tables are present."""
    configured_paths = _configured_source_paths(
        "berka", raw_dir, default_argument="data/raw/berka"
    )
    table_paths = _resolve_berka_configured_tables(configured_paths)
    if all(table_paths):
        print("Berka tables already exist; skipping download.")
        return [path for path in table_paths if path is not None]

    source = load_data_config()["sources"]["berka"]
    if len(configured_paths) != len(source.get("files", [])) or not configured_paths:
        raise ValueError("Berka source config must list its required table files")
    destination = configured_paths[0].parent
    destination.mkdir(parents=True, exist_ok=True)
    dataset_slug = source["kaggle"].rsplit("/", 1)[-1]
    archive = destination / f"{dataset_slug}.zip"
    print("Downloading Berka dataset from Kaggle...")
    _run_kaggle(
        ["kaggle", "datasets", "download", "-d", source["kaggle"], "-p", str(destination)],
        "Berka",
    )
    extract_zip(archive, destination)
    resolved = _resolve_berka_configured_tables(configured_paths)
    missing = [str(configured) for configured, found in zip(configured_paths, resolved) if found is None]
    if missing:
        raise FileNotFoundError(
            f"Berka extraction is incomplete; required configured tables are missing or empty: {missing}"
        )
    _remove_archive(archive)
    return [path for path in resolved if path is not None]


def _current_downloaders() -> dict[str, Callable[..., object]]:
    """Build a call-time map so monkeypatched/public downloader functions remain usable."""
    return {
        "indian_bank": download_bank_transactions,
        "santander": download_santander,
        "berka": download_berka,
    }


DOWNLOADERS = _current_downloaders()


def download_datasets(tracks: list[str] | None = None) -> None:
    """Download configured sources selected by track or source key.

    Default behavior covers the configured tracks. Calling dataset functions
    without explicit paths keeps configured paths authoritative, while Airflow
    can call ``DOWNLOADERS[source](data_path("raw_dir"))`` directly.
    """
    config = load_data_config()
    track_sources = {
        track: spec["source"] for track, spec in config.get("tracks", {}).items()
    }
    requested = list(track_sources) if tracks is None else list(dict.fromkeys(tracks))
    selected_sources: list[str] = []
    unknown: list[str] = []
    for item in requested:
        source = track_sources.get(item, item)
        if source not in DOWNLOADERS:
            unknown.append(item)
        elif source not in selected_sources:
            selected_sources.append(source)
    if unknown:
        raise ValueError(f"Unknown track or source key(s) for ingestion: {unknown}")
    if not selected_sources:
        raise ValueError("At least one configured track must be selected for ingestion.")

    print("Starting raw dataset ingestion...")
    downloaders = _current_downloaders()
    for source in selected_sources:
        downloaders[source]()
    print(f"Raw dataset ingestion complete; required files verified for: {', '.join(selected_sources)}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        metavar="TRACK",
        help="Ingest selected configured track/source keys only (default: all configured tracks).",
    )
    args = parser.parse_args()
    download_datasets(args.tracks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
