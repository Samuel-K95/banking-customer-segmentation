import subprocess
import zipfile
from pathlib import Path

import pytest

from customer_segmentation.ingest import download_datasets


def test_downloaders_skip_when_required_raw_files_exist(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    santander_dir = raw_dir / "santander"
    berka_dir = raw_dir / "berka"
    santander_dir.mkdir(parents=True)
    berka_dir.mkdir(parents=True)
    (raw_dir / "bank_transactions.csv").write_text("header\n", encoding="utf-8")
    (santander_dir / "train_ver2.csv").write_text("header\n", encoding="utf-8")
    for table in ("trans", "account", "client"):
        (berka_dir / f"{table}.asc").write_text("header\n", encoding="utf-8")

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Kaggle should not be called when raw files exist")

    monkeypatch.setattr(download_datasets.subprocess, "run", unexpected_download)

    assert download_datasets.download_bank_transactions(raw_dir).is_file()
    assert download_datasets.download_santander(santander_dir).is_file()
    assert len(download_datasets.download_berka(berka_dir)) == 3


def test_download_dataset_selector_defaults_to_all_and_accepts_track_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(download_datasets, "download_bank_transactions", lambda: calls.append("track_a"))
    monkeypatch.setattr(download_datasets, "download_santander", lambda: calls.append("track_b"))
    monkeypatch.setattr(download_datasets, "download_berka", lambda: calls.append("track_c"))
    monkeypatch.setattr(download_datasets, "download_banksim", lambda: calls.append("track_d"))

    download_datasets.download_datasets(["track_a"])
    assert calls == ["track_a"]

    calls.clear()
    download_datasets.download_datasets()
    assert calls == ["track_a", "track_b", "track_c", "track_d"]

    with pytest.raises(ValueError, match="Unknown track"):
        download_datasets.download_datasets(["track_z"])


def test_kaggle_failure_is_raised_instead_of_reported_as_success(tmp_path, monkeypatch):
    def failed_download(*args, **kwargs):
        raise subprocess.CalledProcessError(2, args[0])

    monkeypatch.setattr(download_datasets.subprocess, "run", failed_download)

    with pytest.raises(RuntimeError, match="download failed"):
        download_datasets.download_bank_transactions(tmp_path / "raw")


def test_missing_kaggle_executable_is_reported(tmp_path, monkeypatch):
    def missing_cli(*args, **kwargs):
        raise FileNotFoundError("kaggle")

    monkeypatch.setattr(download_datasets.subprocess, "run", missing_cli)

    with pytest.raises(RuntimeError, match="Kaggle CLI is unavailable"):
        download_datasets.download_bank_transactions(tmp_path / "raw")


def test_incomplete_extraction_is_detected_and_archive_is_preserved(tmp_path, monkeypatch):
    def write_incomplete_archive(command, check):
        output_dir = Path(command[command.index("-p") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_dir / "bank-customer-segmentation.zip", "w") as archive:
            archive.writestr("readme.txt", "not the expected dataset")

    monkeypatch.setattr(download_datasets.subprocess, "run", write_incomplete_archive)
    raw_dir = tmp_path / "raw"

    with pytest.raises(FileNotFoundError, match="bank_transactions.csv"):
        download_datasets.download_bank_transactions(raw_dir)

    assert (raw_dir / "bank-customer-segmentation.zip").is_file()


def test_extract_zip_rejects_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="archive"):
        download_datasets.extract_zip(tmp_path / "missing.zip", tmp_path / "extract")
