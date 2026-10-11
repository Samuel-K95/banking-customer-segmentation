import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Public mirror of the competition's train_ver2.csv (same 2,292,759,599-byte file).
# Used when the Kaggle account has not accepted the competition rules (HTTP 403).
SANTANDER_MIRROR = "silviosantana/s-product-recomendation"


class DownloadError(RuntimeError):
    """Raised when a dataset could not be landed in data/raw."""


def extract_zip(zip_path, extract_to):
    """Extracts a zip file and then deletes it."""
    zip_file = Path(zip_path)
    if zip_file.exists():
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_file)
        print(f"Extracted and removed {zip_file}")
    else:
        print(f"Zip file not found: {zip_path}")


def require_file(path: Path, dataset: str) -> None:
    """Fails closed if a download did not produce the expected file."""
    if not path.exists() or path.stat().st_size == 0:
        raise DownloadError(f"{dataset}: expected {path} after download, but it is missing or empty.")


def download_bank_transactions():
    """Downloads the Indian Bank Transactions dataset from Kaggle."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_csv = raw_dir / "bank_transactions.csv"
    if target_csv.exists():
        print("Indian Bank dataset already exists. Skipping download.")
        return

    print("Downloading Indian Bank dataset...")
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'shivamb/bank-customer-segmentation', '-p', str(raw_dir)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise DownloadError("Failed to download the Indian Bank dataset from Kaggle.") from e
    extract_zip(raw_dir / "bank-customer-segmentation.zip", raw_dir)
    require_file(target_csv, "Indian Bank")


def download_santander():
    """Downloads the Santander dataset from Kaggle, falling back to a public mirror."""
    santander_dir = Path("data/raw/santander")
    santander_dir.mkdir(parents=True, exist_ok=True)

    target_csv = santander_dir / "train_ver2.csv"
    if target_csv.exists():
        print("Santander dataset already exists. Skipping download.")
        return

    print("Downloading Santander dataset...")
    try:
        subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', 'santander-product-recommendation', '-p', str(santander_dir)],
            check=True
        )
        extract_zip(santander_dir / "santander-product-recommendation.zip", santander_dir)
    except subprocess.CalledProcessError:
        print(f"Competition download failed (rules not accepted?). Falling back to mirror {SANTANDER_MIRROR}...")
        try:
            subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', SANTANDER_MIRROR, '-f', 'train_ver2.csv', '-p', str(santander_dir)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise DownloadError("Failed to download Santander from both the competition and the mirror.") from e

    inner_zip = santander_dir / "train_ver2.csv.zip"
    if inner_zip.exists():
        extract_zip(inner_zip, santander_dir)
    require_file(target_csv, "Santander")


def download_berka():
    """Downloads the Berka dataset (Czech Banking) from Kaggle."""
    berka_dir = Path("data/raw/berka")
    berka_dir.mkdir(parents=True, exist_ok=True)

    # Berka contains multiple files, typically trans.csv, account.csv, etc.
    target_csv = berka_dir / "trans.csv"
    if target_csv.exists():
        print("Berka dataset already exists. Skipping download.")
        return

    print("Downloading Berka dataset...")
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'marceloventura/the-berka-dataset', '-p', str(berka_dir)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise DownloadError("Failed to download Berka from Kaggle.") from e
    extract_zip(berka_dir / "the-berka-dataset.zip", berka_dir)
    require_file(target_csv, "Berka")
    require_file(berka_dir / "account.csv", "Berka")


def main() -> int:
    print("Starting Dataset Ingestion...")
    failures = []
    for download in (download_bank_transactions, download_santander, download_berka):
        try:
            download()
        except DownloadError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            failures.append(str(e))
    if failures:
        print(f"Ingestion failed for {len(failures)} dataset(s).", file=sys.stderr)
        return 1
    print("Ingestion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
