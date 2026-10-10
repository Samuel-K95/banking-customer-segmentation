import os
import subprocess
import zipfile
from pathlib import Path

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
        extract_zip(raw_dir / "bank-customer-segmentation.zip", raw_dir)
    except subprocess.CalledProcessError:
        print("\nERROR: Failed to download from Kaggle.")

def download_santander():
    """Downloads the Santander dataset from Kaggle."""
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
        inner_zip = santander_dir / "train_ver2.csv.zip"
        if inner_zip.exists():
            extract_zip(inner_zip, santander_dir)
            
    except subprocess.CalledProcessError:
        print("\nERROR: Failed to download Santander from Kaggle.")

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
        extract_zip(berka_dir / "the-berka-dataset.zip", berka_dir)
    except subprocess.CalledProcessError:
        print("\nERROR: Failed to download Berka from Kaggle.")

if __name__ == "__main__":
    print("Starting Dataset Ingestion...")
    download_bank_transactions()
    download_santander()
    download_berka()
    print("Ingestion complete.")
