from pathlib import Path

import pandas as pd


def load_indian_bank_raw_data(data_path: str = "data/raw/bank_transactions.csv") -> pd.DataFrame:
    """Loads the raw Indian Bank transactions dataset."""
    path = Path(data_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset not found or empty at {path}")
    return pd.read_csv(path)
