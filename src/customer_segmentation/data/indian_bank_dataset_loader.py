from pathlib import Path

import pandas as pd


def load_indian_bank_raw_data(data_path: str = "data/raw/bank_transactions.csv") -> pd.DataFrame:
    """Loads the raw Indian Bank transactions dataset."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)
