import pandas as pd
from pathlib import Path

def load_indian_bank_raw_data(data_path: str = "data/bank_transactions.csv") -> pd.DataFrame:
    """Loads the raw Indian Bank transactions dataset."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)
