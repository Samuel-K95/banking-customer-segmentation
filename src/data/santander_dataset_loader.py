import pandas as pd
from pathlib import Path

def load_santander_raw_data(data_path: str = "data/santander/train_ver2.csv", **kwargs) -> pd.DataFrame:
    """Loads the raw Santander dataset. 
    Accepts kwargs for chunking or specifying dtypes to handle memory efficiency.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, **kwargs)
