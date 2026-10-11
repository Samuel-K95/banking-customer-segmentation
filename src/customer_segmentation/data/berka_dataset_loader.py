from pathlib import Path

import pandas as pd


def load_berka_raw_data(berka_dir: str = "data/raw/berka") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the essential tables from the Berka dataset:
    - trans (transactions)
    - account (accounts)
    - client (clients)
    """
    path = Path(berka_dir)
    if not path.exists():
        raise FileNotFoundError(f"Berka directory not found at {path}")
        
    # The Berka dataset files are typically delimited by ';' or ','
    try:
        trans_df = pd.read_csv(path / "trans.csv", sep=";", low_memory=False)
    except FileNotFoundError:
        # Sometimes Kaggle extracts them without .csv or with different separators, handle gracefully
        trans_df = pd.read_csv(path / "trans.asc", sep=";", low_memory=False)
        
    try:
        account_df = pd.read_csv(path / "account.csv", sep=";")
    except FileNotFoundError:
        account_df = pd.read_csv(path / "account.asc", sep=";")
        
    try:
        client_df = pd.read_csv(path / "client.csv", sep=";")
    except FileNotFoundError:
        client_df = pd.read_csv(path / "client.asc", sep=";")
        
    return trans_df, account_df, client_df
