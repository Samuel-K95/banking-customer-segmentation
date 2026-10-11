from pathlib import Path

import pandas as pd


def _read_table(directory: Path, table_name: str) -> pd.DataFrame:
    for suffix in (".csv", ".asc"):
        table_path = directory / f"{table_name}{suffix}"
        if table_path.is_file() and table_path.stat().st_size > 0:
            return pd.read_csv(table_path, sep=";")
    raise FileNotFoundError(
        f"Berka table '{table_name}' not found as a non-empty .csv or .asc in {directory}"
    )

def load_berka_raw_data(berka_dir: str = "data/raw/berka") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load transaction, account, and client tables.

    The source ``disp`` table links accounts to clients, but this loader does
    not return or aggregate it. The current feature builder therefore remains
    explicitly account-grain.
    """
    path = Path(berka_dir)
    if not path.exists():
        raise FileNotFoundError(f"Berka directory not found at {path}")
        
    trans_df = _read_table(path, "trans")
    account_df = _read_table(path, "account")
    client_df = _read_table(path, "client")
    return trans_df, account_df, client_df
