"""Load the BankSim synthetic payments table (ealaxi/banksim1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BANKSIM_COLUMNS = ["step", "customer", "age", "gender", "merchant", "category", "amount", "fraud"]


def load_banksim_raw_data(data_path: str | Path = "data/raw/banksim/bs140513_032310.csv") -> pd.DataFrame:
    """Loads BankSim transactions; the source wraps every text value in single quotes."""
    path = Path(data_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset not found or empty at {path}")
    df = pd.read_csv(
        path,
        quotechar="'",
        usecols=BANKSIM_COLUMNS,
        dtype={"customer": "string", "age": "string", "gender": "string",
               "merchant": "string", "category": "string"},
    )
    return df
