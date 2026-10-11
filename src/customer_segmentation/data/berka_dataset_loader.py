"""Load Berka relational tables from data/raw."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BERKA_DIR_CANDIDATES = (
    Path("data/raw/berka-dataset"),
    Path("data/raw/berka"),
    Path("data/berka"),
)


def resolve_berka_dir(berka_dir: str | Path | None = None) -> Path:
    if berka_dir is not None:
        path = Path(berka_dir)
        if not path.exists():
            raise FileNotFoundError(f"Berka directory not found at {path}")
        return path
    for cand in BERKA_DIR_CANDIDATES:
        if not cand.exists():
            continue
        if (cand / "trans.csv").exists() or (cand / "trans.asc").exists():
            return cand
    raise FileNotFoundError(
        "Berka directory not found. Expected data/raw/berka-dataset/ with trans.csv"
    )


def _read_table(directory: Path, name: str) -> pd.DataFrame:
    for fname in (f"{name}.csv", f"{name}.asc"):
        path = directory / fname
        if not path.exists():
            continue
        for sep in (";", ","):
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
    raise FileNotFoundError(f"{name}.csv/.asc not readable under {directory}")


def load_berka_raw_data(
    berka_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load trans, account, client, loan, card tables."""
    path = resolve_berka_dir(berka_dir)
    trans_df = _read_table(path, "trans")
    account_df = _read_table(path, "account")
    client_df = _read_table(path, "client")
    try:
        loan_df = _read_table(path, "loan")
    except FileNotFoundError:
        loan_df = pd.DataFrame(columns=["account_id"])
    try:
        card_df = _read_table(path, "card")
    except FileNotFoundError:
        card_df = pd.DataFrame(columns=["disp_id"])
    return trans_df, account_df, client_df, loan_df, card_df
