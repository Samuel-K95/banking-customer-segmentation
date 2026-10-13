"""Validated loader for the account-grain Berka source tables.

Also provides optional loan/card/disp helpers for the champion segmentation
feature bundle (product flags) without expanding the account grain.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from customer_segmentation.config import load_data_config, source_files

DEFAULT_BERKA_DIRECTORY = Path("data/raw/berka")
BERKA_TABLES = ("trans", "account", "client")
BERKA_DIR_CANDIDATES = (
    Path("data/raw/berka-dataset"),
    Path("data/raw/berka"),
    Path("data/berka"),
)


def _read_table(directory: Path, table_name: str) -> pd.DataFrame:
    """Read a non-empty configured Berka table, accepting CSV or ASC extracts."""
    source_config = load_data_config()["sources"]["berka"]
    separator = source_config.get("sep", ";")
    table_path: Path | None = None
    for suffix in (".csv", ".asc"):
        candidate = directory / f"{table_name}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            table_path = candidate
            break
    if table_path is None:
        raise FileNotFoundError(
            f"Berka table '{table_name}' not found as a non-empty .csv or .asc in {directory}"
        )

    table = pd.read_csv(table_path, sep=separator, low_memory=False)
    if table.empty:
        raise ValueError(f"Berka table '{table_path}' contains a header but no data rows")

    return table


def _resolve_berka_directory(berka_dir: str | Path) -> Path:
    requested = Path(berka_dir)
    if requested == DEFAULT_BERKA_DIRECTORY:
        # Resolve the default through config so BCIP_PROJECT_DIR and data.yaml
        # remain authoritative for both local runs and Airflow-mounted projects.
        return source_files("berka")[0].parent
    return requested


def resolve_berka_dir(berka_dir: str | Path | None = None) -> Path:
    """Resolve a Berka raw directory, including local berka-dataset layouts."""
    if berka_dir is not None:
        path = _resolve_berka_directory(berka_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Berka directory not found at {path}")
        return path
    for cand in BERKA_DIR_CANDIDATES:
        if not cand.exists():
            continue
        if (cand / "trans.csv").exists() or (cand / "trans.asc").exists():
            return cand
    # Fall back to config-resolved default
    path = source_files("berka")[0].parent
    if path.is_dir():
        return path
    raise FileNotFoundError(
        "Berka directory not found. Expected data/raw/berka/ or data/raw/berka-dataset/ with trans.csv"
    )


def load_berka_raw_data(
    berka_dir: str | Path = "data/raw/berka",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load transaction, account, and client tables, without disposition joins.

    Keeping ``account_id`` as the row grain is intentional: ``disp.csv`` maps
    accounts to owners/disponents but is not returned or used to expand rows.
    """
    path = resolve_berka_dir(berka_dir)
    trans_df, account_df, client_df = (_read_table(path, name) for name in BERKA_TABLES)
    return trans_df, account_df, client_df


def load_berka_product_tables(
    berka_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Optionally load loan, card, and disp for champion product flags.

    Missing tables become empty frames / None so feature builders can zero-fill.
    """
    path = resolve_berka_dir(berka_dir)
    try:
        loan_df = _read_table(path, "loan")
    except FileNotFoundError:
        loan_df = pd.DataFrame(columns=["account_id"])
    try:
        card_df = _read_table(path, "card")
    except FileNotFoundError:
        card_df = pd.DataFrame(columns=["disp_id"])
    try:
        disp_df = _read_table(path, "disp")
    except FileNotFoundError:
        disp_df = None
    return loan_df, card_df, disp_df
