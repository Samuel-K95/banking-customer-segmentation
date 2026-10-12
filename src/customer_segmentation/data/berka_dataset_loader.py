"""Validated loader for the account-grain Berka source tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from customer_segmentation.config import load_data_config, source_files

DEFAULT_BERKA_DIRECTORY = Path("data/raw/berka")
BERKA_TABLES = ("trans", "account", "client")


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


def load_berka_raw_data(
    berka_dir: str = "data/raw/berka",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load transaction, account, and client tables, without disposition joins.

    Keeping ``account_id`` as the row grain is intentional: ``disp.csv`` maps
    accounts to owners/disponents but is not returned or used to expand rows.
    """
    path = _resolve_berka_directory(berka_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Berka directory not found at {path}")

    trans_df, account_df, client_df = (_read_table(path, name) for name in BERKA_TABLES)
    return trans_df, account_df, client_df
