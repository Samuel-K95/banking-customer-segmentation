"""Chunked loader and source-to-feature map for the Santander snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from customer_segmentation.config import load_data_config, source_files

SANTANDER_PRODUCT_COLUMN_MAP = {
    "ind_ahor_fin_ult1": "has_savings_account",
    "ind_aval_fin_ult1": "has_bank_guarantees",
    "ind_cco_fin_ult1": "has_current_account",
    "ind_cder_fin_ult1": "has_derivada_account",
    "ind_cno_fin_ult1": "has_payroll_account",
    "ind_ctju_fin_ult1": "has_junior_account",
    "ind_ctma_fin_ult1": "has_mas_particular_account",
    "ind_ctop_fin_ult1": "has_particular_account",
    "ind_ctpp_fin_ult1": "has_particular_plus_account",
    "ind_deco_fin_ult1": "has_short_term_deposits",
    "ind_deme_fin_ult1": "has_medium_term_deposits",
    "ind_dela_fin_ult1": "has_long_term_deposits",
    "ind_ecue_fin_ult1": "has_e_account",
    "ind_fond_fin_ult1": "has_funds",
    "ind_hip_fin_ult1": "has_mortgage",
    "ind_plan_fin_ult1": "has_pensions",
    "ind_pres_fin_ult1": "has_loans",
    "ind_reca_fin_ult1": "has_taxes",
    "ind_tjcr_fin_ult1": "has_credit_card",
    "ind_valo_fin_ult1": "has_securities",
    "ind_viv_fin_ult1": "has_home_account",
    "ind_nomina_ult1": "has_payroll",
    "ind_nom_pens_ult1": "has_pensions_nom",
    "ind_recibo_ult1": "has_direct_debit",
}
SANTANDER_PRODUCT_FEATURES = tuple(SANTANDER_PRODUCT_COLUMN_MAP.values())

SANTANDER_BASE_DTYPES = {
    "ncodpers": np.int32,
    "age": object,
    "antiguedad": object,
    "renta": np.float32,
    "ind_nuevo": np.float32,
    "indrel": np.float32,
    "indrel_1mes": object,
    "indext": object,
    "conyuemp": object,
    "indfall": object,
    "tipodom": np.float32,
    "ind_actividad_cliente": np.float32,
    "segmento": object,
    "sexo": object,
    "tiprel_1mes": object,
    "nomprov": str,
    "fecha_dato": str,
}
SANTANDER_SOURCE_DTYPES = {
    **SANTANDER_BASE_DTYPES,
    **{column: np.float32 for column in SANTANDER_PRODUCT_COLUMN_MAP},
}
SANTANDER_COLUMN_RENAME = {
    "ncodpers": "customer_id",
    "age": "age",
    "antiguedad": "seniority_months",
    "renta": "gross_income",
    "segmento": "customer_segment",
    "sexo": "gender",
    "nomprov": "province_name",
    "fecha_dato": "snapshot_date",
    "ind_nuevo": "new_customer_index",
    "indrel": "customer_type",
    "indrel_1mes": "customer_type_primary",
    "indext": "foreigner_index",
    "conyuemp": "spouse_index",
    "indfall": "deceased_index",
    "tipodom": "address_type",
    "ind_actividad_cliente": "activity_index",
    "tiprel_1mes": "customer_relation_type",
    **SANTANDER_PRODUCT_COLUMN_MAP,
}
DEFAULT_SANTANDER_PATH = Path("data/raw/santander/train_ver2.csv")
DEFAULT_SNAPSHOT_DATE = "2016-05-28"


class _UseConfiguredSnapshotDate:
    """Sentinel distinguishing an omitted target_date from an explicit value."""

    __slots__ = ()


_USE_CONFIGURED_SNAPSHOT_DATE = _UseConfiguredSnapshotDate()


def load_santander_raw_data(
    data_path: str = "data/raw/santander/train_ver2.csv",
    target_date: str | None | _UseConfiguredSnapshotDate = _USE_CONFIGURED_SNAPSHOT_DATE,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Load a validated Santander customer snapshot in bounded-memory chunks.

    An omitted ``target_date`` uses the configured snapshot date. Explicit dates,
    including the historical default date, are honored as given; ``None`` loads
    all dates. Explicit paths and the established positional parameter order are
    retained for experiments and tests.
    """
    config = load_data_config()["sources"]["santander"]
    path = source_files("santander")[0] if Path(data_path) == DEFAULT_SANTANDER_PATH else Path(data_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset not found or empty at {path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if isinstance(target_date, _UseConfiguredSnapshotDate):
        target_date = config.get("snapshot_date", DEFAULT_SNAPSHOT_DATE)

    separator = config.get("sep", ",")
    header = pd.read_csv(path, sep=separator, nrows=0).columns
    missing_columns = sorted(set(SANTANDER_SOURCE_DTYPES).difference(header))
    if missing_columns:
        missing_products = sorted(set(SANTANDER_PRODUCT_COLUMN_MAP).intersection(missing_columns))
        detail = f" Missing product source columns: {missing_products}." if missing_products else ""
        raise ValueError(f"Santander source schema is missing required columns: {missing_columns}.{detail}")

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep=separator,
        dtype=SANTANDER_SOURCE_DTYPES,
        usecols=list(SANTANDER_SOURCE_DTYPES),
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk.rename(columns=SANTANDER_COLUMN_RENAME, inplace=True)
        if target_date is not None:
            chunk = chunk.loc[chunk["snapshot_date"].eq(target_date)]
        chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No data rows were read from Santander dataset at {path}")
    result = pd.concat(chunks, ignore_index=True)
    if result.empty:
        raise ValueError(f"No Santander rows found for target snapshot {target_date!r} in {path}")
    if target_date is not None and (
        result["customer_id"].isna().any() or result["customer_id"].duplicated().any()
    ):
        raise ValueError(f"Santander snapshot {target_date!r} contains null or duplicate customer IDs")
    return result
