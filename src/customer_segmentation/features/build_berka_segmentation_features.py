"""Champion-path Berka features: RFM + balance + loan/card flags.

The data-layer Track C builder (`build_berka_features.py`) stays the Airflow /
preprocessor path. This module keeps the segmentation champion matrix that
includes product holding flags used by `ChampionModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.preprocessing import StandardScaler

from customer_segmentation.data.berka_dataset_loader import (
    load_berka_product_tables,
    load_berka_raw_data,
    resolve_berka_dir,
)
from customer_segmentation.features.io import save_matrices

MODEL_COLUMNS = [
    "Transaction_Count",
    "Account_Balance",
    "Total_Credit",
    "Total_Withdrawal",
    "Recency_Days",
    "Account_Age_Days",
    "has_loan",
    "has_card",
]

segmentation_schema = pa.DataFrameSchema(
    {
        "account_id": pa.Column(int, unique=True, coerce=True),
        "Account_Balance": pa.Column(float, nullable=False, coerce=True),
        "Transaction_Count": pa.Column(int, pa.Check.greater_than_or_equal_to(1), nullable=False, coerce=True),
        "Total_Credit": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
        "Total_Withdrawal": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
        "Recency_Days": pa.Column(float, nullable=False, coerce=True),
        "Account_Age_Days": pa.Column(float, nullable=False, coerce=True),
        "has_loan": pa.Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True),
        "has_card": pa.Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True),
    }
)


@dataclass
class BerkaFeatureBundle:
    interpretable: pd.DataFrame
    model: pd.DataFrame
    scaler: StandardScaler
    feature_columns: list[str]
    id_col: str = "account_id"


def _parse_berka_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.astype(str).str.zfill(6), format="%y%m%d", errors="coerce")


def build_berka_segmentation_features(
    trans_df: pd.DataFrame,
    account_df: pd.DataFrame,
    loan_df: pd.DataFrame | None = None,
    card_df: pd.DataFrame | None = None,
    disp_df: pd.DataFrame | None = None,
) -> BerkaFeatureBundle:
    """Build interpretable + scaled model matrices for the Berka champion path."""
    trans = trans_df.copy()
    account = account_df.copy()

    trans["date"] = _parse_berka_date(trans["date"])
    account["date"] = _parse_berka_date(account["date"])

    trans["is_credit"] = trans["type"].isin(["PRIJEM", "CREDIT"]).astype(int)
    trans["is_withdrawal"] = trans["type"].isin(["VYDAJ", "VYBER", "WITHDRAWAL"]).astype(int)
    trans["amt_credit"] = trans["amount"] * trans["is_credit"]
    trans["amt_withdrawal"] = trans["amount"] * trans["is_withdrawal"]

    max_date = trans["date"].max()

    agg = (
        trans.groupby("account_id")
        .agg(
            Last_Transaction=("date", "max"),
            Transaction_Count=("trans_id", "count"),
            Account_Balance=("balance", "last"),
            Total_Credit=("amt_credit", "sum"),
            Total_Withdrawal=("amt_withdrawal", "sum"),
        )
        .reset_index()
    )
    agg["Recency_Days"] = (max_date - agg["Last_Transaction"]).dt.days
    account["Account_Age_Days"] = (max_date - account["date"]).dt.days

    merged = agg.merge(account[["account_id", "Account_Age_Days"]], on="account_id", how="inner")
    merged = merged.drop(columns=["Last_Transaction"])

    for col in ["Account_Balance", "Total_Credit", "Total_Withdrawal"]:
        merged[col] = merged[col].fillna(0)

    merged["has_loan"] = 0
    if loan_df is not None and len(loan_df) and "account_id" in loan_df.columns:
        loan_accounts = set(loan_df["account_id"].unique())
        merged["has_loan"] = merged["account_id"].isin(loan_accounts).astype(int)

    merged["has_card"] = 0
    if (
        card_df is not None
        and disp_df is not None
        and len(card_df)
        and "disp_id" in card_df.columns
        and "disp_id" in disp_df.columns
    ):
        card_disp = set(card_df["disp_id"].unique())
        owners = disp_df.loc[disp_df["disp_id"].isin(card_disp), "account_id"].unique()
        merged["has_card"] = merged["account_id"].isin(owners).astype(int)
    elif card_df is not None and "account_id" in getattr(card_df, "columns", []):
        merged["has_card"] = merged["account_id"].isin(card_df["account_id"].unique()).astype(int)

    core = ["Recency_Days", "Transaction_Count", "Account_Age_Days"]
    merged = merged.dropna(subset=core).reset_index(drop=True)

    interpretable = merged[["account_id"] + MODEL_COLUMNS].copy()
    interpretable = segmentation_schema.validate(interpretable)

    model_df = interpretable.copy()
    for col in ["Account_Balance", "Total_Credit", "Total_Withdrawal", "Account_Age_Days", "Transaction_Count"]:
        model_df[col] = np.log1p(np.maximum(model_df[col].astype(float), 0))
    scaler = StandardScaler()
    model_df[MODEL_COLUMNS] = scaler.fit_transform(model_df[MODEL_COLUMNS].astype(float))

    return BerkaFeatureBundle(
        interpretable=interpretable,
        model=model_df,
        scaler=scaler,
        feature_columns=list(MODEL_COLUMNS),
        id_col="account_id",
    )


def transform_berka_model_matrix(
    interpretable: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """Apply the same log1p + frozen StandardScaler used at champion training time."""
    cols = list(feature_columns or MODEL_COLUMNS)
    frame = interpretable[cols].copy()
    for col in ["Account_Balance", "Total_Credit", "Total_Withdrawal", "Account_Age_Days", "Transaction_Count"]:
        if col in frame.columns:
            frame[col] = np.log1p(np.maximum(frame[col].astype(float), 0))
    return scaler.transform(frame.astype(float)).astype(np.float64)


def build_and_save_berka_features(
    berka_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> BerkaFeatureBundle:
    """Load Berka (+ optional products), build champion bundle, persist matrices."""
    path = resolve_berka_dir(berka_dir)
    trans, account, _client = load_berka_raw_data(path)
    loan, card, disp = load_berka_product_tables(path)
    bundle = build_berka_segmentation_features(trans, account, loan, card, disp)
    out = Path(processed_dir) if processed_dir is not None else None
    save_matrices("track_c", bundle.interpretable, bundle.model, out)
    return bundle


DEFAULT_AS_OF_DATE = "1997-12-31"
DEFAULT_MIN_TENURE_DAYS = 365
LOG1P_COLUMNS = [
    "Account_Balance",
    "Total_Credit",
    "Total_Withdrawal",
    "Account_Age_Days",
    "Transaction_Count",
]


def _log1p_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    for col in columns or LOG1P_COLUMNS:
        if col in out.columns:
            out[col] = np.log1p(np.maximum(out[col].astype(float), 0.0))
    return out


def build_berka_segmentation_features_asof(
    trans_df: pd.DataFrame,
    account_df: pd.DataFrame,
    loan_df: pd.DataFrame | None = None,
    card_df: pd.DataFrame | None = None,
    disp_df: pd.DataFrame | None = None,
    *,
    as_of_date: str | pd.Timestamp = DEFAULT_AS_OF_DATE,
    minimum_tenure_days: int = DEFAULT_MIN_TENURE_DAYS,
) -> pd.DataFrame:
    """Build the 8 CRM columns with fixed as-of cutoff (no scaler fit).

    Transactions, loans, and cards after ``as_of_date`` are excluded. Accounts
    must be open by the cutoff and satisfy ``Account_Age_Days >= minimum_tenure_days``.
    """
    cutoff = pd.Timestamp(as_of_date).normalize()
    trans = trans_df.copy()
    account = account_df.copy()

    trans["date"] = _parse_berka_date(trans["date"]).dt.normalize()
    account["date"] = _parse_berka_date(account["date"]).dt.normalize()

    if trans["date"].isna().any():
        raise ValueError("Transaction dates contain nulls after parsing")
    if (trans["date"] > cutoff).any():
        # Drop post-cutoff rows rather than fail — caller may pass full history.
        trans = trans.loc[trans["date"] <= cutoff].copy()
    account = account.loc[account["date"] <= cutoff].copy()
    if account.empty or trans.empty:
        raise ValueError(f"No eligible accounts/transactions on or before {cutoff.date()}")

    trans = trans.sort_values(["account_id", "date"], kind="mergesort")
    trans["is_credit"] = trans["type"].isin(["PRIJEM", "CREDIT"]).astype(int)
    trans["is_withdrawal"] = trans["type"].isin(["VYDAJ", "VYBER", "WITHDRAWAL"]).astype(int)
    trans["amt_credit"] = trans["amount"].astype(float) * trans["is_credit"]
    trans["amt_withdrawal"] = trans["amount"].astype(float) * trans["is_withdrawal"]

    agg = (
        trans.groupby("account_id", sort=False)
        .agg(
            Last_Transaction=("date", "max"),
            Transaction_Count=("date", "count"),
            Account_Balance=("balance", "last"),
            Total_Credit=("amt_credit", "sum"),
            Total_Withdrawal=("amt_withdrawal", "sum"),
        )
        .reset_index()
    )
    agg["Recency_Days"] = (cutoff - agg["Last_Transaction"]).dt.days.astype(float)
    account = account.copy()
    account["Account_Age_Days"] = (cutoff - account["date"]).dt.days.astype(float)

    merged = agg.merge(account[["account_id", "Account_Age_Days"]], on="account_id", how="inner")
    merged = merged.drop(columns=["Last_Transaction"])

    for col in ["Account_Balance", "Total_Credit", "Total_Withdrawal"]:
        merged[col] = merged[col].fillna(0.0)

    merged["has_loan"] = 0
    if loan_df is not None and len(loan_df) and "account_id" in loan_df.columns:
        loan = loan_df.copy()
        if "date" in loan.columns:
            loan["date"] = _parse_berka_date(loan["date"]).dt.normalize()
            loan = loan.loc[loan["date"].isna() | (loan["date"] <= cutoff)]
        merged["has_loan"] = merged["account_id"].isin(set(loan["account_id"].unique())).astype(int)

    merged["has_card"] = 0
    if card_df is not None and len(card_df) and "disp_id" in card_df.columns and disp_df is not None:
        card = card_df.copy()
        if "issued" in card.columns:
            card["issued"] = _parse_berka_date(card["issued"]).dt.normalize()
            card = card.loc[card["issued"].isna() | (card["issued"] <= cutoff)]
        card_disp = set(card["disp_id"].unique())
        owners = disp_df.loc[disp_df["disp_id"].isin(card_disp), "account_id"].unique()
        merged["has_card"] = merged["account_id"].isin(owners).astype(int)
    elif card_df is not None and "account_id" in getattr(card_df, "columns", []):
        merged["has_card"] = merged["account_id"].isin(card_df["account_id"].unique()).astype(int)

    merged = merged.dropna(subset=["Recency_Days", "Transaction_Count", "Account_Age_Days"])
    if minimum_tenure_days > 0:
        merged = merged.loc[merged["Account_Age_Days"] >= float(minimum_tenure_days)]
    merged = merged.loc[merged["Transaction_Count"] >= 1].reset_index(drop=True)

    interpretable = merged[["account_id"] + MODEL_COLUMNS].copy()
    return segmentation_schema.validate(interpretable)


def fit_train_scaler(train_interpretable: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on log1p-transformed train rows only."""
    frame = _log1p_frame(train_interpretable[MODEL_COLUMNS])
    scaler = StandardScaler()
    scaler.fit(frame.astype(float))
    return scaler


def transform_with_train_scaler(
    interpretable: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply champion log1p + a train-fitted StandardScaler."""
    frame = _log1p_frame(interpretable[MODEL_COLUMNS])
    return scaler.transform(frame.astype(float)).astype(np.float64)
