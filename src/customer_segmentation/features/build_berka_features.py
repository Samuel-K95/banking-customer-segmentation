"""Berka feature engineering: RFM + balance + loan/card flags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


@dataclass
class BerkaFeatureBundle:
    interpretable: pd.DataFrame
    model: pd.DataFrame
    scaler: StandardScaler
    feature_columns: list[str]
    id_col: str = "account_id"


def _parse_berka_date(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series.astype(str).str.zfill(6), format="%y%m%d", errors="coerce")


def build_berka_features(
    trans_df: pd.DataFrame,
    account_df: pd.DataFrame,
    loan_df: pd.DataFrame | None = None,
    card_df: pd.DataFrame | None = None,
    disp_df: pd.DataFrame | None = None,
) -> BerkaFeatureBundle:
    """Build interpretable + scaled model matrices for Track C."""
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

    # Monetary NaNs: no transactions after join edge cases → 0
    for col in ["Account_Balance", "Total_Credit", "Total_Withdrawal"]:
        merged[col] = merged[col].fillna(0)

    # Product flags
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

    model_df = interpretable.copy()
    # Log-compress skewed monetary / age (shift negatives on balance)
    for col in ["Account_Balance", "Total_Credit", "Total_Withdrawal", "Account_Age_Days", "Transaction_Count"]:
        model_df[col] = np.log1p(np.maximum(model_df[col].astype(float), 0))
    # Recency: leave as-is then scale; binaries untouched before scale
    scaler = StandardScaler()
    model_df[MODEL_COLUMNS] = scaler.fit_transform(model_df[MODEL_COLUMNS].astype(float))

    return BerkaFeatureBundle(
        interpretable=interpretable,
        model=model_df,
        scaler=scaler,
        feature_columns=list(MODEL_COLUMNS),
        id_col="account_id",
    )


def build_and_save_berka_features(
    berka_dir: str | Path | None = None,
    processed_dir: str | Path = "data/processed",
) -> BerkaFeatureBundle:
    from customer_segmentation.data.berka_dataset_loader import load_berka_raw_data, resolve_berka_dir, _read_table

    trans, account, _client, loan, card = load_berka_raw_data(berka_dir)
    path = resolve_berka_dir(berka_dir)
    try:
        disp = _read_table(path, "disp")
    except FileNotFoundError:
        disp = None

    bundle = build_berka_features(trans, account, loan, card, disp)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle.interpretable.to_csv(out / "track_c_berka_interpretable.csv", index=False)
    bundle.model.to_csv(out / "track_c_berka_model.csv", index=False)
    return bundle
