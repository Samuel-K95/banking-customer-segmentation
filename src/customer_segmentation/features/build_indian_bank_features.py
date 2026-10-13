"""Build customer-grain Indian Bank aggregates and their model matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from customer_segmentation.config import source_files
from customer_segmentation.data.indian_bank_dataset_loader import (
    load_indian_bank_raw_data,
)
from customer_segmentation.features.io import save_matrices
from customer_segmentation.features.preprocessing import prepare_model_matrix

DEFAULT_PREPROCESSOR = Path("artifacts/preprocessors/track_a.joblib")
INDIAN_BANK_REQUIRED_COLUMNS = {
    "TransactionID",
    "CustomerID",
    "CustomerDOB",
    "CustGender",
    "CustLocation",
    "CustAccountBalance",
    "TransactionDate",
    "TransactionTime",
    "TransactionAmount (INR)",
}

track_a_schema = pa.DataFrameSchema(
    {
        "CustomerID": pa.Column(str, unique=True, nullable=False),
        "Age": pa.Column(float, pa.Check.in_range(18.0, 100.0), nullable=True, coerce=True),
        "AccountBalance": pa.Column(float, nullable=False, coerce=True),
        "Transaction_Count": pa.Column(int, pa.Check.greater_than_or_equal_to(1), nullable=False, coerce=True),
        "Total_Spend": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
    }
)


def build_rfm_features(
    df: pd.DataFrame,
    *,
    preprocessor_path: str | Path | None = DEFAULT_PREPROCESSOR,
    fit_preprocessor: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the existing customer RFM/behavioral features and model matrix.

    Historical semantics are preserved: complete-case input rows are used, age
    follows the 2016 DOB convention, and recency is relative to the maximum valid
    transaction date in the provided input. No future/as-of policy is inferred.
    """
    missing = sorted(INDIAN_BANK_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Indian Bank input is missing required columns: {missing}")
    df = df.dropna().copy()
    if df.empty:
        raise ValueError("Indian Bank input has no complete transaction rows")

    # Demographics retained for profiling, not as clustering inputs.
    df["DOB"] = pd.to_datetime(df["CustomerDOB"], format="%d/%m/%y", errors="coerce")
    df.loc[df["DOB"].dt.year > 2005, "DOB"] -= pd.DateOffset(years=100)
    df["Age"] = 2016 - df["DOB"].dt.year
    df["Age"] = np.where((df["Age"] < 18) | (df["Age"] > 100), np.nan, df["Age"])

    if "CustGender" in df.columns:
        df = pd.get_dummies(df, columns=["CustGender"], drop_first=True, dtype=int)

    tier_1 = [
        "MUMBAI",
        "DELHI",
        "NEW DELHI",
        "BANGALORE",
        "CHENNAI",
        "HYDERABAD",
        "KOLKATA",
        "PUNE",
        "AHMEDABAD",
    ]
    if "CustLocation" in df.columns:
        df["Is_Tier_1_City"] = df["CustLocation"].str.upper().isin(tier_1).astype(int)

    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"], format="%d/%m/%y", errors="coerce"
    )
    df["Is_Weekend"] = df["TransactionDate"].dt.dayofweek.isin([5, 6]).astype(int)
    if "TransactionTime" in df.columns:
        df["HourOfDay"] = df["TransactionTime"] // 10000
        df["Is_Night_Spender"] = ((df["HourOfDay"] >= 20) | (df["HourOfDay"] < 4)).astype(int)
    else:
        df["Is_Night_Spender"] = 0

    max_date = df["TransactionDate"].max()
    if pd.isna(max_date):
        raise ValueError("Indian Bank input has no valid transaction dates")
    df["Spend_Balance_Ratio"] = df["TransactionAmount (INR)"] / (df["CustAccountBalance"] + 1)
    gender_column = "CustGender_M" if "CustGender_M" in df.columns else None

    aggregation = {
        "Age": ("Age", "first"),
        "AccountBalance": ("CustAccountBalance", "mean"),
        "Last_Transaction": ("TransactionDate", "max"),
        "Transaction_Count": ("TransactionID", "count"),
        "Avg_Transaction_Amt": ("TransactionAmount (INR)", "mean"),
        "Std_Transaction_Amt": ("TransactionAmount (INR)", "std"),
        "Total_Spend": ("TransactionAmount (INR)", "sum"),
        "Avg_Spend_Ratio": ("Spend_Balance_Ratio", "mean"),
        "Weekend_Spend_Ratio": ("Is_Weekend", "mean"),
        "Night_Spend_Ratio": ("Is_Night_Spender", "mean"),
    }
    if "Is_Tier_1_City" in df.columns:
        aggregation["Is_Tier_1_City"] = ("Is_Tier_1_City", "first")
    if gender_column:
        aggregation["Gender_M"] = (gender_column, "first")

    customer_df = df.groupby("CustomerID", sort=True).agg(**aggregation)
    customer_df["Std_Transaction_Amt"] = customer_df["Std_Transaction_Amt"].fillna(0)
    customer_df = customer_df.dropna()
    customer_df["Recency_Days"] = (max_date - customer_df["Last_Transaction"]).dt.days
    customer_df = customer_df.drop(columns=["Last_Transaction"]).reset_index()

    interpretable_df = track_a_schema.validate(customer_df.copy())
    features = [
        "Age",
        "AccountBalance",
        "Transaction_Count",
        "Avg_Transaction_Amt",
        "Std_Transaction_Amt",
        "Total_Spend",
        "Avg_Spend_Ratio",
        "Recency_Days",
        "Weekend_Spend_Ratio",
        "Night_Spend_Ratio",
    ]
    features = [feature for feature in features if feature in interpretable_df.columns]
    model_df = prepare_model_matrix(
        interpretable_df,
        track="track_a",
        identifier_columns=["CustomerID"],
        log1p_columns=[
            column
            for column in ["AccountBalance", "Avg_Transaction_Amt", "Std_Transaction_Amt", "Total_Spend"]
            if column in features
        ],
        scaled_features=features,
        artifact_path=preprocessor_path,
        fit=fit_preprocessor,
    )
    return interpretable_df, model_df


def run(
    out_dir: str | Path | None = None,
    *,
    preprocessor_path: str | Path | None = DEFAULT_PREPROCESSOR,
    fit_preprocessor: bool = True,
) -> dict:
    """Configured Track A flow for command-line and Airflow data-layer use."""
    (raw_csv,) = source_files("indian_bank")
    print(f"Loading raw Indian Bank data from {raw_csv}...")
    raw_df = load_indian_bank_raw_data(raw_csv)
    interpretable_df, model_df = build_rfm_features(
        raw_df,
        preprocessor_path=preprocessor_path,
        fit_preprocessor=fit_preprocessor,
    )
    return save_matrices("track_a", interpretable_df, model_df, Path(out_dir) if out_dir else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track A customer RFM features.")
    parser.add_argument("--preprocessor", default=str(DEFAULT_PREPROCESSOR))
    parser.add_argument("--reuse-preprocessor", action="store_true")
    args = parser.parse_args()
    run(
        preprocessor_path=args.preprocessor,
        fit_preprocessor=not args.reuse_preprocessor,
    )


if __name__ == "__main__":
    main()
