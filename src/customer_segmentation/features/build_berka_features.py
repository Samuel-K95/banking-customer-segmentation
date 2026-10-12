"""Build account-grain Berka features and their persisted model matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from customer_segmentation.config import source_files
from customer_segmentation.data.berka_dataset_loader import load_berka_raw_data
from customer_segmentation.features.io import save_matrices
from customer_segmentation.features.preprocessing import prepare_model_matrix

DEFAULT_PREPROCESSOR = Path("artifacts/preprocessors/track_c.joblib")
BERKA_TRANSACTION_COLUMNS = {"account_id", "trans_id", "date", "type", "amount", "balance"}
BERKA_ACCOUNT_COLUMNS = {"account_id", "date"}

track_c_schema = pa.DataFrameSchema(
    {
        "Account_Balance": pa.Column(float, nullable=False, coerce=True),
        "Transaction_Count": pa.Column(int, pa.Check.greater_than_or_equal_to(1), nullable=False, coerce=True),
        "Total_Credit": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
        "Total_Withdrawal": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
    }
)


def build_berka_features(
    trans_df: pd.DataFrame,
    account_df: pd.DataFrame,
    *,
    preprocessor_path: str | Path | None = DEFAULT_PREPROCESSOR,
    fit_preprocessor: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unscaled account features and the saved-preprocessor model matrix.

    Historical semantics are retained: account age and recency are referenced to
    the maximum valid transaction date in the supplied input. ``balance_std``-style
    newer behavioral snapshot features are a separate pipeline.
    """
    missing_transactions = sorted(BERKA_TRANSACTION_COLUMNS - set(trans_df.columns))
    missing_accounts = sorted(BERKA_ACCOUNT_COLUMNS - set(account_df.columns))
    if missing_transactions or missing_accounts:
        raise ValueError(
            "Berka input schema is incomplete: "
            f"transactions missing {missing_transactions}; accounts missing {missing_accounts}"
        )

    transactions = trans_df.copy()
    transactions["date"] = pd.to_datetime(transactions["date"], format="%y%m%d", errors="coerce")
    transactions["Is_Credit"] = transactions["type"].isin(["PRIJEM", "CREDIT"]).astype("int8")
    transactions["Is_Withdrawal"] = transactions["type"].isin(
        ["VYDAJ", "VYBER", "WITHDRAWAL"]
    ).astype("int8")
    transactions["amount_credit"] = transactions["amount"] * transactions["Is_Credit"]
    transactions["amount_withdrawal"] = transactions["amount"] * transactions["Is_Withdrawal"]

    max_date = transactions["date"].max()
    if pd.isna(max_date):
        raise ValueError("Berka transactions contain no parseable dates")

    # The source is normally date ordered. An explicit stable order makes the
    # historical `last` balance deterministic and means latest-dated balance.
    transactions = transactions.sort_values(
        ["account_id", "date", "trans_id"], kind="mergesort", na_position="first"
    )
    account_features = (
        transactions.groupby("account_id", sort=True)
        .agg(
            date=("date", "max"),
            trans_id=("trans_id", "count"),
            balance=("balance", "last"),
            amount_credit=("amount_credit", "sum"),
            amount_withdrawal=("amount_withdrawal", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "date": "Last_Transaction",
                "trans_id": "Transaction_Count",
                "balance": "Account_Balance",
                "amount_credit": "Total_Credit",
                "amount_withdrawal": "Total_Withdrawal",
            }
        )
    )
    account_features["Recency_Days"] = (
        max_date - account_features["Last_Transaction"]
    ).dt.days

    accounts = account_df[["account_id", "date"]].copy()
    accounts["date"] = pd.to_datetime(accounts["date"], format="%y%m%d", errors="coerce")
    accounts["Account_Age_Days"] = (max_date - accounts["date"]).dt.days
    merged = account_features.merge(
        accounts[["account_id", "Account_Age_Days"]],
        on="account_id",
        how="inner",
        validate="one_to_one",
    ).drop(columns=["Last_Transaction"])

    interpretable_df = track_c_schema.validate(merged)
    features = [column for column in interpretable_df.columns if column != "account_id"]
    log_columns = ["Account_Balance", "Total_Credit", "Total_Withdrawal", "Account_Age_Days"]
    model_df = prepare_model_matrix(
        interpretable_df,
        track="track_c",
        identifier_columns=["account_id"],
        log1p_columns=log_columns,
        clip_nonnegative_columns=log_columns,
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
    """Configured Track C flow for command-line and Airflow data-layer use."""
    (transaction_path,) = source_files("berka")[:1]
    berka_dir = transaction_path.parent
    print(f"Loading raw Berka dataset from {berka_dir}...")
    trans_df, account_df, _ = load_berka_raw_data(berka_dir)
    print("Building account-grain features...")
    interpretable_df, model_df = build_berka_features(
        trans_df,
        account_df,
        preprocessor_path=preprocessor_path,
        fit_preprocessor=fit_preprocessor,
    )
    return save_matrices("track_c", interpretable_df, model_df, Path(out_dir) if out_dir else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track C Berka account-level features.")
    parser.add_argument("--preprocessor", default=str(DEFAULT_PREPROCESSOR))
    parser.add_argument("--reuse-preprocessor", action="store_true")
    args = parser.parse_args()
    run(
        preprocessor_path=args.preprocessor,
        fit_preprocessor=not args.reuse_preprocessor,
    )


# Champion segmentation helpers (loan/card bundle) — kept alongside the data-layer builder.
from customer_segmentation.features.build_berka_segmentation_features import (  # noqa: E402
    MODEL_COLUMNS,
    BerkaFeatureBundle,
    build_and_save_berka_features,
    build_berka_segmentation_features,
    transform_berka_model_matrix,
)

__all__ = [
    "DEFAULT_PREPROCESSOR",
    "build_berka_features",
    "run",
    "main",
    "MODEL_COLUMNS",
    "BerkaFeatureBundle",
    "build_and_save_berka_features",
    "build_berka_segmentation_features",
    "transform_berka_model_matrix",
]


if __name__ == "__main__":
    main()
