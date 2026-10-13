"""Build customer-grain BankSim features (Track D) and their persisted model matrix.

BankSim covers 180 daily steps of card payments. Each customer becomes one row of
behavioral features: activity, spend level and spread, recency/tenure, rhythm, merchant
breadth and how spend splits across category groups.

Columns that must never drive the segmentation stay out of the model matrix:
``age_band`` and ``gender`` (profiling and fairness) and the fraud columns (an outcome
used only to validate segments after the fact).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from customer_segmentation.config import source_files
from customer_segmentation.data.banksim_dataset_loader import (
    BANKSIM_COLUMNS,
    load_banksim_raw_data,
)
from customer_segmentation.features.io import save_matrices
from customer_segmentation.features.preprocessing import prepare_model_matrix

DEFAULT_PREPROCESSOR = Path("artifacts/preprocessors/track_d.joblib")

# 15 source categories -> 6 spend groups (shares of a customer's total amount)
CATEGORY_GROUPS = {
    "es_transportation": "transport",
    "es_food": "food",
    "es_health": "health_beauty",
    "es_wellnessandbeauty": "health_beauty",
    "es_fashion": "shopping",
    "es_hyper": "shopping",
    "es_sportsandtoys": "shopping",
    "es_tech": "shopping",
    "es_home": "shopping",
    "es_contents": "shopping",
    "es_barsandrestaurants": "leisure_travel",
    "es_hotelservices": "leisure_travel",
    "es_travel": "leisure_travel",
    "es_leisure": "leisure_travel",
    "es_otherservices": "other",
}
SPEND_GROUPS = ["transport", "food", "health_beauty", "shopping", "leisure_travel", "other"]
SHARE_FEATURES = [f"share_{group}" for group in SPEND_GROUPS]

MONEY_FEATURES = ["total_amount", "avg_amount", "std_amount", "max_amount"]
MODEL_FEATURES = [
    "txn_count",
    "active_days",
    *MONEY_FEATURES,
    "recency_days",
    "tenure_days",
    "interarrival_days_median",
    "n_merchants",
    "n_categories",
    *SHARE_FEATURES,
]
PROFILE_COLUMNS = ["age_band", "gender"]
OUTCOME_COLUMNS = ["fraud_txn_count", "any_fraud"]

track_d_schema = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(str, unique=True, nullable=False),
        "txn_count": pa.Column(int, pa.Check.ge(1), coerce=True),
        "active_days": pa.Column(int, pa.Check.ge(1), coerce=True),
        "total_amount": pa.Column(float, pa.Check.ge(0.0), coerce=True),
        "recency_days": pa.Column(int, pa.Check.ge(0), coerce=True),
        "tenure_days": pa.Column(int, pa.Check.ge(0), coerce=True),
        **{share: pa.Column(float, pa.Check.in_range(0.0, 1.0), coerce=True) for share in SHARE_FEATURES},
        "any_fraud": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
    }
)


def build_banksim_features(
    transactions: pd.DataFrame,
    *,
    preprocessor_path: str | Path | None = DEFAULT_PREPROCESSOR,
    fit_preprocessor: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (interpretable customer features, model matrix) for BankSim."""
    missing = sorted(set(BANKSIM_COLUMNS) - set(transactions.columns))
    if missing:
        raise ValueError(f"BankSim input schema is incomplete: missing {missing}")

    tx = transactions.copy()
    unknown = sorted(set(tx["category"].dropna().unique()) - set(CATEGORY_GROUPS))
    if unknown:
        raise ValueError(f"Unmapped BankSim categories: {unknown}")
    tx["spend_group"] = tx["category"].map(CATEGORY_GROUPS)
    last_step = int(tx["step"].max())

    grouped = tx.groupby("customer", sort=True)
    features = grouped.agg(
        txn_count=("amount", "size"),
        active_days=("step", "nunique"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        max_amount=("amount", "max"),
        first_step=("step", "min"),
        last_step=("step", "max"),
        n_merchants=("merchant", "nunique"),
        n_categories=("category", "nunique"),
        age_band=("age", "first"),
        gender=("gender", "first"),
        fraud_txn_count=("fraud", "sum"),
    )
    features["std_amount"] = features["std_amount"].fillna(0.0)  # single-transaction customers
    features["recency_days"] = last_step - features["last_step"]
    features["tenure_days"] = last_step - features["first_step"]
    features["any_fraud"] = (features["fraud_txn_count"] > 0).astype(int)

    # Median gap between consecutive active days (0 when active on a single day)
    days = tx[["customer", "step"]].drop_duplicates().sort_values(["customer", "step"])
    days["gap"] = days.groupby("customer")["step"].diff()
    features["interarrival_days_median"] = days.groupby("customer")["gap"].median().fillna(0.0)

    spend = tx.pivot_table(index="customer", columns="spend_group", values="amount", aggfunc="sum", fill_value=0.0)
    spend = spend.reindex(columns=SPEND_GROUPS, fill_value=0.0)
    totals = spend.sum(axis=1).replace(0.0, np.nan)  # zero-amount customers get zero shares
    shares = spend.div(totals, axis=0).fillna(0.0).add_prefix("share_")
    features = features.join(shares)

    interpretable = (
        features.reset_index()
        .rename(columns={"customer": "customer_id"})
        [["customer_id", *MODEL_FEATURES, *PROFILE_COLUMNS, *OUTCOME_COLUMNS]]
    )
    interpretable["customer_id"] = interpretable["customer_id"].astype(str)
    interpretable = track_d_schema.validate(interpretable)

    model_df = prepare_model_matrix(
        interpretable[["customer_id", *MODEL_FEATURES]],
        track="track_d",
        identifier_columns=["customer_id"],
        log1p_columns=MONEY_FEATURES,
        clip_nonnegative_columns=MONEY_FEATURES,
        scaled_features=MODEL_FEATURES,
        artifact_path=preprocessor_path,
        fit=fit_preprocessor,
    )
    return interpretable, model_df


def run(
    out_dir: str | Path | None = None,
    *,
    preprocessor_path: str | Path | None = DEFAULT_PREPROCESSOR,
    fit_preprocessor: bool = True,
) -> dict:
    """Configured Track D flow for command-line and Airflow data-layer use."""
    (raw_csv,) = source_files("banksim")
    print(f"Loading raw BankSim payments from {raw_csv}...")
    transactions = load_banksim_raw_data(raw_csv)
    print("Building customer-grain features...")
    interpretable_df, model_df = build_banksim_features(
        transactions, preprocessor_path=preprocessor_path, fit_preprocessor=fit_preprocessor
    )
    return save_matrices("track_d", interpretable_df, model_df, Path(out_dir) if out_dir else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track D BankSim customer features.")
    parser.add_argument("--preprocessor", default=str(DEFAULT_PREPROCESSOR))
    parser.add_argument("--reuse-preprocessor", action="store_true")
    args = parser.parse_args()
    run(preprocessor_path=args.preprocessor, fit_preprocessor=not args.reuse_preprocessor)


if __name__ == "__main__":
    main()
