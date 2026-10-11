"""Build the Santander customer portfolio snapshot and its fitted model matrix."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from customer_segmentation.data.santander_dataset_loader import (
    SANTANDER_PRODUCT_FEATURES,
)
from customer_segmentation.features.preprocessing import prepare_model_matrix

DEFAULT_PREPROCESSOR = Path("artifacts/preprocessors/track_b.joblib")

track_b_schema = pa.DataFrameSchema(
    {
        "age": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=True, coerce=True),
        "gross_income": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=True, coerce=True),
        "total_products_owned": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
    }
)


def _validated_product_flags(frame: pd.DataFrame, product_columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(product_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Santander snapshot is missing required product flags: {missing}")

    flags = pd.DataFrame(index=frame.index)
    invalid_values: dict[str, list[object]] = {}
    for column in product_columns:
        source = frame[column]
        numeric = pd.to_numeric(source, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        invalid = source.notna() & (numeric.isna() | ~numeric.isin([0, 1]) | ~np.isfinite(values))
        if invalid.any():
            invalid_values[column] = source.loc[invalid].drop_duplicates().tolist()[:5]
        flags[column] = numeric.fillna(0).astype(np.int8)

    if invalid_values:
        details = "; ".join(f"{column}: {values}" for column, values in invalid_values.items())
        raise ValueError(f"Santander product flags must contain only 0, 1, or missing values: {details}")
    return flags


def build_product_portfolio_features(
    df: pd.DataFrame,
    *,
    preprocessor_path: str | Path | None = DEFAULT_PREPROCESSOR,
    fit_preprocessor: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an unscaled profile matrix and a matrix transformed by saved preprocessing."""
    required = {"customer_id", "age", "seniority_months", "gross_income", "gender"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Santander snapshot is missing required columns: {missing}")

    df_current = df.copy()
    if df_current["customer_id"].isna().any() or df_current["customer_id"].duplicated().any():
        raise ValueError("Santander customer_id must be present and unique within the snapshot.")

    products = list(SANTANDER_PRODUCT_FEATURES)
    validated_products = _validated_product_flags(df_current, products)
    df_current[products] = validated_products

    df_current["age"] = pd.to_numeric(df_current["age"], errors="coerce")
    df_current["seniority_months"] = pd.to_numeric(df_current["seniority_months"], errors="coerce")
    df_current["gross_income"] = pd.to_numeric(df_current["gross_income"], errors="coerce")

    if "province_name" in df_current.columns:
        province_medians = df_current.groupby("province_name", dropna=False)["gross_income"].transform("median")
        df_current["gross_income"] = df_current["gross_income"].fillna(province_medians)

    for column in ["gross_income", "age", "seniority_months"]:
        df_current[column] = df_current[column].fillna(df_current[column].median())

    df_current["total_products_owned"] = df_current[products].sum(axis=1).astype(np.int16)

    if "customer_segment" in df_current.columns:
        df_current = pd.get_dummies(
            df_current, columns=["customer_segment"], dummy_na=False, dtype=np.int8
        )

    normalized_gender = df_current["gender"].astype("string").str.strip().str.upper()
    df_current["gender"] = normalized_gender
    # Santander source coding: H=female, V=male. Unknown/missing values remain NA.
    df_current["is_female"] = normalized_gender.map({"H": 1, "V": 0}).astype("Int8")

    for column in ["foreigner_index", "deceased_index", "spouse_index"]:
        if column in df_current.columns:
            mapped = df_current[column].astype("string").str.strip().str.upper().map({"S": 1, "N": 0})
            df_current[column] = mapped.fillna(0).astype(np.int8)

    for column in ["new_customer_index", "activity_index"]:
        if column in df_current.columns:
            df_current[column] = pd.to_numeric(df_current[column], errors="coerce").fillna(0).astype(np.int8)

    cols_to_drop = [
        "province_name",
        "snapshot_date",
        "customer_type",
        "customer_type_primary",
        "customer_relation_type",
        "address_type",
    ]
    df_current = df_current.drop(columns=[column for column in cols_to_drop if column in df_current.columns])

    # Encode any residual numeric-like source flags; keep gender categorical for fairness review.
    for column in df_current.columns:
        if column not in {"gender", "customer_id"} and df_current[column].dtype == "object":
            df_current[column] = pd.to_numeric(df_current[column], errors="coerce").fillna(0)

    interpretable_df = track_b_schema.validate(df_current.copy())
    continuous_features = ["age", "gross_income", "seniority_months", "total_products_owned"]
    model_df = prepare_model_matrix(
        interpretable_df,
        track="track_b",
        identifier_columns=["customer_id"],
        log1p_columns=[],
        scaled_features=continuous_features,
        artifact_path=preprocessor_path,
        fit=fit_preprocessor,
    )
    return interpretable_df, model_df


def main() -> None:
    from customer_segmentation.data.santander_dataset_loader import (
        load_santander_raw_data,
    )

    parser = argparse.ArgumentParser(description="Build Track B Santander product portfolio features.")
    parser.add_argument("--preprocessor", default=str(DEFAULT_PREPROCESSOR))
    parser.add_argument("--reuse-preprocessor", action="store_true")
    args = parser.parse_args()

    raw_df = load_santander_raw_data("data/raw/santander/train_ver2.csv", target_date="2016-05-28")
    interpretable_df, model_df = build_product_portfolio_features(
        raw_df,
        preprocessor_path=args.preprocessor,
        fit_preprocessor=not args.reuse_preprocessor,
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    interpretable_df.to_csv("data/processed/track_b_features_interpretable.csv", index=False)
    model_df.to_csv("data/processed/track_b_features_model.csv", index=False)
    print(f"Saved Track B matrices and preprocessor to data/processed/ and {args.preprocessor}")


if __name__ == "__main__":
    main()
