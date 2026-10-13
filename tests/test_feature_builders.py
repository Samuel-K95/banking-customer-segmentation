import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from customer_segmentation.data.santander_dataset_loader import (
    SANTANDER_PRODUCT_FEATURES,
)
from customer_segmentation.features.build_berka_features import build_berka_features
from customer_segmentation.features.build_indian_bank_features import (
    build_rfm_features,
    track_a_schema,
)
from customer_segmentation.features.build_santander_features import (
    build_product_portfolio_features,
)


def _indian_bank_transactions():
    rows = [
        ("T1", "C1", "14/09/73", "M", "MUMBAI", 1000.0, "1/1/16", 120000, 100.0),
        ("T2", "C1", "14/09/73", "M", "MUMBAI", 1300.0, "2/1/16", 130000, 200.0),
        ("T3", "C2", "01/01/80", "F", "DELHI", 900.0, "1/1/16", 210000, 250.0),
        ("T4", "C2", "01/01/80", "F", "DELHI", 1000.0, "3/1/16", 10000, 400.0),
        ("T5", "C2", "01/01/80", "F", "DELHI", 1050.0, "4/1/16", 110000, 350.0),
        ("T6", "C3", "20/03/68", "M", "PUNE", 700.0, "4/1/16", 220000, 500.0),
    ]
    columns = [
        "TransactionID",
        "CustomerID",
        "CustomerDOB",
        "CustGender",
        "CustLocation",
        "CustAccountBalance",
        "TransactionDate",
        "TransactionTime",
        "TransactionAmount (INR)",
    ]
    return pd.DataFrame(rows, columns=columns)


def _santander_snapshot():
    row_count = 6
    frame = pd.DataFrame(
        {
            "customer_id": [1000 + index for index in range(row_count)],
            "age": [25, 30, 41, 54, 62, 37],
            "seniority_months": [12, 24, 36, 48, 60, 72],
            "gross_income": [50_000.0, np.nan, 72_000.0, 80_000.0, 90_000.0, 61_000.0],
            "province_name": ["MADRID"] * row_count,
            "customer_segment": ["02 - PARTICULARES", "01 - TOP"] * 3,
            "gender": ["H", "V", None, "X", "H", "V"],
            "foreigner_index": ["N", "S", "N", "N", "S", "N"],
            "deceased_index": ["N"] * row_count,
            "spouse_index": ["N", "S", "N", "N", "N", "S"],
            "new_customer_index": [0, 0, 1, 0, 1, 0],
            "activity_index": [1, 1, 0, 1, 0, 1],
        }
    )
    for product_index, feature in enumerate(SANTANDER_PRODUCT_FEATURES, start=1):
        frame[feature] = [(product_index >> row) & 1 for row in range(row_count)]
    return frame


def _complete_santander_product_flags(frame):
    """Complete abbreviated test snapshots to the required source contract."""
    result = frame.copy()
    for feature in SANTANDER_PRODUCT_FEATURES:
        if feature not in result:
            result[feature] = 0
    return result


def _berka_tables():
    transactions = pd.DataFrame(
        [
            (1, 101, "930101", "PRIJEM", 100.0, 500.0),
            (2, 101, "930102", "VYDAJ", 50.0, 450.0),
            (3, 101, "930103", "PRIJEM", 70.0, 520.0),
            (4, 202, "930101", "VYBER", 80.0, 200.0),
            (5, 202, "930104", "PRIJEM", 20.0, 220.0),
        ],
        columns=["trans_id", "account_id", "date", "type", "amount", "balance"],
    )
    accounts = pd.DataFrame(
        [(101, "920101"), (202, "930101")],
        columns=["account_id", "date"],
    )
    return transactions, accounts


def _assert_aligned_numeric_outputs(interpretable, model, key):
    assert key in interpretable.columns
    assert key in model.columns
    assert interpretable[key].tolist() == model[key].tolist()
    assert len(interpretable) == len(model)
    assert not interpretable[key].duplicated().any()
    assert interpretable.columns.tolist() == model.columns.tolist()
    numeric = model.drop(columns=["is_female"], errors="ignore").select_dtypes(include=["number"]).to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_indian_bank_feature_aggregates_keep_customer_grain_and_city_profile(tmp_path):
    mock_data = pd.DataFrame(
        {
            "TransactionID": ["T1", "T2", "T3"],
            "CustomerID": ["C1", "C1", "C2"],
            "CustomerDOB": ["14/09/73", "14/09/73", "02/03/90"],
            "CustGender": ["M", "M", "F"],
            "CustLocation": ["MUMBAI", "MUMBAI", "NEW DELHI"],
            "CustAccountBalance": [1000.0, 1500.0, 800.0],
            "TransactionDate": ["1/1/16", "2/1/16", "3/1/16"],
            "TransactionTime": [120000, 130000, 220000],
            "TransactionAmount (INR)": [100.0, 200.0, 50.0],
        }
    )
    interpretable, model = build_rfm_features(
        mock_data, preprocessor_path=tmp_path / "track_a_stage_smoke.joblib"
    )

    assert len(interpretable) == len(model) == 2
    assert set(interpretable["CustomerID"]) == {"C1", "C2"}
    assert model["CustomerID"].tolist() == interpretable["CustomerID"].tolist()
    c1 = interpretable.set_index("CustomerID").loc["C1"]
    assert c1["Transaction_Count"] == 2
    assert c1["Total_Spend"] == 300.0
    assert c1["Age"] == 43
    assert interpretable.set_index("CustomerID").loc["C2", "Is_Tier_1_City"] == 1


def test_track_a_schema_rejects_duplicate_customers():
    duplicate = pd.DataFrame(
        {
            "CustomerID": ["C1", "C1"],
            "Age": [30.0, 30.0],
            "AccountBalance": [1.0, 1.0],
            "Transaction_Count": [1, 1],
            "Total_Spend": [1.0, 1.0],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        track_a_schema.validate(duplicate)


def test_indian_bank_builder_returns_aligned_outputs_with_unscaled_customer_id(tmp_path):
    artifact_path = tmp_path / "track_a.joblib"
    interpretable, model = build_rfm_features(
        _indian_bank_transactions(), preprocessor_path=artifact_path
    )

    _assert_aligned_numeric_outputs(interpretable, model, "CustomerID")
    assert set(interpretable["CustomerID"]) == {"C1", "C2", "C3"}
    assert interpretable["Total_Spend"].tolist() != model["Total_Spend"].tolist()
    artifact = joblib.load(artifact_path)
    assert artifact["version"] == 1
    assert artifact["track"] == "track_a"
    assert artifact["feature_order"] == [column for column in interpretable if column != "CustomerID"]
    assert "CustomerID" not in artifact["scaler"].feature_names_in_.tolist()


def test_indian_bank_builder_reuses_saved_preprocessor_without_refitting(tmp_path):
    artifact_path = tmp_path / "track_a.joblib"
    raw = _indian_bank_transactions()
    build_rfm_features(raw, preprocessor_path=artifact_path)
    original_mean = joblib.load(artifact_path)["scaler"].mean_.copy()
    changed = raw.copy()
    changed["TransactionAmount (INR)"] *= 4
    _, reused_model = build_rfm_features(
        changed, preprocessor_path=artifact_path, fit_preprocessor=False
    )

    assert np.array_equal(joblib.load(artifact_path)["scaler"].mean_, original_mean)
    assert not np.allclose(reused_model["Total_Spend"], 0.0)
    with pytest.raises(ValueError, match="not a model matrix"):
        from customer_segmentation.features.preprocessing import prepare_model_matrix

        prepare_model_matrix(
            reused_model,
            track="track_a",
            identifier_columns=["CustomerID"],
            log1p_columns=[],
            scaled_features=[],
            artifact_path=artifact_path,
            fit=False,
        )


def test_santander_builder_preserves_gender_unknowns_and_all_24_product_flags(tmp_path):
    artifact_path = tmp_path / "track_b.joblib"
    interpretable, model = build_product_portfolio_features(
        _santander_snapshot(), preprocessor_path=artifact_path
    )

    _assert_aligned_numeric_outputs(interpretable, model, "customer_id")
    assert len(SANTANDER_PRODUCT_FEATURES) == 24
    assert set(SANTANDER_PRODUCT_FEATURES).issubset(interpretable.columns)
    assert interpretable["is_female"].iloc[0] == 1
    assert interpretable["is_female"].iloc[1] == 0
    assert pd.isna(interpretable["is_female"].iloc[2])
    assert interpretable["gender"].iloc[3] == "X"
    assert pd.isna(interpretable["is_female"].iloc[3])
    assert model["customer_id"].tolist() == interpretable["customer_id"].tolist()
    for feature in SANTANDER_PRODUCT_FEATURES:
        assert set(interpretable[feature].unique()).issubset({0, 1})
        assert model[feature].tolist() == interpretable[feature].tolist()
    artifact = joblib.load(artifact_path)
    assert "customer_id" not in artifact["scaler"].feature_names_in_.tolist()


def test_santander_builder_rejects_non_binary_product_values(tmp_path):
    raw = _santander_snapshot()
    raw.loc[0, "has_savings_account"] = 2

    with pytest.raises(ValueError, match="has_savings_account"):
        build_product_portfolio_features(raw, preprocessor_path=tmp_path / "track_b.joblib")


def test_santander_builder_reuses_saved_preprocessor(tmp_path):
    artifact_path = tmp_path / "track_b.joblib"
    raw = _santander_snapshot()
    build_product_portfolio_features(raw, preprocessor_path=artifact_path)
    original_mean = joblib.load(artifact_path)["scaler"].mean_.copy()
    changed = raw.copy()
    changed["age"] += 20

    _, model = build_product_portfolio_features(
        changed, preprocessor_path=artifact_path, fit_preprocessor=False
    )

    assert np.array_equal(joblib.load(artifact_path)["scaler"].mean_, original_mean)
    assert model["customer_id"].tolist() == raw["customer_id"].tolist()


def test_santander_feature_builder_and_province_income_imputation(tmp_path):
    snapshot = _complete_santander_product_flags(
        pd.DataFrame(
            {
                "customer_id": [1, 2],
                "age": ["25", " 30"],
                "seniority_months": ["12", "24"],
                "gross_income": [50000.0, np.nan],
                "province_name": ["MADRID", "MADRID"],
                "customer_segment": ["02 - PARTICULARES", "01 - TOP"],
                "gender": ["V", "H"],
                "has_current_account": [1.0, 0.0],
            }
        )
    )
    interpretable, model = build_product_portfolio_features(
        snapshot, preprocessor_path=tmp_path / "track_b_stage_smoke.joblib"
    )

    assert len(interpretable) == len(model) == 2
    assert interpretable["gross_income"].tolist() == [50000.0, 50000.0]
    assert not model["gross_income"].isna().any()
    assert "has_current_account" in interpretable.columns
    assert "is_female" in interpretable.columns
    assert "customer_id" in model.columns


def test_santander_seniority_and_age_sentinels_are_imputed(tmp_path):
    snapshot = _complete_santander_product_flags(
        pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "age": ["25", "164", "40"],
                "seniority_months": ["12", "-999999", "36"],
                "gross_income": [50000.0, 60000.0, 70000.0],
                "gender": ["H", "V", "H"],
            }
        )
    )
    interpretable, _ = build_product_portfolio_features(
        snapshot, preprocessor_path=tmp_path / "track_b_sentinels.joblib"
    )

    row = interpretable.set_index("customer_id").loc[2]
    assert row["age"] == 32.5
    assert row["seniority_months"] == 24.0


def test_berka_builder_keeps_account_grain_and_persisted_preprocessing(tmp_path):
    artifact_path = tmp_path / "track_c.joblib"
    interpretable, model = build_berka_features(
        *_berka_tables(), preprocessor_path=artifact_path
    )

    _assert_aligned_numeric_outputs(interpretable, model, "account_id")
    assert interpretable["account_id"].tolist() == [101, 202]
    artifact = joblib.load(artifact_path)
    assert artifact["track"] == "track_c"
    assert "account_id" not in artifact["scaler"].feature_names_in_.tolist()
    assert artifact["transformations"]["clip_nonnegative_columns"] == [
        "Account_Balance",
        "Total_Credit",
        "Total_Withdrawal",
        "Account_Age_Days",
    ]


def test_berka_builder_reuses_saved_preprocessor_without_refitting(tmp_path):
    artifact_path = tmp_path / "track_c.joblib"
    transactions, accounts = _berka_tables()
    build_berka_features(transactions, accounts, preprocessor_path=artifact_path)
    original_mean = joblib.load(artifact_path)["scaler"].mean_.copy()
    changed = transactions.copy()
    changed["amount"] *= 5

    _, model = build_berka_features(
        changed,
        accounts,
        preprocessor_path=artifact_path,
        fit_preprocessor=False,
    )

    assert np.array_equal(joblib.load(artifact_path)["scaler"].mean_, original_mean)
    assert model["account_id"].tolist() == [101, 202]


def test_berka_feature_builder_preserves_credit_debit_and_latest_balance(tmp_path):
    transactions = pd.DataFrame(
        {
            "trans_id": [1, 2, 3, 4],
            "account_id": [10, 10, 10, 20],
            "date": [930101, 930105, 930110, 930110],
            "type": ["PRIJEM", "VYDAJ", "PRIJEM", "PRIJEM"],
            "amount": [1000.0, 300.0, 500.0, 200.0],
            "balance": [1000.0, 700.0, 1200.0, 200.0],
        }
    )
    accounts = pd.DataFrame({"account_id": [10, 20], "date": [921201, 930101]})

    interpretable, model = build_berka_features(
        transactions, accounts, preprocessor_path=tmp_path / "track_c_stage_smoke.joblib"
    )
    indexed = interpretable.set_index("account_id")

    assert len(indexed) == 2
    assert indexed.loc[10, "Transaction_Count"] == 3
    assert indexed.loc[10, "Total_Credit"] == 1500.0
    assert indexed.loc[10, "Total_Withdrawal"] == 300.0
    assert indexed.loc[10, "Account_Balance"] == 1200.0
    assert indexed.loc[20, "Recency_Days"] == 0
    assert model["account_id"].tolist() == interpretable["account_id"].tolist()
