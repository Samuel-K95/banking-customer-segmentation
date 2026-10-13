import numpy as np
import pandas as pd
import pytest

from customer_segmentation.features.build_banksim_features import (
    MODEL_FEATURES,
    SHARE_FEATURES,
    build_banksim_features,
)


def transactions():
    rows = [
        # C1: commuter, active days 0, 1, 3; one fraud payment
        (0, "C1", "2", "F", "M1", "es_transportation", 10.0, 0),
        (1, "C1", "2", "F", "M1", "es_transportation", 10.0, 0),
        (3, "C1", "2", "F", "M2", "es_food", 20.0, 1),
        # C2: single leisure payment on the last day
        (4, "C2", "5", "M", "M3", "es_leisure", 40.0, 0),
        # C3: two payments on day 2, shopping + health
        (2, "C3", "U", "U", "M4", "es_fashion", 30.0, 0),
        (2, "C3", "U", "U", "M5", "es_health", 10.0, 0),
    ]
    return pd.DataFrame(rows, columns=["step", "customer", "age", "gender", "merchant", "category", "amount", "fraud"])


def test_customer_features_are_computed_per_customer(tmp_path):
    interpretable, _ = build_banksim_features(transactions(), preprocessor_path=tmp_path / "pre.joblib")
    by_id = interpretable.set_index("customer_id")

    c1 = by_id.loc["C1"]
    assert c1["txn_count"] == 3 and c1["active_days"] == 3
    assert c1["total_amount"] == 40.0 and c1["max_amount"] == 20.0
    assert c1["recency_days"] == 1 and c1["tenure_days"] == 4  # last step in the data is 4
    assert c1["interarrival_days_median"] == 1.5  # gaps 1 and 2
    assert c1["share_transport"] == 0.5 and c1["share_food"] == 0.5
    assert c1["any_fraud"] == 1 and c1["fraud_txn_count"] == 1

    c2 = by_id.loc["C2"]
    assert c2["std_amount"] == 0.0 and c2["interarrival_days_median"] == 0.0
    assert c2["recency_days"] == 0 and c2["share_leisure_travel"] == 1.0

    c3 = by_id.loc["C3"]
    assert c3["active_days"] == 1 and c3["n_merchants"] == 2
    assert c3["share_shopping"] == 0.75 and c3["share_health_beauty"] == 0.25

    assert np.allclose(interpretable[SHARE_FEATURES].sum(axis=1), 1.0)


def test_demographics_and_outcomes_stay_out_of_the_model_matrix(tmp_path):
    interpretable, model = build_banksim_features(transactions(), preprocessor_path=tmp_path / "pre.joblib")
    assert list(model.columns) == ["customer_id", *MODEL_FEATURES]
    for column in ["age_band", "gender", "fraud_txn_count", "any_fraud"]:
        assert column in interpretable.columns
        assert column not in model.columns


def test_saved_preprocessor_is_reused(tmp_path):
    path = tmp_path / "pre.joblib"
    _, fitted = build_banksim_features(transactions(), preprocessor_path=path)
    _, reused = build_banksim_features(transactions(), preprocessor_path=path, fit_preprocessor=False)
    pd.testing.assert_frame_equal(fitted, reused)


def test_unmapped_category_is_rejected(tmp_path):
    tx = transactions()
    tx.loc[0, "category"] = "es_crypto"
    with pytest.raises(ValueError, match="Unmapped BankSim categories"):
        build_banksim_features(tx, preprocessor_path=tmp_path / "pre.joblib")
