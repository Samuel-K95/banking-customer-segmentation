import numpy as np
import pandas as pd
import pytest

from customer_segmentation.models.berka_behavioral_baselines import EXPECTED_FEATURES
from customer_segmentation.models.berka_feature_sensitivity import (
    _preprocessor_for_features,
    align_to_gross_movement,
    compare_held_out_assignments,
    make_feature_configurations,
)

EXCLUSIONS = {
    "original_13": [],
    "without_gross_money_movement": ["gross_money_movement"],
    "without_balance_volatility": ["balance_std"],
    "without_gross_and_balance_volatility": ["gross_money_movement", "balance_std"],
}
BASELINE_PREPROCESSING = {
    "positive_log1p": [
        "account_tenure_days",
        "recency_days",
        "credit_frequency_per_day",
        "debit_frequency_per_day",
        "gross_money_movement",
        "transaction_amount_median",
        "balance_std",
        "interarrival_days_median",
        "interarrival_days_std",
        "recent_90d_transaction_frequency_per_day",
    ],
    "signed_asinh": ["net_cash_flow", "activity_change_count"],
    "bounded_unchanged": ["negative_balance_frequency"],
    "clipping_quantiles": [0.01, 0.99],
}


def _selected_frame(rows=36):
    rng = np.random.default_rng(77)
    values = {
        "credit_amount_total": rng.lognormal(12, 0.5, rows),
        "debit_amount_total": rng.lognormal(12, 0.5, rows),
    }
    for feature in EXPECTED_FEATURES:
        if feature == "gross_money_movement":
            continue
        elif feature in {"net_cash_flow", "activity_change_count"}:
            values[feature] = rng.normal(0, 100, rows)
        elif feature == "negative_balance_frequency":
            values[feature] = rng.uniform(0, 1, rows)
        elif feature == "account_tenure_days":
            values[feature] = rng.integers(366, 1800, rows)
        else:
            values[feature] = rng.uniform(0, 100, rows)
    frame = pd.DataFrame(values)
    frame["gross_money_movement"] = frame["credit_amount_total"] + frame["debit_amount_total"]
    return frame.loc[:, EXPECTED_FEATURES]


def test_four_feature_sets_remove_only_the_configured_dominant_variables():
    configurations = make_feature_configurations(EXPECTED_FEATURES, EXCLUSIONS)

    assert list(configurations) == list(EXCLUSIONS)
    assert len(configurations["original_13"]) == 13
    assert len(configurations["without_gross_money_movement"]) == 12
    assert len(configurations["without_balance_volatility"]) == 12
    assert len(configurations["without_gross_and_balance_volatility"]) == 11
    assert configurations["without_gross_money_movement"] == [
        feature for feature in EXPECTED_FEATURES if feature != "gross_money_movement"
    ]
    with pytest.raises(ValueError, match="unknown features"):
        make_feature_configurations(EXPECTED_FEATURES, {"bad": ["unknown"]})


def test_each_ablation_gets_train_fitted_preprocessor_in_its_own_feature_space():
    frame = _selected_frame()
    train = frame.iloc[:27].copy()
    held_out = frame.iloc[27:].copy()
    configurations = make_feature_configurations(EXPECTED_FEATURES, EXCLUSIONS)

    for feature_order in configurations.values():
        train_subset = train.loc[:, feature_order]
        holdout_subset = held_out.loc[:, feature_order]
        processor = _preprocessor_for_features(BASELINE_PREPROCESSING, feature_order).fit(
            train_subset, np.arange(100, 127)
        )
        fit_state = processor.metadata()
        transformed = processor.transform(holdout_subset)
        altered = holdout_subset.copy()
        if "gross_money_movement" in feature_order:
            altered["gross_money_movement"] = altered["gross_money_movement"] * 100
        if "balance_std" in feature_order:
            altered["balance_std"] = altered["balance_std"] * 100

        assert transformed.shape == (9, len(feature_order))
        assert np.isfinite(transformed).all()
        assert processor.metadata() == fit_state
        assert fit_state["fit_rows"] == 27
        assert fit_state["fit_scope"] == "training accounts only"
        assert fit_state["feature_order"] == feature_order
        assert len(processor.transform(altered)) == 9


def test_profile_orientation_is_consistent_when_raw_cluster_numbers_flip():
    labels = np.array([1, 1, 0, 0])
    gross = np.array([10.0, 20.0, 100.0, 120.0])
    first_codes, first_map = align_to_gross_movement(labels, gross)
    second_codes, second_map = align_to_gross_movement(1 - labels, gross)

    assert first_codes.tolist() == second_codes.tolist()
    assert first_map[1] == second_map[0] == "lower_gross_movement"
    assert first_map[0] == second_map[1] == "higher_gross_movement"


def test_configuration_ari_requires_the_same_ordered_held_out_accounts():
    ids = np.array(["a", "b", "c", "d"])
    labels_left = np.array([0, 0, 1, 1])
    labels_right = np.array([1, 1, 0, 0])

    assert compare_held_out_assignments(labels_left, ids, labels_right, ids.copy()) == 1.0
    with pytest.raises(ValueError, match="identical ordered held-out"):
        compare_held_out_assignments(labels_left, ids, labels_right, ids[::-1])
