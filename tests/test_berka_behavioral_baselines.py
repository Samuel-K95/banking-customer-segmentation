import json

import joblib
import numpy as np
import pandas as pd
import pytest

from customer_segmentation.models.berka_behavioral_baselines import (
    EXPECTED_FEATURES,
    BerkaTrainFittedPreprocessor,
    load_snapshot,
    make_shared_split,
    select_behavioral_features,
)

POSITIVE = [
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
]
SIGNED = ["net_cash_flow", "activity_change_count"]
BOUNDED = ["negative_balance_frequency"]


def _snapshot(rows=40):
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "account_id": np.arange(1000, 1000 + rows),
            "account_tenure_days": rng.integers(366, 1800, rows),
            "recency_days": rng.integers(0, 20, rows),
            "credit_frequency_per_day": rng.uniform(0.01, 0.15, rows),
            "debit_frequency_per_day": rng.uniform(0.01, 0.20, rows),
            "credit_amount_total": rng.lognormal(12, 0.5, rows),
            "debit_amount_total": rng.lognormal(12, 0.5, rows),
            "net_cash_flow": rng.normal(0, 50000, rows),
            "transaction_amount_median": rng.lognormal(7, 0.6, rows),
            "balance_std": rng.lognormal(9, 0.4, rows),
            "negative_balance_frequency": rng.uniform(0, 0.3, rows),
            "interarrival_days_median": rng.uniform(0, 20, rows),
            "interarrival_days_std": rng.uniform(0, 25, rows),
            "recent_90d_transaction_frequency_per_day": rng.uniform(0, 0.5, rows),
            "activity_change_count": rng.integers(-15, 16, rows).astype(float),
        }
    )
    if rows > 2:
        frame.loc[2, "activity_change_count"] = np.nan
    return frame


def _preprocessor():
    return BerkaTrainFittedPreprocessor(
        EXPECTED_FEATURES,
        positive_log1p=POSITIVE,
        signed_asinh=SIGNED,
        bounded_unchanged=BOUNDED,
    )


def test_shared_account_split_is_deterministic_disjoint_and_seed_independent():
    eligible = _snapshot()
    train, held_out = make_shared_split(eligible, random_seed=8675309)
    shuffled_train, shuffled_hold_out = make_shared_split(
        eligible.sample(frac=1, random_state=8), random_seed=8675309
    )

    assert len(train) == 30
    assert len(held_out) == 10
    assert set(train.account_id).isdisjoint(held_out.account_id)
    assert train.account_id.tolist() == shuffled_train.account_id.tolist()
    assert held_out.account_id.tolist() == shuffled_hold_out.account_id.tolist()


def test_feature_selection_preserves_reviewed_order_and_excludes_identifier():
    snapshot = _snapshot()
    selected = select_behavioral_features(snapshot)

    assert selected.columns.tolist() == EXPECTED_FEATURES
    assert "account_id" not in selected
    assert selected["gross_money_movement"].equals(
        snapshot["credit_amount_total"] + snapshot["debit_amount_total"]
    )
    with pytest.raises(ValueError, match="reviewed Berka"):
        select_behavioral_features(snapshot, list(reversed(EXPECTED_FEATURES)))


def test_preprocessor_fits_imputation_clipping_and_scaling_only_on_training_rows():
    frame = _snapshot()
    train, hold_out = make_shared_split(frame, random_seed=23)
    train_features = select_behavioral_features(train)
    held_features = select_behavioral_features(hold_out)
    processor = _preprocessor().fit(train_features, train.account_id.to_numpy())
    metadata_before = processor.metadata()
    matrix = processor.transform(held_features)

    changed_holdout = held_features.copy()
    changed_holdout.loc[:, "gross_money_movement"] = 1e15
    changed_holdout.loc[:, "net_cash_flow"] = -1e15
    changed_matrix = processor.transform(changed_holdout)

    assert matrix.shape == (10, len(EXPECTED_FEATURES))
    assert np.isfinite(matrix).all()
    assert not np.array_equal(matrix, changed_matrix)
    assert processor.metadata() == metadata_before
    assert metadata_before["fit_rows"] == len(train)
    assert metadata_before["fit_scope"] == "training accounts only"
    assert len(metadata_before["training_fitted_clip_bounds"]) == len(EXPECTED_FEATURES)


def test_preprocessor_requires_finite_feature_schema_and_valid_bounded_values():
    with pytest.raises(ValueError, match="cover the selected feature set"):
        BerkaTrainFittedPreprocessor(EXPECTED_FEATURES, positive_log1p=POSITIVE)

    frame = select_behavioral_features(_snapshot())
    frame.loc[0, "negative_balance_frequency"] = 1.2
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        _preprocessor().fit(frame.iloc[:20], np.arange(20))


def test_preprocessor_state_roundtrip_preserves_held_out_transform(tmp_path):
    frame = _snapshot()
    train, held_out = make_shared_split(frame, random_seed=37)
    train_features = select_behavioral_features(train)
    held_features = select_behavioral_features(held_out)
    processor = _preprocessor().fit(train_features, train.account_id.to_numpy())
    expected = processor.transform(held_features)
    path = tmp_path / "preprocessor_state.joblib"
    joblib.dump(processor.state_dict(), path)

    restored = BerkaTrainFittedPreprocessor.from_state_dict(joblib.load(path))

    assert np.array_equal(restored.transform(held_features), expected)
    assert restored.metadata() == processor.metadata()


def test_snapshot_loader_checks_date_version_and_unique_account_keys(tmp_path):
    frame = _snapshot(rows=2)
    feature_definitions = {column: "synthetic" for column in frame.columns if column != "account_id"}
    metadata = {
        "as_of_date": "1997-12-31",
        "feature_version": "berka_account_snapshot_v1",
        "summary": {"rows": 2},
        "feature_definitions": feature_definitions,
    }
    (tmp_path / "berka_account_snapshot_features.csv").write_text(frame.to_csv(index=False), encoding="utf-8")
    (tmp_path / "berka_account_snapshot_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    loaded, loaded_metadata = load_snapshot(tmp_path, expected_as_of_date="1997-12-31")
    assert len(loaded) == 2
    assert loaded_metadata["as_of_date"] == "1997-12-31"
    with pytest.raises(ValueError, match="as_of_date mismatch"):
        load_snapshot(tmp_path, expected_as_of_date="1998-12-31")

    frame.loc[1, "account_id"] = frame.loc[0, "account_id"]
    frame.to_csv(tmp_path / "berka_account_snapshot_features.csv", index=False)
    with pytest.raises(ValueError, match="one non-null row per account_id"):
        load_snapshot(tmp_path, expected_as_of_date="1997-12-31")
