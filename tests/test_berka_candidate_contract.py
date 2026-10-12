from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.models.berka_behavioral_baselines import (
    EXPECTED_FEATURES,
    select_behavioral_features,
)
from customer_segmentation.models.berka_candidate_contract import (
    AS_OF_DATE,
    MODEL_VERSION,
    SNAPSHOT_VERSION,
    predict_batch,
)
from customer_segmentation.models.berka_candidate_inference import (
    DEFAULT_INTEGRITY_MANIFEST,
    load_berka_kmeans_candidate,
)


def _features(**overrides):
    values = OrderedDict(
        [
            ("account_tenure_days", 500),
            ("recency_days", 20),
            ("credit_frequency_per_day", 0.08),
            ("debit_frequency_per_day", 0.10),
            ("gross_money_movement", 1000.0),
            ("net_cash_flow", 100.0),
            ("transaction_amount_median", 40.0),
            ("balance_std", 20.0),
            ("negative_balance_frequency", 0.05),
            ("interarrival_days_median", 4.0),
            ("interarrival_days_std", 2.0),
            ("recent_90d_transaction_frequency_per_day", 0.20),
            ("activity_change_count", 1),
        ]
    )
    assert list(values) == EXPECTED_FEATURES
    values.update(overrides)
    return values


class _FakeEstimator:
    def __init__(self):
        self.matrices = []

    def predict(self, matrix):
        values = np.asarray(matrix)
        self.matrices.append(values.copy())
        return (values[:, EXPECTED_FEATURES.index("net_cash_flow")] > 0).astype(int)


class _FakeCandidate:
    model_version = MODEL_VERSION
    snapshot_version = SNAPSHOT_VERSION
    as_of_date = AS_OF_DATE
    feature_order = tuple(EXPECTED_FEATURES)

    def __init__(self):
        self.estimator = _FakeEstimator()
        self.calls = 0

    def predict_features(self, frame):
        self.calls += 1
        matrix = frame.loc[:, list(self.feature_order)].to_numpy(dtype=float)
        assert matrix.shape[1] == 13
        labels = self.estimator.predict(matrix)
        return pd.DataFrame(
            {"account_id": frame["account_id"].to_numpy(), "cluster_id": labels},
            index=frame.index,
        )


def _request(accounts):
    return {
        "model_version": MODEL_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "as_of_date": AS_OF_DATE,
        "accounts": accounts,
    }


def test_valid_batch_preserves_request_order_and_returns_only_model_cluster_ids():
    candidate = _FakeCandidate()
    input_features = _features(net_cash_flow=-50.0)
    request = _request(
        [
            {"account_id": "account-z", "features": input_features},
            {"account_id": "account-a", "features": _features(net_cash_flow=75.0)},
        ]
    )

    response = predict_batch(candidate, request)
    repeated_response = predict_batch(candidate, request)

    assert [item["account_id"] for item in response["predictions"]] == ["account-z", "account-a"]
    assert [item["cluster_id"] for item in response["predictions"]] == [0, 1]
    for item in response["predictions"]:
        assert item["model_version"] == MODEL_VERSION
        assert item["snapshot_version"] == SNAPSHOT_VERSION
        assert item["as_of_date"] == "1997-12-31"
        assert item["eligibility_status"] == "assigned"
        assert set(item) == {
            "account_id", "model_version", "snapshot_version", "as_of_date", "eligibility_status", "cluster_id"
        }
    assert candidate.calls == 2
    assert candidate.estimator.matrices[0].shape == (2, 13)
    assert request["accounts"][0]["features"] == input_features
    assert repeated_response == response


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_version", "other-model", "model_version"),
        ("snapshot_version", "other-snapshot", "snapshot_version"),
        ("as_of_date", "1998-12-31", "as_of_date"),
    ],
)
def test_request_rejects_nonbaseline_versions(field, value, message):
    candidate = _FakeCandidate()
    request = _request([{"account_id": "a", "features": _features()}])
    request[field] = value

    with pytest.raises(ValueError, match=message):
        predict_batch(candidate, request)
    assert candidate.calls == 0


@pytest.mark.parametrize("attribute, value", [("model_version", "unknown"), ("snapshot_version", "unknown")])
def test_candidate_artifact_versions_must_match_contract(attribute, value):
    candidate = _FakeCandidate()
    setattr(candidate, attribute, value)

    with pytest.raises(ValueError, match="Unsupported Berka"):
        predict_batch(candidate, _request([{"account_id": "a", "features": _features()}]))
    assert candidate.calls == 0


def test_eligible_accounts_require_exact_feature_names_and_order():
    candidate = _FakeCandidate()
    missing = _features()
    del missing["balance_std"]
    with pytest.raises(ValueError, match="missing required features"):
        predict_batch(candidate, _request([{"account_id": "a", "features": missing}]))

    extra = _features(unexpected_value=1.0)
    with pytest.raises(ValueError, match="unknown names"):
        predict_batch(candidate, _request([{"account_id": "a", "features": extra}]))

    reordered = OrderedDict(reversed(list(_features().items())))
    with pytest.raises(ValueError, match="13-feature order"):
        predict_batch(candidate, _request([{"account_id": "a", "features": reordered}]))
    assert candidate.calls == 0


@pytest.mark.parametrize(
    ("feature", "value", "error", "message"),
    [
        ("net_cash_flow", True, TypeError, "booleans"),
        ("net_cash_flow", "100", TypeError, "real number"),
        ("net_cash_flow", float("nan"), ValueError, "finite"),
        ("net_cash_flow", float("inf"), ValueError, "finite"),
        ("debit_frequency_per_day", -0.1, ValueError, "nonnegative"),
        ("negative_balance_frequency", 1.1, ValueError, r"within \[0, 1\]"),
        ("activity_change_count", 1.5, ValueError, "integer-valued"),
        ("recency_days", 501, ValueError, "cannot exceed"),
        ("gross_money_movement", 20.0, ValueError, "smaller than absolute net_cash_flow"),
    ],
)
def test_eligible_accounts_reject_invalid_values_and_domain_ranges(feature, value, error, message):
    candidate = _FakeCandidate()
    with pytest.raises(error, match=message):
        predict_batch(
            candidate,
            _request([{"account_id": "a", "features": _features(**{feature: value})}]),
        )
    assert candidate.calls == 0


def test_duplicate_empty_and_nonstring_account_ids_are_rejected():
    candidate = _FakeCandidate()
    for ids in (["a", "a"], [" ", "b"], [123]):
        request = _request([{"account_id": account_id, "features": _features()} for account_id in ids])
        with pytest.raises(ValueError if ids != [123] else TypeError):
            predict_batch(candidate, request)
    assert candidate.calls == 0


def test_ineligible_account_may_omit_other_features_and_is_never_scored():
    candidate = _FakeCandidate()
    response = predict_batch(
        candidate,
        _request(
            [
                {"account_id": "short-history", "features": {"account_tenure_days": 364}},
                {"account_id": "eligible", "features": _features()},
            ]
        ),
    )

    insufficient, assigned = response["predictions"]
    assert insufficient["eligibility_status"] == "insufficient_history"
    assert insufficient["cluster_id"] is None
    assert insufficient["account_id"] == "short-history"
    assert assigned["eligibility_status"] == "assigned"
    assert candidate.calls == 1
    assert candidate.estimator.matrices[0].shape == (1, 13)


def test_ineligible_account_still_requires_valid_tenure_and_known_partial_features():
    candidate = _FakeCandidate()
    no_tenure = {"account_id": "short", "features": {}}
    unknown = {"account_id": "short", "features": {"account_tenure_days": 100, "foo": 1.0}}
    invalid = {"account_id": "short", "features": {"account_tenure_days": 100, "net_cash_flow": np.nan}}
    for account in (no_tenure, unknown, invalid):
        with pytest.raises((TypeError, ValueError)):
            predict_batch(candidate, _request([account]))
    assert candidate.calls == 0


def test_request_rejects_unexpected_top_level_and_account_fields():
    candidate = _FakeCandidate()
    request = _request([{"account_id": "a", "features": _features()}])
    request["debug"] = True
    with pytest.raises(ValueError, match="request schema mismatch"):
        predict_batch(candidate, request)
    request = _request([{"account_id": "a", "features": _features()}])
    request["transaction_date_provenance"] = "verified"
    with pytest.raises(ValueError, match="request schema mismatch"):
        predict_batch(candidate, request)
    request = _request([{"account_id": "a", "features": _features(), "cluster_id": 1}])
    with pytest.raises(ValueError, match=r"accounts\[0\] schema mismatch"):
        predict_batch(candidate, request)
    assert candidate.calls == 0


def test_frozen_historical_contract_to_model_replay_when_artifacts_exist():
    root = Path.cwd()
    baseline = root / "artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31"
    snapshot_path = root / "data/processed/berka_snapshot/v1/asof_1997-12-31/berka_account_snapshot_features.csv"
    assignments_path = baseline / "held_out_assignments.csv"
    required = [
        root / DEFAULT_INTEGRITY_MANIFEST,
        baseline / "models/kmeans_k2_seed42.joblib",
        baseline / "preprocessor_train_fitted.joblib",
        snapshot_path,
        baseline / "account_split.csv",
        assignments_path,
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Optional Berka historical research artifacts are not available locally")

    candidate = load_berka_kmeans_candidate(root / DEFAULT_INTEGRITY_MANIFEST, project_root=root)
    split = pd.read_csv(baseline / "account_split.csv")
    held_ids = split.loc[split["split"].eq("held_out"), "account_id"].to_numpy()
    snapshot = pd.read_csv(snapshot_path).set_index("account_id").loc[held_ids].reset_index()
    features = select_behavioral_features(snapshot, list(EXPECTED_FEATURES))
    request = _request(
        [
            {"account_id": str(account_id), "features": row.to_dict()}
            for account_id, (_, row) in zip(held_ids, features.iterrows(), strict=True)
        ]
    )

    response = predict_batch(candidate, request)

    saved = pd.read_csv(assignments_path)
    frozen = saved.loc[
        saved["algorithm"].eq("kmeans") & saved["k"].eq(2) & saved["seed"].eq(42)
    ].set_index("account_id").loc[held_ids, "cluster"].to_numpy()
    predictions = response["predictions"]
    assert len(predictions) == 901
    assert [item["account_id"] for item in predictions] == [str(value) for value in held_ids]
    assert [item["cluster_id"] for item in predictions] == frozen.tolist()
    assert all(item["eligibility_status"] == "assigned" for item in predictions)
    assert response["as_of_date"] == "1997-12-31"
    assert response["model_version"] == candidate.model_version
    assert response["snapshot_version"] == SNAPSHOT_VERSION
    assert "provenance_verified" not in response
    assert "transaction_date_provenance" not in response
    assert all("transaction_date_provenance" not in item for item in predictions)
