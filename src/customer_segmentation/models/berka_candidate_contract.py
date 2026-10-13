"""Strict batch request/response contract for the read-only Berka candidate."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from customer_segmentation.models.berka_behavioral_baselines import EXPECTED_FEATURES
from customer_segmentation.models.berka_candidate_inference import BerkaKMeansCandidate

MODEL_VERSION = "berka_behavioral_baselines_v1_1:kmeans:k2:seed42"
SNAPSHOT_VERSION = "berka_account_snapshot_v1"
AS_OF_DATE = "1997-12-31"
MINIMUM_TENURE_DAYS = 365

REQUEST_FIELDS = {"model_version", "snapshot_version", "as_of_date", "accounts"}
ACCOUNT_FIELDS = {"account_id", "features"}
INTEGER_FEATURES = {"account_tenure_days", "recency_days", "activity_change_count"}
NONNEGATIVE_FEATURES = {
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
}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    observed = set(value)
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if missing or extra:
        raise ValueError(f"{context} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def _numeric_feature(name: str, value: Any) -> Any:
    """Validate a JSON-compatible finite number without bool/string coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"Feature {name} must be a real number; booleans and numeric strings are invalid")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Feature {name} must be finite")
    if name in INTEGER_FEATURES and not numeric.is_integer():
        raise ValueError(f"Feature {name} must be integer-valued")
    if name in NONNEGATIVE_FEATURES and numeric < 0:
        raise ValueError(f"Feature {name} must be nonnegative")
    if name == "negative_balance_frequency" and not 0 <= numeric <= 1:
        raise ValueError("Feature negative_balance_frequency must be within [0, 1]")
    return value


def _validate_features(features: Any, *, eligible: bool) -> dict[str, Any]:
    if not isinstance(features, Mapping):
        raise TypeError("features must be an object keyed by the saved feature names")
    expected = set(EXPECTED_FEATURES)
    provided = set(features)
    extra = provided.difference(expected)
    missing = expected.difference(provided)
    if extra:
        raise ValueError(f"features contains unknown names: {sorted(extra)}")
    if "account_tenure_days" not in features:
        raise ValueError("features must include account_tenure_days to determine eligibility")
    if eligible and missing:
        raise ValueError(f"eligible account is missing required features: {sorted(missing)}")
    if eligible and list(features) != list(EXPECTED_FEATURES):
        raise ValueError("eligible account features must use the saved 13-feature order exactly")
    return {name: _numeric_feature(name, features[name]) for name in EXPECTED_FEATURES if name in features}


def _validate_feature_relationships(features: Mapping[str, float | int]) -> None:
    if "recency_days" in features and features["recency_days"] > features["account_tenure_days"]:
        raise ValueError("recency_days cannot exceed account_tenure_days")
    if "gross_money_movement" in features and "net_cash_flow" in features:
        gross = float(features["gross_money_movement"])
        net = abs(float(features["net_cash_flow"]))
        tolerance = 1e-9 * max(1.0, gross)
        if gross + tolerance < net:
            raise ValueError("gross_money_movement cannot be smaller than absolute net_cash_flow")


def _validate_candidate(candidate: BerkaKMeansCandidate) -> None:
    if candidate.model_version != MODEL_VERSION:
        raise ValueError(f"Unsupported Berka model_version: {candidate.model_version}")
    if candidate.as_of_date != AS_OF_DATE:
        raise ValueError(f"Unsupported Berka as_of_date: {candidate.as_of_date}")
    if tuple(candidate.feature_order) != tuple(EXPECTED_FEATURES):
        raise ValueError("Loaded candidate feature order differs from the original 13-feature contract")
    model_path = getattr(candidate, "model_path", None)
    if model_path is not None:
        run_metadata_path = model_path.parent.parent / "run_metadata.json"
        if not run_metadata_path.is_file():
            raise FileNotFoundError(f"Berka baseline run metadata not found: {run_metadata_path}")
        saved_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if (
            saved_metadata.get("experiment") != "berka_behavioral_baselines_v1_1"
            or saved_metadata.get("as_of_date") != AS_OF_DATE
            or saved_metadata.get("feature_order") != list(EXPECTED_FEATURES)
        ):
            raise ValueError("Candidate does not match the saved Berka baseline run metadata")
        candidate_snapshot = saved_metadata.get("snapshot_version")
    else:
        # Synthetic/local unit fixtures expose this value directly; real candidates
        # resolve it from their saved run_metadata.json above.
        candidate_snapshot = getattr(candidate, "snapshot_version", None)
    if candidate_snapshot != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported Berka snapshot_version: {candidate_snapshot}")


def predict_batch(candidate: BerkaKMeansCandidate, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a versioned request and call the frozen candidate for eligible rows.

    Ineligible accounts must provide an account ID and finite integer tenure, but
    may omit other known feature values. Supplied values are still domain-checked;
    incomplete features are ignored because the account is not scored. A precomputed
    feature object cannot prove that its source transactions respected the as-of
    cutoff; upstream aggregation must enforce and attest to that provenance.
    """
    _validate_candidate(candidate)
    if not isinstance(request, Mapping):
        raise TypeError("Batch request must be an object")
    _require_exact_keys(request, REQUEST_FIELDS, context="request")
    for field, expected in (
        ("model_version", MODEL_VERSION),
        ("snapshot_version", SNAPSHOT_VERSION),
        ("as_of_date", AS_OF_DATE),
    ):
        if not isinstance(request[field], str) or request[field] != expected:
            raise ValueError(f"Unsupported {field}: expected {expected!r}")

    accounts = request["accounts"]
    if not isinstance(accounts, Sequence) or isinstance(accounts, (str, bytes, bytearray)):
        raise TypeError("accounts must be an ordered list of account request objects")
    if len(accounts) == 0:
        raise ValueError("accounts batch must not be empty")

    seen_ids: set[str] = set()
    validated: list[tuple[str, dict[str, Any], bool]] = []
    for position, account in enumerate(accounts):
        if not isinstance(account, Mapping):
            raise TypeError(f"accounts[{position}] must be an object")
        _require_exact_keys(account, ACCOUNT_FIELDS, context=f"accounts[{position}]")
        account_id = account["account_id"]
        if not isinstance(account_id, str):
            raise TypeError(f"accounts[{position}].account_id must be a string")
        if not account_id.strip():
            raise ValueError(f"accounts[{position}].account_id must be a non-empty string")
        if account_id in seen_ids:
            raise ValueError(f"Duplicate account_id in request: {account_id!r}")
        seen_ids.add(account_id)

        features = account["features"]
        if not isinstance(features, Mapping):
            raise TypeError(f"accounts[{position}].features must be an object")
        if "account_tenure_days" not in features:
            raise ValueError(f"accounts[{position}].features must include account_tenure_days")
        tenure = _numeric_feature("account_tenure_days", features["account_tenure_days"])
        eligible = tenure >= MINIMUM_TENURE_DAYS
        normalized = _validate_features(features, eligible=eligible)
        _validate_feature_relationships(normalized)
        validated.append((account_id, normalized, eligible))

    eligible_rows = [(account_id, values) for account_id, values, eligible in validated if eligible]
    scored: dict[str, int] = {}
    if eligible_rows:
        feature_frame = pd.DataFrame(
            [values for _, values in eligible_rows],
            columns=list(EXPECTED_FEATURES),
        )
        feature_frame.insert(0, "account_id", [account_id for account_id, _ in eligible_rows])
        prediction_frame = candidate.predict_features(feature_frame)
        if not isinstance(prediction_frame, pd.DataFrame) or "cluster_id" not in prediction_frame:
            raise TypeError("candidate.predict_features must return a DataFrame containing cluster_id")
        if len(prediction_frame) != len(eligible_rows):
            raise ValueError("Candidate returned a prediction count that does not match eligible accounts")
        if "account_id" in prediction_frame and prediction_frame["account_id"].tolist() != [
            account_id for account_id, _ in eligible_rows
        ]:
            raise ValueError("Candidate predictions changed eligible account order or identity")
        for (account_id, _), label in zip(eligible_rows, prediction_frame["cluster_id"], strict=True):
            if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) or int(label) not in (0, 1):
                raise ValueError("K-Means cluster_id must be integer 0 or 1")
            scored[account_id] = int(label)

    predictions = [
        {
            "account_id": account_id,
            "model_version": MODEL_VERSION,
            "snapshot_version": SNAPSHOT_VERSION,
            "as_of_date": AS_OF_DATE,
            "eligibility_status": "assigned" if eligible else "insufficient_history",
            "cluster_id": scored[account_id] if eligible else None,
        }
        for account_id, _features, eligible in validated
    ]
    return {
        "model_version": MODEL_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "as_of_date": AS_OF_DATE,
        "predictions": predictions,
    }
