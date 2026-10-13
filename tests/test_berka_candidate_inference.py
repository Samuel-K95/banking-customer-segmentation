import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from customer_segmentation.models import berka_candidate_inference as inference
from customer_segmentation.models.berka_behavioral_baselines import (
    EXPECTED_FEATURES,
    BerkaTrainFittedPreprocessor,
    _sha256_ids,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_artifacts(root: Path) -> tuple[Path, pd.DataFrame]:
    model_path = root / "artifacts" / "models" / "berka_kmeans.joblib"
    preprocessor_path = root / "artifacts" / "preprocessor.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(812)
    feature_values = {
        "account_tenure_days": rng.integers(366, 1800, 16).astype(float),
        "recency_days": rng.integers(0, 40, 16).astype(float),
        "credit_frequency_per_day": rng.uniform(0.01, 0.15, 16),
        "debit_frequency_per_day": rng.uniform(0.01, 0.20, 16),
        "gross_money_movement": rng.lognormal(12, 0.4, 16),
        "net_cash_flow": rng.normal(5000, 20000, 16),
        "transaction_amount_median": rng.lognormal(7, 0.5, 16),
        "balance_std": rng.lognormal(9, 0.4, 16),
        "negative_balance_frequency": rng.uniform(0, 0.2, 16),
        "interarrival_days_median": rng.uniform(0, 12, 16),
        "interarrival_days_std": rng.uniform(0, 15, 16),
        "recent_90d_transaction_frequency_per_day": rng.uniform(0, 0.4, 16),
        "activity_change_count": rng.integers(-10, 11, 16).astype(float),
    }
    train = pd.DataFrame(feature_values, columns=EXPECTED_FEATURES)
    training_ids = np.asarray([f"train-{index}" for index in range(len(train))])
    preprocessor = BerkaTrainFittedPreprocessor(
        EXPECTED_FEATURES,
        positive_log1p=POSITIVE,
        signed_asinh=SIGNED,
        bounded_unchanged=BOUNDED,
    ).fit(train, training_ids)
    # Minimal initialized KMeans state avoids fitting a candidate in these tests.
    estimator = KMeans(n_clusters=2, random_state=42)
    estimator.cluster_centers_ = np.vstack(
        [np.full(len(EXPECTED_FEATURES), -0.5), np.full(len(EXPECTED_FEATURES), 0.5)]
    )
    estimator.n_features_in_ = len(EXPECTED_FEATURES)
    estimator._n_threads = 1
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"preprocessor_state": preprocessor.state_dict(), "metadata": preprocessor.metadata()},
        preprocessor_path,
    )
    heldout_ids = np.asarray(["acct-z", "acct-a", "acct-m"])
    model_metadata = {
        "experiment": "synthetic_fixture",
        "algorithm": "kmeans",
        "k": 2,
        "seed": 42,
        "as_of_date": "1997-12-31",
        "feature_order": list(EXPECTED_FEATURES),
        "training_account_ids_sha256": _sha256_ids(training_ids),
        "held_out_account_ids_sha256": _sha256_ids(heldout_ids),
        "preprocessor_path": "artifacts/preprocessor.joblib",
        "training_rows": len(train),
        "held_out_rows": len(heldout_ids),
    }
    joblib.dump({"estimator": estimator, "metadata": model_metadata}, model_path)
    manifest = {
        "kmeans": {
            "path": "artifacts/models/berka_kmeans.joblib",
            "sha256": _sha256(model_path),
            "algorithm": "kmeans",
            "k": 2,
            "seed": 42,
            "training_account_ids_sha256": _sha256_ids(training_ids),
            "held_out_account_ids_sha256": _sha256_ids(heldout_ids),
            "replay_matches_frozen_assignments": True,
        },
        "preprocessor": {
            "path": "artifacts/preprocessor.joblib",
            "sha256": _sha256(preprocessor_path),
            "feature_order": list(EXPECTED_FEATURES),
            "fit_rows": len(train),
            "fit_account_ids_sha256": _sha256_ids(training_ids),
            "fit_scope": "training accounts only",
        },
    }
    manifest_path = root / "integrity.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions = pd.DataFrame(
        rng.uniform(1, 10, size=(len(heldout_ids), len(EXPECTED_FEATURES))),
        columns=EXPECTED_FEATURES,
        index=[20, 4, 13],
    )
    predictions["negative_balance_frequency"] = rng.uniform(0, 0.2, len(heldout_ids))
    predictions.insert(0, "account_id", heldout_ids)
    return manifest_path, predictions


def test_verified_loading_prediction_excludes_account_id_and_preserves_order(tmp_path):
    manifest_path, frame = _synthetic_artifacts(tmp_path)
    candidate = inference.load_berka_kmeans_candidate(manifest_path, project_root=tmp_path)

    first = candidate.predict_features(frame)
    second = candidate.predict_features(frame)

    assert first.account_id.tolist() == frame.account_id.tolist()
    assert first.index.tolist() == frame.index.tolist()
    assert first.cluster_id.tolist() == second.cluster_id.tolist()
    assert set(first.cluster_id).issubset({0, 1})
    matrix = candidate.preprocessor.transform(frame.loc[:, list(EXPECTED_FEATURES)])
    assert matrix.shape == (len(frame), 13)
    assert candidate.model_sha256 == _sha256(candidate.model_path)
    assert candidate.preprocessor_sha256 == _sha256(candidate.preprocessor_path)


@pytest.mark.parametrize("artifact_key", ["kmeans", "preprocessor"])
def test_hash_mismatch_rejected_before_any_deserialization(tmp_path, monkeypatch, artifact_key):
    manifest_path, _ = _synthetic_artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = tmp_path / manifest[artifact_key]["path"]
    with artifact_path.open("ab") as changed:
        changed.write(b"tampered")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    deserialization_called = False

    def unexpected_load(*_args, **_kwargs):
        nonlocal deserialization_called
        deserialization_called = True
        raise AssertionError("joblib.load must not run after an integrity mismatch")

    monkeypatch.setattr(inference.joblib, "load", unexpected_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inference.load_berka_kmeans_candidate(manifest_path, project_root=tmp_path)
    assert not deserialization_called


def test_historical_frozen_holdout_replay_when_local_research_artifacts_exist():
    root = Path.cwd()
    manifest = root / inference.DEFAULT_INTEGRITY_MANIFEST
    baseline = root / "artifacts/berka_behavioral_baselines/v1.1/asof_1997-12-31"
    snapshot_path = root / "data/processed/berka_snapshot/v1/asof_1997-12-31/berka_account_snapshot_features.csv"
    assignments_path = baseline / "held_out_assignments.csv"
    required = [manifest, baseline / "models/kmeans_k2_seed42.joblib", baseline / "preprocessor_train_fitted.joblib", snapshot_path, assignments_path]
    if not all(path.is_file() for path in required):
        pytest.skip("Optional Berka historical research artifacts are not available locally")

    candidate = inference.load_berka_kmeans_candidate(manifest, project_root=root)
    split = pd.read_csv(baseline / "account_split.csv")
    held_out_ids = split.loc[split["split"].eq("held_out"), "account_id"].to_numpy()
    snapshot = pd.read_csv(snapshot_path).set_index("account_id").loc[held_out_ids].reset_index()
    interpreted = select_behavioral_features(snapshot, list(EXPECTED_FEATURES))
    interpreted.insert(0, "account_id", held_out_ids)
    predicted = candidate.predict_features(interpreted)

    saved = pd.read_csv(assignments_path)
    saved = saved.loc[
        saved["algorithm"].eq("kmeans") & saved["k"].eq(2) & saved["seed"].eq(42)
    ].set_index("account_id").loc[held_out_ids, "cluster"].to_numpy()
    assert len(predicted) == 901
    assert np.array_equal(predicted["cluster_id"].to_numpy(), saved)
