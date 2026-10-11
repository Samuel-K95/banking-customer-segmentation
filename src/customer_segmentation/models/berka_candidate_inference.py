"""Read-only inference for the saved Berka K-Means k=2 seed-42 candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from customer_segmentation.models.berka_behavioral_baselines import (
    EXPECTED_FEATURES,
    BerkaTrainFittedPreprocessor,
)

DEFAULT_INTEGRITY_MANIFEST = Path(
    "artifacts/berka_temporal_behavior_validation/v1/asof_1997-12-31/frozen_model_integrity.json"
)
SUPPORTED_AS_OF_DATE = "1997-12-31"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(value: str, project_root: Path) -> Path:
    # Temporal metadata was produced on Windows; normalize separators for portability.
    normalized = Path(value.replace("\\", "/"))
    return normalized if normalized.is_absolute() else project_root / normalized


def _verify_file_hash(path: Path, expected_sha256: str, artifact_name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Berka {artifact_name} artifact not found: {path}")
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256):
        raise ValueError(f"Integrity manifest has an invalid SHA-256 for {artifact_name}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Berka {artifact_name} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


@dataclass
class BerkaKMeansCandidate:
    """Loaded immutable-by-interface K-Means candidate with fitted transforms."""

    estimator: KMeans
    preprocessor: BerkaTrainFittedPreprocessor
    feature_order: tuple[str, ...]
    model_path: Path
    preprocessor_path: Path
    model_sha256: str
    preprocessor_sha256: str
    model_version: str
    as_of_date: str

    def predict_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict labels without fitting; preserve row order and exclude account_id.

        ``frame`` must contain exactly the 13 original model features and may also
        contain an ``account_id`` alignment key. Feature columns are selected in
        saved contract order regardless of their input column order.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Berka inference input must be a pandas DataFrame")
        if not frame.columns.is_unique:
            raise ValueError("Berka inference input must not have duplicate column names")
        expected = set(self.feature_order)
        observed = set(frame.columns)
        permitted = expected | {"account_id"}
        missing = expected.difference(observed)
        extra = observed.difference(permitted)
        if missing or extra:
            raise ValueError(
                f"Berka feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if "account_id" in frame:
            account_ids = frame["account_id"]
            if account_ids.isna().any() or account_ids.duplicated().any():
                raise ValueError("account_id must be non-null and unique within a prediction batch")

        ordered_features = frame.loc[:, list(self.feature_order)]
        matrix = self.preprocessor.transform(ordered_features)
        labels = self.estimator.predict(matrix)
        if labels.ndim != 1 or len(labels) != len(frame):
            raise RuntimeError("K-Means returned predictions that do not align with the input rows")
        result = pd.DataFrame({"cluster_id": labels.astype(np.int64)}, index=frame.index)
        if "account_id" in frame:
            result.insert(0, "account_id", frame["account_id"].to_numpy(copy=True))
        return result


def load_berka_kmeans_candidate(
    integrity_manifest: str | Path = DEFAULT_INTEGRITY_MANIFEST,
    *,
    project_root: str | Path | None = None,
) -> BerkaKMeansCandidate:
    """Verify saved-file digests before deserializing the trusted local artifacts."""
    root = Path.cwd() if project_root is None else Path(project_root)
    manifest_path = Path(integrity_manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        model_entry = manifest["kmeans"]
        preprocessor_entry = manifest["preprocessor"]
        model_path = _resolve_manifest_path(model_entry["path"], root)
        preprocessor_path = _resolve_manifest_path(preprocessor_entry["path"], root)
        expected_model_sha = model_entry["sha256"]
        expected_preprocessor_sha = preprocessor_entry["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Integrity manifest is missing Berka K-Means/preprocessor entries") from exc

    # Hash both payloads before loading either pickle/joblib object.
    model_sha = _verify_file_hash(model_path, expected_model_sha, "K-Means")
    preprocessor_sha = _verify_file_hash(preprocessor_path, expected_preprocessor_sha, "preprocessor")
    model_artifact = joblib.load(model_path)
    preprocessor_artifact = joblib.load(preprocessor_path)
    if not isinstance(model_artifact, dict) or not {"estimator", "metadata"}.issubset(model_artifact):
        raise ValueError("Invalid Berka K-Means artifact envelope")
    if not isinstance(preprocessor_artifact, dict) or not {
        "preprocessor_state",
        "metadata",
    }.issubset(preprocessor_artifact):
        raise ValueError("Invalid Berka preprocessor artifact envelope")

    estimator = model_artifact["estimator"]
    model_metadata = model_artifact["metadata"]
    if not isinstance(estimator, KMeans):
        raise TypeError("The verified Berka estimator artifact is not a scikit-learn KMeans model")
    if (
        estimator.n_clusters != 2
        or estimator.random_state != 42
        or estimator.n_features_in_ != len(EXPECTED_FEATURES)
        or model_metadata.get("algorithm") != "kmeans"
        or model_metadata.get("k") != 2
        or model_metadata.get("seed") != 42
    ):
        raise ValueError("The loaded estimator does not match the Berka K-Means k=2 seed-42 candidate")
    for field in (
        "algorithm",
        "k",
        "seed",
        "training_account_ids_sha256",
        "held_out_account_ids_sha256",
    ):
        if model_metadata.get(field) != model_entry.get(field):
            raise ValueError(f"K-Means artifact metadata field {field!r} does not match integrity metadata")
    if model_entry.get("replay_matches_frozen_assignments") is not True:
        raise ValueError("Integrity metadata does not verify the frozen K-Means assignment replay")
    if model_metadata.get("as_of_date") != SUPPORTED_AS_OF_DATE:
        raise ValueError("The loaded Berka candidate is not tied to the supported 1997-12-31 snapshot")
    feature_order = tuple(model_metadata.get("feature_order", ()))
    if feature_order != tuple(EXPECTED_FEATURES):
        raise ValueError("The loaded model feature order differs from the original 13-feature contract")

    preprocessor = BerkaTrainFittedPreprocessor.from_state_dict(preprocessor_artifact["preprocessor_state"])
    preprocessor_metadata = preprocessor.metadata()
    if preprocessor_metadata != preprocessor_artifact["metadata"]:
        raise ValueError("Preprocessor state does not match its embedded metadata")
    manifest_features = tuple(preprocessor_entry.get("feature_order", ()))
    if manifest_features != feature_order or tuple(preprocessor_metadata["feature_order"]) != feature_order:
        raise ValueError("Saved preprocessor feature order does not match the K-Means model")
    if preprocessor_metadata["fit_scope"] != "training accounts only":
        raise ValueError("Saved preprocessor fit scope is not training-only")
    if preprocessor_metadata["fit_rows"] != model_metadata.get("training_rows"):
        raise ValueError("Saved preprocessor fit row count does not match model metadata")
    if preprocessor_metadata["fit_account_ids_sha256"] != model_metadata.get("training_account_ids_sha256"):
        raise ValueError("Saved preprocessor training IDs do not match model metadata")
    if preprocessor_metadata["fit_rows"] != preprocessor_entry.get("fit_rows"):
        raise ValueError("Saved preprocessor fit row count does not match integrity metadata")
    if preprocessor_metadata["fit_account_ids_sha256"] != preprocessor_entry.get("fit_account_ids_sha256"):
        raise ValueError("Saved preprocessor training ID hash does not match integrity metadata")

    model_preprocessor_path = model_metadata.get("preprocessor_path", "").replace("\\", "/")
    manifest_preprocessor_path = preprocessor_entry["path"].replace("\\", "/")
    if model_preprocessor_path != manifest_preprocessor_path:
        raise ValueError("Model metadata points to a different preprocessing artifact")

    return BerkaKMeansCandidate(
        estimator=estimator,
        preprocessor=preprocessor,
        feature_order=feature_order,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        model_sha256=model_sha,
        preprocessor_sha256=preprocessor_sha,
        model_version=f"{model_metadata['experiment']}:kmeans:k2:seed42",
        as_of_date=SUPPORTED_AS_OF_DATE,
    )
