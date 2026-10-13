"""Reusable, feature-contract-aware clustering and evaluation helpers."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class SampledCustomers:
    """A deterministic, ordered ID/features sample from a model CSV."""

    name: str
    row_positions: np.ndarray
    customer_ids: np.ndarray
    features: np.ndarray


def validate_feature_contract(
    track_spec: dict[str, Any],
    model_columns: list[str],
    requested_features: list[str],
) -> list[str]:
    """Enforce exact configured feature order and reject profile/ID leakage."""
    contract_features = list(track_spec.get("model_features", []))
    if requested_features != contract_features:
        raise ValueError(
            "Requested model features/order must exactly match the track contract; "
            f"expected {contract_features}, got {requested_features}."
        )

    identifiers = set(track_spec.get("identifier_columns", []))
    excluded = set(track_spec.get("excluded_features", []))
    patterns = [re.compile(value) for value in track_spec.get("excluded_column_patterns", [])]
    prohibited = identifiers | excluded
    invalid = [
        feature
        for feature in requested_features
        if feature in prohibited or any(pattern.search(feature) for pattern in patterns)
    ]
    if invalid:
        raise ValueError(f"Identifier/profile/excluded columns cannot be clustering inputs: {invalid}")

    missing = [feature for feature in requested_features if feature not in model_columns]
    if missing:
        raise ValueError(f"Model matrix is missing contract features: {missing}")
    return contract_features


def select_baseline_features(
    frame: pd.DataFrame,
    track_spec: dict[str, Any],
    requested_features: list[str],
) -> pd.DataFrame:
    """Select only the contract features, preserving contract order."""
    feature_order = validate_feature_contract(
        track_spec, list(frame.columns), requested_features
    )
    selected = frame.loc[:, feature_order]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
        raise TypeError("All baseline clustering inputs must be numeric.")
    values = selected.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Baseline clustering inputs must contain only finite values.")
    return selected


def make_deterministic_sample_indices(
    row_count: int,
    *,
    training_count: int,
    evaluation_count: int,
    training_seed: int,
    evaluation_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample sorted row positions with a deterministic, disjoint holdout."""
    if row_count < 1:
        raise ValueError("row_count must be positive.")
    if training_count < 1 or evaluation_count < 1:
        raise ValueError("Training and evaluation sample sizes must be positive.")
    if training_count + evaluation_count > row_count:
        raise ValueError("Training and evaluation samples must fit within the dataset.")

    evaluation = np.sort(
        np.random.default_rng(evaluation_seed).choice(
            row_count, size=evaluation_count, replace=False
        )
    ).astype(np.int64, copy=False)
    available = np.ones(row_count, dtype=bool)
    available[evaluation] = False
    remaining = np.flatnonzero(available)
    training = np.sort(
        np.random.default_rng(training_seed).choice(
            remaining, size=training_count, replace=False
        )
    ).astype(np.int64, copy=False)
    if np.intersect1d(training, evaluation, assume_unique=True).size:
        raise RuntimeError("Internal sampling error: training and evaluation positions overlap.")
    return training, evaluation


def load_sampled_rows(
    path: str | Path,
    *,
    identifier_column: str,
    feature_order: list[str],
    requested_positions: dict[str, np.ndarray],
    expected_row_count: int,
    chunk_rows: int = 50_000,
) -> dict[str, SampledCustomers]:
    """Read only requested IDs/features and selected rows from a large CSV."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    if len(feature_order) != len(set(feature_order)):
        raise ValueError("Feature order contains duplicate names.")
    if identifier_column in feature_order:
        raise ValueError("Identifier columns are not valid model features.")

    normalized_positions: dict[str, np.ndarray] = {}
    for sample_name, positions in requested_positions.items():
        position_array = np.asarray(positions, dtype=np.int64)
        if position_array.ndim != 1 or np.any(position_array < 0):
            raise ValueError(f"Invalid requested row positions for sample {sample_name!r}.")
        if position_array.size > 1 and np.any(position_array[1:] <= position_array[:-1]):
            raise ValueError(f"Sample {sample_name!r} positions must be sorted and unique.")
        if position_array.size and position_array[-1] >= expected_row_count:
            raise ValueError(f"Sample {sample_name!r} contains an out-of-range row position.")
        normalized_positions[sample_name] = position_array

    usecols = [identifier_column, *feature_order]
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in normalized_positions}
    observed_rows = 0
    for frame in pd.read_csv(
        path,
        usecols=usecols,
        dtype={identifier_column: "string"},
        chunksize=chunk_rows,
    ):
        start = observed_rows
        end = start + len(frame)
        observed_rows = end
        frame = frame.loc[:, usecols]
        for sample_name, positions in normalized_positions.items():
            left = int(np.searchsorted(positions, start, side="left"))
            right = int(np.searchsorted(positions, end, side="left"))
            if right > left:
                local_positions = positions[left:right] - start
                parts[sample_name].append(frame.iloc[local_positions].copy())

    if observed_rows != expected_row_count:
        raise ValueError(
            f"Model CSV row count changed during sampling: expected {expected_row_count}, "
            f"observed {observed_rows}."
        )

    samples: dict[str, SampledCustomers] = {}
    for sample_name, positions in normalized_positions.items():
        if not parts[sample_name]:
            raise ValueError(f"No model rows were collected for sample {sample_name!r}.")
        selected = pd.concat(parts[sample_name], ignore_index=True)
        if len(selected) != len(positions):
            raise ValueError(
                f"Sample {sample_name!r} row count mismatch: requested {len(positions)}, "
                f"collected {len(selected)}."
            )
        identifiers = selected[identifier_column]
        if identifiers.isna().any() or identifiers.duplicated().any():
            raise ValueError(f"Sample {sample_name!r} has null or duplicate customer IDs.")
        feature_frame = selected.loc[:, feature_order]
        matrix = feature_frame.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Sample {sample_name!r} contains non-finite model features.")
        samples[sample_name] = SampledCustomers(
            name=sample_name,
            row_positions=positions,
            customer_ids=identifiers.astype("string").to_numpy(dtype=str),
            features=np.ascontiguousarray(matrix),
        )
    return samples


def train_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    n_init: int = 10,
    max_iter: int = 300,
    algorithm: str = "lloyd",
) -> KMeans:
    _validate_training_data(features, n_clusters)
    estimator = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
    )
    estimator.fit(features)
    return estimator


def train_gaussian_mixture(
    features: np.ndarray,
    *,
    n_components: int,
    covariance_type: str = "diag",
    reg_covar: float = 1e-6,
    random_state: int,
    n_init: int = 1,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> GaussianMixture:
    _validate_training_data(features, n_components)
    if reg_covar <= 0:
        raise ValueError("reg_covar must be positive.")
    if covariance_type not in {"full", "tied", "diag", "spherical"}:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    estimator = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )
    estimator.fit(features)
    return estimator


def _validate_training_data(features: np.ndarray, n_clusters: int) -> None:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Training features must be a non-empty 2D matrix with at least two rows.")
    if n_clusters < 2 or n_clusters >= values.shape[0]:
        raise ValueError("Cluster/component count must be >=2 and smaller than training rows.")
    if not np.isfinite(values).all():
        raise ValueError("Training features must contain only finite values.")


def evaluate_assignments(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_silhouette_rows: int = 3_000,
    random_state: int = 13579,
) -> dict[str, Any]:
    """Evaluate held-out labels with a bounded, deterministic silhouette sample."""
    values = np.asarray(features)
    labels = np.asarray(labels)
    if values.ndim != 2 or labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("Evaluation features and labels have incompatible shapes.")
    if len(values) < 2 or max_silhouette_rows < 2:
        raise ValueError("Evaluation and silhouette sample sizes must be at least two.")
    if not np.isfinite(values).all():
        raise ValueError("Evaluation features must contain only finite values.")

    unique_labels, counts = np.unique(labels, return_counts=True)
    membership = {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}
    percentages = {key: 100.0 * count / len(labels) for key, count in membership.items()}
    result: dict[str, Any] = {
        "evaluation_rows": len(values),
        "observed_clusters": len(unique_labels),
        "segment_counts": membership,
        "segment_percentages": percentages,
        "smallest_segment_pct": min(percentages.values()) if percentages else None,
        "silhouette_sample_rows": 0,
        "silhouette_score": None,
        "davies_bouldin_index": None,
        "calinski_harabasz_index": None,
        "metric_status": "ok",
        "metric_error": None,
    }
    if len(unique_labels) < 2 or len(unique_labels) >= len(values):
        result["metric_status"] = "degenerate"
        result["metric_error"] = "Held-out labels contain fewer than two clusters or one cluster per row."
        return result

    silhouette_count = min(len(values), max_silhouette_rows)
    silhouette_indices = np.sort(
        np.random.default_rng(random_state).choice(
            len(values), size=silhouette_count, replace=False
        )
    )
    result["silhouette_sample_rows"] = int(silhouette_count)
    try:
        result["silhouette_score"] = float(
            silhouette_score(values[silhouette_indices], labels[silhouette_indices])
        )
        result["davies_bouldin_index"] = float(davies_bouldin_score(values, labels))
        result["calinski_harabasz_index"] = float(calinski_harabasz_score(values, labels))
    except ValueError as exc:
        result["metric_status"] = "metric_error"
        result["metric_error"] = f"{type(exc).__name__}: {exc}"
    return result


def aligned_adjusted_rand_index(
    labels_left: np.ndarray,
    ids_left: np.ndarray,
    labels_right: np.ndarray,
    ids_right: np.ndarray,
) -> float:
    """Compare seeded runs only when predictions use the same ordered holdout IDs."""
    ids_left = np.asarray(ids_left, dtype=str)
    ids_right = np.asarray(ids_right, dtype=str)
    labels_left = np.asarray(labels_left)
    labels_right = np.asarray(labels_right)
    if not np.array_equal(ids_left, ids_right):
        raise ValueError("Stability ARI requires identical customer IDs in identical order.")
    if labels_left.shape != labels_right.shape or labels_left.shape != ids_left.shape:
        raise ValueError("Stability labels must align one-to-one with the evaluation IDs.")
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(labels_left, labels_right))


def make_assignment_frame(
    customer_ids: np.ndarray,
    labels: np.ndarray,
    *,
    run_id: str,
    split: str,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    customer_ids = np.asarray(customer_ids, dtype=str)
    labels = np.asarray(labels)
    if customer_ids.shape != labels.shape:
        raise ValueError("Assignment labels must align one-to-one with CustomerID values.")
    frame = pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "run_id": run_id,
            "split": split,
            "cluster": labels.astype(np.int32),
        }
    )
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != labels.shape:
            raise ValueError("Assignment probabilities must align one-to-one with labels.")
        frame["max_component_probability"] = probabilities
    return frame


def ordered_id_sha256(customer_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for customer_id in np.asarray(customer_ids, dtype=str):
        digest.update(customer_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def save_model_artifact(path: str | Path, estimator: Any, metadata: dict[str, Any]) -> Path:
    """Atomically save a fitted estimator and its reproducibility metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing model artifact: {destination}")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{destination.stem}-", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = temporary.name
        joblib.dump({"estimator": estimator, "metadata": metadata}, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return destination


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """Load a fitted candidate artifact and validate its envelope."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"estimator", "metadata"}.issubset(artifact):
        raise ValueError(f"Invalid clustering model artifact: {path}")
    return artifact
