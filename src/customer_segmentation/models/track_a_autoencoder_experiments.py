"""Run deterministic Track A PyTorch autoencoder and latent K-Means experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from customer_segmentation.models.autoencoder import (
    AutoencoderTrainingConfig,
    CustomerAutoencoder,
    encode_features,
    load_autoencoder_checkpoint,
    make_internal_validation_indices,
    reconstruction_errors,
    save_autoencoder_checkpoint,
    select_autoencoder_feature_order,
    train_autoencoder,
    validate_train_evaluation_ids,
)
from customer_segmentation.models.clustering_baselines import (
    SampledCustomers,
    aligned_adjusted_rand_index,
    evaluate_assignments,
    load_sampled_rows,
    make_assignment_frame,
    ordered_id_sha256,
    save_model_artifact,
    validate_feature_contract,
)
from customer_segmentation.models.preflight import run_preflight
from customer_segmentation.models.track_a_experiments import (
    BOARD_FIELDS,
    _atomic_write,
    _json,
    _mlflow_log,
    _persist_comparison_rows,
    _sha256_file,
    _versions,
    _WorkingSetSampler,
    validate_p1_preprocessor,
)


def _write_joblib_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = temporary.name
        joblib.dump(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _read_manifest(path: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={"CustomerID": "string"})
    required = {"row_position", "CustomerID"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Sample manifest is missing columns: {sorted(required - set(frame.columns))}")
    positions = frame["row_position"].to_numpy(dtype=np.int64)
    customer_ids = frame["CustomerID"].astype("string").to_numpy(dtype=str)
    if len(positions) != expected_rows:
        raise ValueError(f"Manifest {path} has {len(positions)} rows; expected {expected_rows}.")
    if positions.ndim != 1 or np.any(positions < 0):
        raise ValueError(f"Manifest {path} has invalid row positions.")
    if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
        raise ValueError(f"Manifest {path} positions must be sorted and unique.")
    if pd.isna(customer_ids).any() or len(np.unique(customer_ids)) != len(customer_ids):
        raise ValueError(f"Manifest {path} has null or duplicate CustomerID values.")
    return positions, customer_ids


def _load_exact_samples(
    root: Path,
    config: dict[str, Any],
    *,
    seeds: list[int],
) -> tuple[dict[int, SampledCustomers], SampledCustomers, dict[str, str]]:
    data = config["data"]
    sampling = config["sampling"]
    full_features = list(data["features"])
    contracts_path = root / data["feature_contract_file"]
    with contracts_path.open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)["tracks"]["track_a"]
    validate_feature_contract(contract, full_features, full_features)

    train_manifest_paths = {
        seed: root / sampling["train_manifests"][str(seed)] for seed in seeds
    }
    evaluation_manifest_path = root / sampling["evaluation_manifest"]
    expected_train_rows = int(sampling["expected_training_rows"])
    expected_eval_rows = int(sampling["expected_evaluation_rows"])
    manifests: dict[str, tuple[np.ndarray, np.ndarray]] = {
        f"train_seed{seed}": _read_manifest(path, expected_train_rows)
        for seed, path in train_manifest_paths.items()
    }
    manifests["evaluation"] = _read_manifest(evaluation_manifest_path, expected_eval_rows)

    requested_positions = {
        name: position_ids[0] for name, position_ids in manifests.items()
    }
    samples = load_sampled_rows(
        root / data["model_file"],
        identifier_column=str(data["identifier_column"]),
        feature_order=full_features,
        requested_positions=requested_positions,
        expected_row_count=int(data["row_count"]),
        chunk_rows=int(data["chunk_rows"]),
    )
    for name, sample in samples.items():
        expected_ids = manifests[name][1]
        if not np.array_equal(sample.customer_ids, expected_ids):
            raise ValueError(f"CustomerID sequence in the model CSV does not match {name} manifest.")

    evaluation = samples["evaluation"]
    training_samples: dict[int, SampledCustomers] = {}
    for seed in seeds:
        training = samples[f"train_seed{seed}"]
        validate_train_evaluation_ids(training.customer_ids, evaluation.customer_ids)
        training_samples[seed] = training
    for left, right in itertools.pairwise(seeds):
        if np.array_equal(training_samples[left].customer_ids, training_samples[right].customer_ids):
            raise ValueError("Independent training seeds unexpectedly share the same CustomerID sequence.")

    manifest_paths = {
        **{f"training_seed{seed}": _relative(path, root) for seed, path in train_manifest_paths.items()},
        "evaluation": _relative(evaluation_manifest_path, root),
    }
    return training_samples, evaluation, manifest_paths


def _load_evaluation_transaction_counts(
    path: Path,
    *,
    identifier_column: str,
    customer_ids: np.ndarray,
    chunk_rows: int,
) -> dict[str, int]:
    """Read only evaluation customers' interpretable transaction counts."""
    requested = set(np.asarray(customer_ids, dtype=str))
    found: dict[str, int] = {}
    for frame in pd.read_csv(
        path,
        usecols=[identifier_column, "Transaction_Count"],
        dtype={identifier_column: "string"},
        chunksize=chunk_rows,
    ):
        selected = frame.loc[frame[identifier_column].isin(requested)]
        for customer_id, count in zip(
            selected[identifier_column].astype("string"), selected["Transaction_Count"]
        ):
            key = str(customer_id)
            if key in found:
                raise ValueError(f"Duplicate interpretable Transaction_Count for CustomerID {key}.")
            found[key] = int(count)
        if len(found) == len(requested):
            break
    if set(found) != requested:
        missing = sorted(requested - set(found))[:5]
        raise ValueError(f"Could not find interpretable transaction counts for evaluation IDs: {missing}")
    return found


def _save_internal_split(
    path: Path,
    *,
    training_sample: SampledCustomers,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> None:
    labels = np.full(len(training_sample.customer_ids), "internal_train", dtype=object)
    labels[validation_indices] = "internal_validation"
    frame = pd.DataFrame(
        {
            "row_position": training_sample.row_positions,
            "CustomerID": training_sample.customer_ids,
            "internal_split": labels,
        }
    )
    if len(training_indices) + len(validation_indices) != len(frame):
        raise RuntimeError("Internal train/validation split does not cover the training manifest.")
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_write(path, content)


def _segment_profile(
    customer_ids: np.ndarray,
    labels: np.ndarray,
    transaction_counts: dict[str, int],
) -> pd.DataFrame:
    customer_ids = np.asarray(customer_ids, dtype=str)
    labels = np.asarray(labels)
    if customer_ids.shape != labels.shape:
        raise ValueError("Evaluation labels do not align with CustomerID values.")
    rows = []
    for label in np.unique(labels):
        mask = labels == label
        ids = customer_ids[mask]
        singleton_count = sum(transaction_counts[customer_id] == 1 for customer_id in ids)
        rows.append(
            {
                "cluster": int(label),
                "customers": int(mask.sum()),
                "evaluation_percent": float(100 * mask.sum() / len(labels)),
                "single_transaction_customers": int(singleton_count),
                "single_transaction_percent": float(100 * singleton_count / mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_frame_new(path: Path, frame: pd.DataFrame) -> None:
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_write(path, content)


def _version_metadata() -> dict[str, str]:
    versions = _versions()
    versions["torch"] = str(torch.__version__)
    versions["scipy"] = __import__("scipy").__version__
    versions["threadpoolctl"] = __import__("threadpoolctl").__version__
    versions["cuda_available"] = str(torch.cuda.is_available())
    return versions


def _fit_config(
    settings: dict[str, Any],
    *,
    input_dim: int,
    seed: int,
) -> AutoencoderTrainingConfig:
    return AutoencoderTrainingConfig(
        input_dim=input_dim,
        hidden_dims=tuple(int(value) for value in settings["hidden_dims"]),
        latent_dim=int(settings["latent_dim"]),
        batch_size=int(settings["batch_size"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        max_epochs=int(settings["max_epochs"]),
        patience=int(settings["early_stopping_patience"]),
        min_delta=float(settings["early_stopping_min_delta"]),
        huber_beta=float(settings["huber_beta"]),
        divergence_factor=float(settings["loss_divergence_factor"]),
        max_loss=float(settings["max_loss"]),
        seed=seed,
        num_threads=int(settings["num_threads"]),
    )


def _feature_error_by_name(
    errors: dict[str, dict[str, float]], feature_order: list[str]
) -> dict[str, dict[str, float]]:
    return {feature: errors[str(index)] for index, feature in enumerate(feature_order)}


def _failure_row(
    *,
    root: Path,
    config: dict[str, Any],
    candidate_id: str,
    phase: str,
    family: str,
    algorithm: str,
    feature_order: list[str],
    seed: int,
    k: int,
    training_sample: SampledCustomers,
    evaluation_sample: SampledCustomers,
    model_matrix_hash: str,
    preprocessor_hash: str,
    metric_space: str,
    error: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_id = f"{candidate_id}__{metric_space}"
    metadata_path = root / config["execution"]["run_metadata_directory"] / f"{row_id}.json"
    parameters = {
        "family": family,
        "feature_set": family,
        "metric_space": metric_space,
        "input_feature_order": feature_order,
        "kmeans": config["clustering"],
        "seed": seed,
    }
    row = {field: None for field in BOARD_FIELDS}
    row.update(
        {
            "run_id": row_id,
            "phase": phase,
            "algorithm": algorithm,
            "training_rows": len(training_sample.customer_ids),
            "evaluation_rows": len(evaluation_sample.customer_ids),
            "k": k,
            "seed": seed,
            "train_customer_id_sha256": ordered_id_sha256(training_sample.customer_ids),
            "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_sample.customer_ids),
            "input_model_matrix_sha256": model_matrix_hash,
            "p1_preprocessor_sha256": preprocessor_hash,
            "feature_order_json": _json(feature_order),
            "parameters_json": _json(parameters),
            "metric_status": "failed",
            "run_status": "failed",
            "run_metadata_path": _relative(metadata_path, root),
            "mlflow_status": "not_attempted",
            "error": error,
        }
    )
    metadata = {
        "run_id": row_id,
        "candidate_id": candidate_id,
        "phase": phase,
        "algorithm": algorithm,
        "feature_set": family,
        "metric_space": metric_space,
        "feature_order": feature_order,
        "seed": seed,
        "k": k,
        "error": error,
        "versions": _version_metadata(),
        "experiment_config": config,
    }
    return row, {"path": metadata_path, "payload": metadata}


def _fit_kmeans_candidate(
    *,
    root: Path,
    config: dict[str, Any],
    candidate_id: str,
    phase: str,
    family: str,
    algorithm_name: str,
    feature_order: list[str],
    seed: int,
    k: int,
    training_sample: SampledCustomers,
    evaluation_sample: SampledCustomers,
    clustering_training_features: np.ndarray,
    clustering_evaluation_features: np.ndarray,
    metric_spaces: dict[str, tuple[np.ndarray, list[str]]],
    singleton_counts: dict[str, int],
    model_matrix_hash: str,
    preprocessor_hash: str,
    base_training_seconds: float,
    base_memory: tuple[int | None, int | None, int | None],
    autoencoder_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    model_directory = root / config["execution"]["model_directory"]
    assignment_directory = root / config["execution"]["assignment_directory"]
    profile_directory = root / config["execution"]["profile_directory"]
    model_path = model_directory / f"{candidate_id}_kmeans.joblib"
    assignment_path = assignment_directory / f"{candidate_id}_assignments.csv"
    profile_path = profile_directory / f"{candidate_id}_evaluation_segment_profile.csv"

    memory_sampler = _WorkingSetSampler()
    kmeans_start = time.perf_counter()
    with threadpool_limits(limits=int(config["autoencoder"]["num_threads"])), memory_sampler:
        estimator = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=int(config["clustering"]["n_init"]),
            max_iter=int(config["clustering"]["max_iter"]),
            algorithm=str(config["clustering"]["algorithm_variant"]),
            init=str(config["clustering"]["init"]),
        ).fit(clustering_training_features)
    kmeans_seconds = time.perf_counter() - kmeans_start
    train_labels = estimator.predict(clustering_training_features).astype(np.int32)
    evaluation_labels = estimator.predict(clustering_evaluation_features).astype(np.int32)
    if not np.isfinite(estimator.cluster_centers_).all() or not np.isfinite(estimator.inertia_):
        raise ValueError("K-Means produced non-finite centers or inertia.")

    segment_profile = _segment_profile(
        evaluation_sample.customer_ids, evaluation_labels, singleton_counts
    )
    train_assignments = make_assignment_frame(
        training_sample.customer_ids, train_labels, run_id=candidate_id, split="train"
    )
    evaluation_assignments = make_assignment_frame(
        evaluation_sample.customer_ids,
        evaluation_labels,
        run_id=candidate_id,
        split="evaluation",
    )
    _write_frame_new(
        assignment_path,
        pd.concat([train_assignments, evaluation_assignments], ignore_index=True),
    )
    _write_frame_new(profile_path, segment_profile)

    model_metadata = {
        "candidate_id": candidate_id,
        "phase": phase,
        "family": family,
        "algorithm": algorithm_name,
        "input_feature_order": feature_order,
        "metric_space_ids": list(metric_spaces),
        "seed": seed,
        "k": k,
        "parameters": config["clustering"],
        "training_customer_id_sha256": ordered_id_sha256(training_sample.customer_ids),
        "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_sample.customer_ids),
        "assignment_path": _relative(assignment_path, root),
        "segment_profile_path": _relative(profile_path, root),
        "autoencoder_artifacts": autoencoder_metadata,
        "versions": _version_metadata(),
    }
    save_model_artifact(model_path, estimator, model_metadata)
    reloaded = joblib.load(model_path)["estimator"]
    reload_predictions_identical = bool(
        np.array_equal(evaluation_labels, reloaded.predict(clustering_evaluation_features))
    )
    if not reload_predictions_identical:
        raise RuntimeError("Reloaded K-Means produced different evaluation predictions.")

    base_before, base_peak, _ = base_memory
    working_set_before = base_before or memory_sampler.before_bytes
    working_set_after = memory_sampler.after_bytes
    peak_candidates = [value for value in (base_peak, memory_sampler.peak_bytes) if value is not None]
    peak_working_set = max(peak_candidates) if peak_candidates else None
    total_training_seconds = base_training_seconds + kmeans_seconds

    metric_rows: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    for metric_space, (evaluation_features, metric_feature_order) in metric_spaces.items():
        metrics = evaluate_assignments(
            evaluation_features,
            evaluation_labels,
            max_silhouette_rows=int(config["metrics"]["silhouette_max_rows"]),
            random_state=int(config["metrics"]["silhouette_seed"]),
        )
        row_id = f"{candidate_id}__{metric_space}"
        metadata_path = root / config["execution"]["run_metadata_directory"] / f"{row_id}.json"
        parameters = metric_parameters(
            phase=phase,
            family=family,
            input_feature_order=feature_order,
            metric_space=metric_space,
            metric_feature_order=metric_feature_order,
            seed=seed,
            k=k,
            clustering=config["clustering"],
        )
        parameters["latent_dim"] = (
            int(config["autoencoder"]["latent_dim"])
            if autoencoder_metadata is not None
            else None
        )
        row = {field: None for field in BOARD_FIELDS}
        row.update(
            {
                "run_id": row_id,
                "phase": phase,
                "algorithm": algorithm_name,
                "training_rows": len(training_sample.customer_ids),
                "evaluation_rows": len(evaluation_sample.customer_ids),
                "k": int(k),
                "seed": int(seed),
                "train_customer_id_sha256": ordered_id_sha256(training_sample.customer_ids),
                "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_sample.customer_ids),
                "input_model_matrix_sha256": model_matrix_hash,
                "p1_preprocessor_sha256": preprocessor_hash,
                "feature_order_json": _json(feature_order),
                "parameters_json": _json(parameters),
                "silhouette_score": metrics["silhouette_score"],
                "silhouette_sample_rows": metrics["silhouette_sample_rows"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "calinski_harabasz_index": metrics["calinski_harabasz_index"],
                "evaluation_segment_counts_json": _json(metrics["segment_counts"]),
                "evaluation_segment_percentages_json": _json(metrics["segment_percentages"]),
                "smallest_segment_pct": metrics["smallest_segment_pct"],
                "kmeans_inertia_train": float(estimator.inertia_),
                "iterations": int(estimator.n_iter_),
                "training_seconds": float(total_training_seconds),
                "working_set_before_bytes": working_set_before,
                "peak_working_set_during_fit_bytes": peak_working_set,
                "working_set_after_bytes": working_set_after,
                "reload_predictions_identical": reload_predictions_identical,
                "metric_status": metrics["metric_status"],
                "run_status": "completed" if metrics["metric_status"] == "ok" else metrics["metric_status"],
                "model_path": _relative(model_path, root),
                "assignment_path": _relative(assignment_path, root),
                "run_metadata_path": _relative(metadata_path, root),
                "mlflow_status": "pending",
                "error": metrics["metric_error"],
            }
        )
        metadata = {
            "run_id": row_id,
            "candidate_id": candidate_id,
            "phase": phase,
            "algorithm": algorithm_name,
            "family": family,
            "seed": seed,
            "k": k,
            "metric_space": metric_space,
            "metric_feature_order": metric_feature_order,
            "input_feature_order": feature_order,
            "metrics": metrics,
            "segment_profiles": segment_profile.to_dict(orient="records"),
            "segment_profile_path": _relative(profile_path, root),
            "assignment_path": _relative(assignment_path, root),
            "kmeans_model_path": _relative(model_path, root),
            "autoencoder_artifacts": autoencoder_metadata,
            "parameters": parameters,
            "versions": _version_metadata(),
        }
        metric_rows.append(row)
        metadata_items.append({"path": metadata_path, "payload": metadata})

    return {
        "candidate_id": candidate_id,
        "family": family,
        "seed": seed,
        "k": k,
        "feature_order": feature_order,
        "evaluation_labels": evaluation_labels,
        "evaluation_ids": evaluation_sample.customer_ids,
        "model_path": model_path,
        "assignment_path": assignment_path,
        "segment_profile_path": profile_path,
        "rows": metric_rows,
        "metadata_items": metadata_items,
        "kmeans_seconds": kmeans_seconds,
    }


def _failed_candidate(
    *,
    root: Path,
    config: dict[str, Any],
    candidate_id: str,
    phase: str,
    family: str,
    algorithm_name: str,
    feature_order: list[str],
    seed: int,
    k: int,
    training_sample: SampledCustomers,
    evaluation_sample: SampledCustomers,
    model_matrix_hash: str,
    preprocessor_hash: str,
    metric_spaces: dict[str, tuple[np.ndarray, list[str]]],
    error: str,
) -> dict[str, Any]:
    rows = []
    metadata_items = []
    for metric_space in metric_spaces:
        row, metadata = _failure_row(
            root=root,
            config=config,
            candidate_id=candidate_id,
            phase=phase,
            family=family,
            algorithm=algorithm_name,
            feature_order=feature_order,
            seed=seed,
            k=k,
            training_sample=training_sample,
            evaluation_sample=evaluation_sample,
            model_matrix_hash=model_matrix_hash,
            preprocessor_hash=preprocessor_hash,
            metric_space=metric_space,
            error=error,
        )
        rows.append(row)
        metadata_items.append(metadata)
    return {
        "candidate_id": candidate_id,
        "family": family,
        "seed": seed,
        "k": k,
        "evaluation_labels": None,
        "evaluation_ids": evaluation_sample.customer_ids,
        "rows": rows,
        "metadata_items": metadata_items,
        "error": error,
    }


def _run_autoencoder_feature_seed(
    *,
    root: Path,
    config: dict[str, Any],
    training_sample: SampledCustomers,
    evaluation_sample: SampledCustomers,
    singleton_counts: dict[str, int],
    seed: int,
    feature_set: str,
    k_values: list[int],
    phase_name: str,
    model_matrix_hash: str,
    preprocessor_hash: str,
    manifest_paths: dict[str, str],
    versions: dict[str, str],
) -> list[dict[str, Any]]:
    all_features = list(config["data"]["features"])
    ratio_feature = str(config["data"]["ablation_feature"])
    if feature_set == "nine_feature":
        feature_order = select_autoencoder_feature_order(all_features)
        index = np.arange(len(all_features), dtype=np.int64)
    elif feature_set == "eight_feature_no_ratio":
        feature_order = select_autoencoder_feature_order(
            all_features, exclude_feature=ratio_feature
        )
        index = np.asarray([all_features.index(feature) for feature in feature_order], dtype=np.int64)
        if len(feature_order) != len(all_features) - 1 or ratio_feature in feature_order:
            raise ValueError("Spending-ratio ablation did not remove exactly the contracted ratio feature.")
    else:
        raise ValueError(f"Unknown autoencoder feature set: {feature_set}")

    training_features = np.ascontiguousarray(training_sample.features[:, index])
    evaluation_features = np.ascontiguousarray(evaluation_sample.features[:, index])
    split_seed = seed + int(config["sampling"]["internal_validation_seed_offset"])
    training_indices, validation_indices = make_internal_validation_indices(
        training_sample.customer_ids,
        validation_fraction=float(config["sampling"]["internal_validation_fraction"]),
        seed=split_seed,
    )
    internal_run_id = _run_id(f"track_a_ae_{feature_set}_n{len(training_sample.customer_ids)}_seed{seed}")
    model_directory = root / config["execution"]["model_directory"]
    history_directory = root / config["execution"]["history_directory"]
    split_path = (
        root
        / config["execution"]["internal_split_directory"]
        / f"{internal_run_id}_internal_train_validation.csv"
    )
    checkpoint_path = model_directory / f"{internal_run_id}_best.pt"
    history_path = history_directory / f"{internal_run_id}_history.csv"
    scaler_path = model_directory / f"{internal_run_id}_latent_scaler.joblib"
    _save_internal_split(
        split_path,
        training_sample=training_sample,
        training_indices=training_indices,
        validation_indices=validation_indices,
    )

    ae_config = _fit_config(config["autoencoder"], input_dim=len(feature_order), seed=seed)
    ae_memory = _WorkingSetSampler()
    ae_start = time.perf_counter()
    with ae_memory:
        training_result = train_autoencoder(
            training_features[training_indices],
            training_features[validation_indices],
            ae_config,
        )
    autoencoder_seconds = time.perf_counter() - ae_start

    training_errors = _feature_error_by_name(
        reconstruction_errors(
            training_result.model, training_features[training_indices]
        ),
        feature_order,
    )
    validation_errors = _feature_error_by_name(
        reconstruction_errors(
            training_result.model, training_features[validation_indices]
        ),
        feature_order,
    )
    training_summary = {
        "best_epoch": training_result.best_epoch,
        "epochs_completed": training_result.epochs_completed,
        "best_train_huber_loss": training_result.best_train_loss,
        "best_validation_huber_loss": training_result.best_validation_loss,
        "last_train_huber_loss": float(training_result.history[-1]["training_loss"]),
        "last_validation_huber_loss": float(training_result.history[-1]["validation_loss"]),
        "stopped_early": training_result.stopped_early,
        "history_path": _relative(history_path, root),
        "training_reconstruction_errors": training_errors,
        "internal_validation_reconstruction_errors": validation_errors,
    }
    _write_frame_new(history_path, pd.DataFrame(training_result.history))
    save_autoencoder_checkpoint(
        checkpoint_path,
        training_result.model,
        training_config=ae_config,
        training_summary=training_summary,
    )
    reloaded_checkpoint = load_autoencoder_checkpoint(checkpoint_path)
    reloaded_model: CustomerAutoencoder = reloaded_checkpoint["model"]

    train_latent = encode_features(reloaded_model, training_features)
    evaluation_latent = encode_features(reloaded_model, evaluation_features)
    train_latent_scaler = fit_train_only_latent_scaler(train_latent)
    scaled_train_latent = train_latent_scaler.transform(train_latent)
    scaled_evaluation_latent = train_latent_scaler.transform(evaluation_latent)
    if not np.isfinite(scaled_train_latent).all() or not np.isfinite(scaled_evaluation_latent).all():
        raise ValueError("Latent scaler produced non-finite values.")
    reloaded_embeddings = encode_features(reloaded_checkpoint["model"], evaluation_features)
    inference_reload_identical = bool(np.array_equal(evaluation_latent, reloaded_embeddings))
    if not inference_reload_identical:
        raise RuntimeError("Reloaded autoencoder produced different evaluation embeddings.")

    scaler_payload = {
        "scaler": train_latent_scaler,
        "metric_space": f"scaled_latent_{ae_config.latent_dim}d",
        "latent_dim": ae_config.latent_dim,
        "fit_rows": len(training_sample.customer_ids),
        "fit_customer_id_sha256": ordered_id_sha256(training_sample.customer_ids),
        "fit_scope": "all training-manifest customer embeddings only",
        "input_feature_order": feature_order,
    }
    _write_joblib_new(scaler_path, scaler_payload)

    architecture = reloaded_model.architecture()
    autoencoder_metadata = {
        "checkpoint_path": _relative(checkpoint_path, root),
        "history_path": _relative(history_path, root),
        "internal_split_manifest": _relative(split_path, root),
        "latent_scaler_path": _relative(scaler_path, root),
        "architecture": architecture,
        "training_config": json.loads(_json(ae_config.__dict__)),
        "training_summary": training_summary,
        "inference_reload_identical": inference_reload_identical,
        "training_manifest": manifest_paths[f"training_seed{seed}"],
        "evaluation_manifest": manifest_paths["evaluation"],
        "train_customer_id_sha256": ordered_id_sha256(training_sample.customer_ids),
        "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_sample.customer_ids),
        "input_model_matrix_sha256": model_matrix_hash,
        "p1_preprocessor_sha256": preprocessor_hash,
        "versions": versions,
        "device": "cpu",
    }

    original_nine_space = (evaluation_sample.features, all_features)
    latent_space = (
        scaled_evaluation_latent,
        [f"latent_{index}" for index in range(ae_config.latent_dim)],
    )
    if feature_set == "nine_feature":
        metric_spaces = {
            str(config["metrics"]["metric_spaces"]["original_nine"]): original_nine_space,
            str(config["metrics"]["metric_spaces"]["latent"]): latent_space,
        }
    else:
        original_eight_space = (evaluation_features, feature_order)
        metric_spaces = {
            str(config["metrics"]["metric_spaces"]["original_eight"]): original_eight_space,
            str(config["metrics"]["metric_spaces"]["original_nine"]): original_nine_space,
            str(config["metrics"]["metric_spaces"]["latent"]): latent_space,
        }

    base_memory = (
        ae_memory.before_bytes,
        ae_memory.peak_bytes,
        ae_memory.after_bytes,
    )
    candidates = []
    family = "ae_nine_feature" if feature_set == "nine_feature" else "ae_eight_feature"
    algorithm_name = "autoencoder_kmeans"
    for k in k_values:
        candidate_id = _run_id(f"{internal_run_id}_k{k}")
        try:
            candidate = _fit_kmeans_candidate(
                root=root,
                config=config,
                candidate_id=candidate_id,
                phase=phase_name,
                family=family,
                algorithm_name=algorithm_name,
                feature_order=feature_order,
                seed=seed,
                k=k,
                training_sample=training_sample,
                evaluation_sample=evaluation_sample,
                clustering_training_features=scaled_train_latent,
                clustering_evaluation_features=scaled_evaluation_latent,
                metric_spaces=metric_spaces,
                singleton_counts=singleton_counts,
                model_matrix_hash=model_matrix_hash,
                preprocessor_hash=preprocessor_hash,
                base_training_seconds=autoencoder_seconds,
                base_memory=base_memory,
                autoencoder_metadata=autoencoder_metadata,
            )
            candidate.update(
                {
                    "feature_set": feature_set,
                    "autoencoder_metadata": autoencoder_metadata,
                    "training_result": training_result,
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed candidate evidence in the ledger
            candidate = _failed_candidate(
                root=root,
                config=config,
                candidate_id=candidate_id,
                phase=phase_name,
                family=family,
                algorithm_name=algorithm_name,
                feature_order=feature_order,
                seed=seed,
                k=k,
                training_sample=training_sample,
                evaluation_sample=evaluation_sample,
                model_matrix_hash=model_matrix_hash,
                preprocessor_hash=preprocessor_hash,
                metric_spaces=metric_spaces,
                error=f"{type(exc).__name__}: {exc}",
            )
        candidates.append(candidate)
    return candidates


def _run_traditional_eight_feature_candidates(
    *,
    root: Path,
    config: dict[str, Any],
    training_samples: dict[int, SampledCustomers],
    evaluation_sample: SampledCustomers,
    singleton_counts: dict[str, int],
    seeds: list[int],
    k_values: list[int],
    phase_name: str,
    model_matrix_hash: str,
    preprocessor_hash: str,
) -> list[dict[str, Any]]:
    all_features = list(config["data"]["features"])
    ratio_feature = str(config["data"]["ablation_feature"])
    feature_order = select_autoencoder_feature_order(
        all_features, exclude_feature=ratio_feature
    )
    indices = np.asarray([all_features.index(feature) for feature in feature_order], dtype=np.int64)
    metric_spaces = {
        str(config["metrics"]["metric_spaces"]["original_eight"]): (
            evaluation_sample.features[:, indices],
            feature_order,
        ),
        str(config["metrics"]["metric_spaces"]["original_nine"]): (
            evaluation_sample.features,
            all_features,
        ),
    }
    results = []
    for seed in seeds:
        training_sample = training_samples[seed]
        training_features = np.ascontiguousarray(training_sample.features[:, indices])
        for k in k_values:
            candidate_id = _run_id(f"track_a_task06_kmeans_eight_feature_n50000_k{k}_seed{seed}")
            try:
                candidate = _fit_kmeans_candidate(
                    root=root,
                    config=config,
                    candidate_id=candidate_id,
                    phase=phase_name,
                    family="kmeans_eight_feature",
                    algorithm_name="kmeans_ablation_8f",
                    feature_order=feature_order,
                    seed=seed,
                    k=k,
                    training_sample=training_sample,
                    evaluation_sample=evaluation_sample,
                    clustering_training_features=training_features,
                    clustering_evaluation_features=evaluation_sample.features[:, indices],
                    metric_spaces=metric_spaces,
                    singleton_counts=singleton_counts,
                    model_matrix_hash=model_matrix_hash,
                    preprocessor_hash=preprocessor_hash,
                    base_training_seconds=0.0,
                    base_memory=(None, None, None),
                    autoencoder_metadata=None,
                )
                candidate["feature_set"] = "eight_feature_no_ratio"
            except Exception as exc:  # noqa: BLE001 - record and continue remaining candidates
                candidate = _failed_candidate(
                    root=root,
                    config=config,
                    candidate_id=candidate_id,
                    phase=phase_name,
                    family="kmeans_eight_feature",
                    algorithm_name="kmeans_ablation_8f",
                    feature_order=feature_order,
                    seed=seed,
                    k=k,
                    training_sample=training_sample,
                    evaluation_sample=evaluation_sample,
                    model_matrix_hash=model_matrix_hash,
                    preprocessor_hash=preprocessor_hash,
                    metric_spaces=metric_spaces,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(candidate)
    return results


def _feature_seed_failure_candidates(
    *,
    root: Path,
    config: dict[str, Any],
    training_sample: SampledCustomers,
    evaluation_sample: SampledCustomers,
    seed: int,
    feature_set: str,
    k_values: list[int],
    phase_name: str,
    model_matrix_hash: str,
    preprocessor_hash: str,
    error: str,
) -> list[dict[str, Any]]:
    all_features = list(config["data"]["features"])
    if feature_set == "nine_feature":
        feature_order = all_features
        family = "ae_nine_feature"
        metric_names = [
            str(config["metrics"]["metric_spaces"]["original_nine"]),
            str(config["metrics"]["metric_spaces"]["latent"]),
        ]
    else:
        feature_order = select_autoencoder_feature_order(
            all_features, exclude_feature=str(config["data"]["ablation_feature"])
        )
        family = "ae_eight_feature"
        metric_names = [
            str(config["metrics"]["metric_spaces"]["original_eight"]),
            str(config["metrics"]["metric_spaces"]["original_nine"]),
            str(config["metrics"]["metric_spaces"]["latent"]),
        ]
    metric_spaces = {name: (evaluation_sample.features, feature_order) for name in metric_names}
    return [
        _failed_candidate(
            root=root,
            config=config,
            candidate_id=_run_id(f"track_a_ae_failed_{feature_set}_seed{seed}_k{k}"),
            phase=phase_name,
            family=family,
            algorithm_name="autoencoder_kmeans",
            feature_order=feature_order,
            seed=seed,
            k=k,
            training_sample=training_sample,
            evaluation_sample=evaluation_sample,
            model_matrix_hash=model_matrix_hash,
            preprocessor_hash=preprocessor_hash,
            metric_spaces=metric_spaces,
            error=error,
        )
        for k in k_values
    ]


def _load_task05_kmeans_labels(
    root: Path,
    *,
    evaluation_ids: np.ndarray,
    seeds: list[int],
    k_values: list[int],
) -> dict[tuple[int, int], np.ndarray]:
    board_path = root / "artifacts/experiments/bakeoffs/comparison_board.csv"
    if not board_path.is_file():
        raise FileNotFoundError("Task 05 comparison ledger is required for baseline comparison.")
    board = pd.read_csv(board_path, dtype={"run_id": "string"})
    expected = set(np.asarray(evaluation_ids, dtype=str))
    labels: dict[tuple[int, int], np.ndarray] = {}
    for seed in seeds:
        for k in k_values:
            match = board.loc[
                (board["phase"] == "comparison")
                & (board["algorithm"] == "kmeans")
                & (board["seed"] == seed)
                & (board["k"] == k)
                & (board["run_status"] == "completed")
            ]
            if len(match) != 1:
                raise ValueError(f"Expected one completed Task 05 K-Means result for seed={seed}, k={k}.")
            assignment_path = root / str(match.iloc[0]["assignment_path"])
            assignments = pd.read_csv(
                assignment_path, dtype={"CustomerID": "string"}
            )
            evaluation = assignments.loc[assignments["split"] == "evaluation"]
            customer_ids = evaluation["CustomerID"].astype("string").to_numpy(dtype=str)
            if not np.array_equal(customer_ids, np.asarray(evaluation_ids, dtype=str)):
                raise ValueError("Task 05 K-Means assignments do not align with the shared evaluation manifest.")
            if set(customer_ids) != expected:
                raise ValueError("Task 05 K-Means evaluation IDs differ from the Task 06 holdout.")
            labels[(seed, k)] = evaluation["cluster"].to_numpy(dtype=np.int32)
    return labels


def _candidate_index(candidates: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (str(candidate["family"]), int(candidate["seed"]), int(candidate["k"])): candidate
        for candidate in candidates
    }


def _set_row_parameter(row: dict[str, Any], key: str, value: Any) -> None:
    parameters = json.loads(row["parameters_json"] or "{}")
    parameters[key] = value
    row["parameters_json"] = _json(parameters)


def fit_train_only_latent_scaler(training_embeddings: np.ndarray) -> StandardScaler:
    """Fit the latent StandardScaler on training embeddings only."""
    values = np.asarray(training_embeddings)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Training embeddings must be a 2D matrix with at least two rows.")
    if not np.isfinite(values).all():
        raise ValueError("Training embeddings contain non-finite values.")
    return StandardScaler().fit(values)


def metric_parameters(
    *,
    phase: str,
    family: str,
    input_feature_order: list[str],
    metric_space: str,
    metric_feature_order: list[str],
    seed: int,
    k: int,
    clustering: dict[str, Any],
) -> dict[str, Any]:
    """Build ledger parameters with a mandatory explicit metric-space ID."""
    if not metric_space:
        raise ValueError("metric_space must be explicitly identified for every metric row.")
    return {
        "experiment": phase,
        "family": family,
        "input_feature_order": list(input_feature_order),
        "metric_space": metric_space,
        "metric_feature_order": list(metric_feature_order),
        "kmeans": clustering,
        "random_state": int(seed),
        "k": int(k),
        "ablation_ari_same_seed": None,
        "stability_ari_across_seeds": None,
    }


def _add_stability_and_ablation(
    *,
    candidates: list[dict[str, Any]],
    baseline_labels: dict[tuple[int, int], np.ndarray],
    evaluation_ids: np.ndarray,
    seeds: list[int],
    k_values: list[int],
) -> dict[str, Any]:
    indexed = _candidate_index(candidates)
    stability_summary: dict[str, Any] = {}
    families = ["ae_nine_feature", "ae_eight_feature", "kmeans_eight_feature"]
    for family in families:
        if len(seeds) < 2:
            continue
        for k in k_values:
            left = indexed.get((family, seeds[0], k))
            right = indexed.get((family, seeds[1], k))
            if not left or not right or left.get("evaluation_labels") is None or right.get("evaluation_labels") is None:
                continue
            ari = aligned_adjusted_rand_index(
                left["evaluation_labels"],
                evaluation_ids,
                right["evaluation_labels"],
                evaluation_ids,
            )
            stability_summary[f"{family}_k{k}"] = ari
            for candidate in (left, right):
                candidate["stability_ari_across_seeds"] = ari
                for row in candidate["rows"]:
                    row["stability_ari_across_seeds"] = ari
                    _set_row_parameter(row, "stability_ari_across_seeds", ari)
                for metadata in candidate["metadata_items"]:
                    metadata["payload"]["stability_ari_across_seeds"] = ari

    ablation_summary: dict[str, Any] = {}
    for seed in seeds:
        for k in k_values:
            ae_nine = indexed.get(("ae_nine_feature", seed, k))
            ae_eight = indexed.get(("ae_eight_feature", seed, k))
            if ae_nine and ae_eight and ae_nine.get("evaluation_labels") is not None and ae_eight.get("evaluation_labels") is not None:
                ari = aligned_adjusted_rand_index(
                    ae_nine["evaluation_labels"],
                    evaluation_ids,
                    ae_eight["evaluation_labels"],
                    evaluation_ids,
                )
                ablation_summary[f"autoencoder_seed{seed}_k{k}"] = ari
                for candidate in (ae_nine, ae_eight):
                    candidate["same_seed_ratio_ablation_ari"] = ari
                    for row in candidate["rows"]:
                        _set_row_parameter(row, "same_seed_ratio_ablation_ari", ari)
                    for metadata in candidate["metadata_items"]:
                        metadata["payload"]["same_seed_ratio_ablation_ari"] = ari

            kmeans_eight = indexed.get(("kmeans_eight_feature", seed, k))
            kmeans_nine_labels = baseline_labels.get((seed, k))
            if kmeans_eight and kmeans_eight.get("evaluation_labels") is not None and kmeans_nine_labels is not None:
                ari = aligned_adjusted_rand_index(
                    kmeans_nine_labels,
                    evaluation_ids,
                    kmeans_eight["evaluation_labels"],
                    evaluation_ids,
                )
                ablation_summary[f"traditional_kmeans_seed{seed}_k{k}"] = ari
                for row in kmeans_eight["rows"]:
                    _set_row_parameter(row, "same_seed_ratio_ablation_ari_vs_task05_kmeans9", ari)
                for metadata in kmeans_eight["metadata_items"]:
                    metadata["payload"]["same_seed_ratio_ablation_ari_vs_task05_kmeans9"] = ari

            if ae_nine and ae_nine.get("evaluation_labels") is not None and kmeans_nine_labels is not None:
                ari = aligned_adjusted_rand_index(
                    kmeans_nine_labels,
                    evaluation_ids,
                    ae_nine["evaluation_labels"],
                    evaluation_ids,
                )
                for row in ae_nine["rows"]:
                    _set_row_parameter(row, "same_feature_ae_vs_task05_kmeans_ari", ari)
                for metadata in ae_nine["metadata_items"]:
                    metadata["payload"]["same_feature_ae_vs_task05_kmeans_ari"] = ari
    return {"stability_ari_across_seeds": stability_summary, "ratio_ablation_ari": ablation_summary}


def _persist_run_records(
    root: Path,
    candidates: list[dict[str, Any]],
    board_path: Path,
    *,
    experiment_name: str,
) -> None:
    rows: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.extend(candidate["rows"])
        metadata_items.extend(candidate["metadata_items"])
    metadata_by_path = {item["path"]: item["payload"] for item in metadata_items}
    if rows:
        for row in rows:
            metadata_path = root / row["run_metadata_path"]
            metadata = metadata_by_path[metadata_path]
            if row["run_status"] == "failed":
                mlflow_status = "not_attempted"
            else:
                mlflow_status = _mlflow_log(
                    experiment_name=experiment_name,
                    run_id=row["run_id"],
                    parameters=json.loads(row["parameters_json"] or "{}"),
                    metrics={
                        key: row[key]
                        for key in (
                            "silhouette_score",
                            "davies_bouldin_index",
                            "calinski_harabasz_index",
                            "smallest_segment_pct",
                            "kmeans_inertia_train",
                        )
                        if row.get(key) is not None
                    },
                    model_path=root / row["model_path"],
                    assignment_path=root / row["assignment_path"],
                )
            row["mlflow_status"] = mlflow_status
            metadata["mlflow_status"] = mlflow_status
        for item in metadata_items:
            _atomic_write(
                item["path"], (_json(item["payload"]) + "\n").encode("utf-8")
            )
        _persist_comparison_rows(board_path, rows)


def _run_synthetic_smoke() -> int:
    """Exercise fit, early stopping, latent scaling, checkpointing, and K-Means."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(7654)
    train = rng.normal(size=(768, 9)).astype(np.float32)
    validation = rng.normal(size=(192, 9)).astype(np.float32)
    train[:4, 5] *= 25
    validation[:2, 5] *= 25
    config = AutoencoderTrainingConfig(
        input_dim=9,
        hidden_dims=(32, 16),
        latent_dim=4,
        batch_size=128,
        learning_rate=1e-3,
        max_epochs=8,
        patience=3,
        seed=42,
        num_threads=1,
    )
    result = train_autoencoder(train, validation, config)
    if not np.isfinite([entry["training_loss"] for entry in result.history]).all():
        raise RuntimeError("Synthetic smoke produced non-finite training loss.")
    with tempfile.TemporaryDirectory(prefix="track-a-autoencoder-smoke-") as directory:
        checkpoint = Path(directory) / "smoke.pt"
        save_autoencoder_checkpoint(
            checkpoint,
            result.model,
            training_config=config,
            training_summary={"best_epoch": result.best_epoch},
        )
        loaded = load_autoencoder_checkpoint(checkpoint)["model"]
        latent_train = encode_features(loaded, train)
        latent_eval = encode_features(loaded, validation)
        if not np.array_equal(latent_eval, encode_features(loaded, validation)):
            raise RuntimeError("Synthetic checkpoint reload inference was not stable.")
        scaler = StandardScaler().fit(latent_train)
        scaled_train = scaler.transform(latent_train)
        scaled_eval = scaler.transform(latent_eval)
        synthetic_kmeans = KMeans(n_clusters=2, random_state=42, n_init=3).fit(scaled_train)
        labels = synthetic_kmeans.labels_
        eval_labels = synthetic_kmeans.predict(scaled_eval)
        if len(labels) != len(train) or len(eval_labels) != len(validation):
            raise RuntimeError("Synthetic smoke K-Means prediction dimensions are invalid.")
    print(
        "Synthetic smoke PASS: finite training, checkpoint reload, deterministic latent inference, "
        "train-only latent scaler, and K-Means fit/predict."
    )
    return 0


def run_experiments(config_path: str | Path, phase: str) -> int:
    if phase not in {"smoke", "comparison"}:
        raise ValueError(f"Unsupported real-data phase: {phase}")
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("track") != "track_a":
        raise ValueError("Autoencoder config must describe track_a.")
    root = config_path.parent.parent.parent
    data = config["data"]
    sampling = config["sampling"]
    execution = config["execution"]

    if data.get("input_representation") != "p1_model_matrix_already_scaled":
        raise ValueError("Task 06 must consume the already-transformed P1 matrix without rescaling it.")
    full_features = list(data["features"])
    preprocessor_path = root / data["preprocessor_artifact"]
    validate_p1_preprocessor(
        preprocessor_path,
        feature_order=full_features,
        identifier_column=str(data["identifier_column"]),
    )
    preflight = run_preflight(
        root / data["feature_contract_file"], selected_tracks=["track_a"]
    )
    if not preflight["ready"]:
        raise ValueError(f"Track A preflight failed: {preflight['tracks']['track_a']['errors']}")
    row_count = int(preflight["tracks"]["track_a"]["rows"]["model"])
    if row_count != int(data["row_count"]):
        raise ValueError(f"Track A row count changed: config={data['row_count']}, observed={row_count}.")

    phase_settings = sampling[phase]
    seeds = [int(seed) for seed in phase_settings["seeds"]]
    k_values = [int(k) for k in phase_settings["k_values"]]
    feature_sets = [str(value) for value in phase_settings["feature_sets"]]
    training_samples, evaluation_sample, manifest_paths = _load_exact_samples(
        root, config, seeds=seeds
    )
    singleton_counts = _load_evaluation_transaction_counts(
        root / data["interpretable_file"],
        identifier_column=str(data["identifier_column"]),
        customer_ids=evaluation_sample.customer_ids,
        chunk_rows=int(data["chunk_rows"]),
    )
    model_matrix_path = root / data["model_file"]
    model_matrix_hash = _sha256_file(model_matrix_path)
    preprocessor_hash = _sha256_file(preprocessor_path)
    versions = _version_metadata()
    torch.set_num_threads(int(config["autoencoder"]["num_threads"]))

    phase_name = "task06_real_smoke" if phase == "smoke" else "task06_real_comparison"
    all_candidates: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        for seed in seeds:
            try:
                candidates = _run_autoencoder_feature_seed(
                    root=root,
                    config=config,
                    training_sample=training_samples[seed],
                    evaluation_sample=evaluation_sample,
                    singleton_counts=singleton_counts,
                    seed=seed,
                    feature_set=feature_set,
                    k_values=k_values,
                    phase_name=phase_name,
                    model_matrix_hash=model_matrix_hash,
                    preprocessor_hash=preprocessor_hash,
                    manifest_paths=manifest_paths,
                    versions=versions,
                )
            except Exception as exc:  # noqa: BLE001 - persist failure and continue other seeds/features
                candidates = _feature_seed_failure_candidates(
                    root=root,
                    config=config,
                    training_sample=training_samples[seed],
                    evaluation_sample=evaluation_sample,
                    seed=seed,
                    feature_set=feature_set,
                    k_values=k_values,
                    phase_name=phase_name,
                    model_matrix_hash=model_matrix_hash,
                    preprocessor_hash=preprocessor_hash,
                    error=f"{type(exc).__name__}: {exc}",
                )
            all_candidates.extend(candidates)

    baseline_labels: dict[tuple[int, int], np.ndarray] = {}
    if phase == "comparison" and bool(
        phase_settings.get("run_eight_feature_traditional_kmeans", False)
    ):
        baseline_labels = _load_task05_kmeans_labels(
            root,
            evaluation_ids=evaluation_sample.customer_ids,
            seeds=seeds,
            k_values=k_values,
        )
        all_candidates.extend(
            _run_traditional_eight_feature_candidates(
                root=root,
                config=config,
                training_samples=training_samples,
                evaluation_sample=evaluation_sample,
                singleton_counts=singleton_counts,
                seeds=seeds,
                k_values=k_values,
                phase_name=phase_name,
                model_matrix_hash=model_matrix_hash,
                preprocessor_hash=preprocessor_hash,
            )
        )

    comparisons = {"stability_ari_across_seeds": {}, "ratio_ablation_ari": {}}
    if phase == "comparison":
        comparisons = _add_stability_and_ablation(
            candidates=all_candidates,
            baseline_labels=baseline_labels,
            evaluation_ids=evaluation_sample.customer_ids,
            seeds=seeds,
            k_values=k_values,
        )

    board_path = root / execution["comparison_board"]
    _persist_run_records(
        root,
        all_candidates,
        board_path,
        experiment_name=str(execution["mlflow_experiment"]),
    )
    rows = [row for candidate in all_candidates for row in candidate["rows"]]
    completed = sum(row["run_status"] == "completed" for row in rows)
    failed = sum(row["run_status"] == "failed" for row in rows)
    degenerate = sum(row["run_status"] == "degenerate" for row in rows)
    print(
        f"Phase {phase}: {len(rows)} metric row(s); completed={completed}, "
        f"failed={failed}, degenerate={degenerate}; candidates={len(all_candidates)}"
    )
    if phase == "comparison":
        print(f"Stability and ablation ARI: {_json(comparisons)}")
    print(f"Comparison board: {board_path}")
    print(f"Autoencoder models: {root / execution['model_directory']}")
    print(f"Candidate assignments: {root / execution['assignment_directory']}")
    return 0 if completed == len(rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/models/track_a/track_a_autoencoder.yaml",
        help="Track A PyTorch autoencoder experiment YAML.",
    )
    parser.add_argument(
        "--phase",
        choices=("synthetic-smoke", "smoke", "comparison"),
        default="synthetic-smoke",
        help="Run synthetic verification, one real-data candidate, or the planned comparison sweep.",
    )
    arguments = parser.parse_args(argv)
    if arguments.phase == "synthetic-smoke":
        return _run_synthetic_smoke()
    return run_experiments(arguments.config, arguments.phase)


if __name__ == "__main__":
    raise SystemExit(main())
