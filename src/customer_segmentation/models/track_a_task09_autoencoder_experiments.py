"""Run the matched Task 09 P2 autoencoder and latent K-Means comparison."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict
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
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from customer_segmentation.models.autoencoder import (
    AutoencoderTrainingConfig,
    CustomerAutoencoder,
    encode_features,
    load_autoencoder_checkpoint,
    make_internal_validation_indices,
    reconstruction_errors,
    save_autoencoder_checkpoint,
    train_autoencoder,
    validate_train_evaluation_ids,
)
from customer_segmentation.models.clustering_baselines import (
    aligned_adjusted_rand_index,
    evaluate_assignments,
    load_model_artifact,
    make_assignment_frame,
    ordered_id_sha256,
    save_model_artifact,
    validate_feature_contract,
)
from customer_segmentation.models.p2_behavioral_pipeline import (
    P2TrainFittedPreprocessor,
)
from customer_segmentation.models.p2_segmentation_experiments import (
    _aggregate_raw_customer_features,
    _load_interpretable_customer_features,
    _verify_p1_aggregation_semantics,
)
from customer_segmentation.models.track_a_autoencoder_experiments import (
    fit_train_only_latent_scaler,
)
from customer_segmentation.models.track_a_experiments import (
    BOARD_FIELDS,
    _atomic_write,
    _json,
    _mlflow_log,
    _persist_comparison_rows,
    _sha256_file,
    _WorkingSetSampler,
)
from customer_segmentation.models.track_a_experiments import _versions as _base_versions


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"task09_track_a_final_ae_{stamp}_{uuid.uuid4().hex[:8]}"


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Task 09 artifact: {path}")
    _atomic_write(path, content)


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    _write_new(path, (_json(payload) + "\n").encode("utf-8"))


def _write_frame_new(path: Path, frame: pd.DataFrame) -> None:
    _write_new(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _write_joblib_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Task 09 artifact: {path}")
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


def _read_manifest(path: Path, *, expected_rows: int, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"CustomerID": "string"})
    required = {"row_position", "CustomerID", "Transaction_Count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} manifest is missing columns: {sorted(missing)}")
    positions = frame["row_position"].to_numpy(dtype=np.int64)
    ids = frame["CustomerID"].astype("string").to_numpy(dtype=str)
    counts = pd.to_numeric(frame["Transaction_Count"], errors="raise").to_numpy(dtype=np.int64)
    if len(frame) != expected_rows:
        raise ValueError(f"{name} manifest has {len(frame)} rows; expected {expected_rows}.")
    if np.any(positions < 0) or (len(positions) > 1 and np.any(np.diff(positions) <= 0)):
        raise ValueError(f"{name} row positions must be non-negative, sorted, and unique.")
    if pd.isna(ids).any() or np.any(ids == "") or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} manifest has null, empty, or duplicate CustomerIDs.")
    if np.any(counts < 2):
        raise ValueError(f"{name} manifest includes customers outside the 2+ cohort.")
    frame["Transaction_Count"] = counts
    return frame


def validate_task08_preprocessor(
    preprocessor: P2TrainFittedPreprocessor,
    wrapper_metadata: dict[str, Any],
    *,
    training_ids: np.ndarray,
    feature_order: list[str],
    seed: int,
) -> dict[str, Any]:
    """Require the saved Task 08 preprocessor to match its train manifest."""
    expected_hash = ordered_id_sha256(training_ids)
    metadata = preprocessor.metadata()
    if preprocessor.feature_order != feature_order or metadata["feature_order"] != feature_order:
        raise ValueError("Task 08 preprocessor feature order differs from the Track A contract.")
    if preprocessor.fit_rows != len(training_ids) or metadata["scaler_fit_rows"] != len(training_ids):
        raise ValueError("Task 08 preprocessor fit rows do not match the train manifest.")
    if preprocessor.fit_customer_id_sha256 != expected_hash:
        raise ValueError("Task 08 preprocessor fit-ID hash does not match the train manifest.")
    if metadata["fit_customer_id_sha256"] != expected_hash:
        raise ValueError("Task 08 scaler metadata fit-ID hash does not match the train manifest.")
    if wrapper_metadata.get("fit_customer_id_sha256") != expected_hash:
        raise ValueError("Task 08 preprocessor artifact metadata has a mismatched fit-ID hash.")
    if wrapper_metadata.get("variant") != "nine_feature_trainfit":
        raise ValueError("Task 09 requires the nine_feature_trainfit Task 08 preprocessor.")
    if wrapper_metadata.get("cohort") != "two_plus":
        raise ValueError("Task 09 requires the two_plus Task 08 preprocessor.")
    if preprocessor.clip_bounds is not None or preprocessor.clip_feature is not None:
        raise ValueError("Task 09 matched preprocessing must not clip Avg_Spend_Ratio.")
    if preprocessor.as_of_date != "2016-11-01":
        raise ValueError("Task 09 must reuse the Task 08 fixed as-of date.")
    return {
        "seed": seed,
        "fit_rows": preprocessor.fit_rows,
        "fit_customer_id_sha256": expected_hash,
        "feature_order": list(feature_order),
        "clip_bounds": None,
        "scaler_fit_scope": metadata["scaler_fit_scope"],
        "as_of_date": preprocessor.as_of_date,
    }


def load_task08_baseline_assignments(
    root: Path,
    board: pd.DataFrame,
    *,
    evaluation_ids: np.ndarray,
    seed: int,
    k: int,
    algorithm: str,
) -> dict[str, Any]:
    """Load one completed same-cohort, same-feature baseline by ordered IDs."""
    if algorithm not in {"kmeans_p2", "gaussian_mixture_p2"}:
        raise ValueError(f"Unsupported Task 08 baseline algorithm: {algorithm}")
    expected_ids = np.asarray(evaluation_ids, dtype=str)
    candidates = board.loc[
        (board["phase"] == "task08_p2_leakage_safe")
        & (board["algorithm"] == algorithm)
        & (board["seed"] == seed)
        & (board["k"] == k)
        & (board["run_status"] == "completed")
    ]
    matches = []
    for _, row in candidates.iterrows():
        parameters = json.loads(row["parameters_json"] or "{}")
        if (
            parameters.get("cohort") == "two_plus"
            and parameters.get("preprocessing_variant") == "nine_feature_trainfit"
        ):
            matches.append((row, parameters))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Task 08 {algorithm} 2+ nine-feature baseline for seed={seed}, k={k}; "
            f"found {len(matches)}."
        )
    row, parameters = matches[0]
    assignments = pd.read_csv(root / str(row["assignment_path"]), dtype={"CustomerID": "string"})
    evaluation = assignments.loc[assignments["split"] == "evaluation"]
    observed_ids = evaluation["CustomerID"].astype("string").to_numpy(dtype=str)
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError(
            f"Task 08 {algorithm} seed={seed} k={k} assignments do not match the ordered holdout."
        )
    if str(row["evaluation_customer_id_sha256"]) != ordered_id_sha256(expected_ids):
        raise ValueError("Task 08 baseline evaluation hash differs from the Task 09 holdout.")
    return {
        "algorithm": algorithm,
        "seed": seed,
        "k": k,
        "candidate_id": str(row["run_id"]),
        "assignment_path": str(row["assignment_path"]),
        "run_metadata_path": str(row["run_metadata_path"]),
        "labels": evaluation["cluster"].to_numpy(dtype=np.int32),
        "saved_metrics": {
            "silhouette_score": float(row["silhouette_score"]),
            "davies_bouldin_index": float(row["davies_bouldin_index"]),
            "calinski_harabasz_index": float(row["calinski_harabasz_index"]),
            "smallest_segment_pct": float(row["smallest_segment_pct"]),
        },
        "parameters": parameters,
    }


def _load_task08_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    task08 = config["task08"]
    feature_order = list(data["feature_order"])
    with (root / data["feature_contract_file"]).open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)["tracks"]["track_a"]
    validate_feature_contract(contract, feature_order, feature_order)
    if len(feature_order) != 9 or int(config["autoencoder"]["input_dim"]) != 9:
        raise ValueError("Task 09 requires the contracted nine-feature input.")
    if int(task08["minimum_transactions"]) != 2:
        raise ValueError("Task 09 must use the Task 08 2+ cohort.")

    training_manifests = {
        int(seed): _read_manifest(
            root / task08["train_manifests"][str(seed)],
            expected_rows=int(task08["training_rows"]),
            name=f"Task 08 seed-{seed} training",
        )
        for seed in config["sampling"]["seeds"]
    }
    evaluation_manifest = _read_manifest(
        root / task08["evaluation_manifest"],
        expected_rows=int(task08["evaluation_rows"]),
        name="Task 08 shared evaluation",
    )
    evaluation_ids = evaluation_manifest["CustomerID"].astype("string").to_numpy(dtype=str)
    training_ids = {
        seed: frame["CustomerID"].astype("string").to_numpy(dtype=str)
        for seed, frame in training_manifests.items()
    }
    if set(training_ids) != {42, 2024}:
        raise ValueError(f"Task 09 requires Task 08 training seeds 42 and 2024; got {sorted(training_ids)}.")
    for ids in training_ids.values():
        validate_train_evaluation_ids(ids, evaluation_ids)

    wanted_ids = set(evaluation_ids)
    for ids in training_ids.values():
        wanted_ids.update(ids)
    raw_aggregates, raw_stats = _aggregate_raw_customer_features(
        root / data["raw_transactions_file"],
        wanted_ids=wanted_ids,
        chunk_rows=int(data["chunk_rows"]),
    )
    interpreted = _load_interpretable_customer_features(
        root / data["interpretable_file"],
        identifier_column=str(data["identifier_column"]),
        features=feature_order,
        wanted_ids=wanted_ids,
        chunk_rows=int(data["chunk_rows"]),
    )
    as_of = pd.Timestamp(data["as_of_date"])
    aggregate_validation = _verify_p1_aggregation_semantics(
        raw_aggregates,
        interpreted,
        feature_order=feature_order,
        p1_global_max_date=pd.Timestamp(raw_stats["p1_global_max_date"]),
        as_of_date=as_of,
    )
    if as_of.strftime("%Y-%m-%d") != "2016-11-01":
        raise ValueError("Task 09 must retain the Task 08 fixed as-of date.")

    evaluation_frame = raw_aggregates.loc[evaluation_ids].reset_index()
    if not np.array_equal(evaluation_frame["CustomerID"].astype(str).to_numpy(), evaluation_ids):
        raise RuntimeError("Raw evaluation features do not preserve Task 08 manifest order.")
    training_frames: dict[int, pd.DataFrame] = {}
    for seed, ids in training_ids.items():
        frame = raw_aggregates.loc[ids].reset_index()
        expected_counts = training_manifests[seed]["Transaction_Count"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame["Transaction_Count"].to_numpy(dtype=np.int64), expected_counts):
            raise ValueError(f"Raw transaction counts mismatch the Task 08 seed-{seed} manifest.")
        if not np.array_equal(frame["CustomerID"].astype(str).to_numpy(), ids):
            raise RuntimeError(f"Raw seed-{seed} features changed Task 08 manifest order.")
        training_frames[seed] = frame
    if not np.array_equal(
        evaluation_frame["Transaction_Count"].to_numpy(dtype=np.int64),
        evaluation_manifest["Transaction_Count"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Raw evaluation transaction counts mismatch the Task 08 manifest.")
    return {
        "feature_order": feature_order,
        "training_manifests": training_manifests,
        "training_ids": training_ids,
        "training_frames": training_frames,
        "evaluation_manifest": evaluation_manifest,
        "evaluation_ids": evaluation_ids,
        "evaluation_frame": evaluation_frame,
        "raw_aggregates": raw_aggregates,
        "raw_stats": raw_stats,
        "aggregate_validation": aggregate_validation,
        "raw_transactions_sha256": _sha256_file(root / data["raw_transactions_file"]),
        "interpretable_sha256": _sha256_file(root / data["interpretable_file"]),
    }


def _fit_config(settings: dict[str, Any], seed: int) -> AutoencoderTrainingConfig:
    return AutoencoderTrainingConfig(
        input_dim=int(settings["input_dim"]),
        hidden_dims=tuple(int(width) for width in settings["hidden_dims"]),
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


def _metrics(features: np.ndarray, labels: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_assignments(
        features,
        labels,
        max_silhouette_rows=int(config["metrics"]["silhouette_max_rows"]),
        random_state=int(config["metrics"]["silhouette_seed"]),
    )
    if result["metric_status"] != "ok":
        raise ValueError(f"Invalid Task 09 clustering metrics: {result['metric_error']}.")
    return result


def _profile_segments(
    evaluation_frame: pd.DataFrame,
    labels: np.ndarray,
    *,
    as_of_date: str,
    seed: int,
    k: int,
) -> pd.DataFrame:
    frame = evaluation_frame.copy()
    last_date = pd.to_datetime(frame["Last_Transaction"], errors="raise")
    as_of = pd.Timestamp(as_of_date)
    if last_date.isna().any() or (last_date > as_of).any():
        raise ValueError("Evaluation transaction dates are invalid for the Task 08 as-of date.")
    frame["Recency_Days"] = (as_of - last_date).dt.days.astype(np.int64)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape != (len(frame),):
        raise ValueError("Evaluation labels do not align with behavior-profile rows.")
    rows: list[dict[str, Any]] = []
    for cluster in np.unique(labels):
        mask = labels == cluster
        segment = frame.loc[mask]
        tx_counts = segment["Transaction_Count"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "seed": seed,
                "k": k,
                "cluster": int(cluster),
                "customers": int(mask.sum()),
                "evaluation_percent": float(100 * mask.mean()),
                "transaction_count_mean": float(segment["Transaction_Count"].mean()),
                "transaction_count_median": float(segment["Transaction_Count"].median()),
                "exactly_two_percent": float(100 * np.mean(tx_counts == 2)),
                "three_or_more_percent": float(100 * np.mean(tx_counts >= 3)),
                "balance_median": float(segment["AccountBalance"].median()),
                "average_transaction_median": float(segment["Avg_Transaction_Amt"].median()),
                "total_spend_median": float(segment["Total_Spend"].median()),
                "ratio_median": float(segment["Avg_Spend_Ratio"].median()),
                "recency_days_median": float(segment["Recency_Days"].median()),
            }
        )
    return pd.DataFrame(rows)


def _feature_error_rows(
    errors: dict[str, dict[str, float]], feature_order: list[str], *, seed: int, split: str
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "split": split,
            "feature": feature,
            "mae": float(errors[str(index)]["mae"]),
            "rmse": float(errors[str(index)]["rmse"]),
        }
        for index, feature in enumerate(feature_order)
    ]


def _add_cross_seed_ari(row: dict[str, Any], ari: float) -> None:
    parameters = json.loads(row["parameters_json"] or "{}")
    parameters["cross_seed_evaluation_ari"] = float(ari)
    row["parameters_json"] = _json(parameters)


def _make_metric_row(
    *,
    root: Path,
    candidate_id: str,
    metric_space: str,
    metric_feature_order: list[str],
    input_feature_order: list[str],
    seed: int,
    k: int,
    training_ids: np.ndarray,
    evaluation_ids: np.ndarray,
    metrics: dict[str, Any],
    training_seconds: float,
    memory_before: int | None,
    memory_peak: int | None,
    memory_after: int | None,
    model_path: Path,
    assignment_path: Path,
    metadata_path: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    row = {field: None for field in BOARD_FIELDS}
    row.update(
        {
            "run_id": f"{candidate_id}__{metric_space}",
            "phase": "task09_final_autoencoder",
            "algorithm": "autoencoder_kmeans",
            "training_rows": len(training_ids),
            "evaluation_rows": len(evaluation_ids),
            "k": k,
            "seed": seed,
            "train_customer_id_sha256": ordered_id_sha256(training_ids),
            "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_ids),
            "input_model_matrix_sha256": None,
            "p1_preprocessor_sha256": None,
            "feature_order_json": _json(input_feature_order),
            "parameters_json": _json(
                {
                    **parameters,
                    "metric_space": metric_space,
                    "metric_feature_order": metric_feature_order,
                }
            ),
            "silhouette_score": metrics["silhouette_score"],
            "silhouette_sample_rows": metrics["silhouette_sample_rows"],
            "davies_bouldin_index": metrics["davies_bouldin_index"],
            "calinski_harabasz_index": metrics["calinski_harabasz_index"],
            "evaluation_segment_counts_json": _json(metrics["segment_counts"]),
            "evaluation_segment_percentages_json": _json(metrics["segment_percentages"]),
            "smallest_segment_pct": metrics["smallest_segment_pct"],
            "training_seconds": float(training_seconds),
            "working_set_before_bytes": memory_before,
            "peak_working_set_during_fit_bytes": memory_peak,
            "working_set_after_bytes": memory_after,
            "reload_predictions_identical": True,
            "metric_status": metrics["metric_status"],
            "run_status": "completed",
            "model_path": _relative(model_path, root),
            "assignment_path": _relative(assignment_path, root),
            "run_metadata_path": _relative(metadata_path, root),
            "mlflow_status": "pending",
            "error": metrics["metric_error"],
        }
    )
    return row


def _version_metadata() -> dict[str, str]:
    versions = _base_versions()
    versions["torch"] = str(torch.__version__)
    versions["cuda_available"] = str(torch.cuda.is_available())
    versions["scipy"] = __import__("scipy").__version__
    versions["threadpoolctl"] = __import__("threadpoolctl").__version__
    return versions


def run_task09(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("track") != "track_a" or config.get("version") != 1:
        raise ValueError("Task 09 config must be a version-1 Track A experiment.")
    root = config_path.parent.parent.parent
    task08 = config["task08"]
    execution = config["execution"]
    run_id = _run_id()
    model_directory = root / execution["model_directory"] / run_id
    history_directory = root / execution["history_directory"] / run_id
    internal_directory = root / execution["internal_split_directory"] / run_id
    profile_directory = root / execution["profile_directory"] / run_id
    metadata_directory = root / execution["run_metadata_directory"] / run_id
    diagnostics_directory = root / execution["diagnostic_directory"] / run_id
    for directory in (
        model_directory,
        history_directory,
        internal_directory,
        profile_directory,
        metadata_directory,
        diagnostics_directory,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    inputs = _load_task08_inputs(root, config)
    feature_order = inputs["feature_order"]
    training_ids = inputs["training_ids"]
    evaluation_ids = inputs["evaluation_ids"]
    evaluation_frame = inputs["evaluation_frame"]
    seeds = [int(value) for value in config["sampling"]["seeds"]]
    k_values = [int(value) for value in config["clustering"]["k_values"]]
    evaluation_hash = ordered_id_sha256(evaluation_ids)
    board_path = root / task08["comparison_board"]
    board = pd.read_csv(board_path, dtype={"run_id": "string"})

    # Enforce the corrected Task 08 audit evidence without rewriting its history.
    audit_directory = root / task08["audit_directory"]
    audit_transform_path = audit_directory / "ratio_clip_transform_audit.csv"
    audit_assignment_path = audit_directory / "ratio_clip_assignment_effect_audit.csv"
    if not audit_transform_path.is_file() or not audit_assignment_path.is_file():
        raise FileNotFoundError("Corrected Task 08 clipping-audit artifacts are required.")
    audit = pd.read_csv(audit_transform_path)
    expected_clip_counts = {42: 101, 2024: 106}
    for seed, expected_count in expected_clip_counts.items():
        entry = audit.loc[
            (audit["cohort"] == "two_plus")
            & (audit["variant"] == "ratio_clip_trainfit")
            & (audit["seed"] == seed)
        ]
        if len(entry) != 1 or int(entry.iloc[0]["evaluation_feature_values_changed"]) != expected_count:
            raise ValueError(f"Corrected Task 08 ratio-clipping audit mismatch for seed={seed}.")

    version_metadata = _version_metadata()
    torch.set_num_threads(int(config["autoencoder"]["num_threads"]))
    manifest_paths = {
        f"training_seed{seed}": _relative(root / task08["train_manifests"][str(seed)], root)
        for seed in seeds
    }
    manifest_paths["evaluation"] = _relative(root / task08["evaluation_manifest"], root)

    preprocessors: dict[int, dict[str, Any]] = {}
    baseline: dict[tuple[str, int, int], dict[str, Any]] = {}
    for seed in seeds:
        artifact_path = root / task08["preprocessors"][str(seed)]
        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, dict) or not {"preprocessor", "metadata"}.issubset(artifact):
            raise ValueError(f"Invalid Task 08 preprocessor artifact: {artifact_path}")
        processor: P2TrainFittedPreprocessor = artifact["preprocessor"]
        fit_metadata = validate_task08_preprocessor(
            processor,
            artifact["metadata"],
            training_ids=training_ids[seed],
            feature_order=feature_order,
            seed=seed,
        )
        train_features = processor.transform(inputs["training_frames"][seed])
        eval_features = processor.transform(evaluation_frame)
        if train_features.shape != (len(training_ids[seed]), 9):
            raise ValueError(f"Seed-{seed} Task 08 train transform shape mismatch.")
        if eval_features.shape != (len(evaluation_ids), 9):
            raise ValueError(f"Seed-{seed} Task 08 evaluation transform shape mismatch.")
        preprocessors[seed] = {
            "processor": processor,
            "path": artifact_path,
            "sha256": _sha256_file(artifact_path),
            "fit_validation": fit_metadata,
            "train_features": train_features,
            "eval_features": eval_features,
        }
        for algorithm in ("kmeans_p2", "gaussian_mixture_p2"):
            for k in k_values:
                entry = load_task08_baseline_assignments(
                    root,
                    board,
                    evaluation_ids=evaluation_ids,
                    seed=seed,
                    k=k,
                    algorithm=algorithm,
                )
                metrics = _metrics(eval_features, entry["labels"], config)
                entry["metrics"] = metrics
                for metric in (
                    "silhouette_score",
                    "davies_bouldin_index",
                    "calinski_harabasz_index",
                ):
                    if not np.isclose(metrics[metric], entry["saved_metrics"][metric], rtol=1e-7, atol=1e-8):
                        raise ValueError(
                            f"Recomputed Task 08 {algorithm} {metric} differs from its saved ledger "
                            f"for seed={seed}, k={k}."
                        )
                baseline[(algorithm, seed, k)] = entry

    rows: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    profile_frames: list[pd.DataFrame] = []
    feature_error_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    candidate_states: dict[tuple[int, int], dict[str, Any]] = {}
    seed_payloads: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        ids = training_ids[seed]
        p2_train = preprocessors[seed]["train_features"]
        p2_eval = preprocessors[seed]["eval_features"]
        split_seed = seed + int(config["sampling"]["internal_validation_seed_offset"])
        internal_train_index, internal_validation_index = make_internal_validation_indices(
            ids,
            validation_fraction=float(config["sampling"]["internal_validation_fraction"]),
            seed=split_seed,
        )
        internal_train_ids = ids[internal_train_index]
        internal_validation_ids = ids[internal_validation_index]
        validate_train_evaluation_ids(internal_train_ids, internal_validation_ids)
        validate_train_evaluation_ids(ids, evaluation_ids)
        if len(internal_train_ids) + len(internal_validation_ids) != len(ids):
            raise RuntimeError("Internal validation split does not cover the training manifest.")

        internal_manifest_path = internal_directory / f"seed{seed}_internal_split.csv"
        split_label = np.full(len(ids), "internal_train", dtype=object)
        split_label[internal_validation_index] = "internal_validation"
        _write_frame_new(
            internal_manifest_path,
            pd.DataFrame(
                {
                    "row_position": inputs["training_manifests"][seed]["row_position"],
                    "CustomerID": ids,
                    "internal_split": split_label,
                }
            ),
        )

        ae_config = _fit_config(config["autoencoder"], seed)
        ae_memory_sampler = _WorkingSetSampler()
        fit_started = time.perf_counter()
        with ae_memory_sampler:
            fit_result = train_autoencoder(
                p2_train[internal_train_index], p2_train[internal_validation_index], ae_config
            )
        ae_fit_seconds = time.perf_counter() - fit_started

        training_errors = reconstruction_errors(fit_result.model, p2_train[internal_train_index])
        validation_errors = reconstruction_errors(fit_result.model, p2_train[internal_validation_index])
        feature_error_rows.extend(
            _feature_error_rows(training_errors, feature_order, seed=seed, split="internal_train")
        )
        feature_error_rows.extend(
            _feature_error_rows(
                validation_errors, feature_order, seed=seed, split="internal_validation"
            )
        )
        training_summary = {
            "best_epoch": fit_result.best_epoch,
            "epochs_completed": fit_result.epochs_completed,
            "best_train_huber_loss": fit_result.best_train_loss,
            "best_validation_huber_loss": fit_result.best_validation_loss,
            "last_train_huber_loss": float(fit_result.history[-1]["training_loss"]),
            "last_validation_huber_loss": float(fit_result.history[-1]["validation_loss"]),
            "stopped_early": fit_result.stopped_early,
            "training_reconstruction_errors": training_errors,
            "internal_validation_reconstruction_errors": validation_errors,
        }
        history_path = history_directory / f"seed{seed}_training_history.csv"
        _write_frame_new(history_path, pd.DataFrame(fit_result.history))
        checkpoint_path = model_directory / f"seed{seed}_best_autoencoder.pt"
        save_autoencoder_checkpoint(
            checkpoint_path,
            fit_result.model,
            training_config=ae_config,
            training_summary=training_summary,
        )
        checkpoint_payload = load_autoencoder_checkpoint(checkpoint_path)
        model: CustomerAutoencoder = checkpoint_payload["model"]
        expected_architecture = {
            "input_dim": 9,
            "hidden_dims": [32, 16],
            "latent_dim": 4,
            "hidden_activation": "ReLU",
            "latent_activation": "linear",
            "reconstruction_activation": "linear",
        }
        if model.architecture() != expected_architecture:
            raise ValueError("Reloaded model is not the required 9-32-16-4-16-32-9 architecture.")

        train_embeddings = encode_features(model, p2_train)
        eval_embeddings = encode_features(model, p2_eval)
        if not np.array_equal(eval_embeddings, encode_features(model, p2_eval)):
            raise RuntimeError("Reloaded autoencoder evaluation inference is not deterministic.")
        latent_scaler = fit_train_only_latent_scaler(train_embeddings)
        scaled_train_embeddings = latent_scaler.transform(train_embeddings)
        scaled_eval_embeddings = latent_scaler.transform(eval_embeddings)
        if int(latent_scaler.n_samples_seen_) != len(ids):
            raise ValueError("Latent scaler was not fit on all and only Task 08 training embeddings.")
        expected_latent_mean = train_embeddings.astype(np.float64).mean(axis=0)
        if not np.allclose(latent_scaler.mean_, expected_latent_mean, rtol=1e-10, atol=1e-10):
            raise ValueError("Latent scaler mean differs from training embeddings.")

        scaler_path = model_directory / f"seed{seed}_latent_scaler.joblib"
        scaler_payload = {
            "scaler": latent_scaler,
            "metric_space": str(config["metrics"]["latent_space"]),
            "latent_dim": 4,
            "fit_rows": len(ids),
            "fit_customer_id_sha256": ordered_id_sha256(ids),
            "fit_scope": "Task 08 training-manifest customer embeddings only",
            "input_feature_order": feature_order,
        }
        _write_joblib_new(scaler_path, scaler_payload)
        saved_scaler_payload = joblib.load(scaler_path)
        saved_scaler = saved_scaler_payload["scaler"]
        if (
            saved_scaler_payload["fit_rows"] != len(ids)
            or saved_scaler_payload["fit_customer_id_sha256"] != ordered_id_sha256(ids)
            or not np.array_equal(saved_scaler.mean_, latent_scaler.mean_)
            or not np.array_equal(saved_scaler.scale_, latent_scaler.scale_)
        ):
            raise RuntimeError("Saved latent scaler failed its reload and fit-ID checks.")

        seed_payloads[seed] = {
            "run_id": run_id,
            "phase": "task09_final_autoencoder",
            "seed": seed,
            "device": "cpu",
            "architecture": model.architecture(),
            "training_config": asdict(ae_config),
            "training_summary": training_summary,
            "training_seconds": ae_fit_seconds,
            "working_set_before_bytes": ae_memory_sampler.before_bytes,
            "peak_working_set_during_fit_bytes": ae_memory_sampler.peak_bytes,
            "working_set_after_bytes": ae_memory_sampler.after_bytes,
            "training_manifest": manifest_paths[f"training_seed{seed}"],
            "evaluation_manifest": manifest_paths["evaluation"],
            "training_customer_id_sha256": ordered_id_sha256(ids),
            "internal_train_customer_id_sha256": ordered_id_sha256(internal_train_ids),
            "internal_validation_customer_id_sha256": ordered_id_sha256(internal_validation_ids),
            "evaluation_customer_id_sha256": evaluation_hash,
            "internal_train_rows": len(internal_train_ids),
            "internal_validation_rows": len(internal_validation_ids),
            "internal_split_seed": split_seed,
            "internal_split_manifest": _relative(internal_manifest_path, root),
            "history_path": _relative(history_path, root),
            "checkpoint_path": _relative(checkpoint_path, root),
            "latent_scaler_path": _relative(scaler_path, root),
            "latent_scaler_fit_rows": int(saved_scaler.n_samples_seen_),
            "latent_scaler_fit_customer_id_sha256": ordered_id_sha256(ids),
            "p2_preprocessor_path": _relative(preprocessors[seed]["path"], root),
            "p2_preprocessor_sha256": preprocessors[seed]["sha256"],
            "p2_preprocessor_validation": preprocessors[seed]["fit_validation"],
            "p2_preprocessor_fit_scope_note": (
                "The exact Task 08 scaler, fit on the full 50,000-row Task 08 training manifest, "
                "is reused. Internal AE validation IDs are a subset of that training manifest; "
                "no Task 08 held-out IDs enter preprocessing or model selection."
            ),
            "inference_reload_identical": True,
            "inference_deterministic": True,
            "task08_clipping_audit_directory": _relative(audit_directory, root),
            "raw_aggregate_validation": inputs["aggregate_validation"],
            "raw_transactions_sha256": inputs["raw_transactions_sha256"],
            "interpretable_sha256": inputs["interpretable_sha256"],
            "versions": version_metadata,
        }

        for k in k_values:
            candidate_id = f"{run_id}_two_plus_ae9_seed{seed}_k{k}"
            kmeans_path = model_directory / f"{candidate_id}_kmeans.joblib"
            assignment_path = root / execution["assignment_directory"] / f"{candidate_id}_assignments.csv"
            profile_path = profile_directory / f"{candidate_id}_evaluation_profile.csv"
            kmeans_memory = _WorkingSetSampler()
            cluster_started = time.perf_counter()
            with threadpool_limits(limits=int(config["autoencoder"]["num_threads"])), kmeans_memory:
                estimator = KMeans(
                    n_clusters=k,
                    random_state=seed,
                    n_init=int(config["clustering"]["n_init"]),
                    max_iter=int(config["clustering"]["max_iter"]),
                    algorithm=str(config["clustering"]["algorithm_variant"]),
                    init=str(config["clustering"]["init"]),
                ).fit(scaled_train_embeddings)
            kmeans_seconds = time.perf_counter() - cluster_started
            train_labels = estimator.predict(scaled_train_embeddings).astype(np.int32)
            eval_labels = estimator.predict(scaled_eval_embeddings).astype(np.int32)
            if not np.isfinite(estimator.cluster_centers_).all():
                raise ValueError("Task 09 K-Means produced non-finite latent centers.")

            profile = _profile_segments(
                evaluation_frame,
                eval_labels,
                as_of_date=str(config["data"]["as_of_date"]),
                seed=seed,
                k=k,
            )
            profile_frames.append(profile)
            _write_frame_new(profile_path, profile)
            assignment_frame = pd.concat(
                [
                    make_assignment_frame(ids, train_labels, run_id=candidate_id, split="train"),
                    make_assignment_frame(
                        evaluation_ids, eval_labels, run_id=candidate_id, split="evaluation"
                    ),
                ],
                ignore_index=True,
            )
            _write_frame_new(assignment_path, assignment_frame)

            original_metrics = _metrics(p2_eval, eval_labels, config)
            latent_metrics = _metrics(scaled_eval_embeddings, eval_labels, config)
            kmeans_model_metadata = {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "phase": "task09_final_autoencoder",
                "cohort": "two_plus",
                "family": "ae_nine_feature",
                "algorithm": "autoencoder_kmeans",
                "seed": seed,
                "k": k,
                "feature_order": feature_order,
                "training_customer_id_sha256": ordered_id_sha256(ids),
                "evaluation_customer_id_sha256": evaluation_hash,
                "autoencoder_checkpoint_path": _relative(checkpoint_path, root),
                "latent_scaler_path": _relative(scaler_path, root),
                "latent_scaler_fit_rows": len(ids),
                "latent_scaler_fit_customer_id_sha256": ordered_id_sha256(ids),
                "p2_preprocessor_path": _relative(preprocessors[seed]["path"], root),
                "p2_preprocessor_fit_customer_id_sha256": preprocessors[seed]["fit_validation"][
                    "fit_customer_id_sha256"
                ],
                "assignment_path": _relative(assignment_path, root),
                "segment_profile_path": _relative(profile_path, root),
                "parameters": config["clustering"],
                "versions": version_metadata,
            }
            save_model_artifact(kmeans_path, estimator, kmeans_model_metadata)
            reloaded_kmeans = load_model_artifact(kmeans_path)["estimator"]
            reloaded_predictions = reloaded_kmeans.predict(scaled_eval_embeddings).astype(np.int32)
            if not np.array_equal(eval_labels, reloaded_predictions):
                raise RuntimeError("Reloaded K-Means changed Task 09 evaluation predictions.")

            inference_started = time.perf_counter()
            reload_embeddings = encode_features(model, p2_eval)
            reload_scaled_embeddings = saved_scaler.transform(reload_embeddings)
            reload_labels = reloaded_kmeans.predict(reload_scaled_embeddings).astype(np.int32)
            inference_seconds = time.perf_counter() - inference_started
            if not np.array_equal(eval_labels, reload_labels):
                raise RuntimeError("Reloaded AE/scaler/K-Means inference changed evaluation labels.")

            peak_values = [
                value for value in (ae_memory_sampler.peak_bytes, kmeans_memory.peak_bytes)
                if value is not None
            ]
            memory_peak = max(peak_values) if peak_values else None
            training_seconds = ae_fit_seconds + kmeans_seconds
            common_parameters = {
                "experiment": "task09_final_autoencoder",
                "candidate_id": candidate_id,
                "task08_run_id": str(task08["run_id"]),
                "cohort": "two_plus",
                "preprocessing_variant": "nine_feature_trainfit",
                "input_feature_order": feature_order,
                "random_state": seed,
                "k": k,
                "kmeans": config["clustering"],
                "autoencoder_architecture": model.architecture(),
                "autoencoder_training_config": asdict(ae_config),
                "internal_validation_rows": len(internal_validation_ids),
                "internal_validation_customer_id_sha256": ordered_id_sha256(internal_validation_ids),
                "p2_preprocessor_path": _relative(preprocessors[seed]["path"], root),
                "p2_preprocessor_sha256": preprocessors[seed]["sha256"],
                "p2_preprocessor_fit_customer_id_sha256": preprocessors[seed]["fit_validation"][
                    "fit_customer_id_sha256"
                ],
                "latent_scaler_path": _relative(scaler_path, root),
                "latent_scaler_fit_scope": "Task 08 training-manifest embeddings only",
                "latent_scaler_fit_customer_id_sha256": ordered_id_sha256(ids),
                "clip_evaluation_changed_rows": 0,
                "same_feature_baseline_phase": "task08_p2_leakage_safe",
            }
            spaces = [
                (str(config["metrics"]["original_space"]), feature_order, original_metrics),
                (
                    str(config["metrics"]["latent_space"]),
                    [f"latent_{index}" for index in range(4)],
                    latent_metrics,
                ),
            ]
            candidate_rows: list[dict[str, Any]] = []
            candidate_metadata: list[dict[str, Any]] = []
            for metric_space, metric_features, score in spaces:
                metadata_path = metadata_directory / f"{candidate_id}__{metric_space}.json"
                row = _make_metric_row(
                    root=root,
                    candidate_id=candidate_id,
                    metric_space=metric_space,
                    metric_feature_order=metric_features,
                    input_feature_order=feature_order,
                    seed=seed,
                    k=k,
                    training_ids=ids,
                    evaluation_ids=evaluation_ids,
                    metrics=score,
                    training_seconds=training_seconds,
                    memory_before=ae_memory_sampler.before_bytes,
                    memory_peak=memory_peak,
                    memory_after=kmeans_memory.after_bytes,
                    model_path=kmeans_path,
                    assignment_path=assignment_path,
                    metadata_path=metadata_path,
                    parameters=common_parameters,
                )
                metadata = {
                    "run_id": row["run_id"],
                    "candidate_id": candidate_id,
                    "phase": "task09_final_autoencoder",
                    "algorithm": "autoencoder_kmeans",
                    "cohort": "two_plus",
                    "seed": seed,
                    "k": k,
                    "metric_space": metric_space,
                    "metric_feature_order": metric_features,
                    "input_feature_order": feature_order,
                    "metrics": score,
                    "training_manifest": manifest_paths[f"training_seed{seed}"],
                    "evaluation_manifest": manifest_paths["evaluation"],
                    "training_customer_id_sha256": ordered_id_sha256(ids),
                    "evaluation_customer_id_sha256": evaluation_hash,
                    "p2_preprocessor_path": _relative(preprocessors[seed]["path"], root),
                    "p2_preprocessor_sha256": preprocessors[seed]["sha256"],
                    "p2_preprocessor_validation": preprocessors[seed]["fit_validation"],
                    "autoencoder_checkpoint_path": _relative(checkpoint_path, root),
                    "latent_scaler_path": _relative(scaler_path, root),
                    "kmeans_model_path": _relative(kmeans_path, root),
                    "assignment_path": _relative(assignment_path, root),
                    "segment_profile_path": _relative(profile_path, root),
                    "inference_reload_identical": True,
                    "inference_seconds_for_10000_rows": inference_seconds,
                    "training_seconds": training_seconds,
                    "memory_before_bytes": ae_memory_sampler.before_bytes,
                    "memory_peak_bytes": memory_peak,
                    "memory_after_bytes": kmeans_memory.after_bytes,
                    "autoencoder_training_summary": training_summary,
                    "reconstruction_errors_path": _relative(
                        diagnostics_directory / "reconstruction_errors.csv", root
                    ),
                    "internal_split_manifest": _relative(internal_manifest_path, root),
                    "task08_baseline_candidate_ids": {
                        "kmeans": baseline[("kmeans_p2", seed, k)]["candidate_id"],
                        "gmm": baseline[("gaussian_mixture_p2", seed, k)]["candidate_id"],
                    },
                    "parameters": json.loads(row["parameters_json"]),
                    "versions": version_metadata,
                }
                rows.append(row)
                metadata_items.append({"path": metadata_path, "payload": metadata})
                candidate_rows.append(row)
                candidate_metadata.append(metadata)
                metric_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "k": k,
                        "metric_space": metric_space,
                        **score,
                        "training_seconds": training_seconds,
                        "inference_seconds_for_10000_rows": inference_seconds,
                        "memory_peak_bytes": memory_peak,
                    }
                )
            candidate_states[(seed, k)] = {
                "candidate_id": candidate_id,
                "labels": eval_labels,
                "train_labels": train_labels,
                "original_metrics": original_metrics,
                "latent_metrics": latent_metrics,
                "kmeans_path": kmeans_path,
                "assignment_path": assignment_path,
                "profile_path": profile_path,
                "profile": profile,
                "kmeans_seconds": kmeans_seconds,
                "inference_seconds": inference_seconds,
                "memory_peak_bytes": memory_peak,
                "rows": candidate_rows,
                "metadata": candidate_metadata,
            }

        seed_payloads[seed] = {
            "run_id": run_id,
            "phase": "task09_final_autoencoder",
            "seed": seed,
            "device": "cpu",
            "architecture": model.architecture(),
            "training_config": asdict(ae_config),
            "training_summary": training_summary,
            "training_seconds": ae_fit_seconds,
            "working_set_before_bytes": ae_memory_sampler.before_bytes,
            "peak_working_set_during_fit_bytes": ae_memory_sampler.peak_bytes,
            "working_set_after_bytes": ae_memory_sampler.after_bytes,
            "training_manifest": manifest_paths[f"training_seed{seed}"],
            "evaluation_manifest": manifest_paths["evaluation"],
            "training_customer_id_sha256": ordered_id_sha256(ids),
            "internal_train_customer_id_sha256": ordered_id_sha256(internal_train_ids),
            "internal_validation_customer_id_sha256": ordered_id_sha256(internal_validation_ids),
            "evaluation_customer_id_sha256": evaluation_hash,
            "internal_train_rows": len(internal_train_ids),
            "internal_validation_rows": len(internal_validation_ids),
            "internal_split_seed": split_seed,
            "internal_split_manifest": _relative(internal_manifest_path, root),
            "history_path": _relative(history_path, root),
            "checkpoint_path": _relative(checkpoint_path, root),
            "latent_scaler_path": _relative(scaler_path, root),
            "latent_scaler_fit_rows": int(saved_scaler.n_samples_seen_),
            "latent_scaler_fit_customer_id_sha256": ordered_id_sha256(ids),
            "p2_preprocessor_path": _relative(preprocessors[seed]["path"], root),
            "p2_preprocessor_sha256": preprocessors[seed]["sha256"],
            "p2_preprocessor_validation": preprocessors[seed]["fit_validation"],
            "p2_preprocessor_fit_scope_note": (
                "Exact Task 08 scaler fit on all Task 08 training-manifest IDs is reused. "
                "Internal validation is drawn only from those IDs; no Task 08 evaluation IDs "
                "enter preprocessing, early stopping, latent scaling, or K-Means fitting."
            ),
            "inference_reload_identical": True,
            "inference_deterministic": True,
            "task08_clipping_audit_directory": _relative(audit_directory, root),
            "raw_aggregate_validation": inputs["aggregate_validation"],
            "raw_transactions_sha256": inputs["raw_transactions_sha256"],
            "interpretable_sha256": inputs["interpretable_sha256"],
            "versions": version_metadata,
        }

    stability: dict[str, dict[int, float]] = {
        "autoencoder_kmeans": {},
        "kmeans_p2": {},
        "gaussian_mixture_p2": {},
    }
    for k in k_values:
        ae_ari = aligned_adjusted_rand_index(
            candidate_states[(seeds[0], k)]["labels"],
            evaluation_ids,
            candidate_states[(seeds[1], k)]["labels"],
            evaluation_ids,
        )
        stability["autoencoder_kmeans"][k] = ae_ari
        for seed in seeds:
            for row in candidate_states[(seed, k)]["rows"]:
                _add_cross_seed_ari(row, ae_ari)
            for metadata in candidate_states[(seed, k)]["metadata"]:
                metadata["cross_seed_evaluation_ari"] = ae_ari
        for algorithm in ("kmeans_p2", "gaussian_mixture_p2"):
            stability[algorithm][k] = aligned_adjusted_rand_index(
                baseline[(algorithm, seeds[0], k)]["labels"],
                evaluation_ids,
                baseline[(algorithm, seeds[1], k)]["labels"],
                evaluation_ids,
            )

    comparison_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for k in k_values:
            state = candidate_states[(seed, k)]
            compare: dict[str, Any] = {
                "seed": seed,
                "k": k,
                "customer_count": len(evaluation_ids),
                "evaluation_customer_id_sha256": evaluation_hash,
                "ae_candidate_id": state["candidate_id"],
                "ae_original_silhouette": state["original_metrics"]["silhouette_score"],
                "ae_original_davies_bouldin": state["original_metrics"]["davies_bouldin_index"],
                "ae_original_calinski_harabasz": state["original_metrics"]["calinski_harabasz_index"],
                "ae_latent_silhouette": state["latent_metrics"]["silhouette_score"],
                "ae_latent_davies_bouldin": state["latent_metrics"]["davies_bouldin_index"],
                "ae_latent_calinski_harabasz": state["latent_metrics"]["calinski_harabasz_index"],
                "ae_cluster_counts": _json(state["original_metrics"]["segment_counts"]),
                "ae_smallest_segment_percent": state["original_metrics"]["smallest_segment_pct"],
                "ae_cross_seed_ari": stability["autoencoder_kmeans"][k],
            }
            for algorithm, prefix in (
                ("kmeans_p2", "task08_kmeans"),
                ("gaussian_mixture_p2", "task08_gmm"),
            ):
                base = baseline[(algorithm, seed, k)]
                score = base["metrics"]
                compare[f"{prefix}_candidate_id"] = base["candidate_id"]
                compare[f"{prefix}_silhouette"] = score["silhouette_score"]
                compare[f"{prefix}_davies_bouldin"] = score["davies_bouldin_index"]
                compare[f"{prefix}_calinski_harabasz"] = score["calinski_harabasz_index"]
                compare[f"ae_vs_{prefix}_ari"] = float(
                    adjusted_rand_score(state["labels"], base["labels"])
                )
                compare[f"{prefix}_cross_seed_ari"] = stability[algorithm][k]
            comparison_rows.append(compare)

    for seed in seeds:
        seed_payloads[seed]["cross_seed_evaluation_ari"] = {
            str(k): stability["autoencoder_kmeans"][k] for k in k_values
        }
        _write_json_new(metadata_directory / f"seed{seed}_autoencoder.json", seed_payloads[seed])

    profile_frame = pd.concat(
        [candidate_states[(seed, k)]["profile"] for seed in seeds for k in k_values],
        ignore_index=True,
    )
    profile_path = diagnostics_directory / "evaluation_behavior_profiles.csv"
    _write_frame_new(profile_path, profile_frame)
    errors_path = diagnostics_directory / "reconstruction_errors.csv"
    _write_frame_new(errors_path, pd.DataFrame(feature_error_rows))
    metrics_path = diagnostics_directory / "autoencoder_metrics.csv"
    _write_frame_new(metrics_path, pd.DataFrame(metric_rows))
    comparison_path = diagnostics_directory / "task08_baseline_comparison.csv"
    _write_frame_new(comparison_path, pd.DataFrame(comparison_rows))

    for item in metadata_items:
        row = next(row for row in rows if row["run_id"] == item["payload"]["run_id"])
        item["payload"]["mlflow_status"] = _mlflow_log(
            experiment_name=str(execution["mlflow_experiment"]),
            run_id=str(row["run_id"]),
            parameters=json.loads(row["parameters_json"]),
            metrics=item["payload"]["metrics"],
            model_path=root / item["payload"]["kmeans_model_path"],
            assignment_path=root / item["payload"]["assignment_path"],
        )
        row["mlflow_status"] = item["payload"]["mlflow_status"]
        _write_json_new(item["path"], item["payload"])
    _persist_comparison_rows(board_path, rows)

    run_summary = {
        "run_id": run_id,
        "phase": "task09_final_autoencoder",
        "status": "completed",
        "candidate_count": len(candidate_states),
        "comparison_board_rows_appended": len(rows),
        "seeds": seeds,
        "k_values": k_values,
        "cohort": "two_plus",
        "training_rows_per_seed": int(task08["training_rows"]),
        "evaluation_rows": int(task08["evaluation_rows"]),
        "feature_order": feature_order,
        "training_customer_id_sha256": {
            str(seed): ordered_id_sha256(training_ids[seed]) for seed in seeds
        },
        "evaluation_customer_id_sha256": evaluation_hash,
        "p2_preprocessor_validation": {
            str(seed): preprocessors[seed]["fit_validation"] for seed in seeds
        },
        "task08_clipping_audit_directory": _relative(audit_directory, root),
        "task08_ratio_clip_evaluation_counts": expected_clip_counts,
        "cross_seed_evaluation_ari": {
            family: {str(k): value for k, value in values.items()}
            for family, values in stability.items()
        },
        "internal_validation": {
            str(seed): {
                "rows": seed_payloads[seed]["internal_validation_rows"],
                "customer_id_sha256": seed_payloads[seed]["internal_validation_customer_id_sha256"],
                "split_seed": seed_payloads[seed]["internal_split_seed"],
            }
            for seed in seeds
        },
        "aggregate_validation": inputs["aggregate_validation"],
        "metrics_path": _relative(metrics_path, root),
        "baseline_comparison_path": _relative(comparison_path, root),
        "profiles_path": _relative(profile_path, root),
        "reconstruction_errors_path": _relative(errors_path, root),
        "model_directory": _relative(model_directory, root),
        "diagnostics_directory": _relative(diagnostics_directory, root),
        "versions": version_metadata,
        "champion_selected": False,
        "p1_pipeline_modified": False,
        "dec_idec_trained": False,
    }
    _write_json_new(diagnostics_directory / "run_summary.json", run_summary)
    print(
        f"Task 09 {run_id}: {len(candidate_states)} AE-K-Means candidates; "
        f"{len(rows)} metric rows appended; completed."
    )
    print(f"Task 09 diagnostics: {diagnostics_directory}")
    return run_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/models/track_a/track_a_task09_final_autoencoder.yaml",
        help="Matched Task 09 autoencoder configuration.",
    )
    args = parser.parse_args(argv)
    run_task09(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
