"""Audit Track A behavioral clusters and run bounded robustness experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from customer_segmentation.models.autoencoder import (
    encode_features,
    load_autoencoder_checkpoint,
    validate_train_evaluation_ids,
)
from customer_segmentation.models.clustering_baselines import (
    SampledCustomers,
    aligned_adjusted_rand_index,
    evaluate_assignments,
    load_model_artifact,
    load_sampled_rows,
    make_assignment_frame,
    ordered_id_sha256,
    save_model_artifact,
)
from customer_segmentation.models.preflight import run_preflight
from customer_segmentation.models.robustness import (
    apply_quantile_clip,
    cohort_segment_profiles,
    fit_quantile_clip_bounds,
    history_assignment_association,
    match_cluster_membership,
    validate_customer_order,
)
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


@dataclass
class ExistingCandidate:
    """A historical fitted candidate and its original ordered assignments."""

    candidate_id: str
    family: str
    source_task: str
    algorithm: str
    seed: int
    k: int
    feature_order: list[str]
    assignment_path: str
    model_path: str
    train_ids: np.ndarray
    train_labels: np.ndarray
    evaluation_ids: np.ndarray
    evaluation_labels: np.ndarray
    metadata: dict[str, Any]


def _new_run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _read_manifest(path: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={"CustomerID": "string"})
    if not {"row_position", "CustomerID"}.issubset(frame.columns):
        raise ValueError(f"Sample manifest is missing required columns: {path}")
    positions = frame["row_position"].to_numpy(dtype=np.int64)
    customer_ids = frame["CustomerID"].astype("string").to_numpy(dtype=str)
    if len(positions) != expected_rows:
        raise ValueError(f"Manifest {path} has {len(positions)} rows, expected {expected_rows}.")
    if np.any(positions < 0) or (len(positions) > 1 and np.any(np.diff(positions) <= 0)):
        raise ValueError(f"Manifest row positions must be sorted, unique, and nonnegative: {path}")
    if len(np.unique(customer_ids)) != len(customer_ids) or np.any(customer_ids == ""):
        raise ValueError(f"Manifest CustomerIDs must be unique and non-empty: {path}")
    return positions, customer_ids


def _load_manifest_samples(
    root: Path,
    config: dict[str, Any],
) -> tuple[dict[int, SampledCustomers], SampledCustomers, dict[str, str]]:
    data = config["data"]
    sampling = config["sampling"]
    features = list(data["features"])
    expected_train = int(sampling["expected_training_rows"])
    expected_evaluation = int(sampling["expected_evaluation_rows"])
    train_paths = {
        int(seed): root / path for seed, path in sampling["training_manifests"].items()
    }
    eval_path = root / sampling["evaluation_manifest"]
    manifest_rows = {
        f"train_seed{seed}": _read_manifest(path, expected_train)
        for seed, path in train_paths.items()
    }
    manifest_rows["evaluation"] = _read_manifest(eval_path, expected_evaluation)
    samples = load_sampled_rows(
        root / data["model_file"],
        identifier_column=str(data["identifier_column"]),
        feature_order=features,
        requested_positions={key: value[0] for key, value in manifest_rows.items()},
        expected_row_count=int(data["row_count"]),
        chunk_rows=int(data["chunk_rows"]),
    )
    for name, sample in samples.items():
        validate_customer_order(manifest_rows[name][1], sample.customer_ids)
    evaluation = samples["evaluation"]
    training_samples = {}
    for seed in train_paths:
        training = samples[f"train_seed{seed}"]
        validate_train_evaluation_ids(training.customer_ids, evaluation.customer_ids)
        training_samples[seed] = training
    manifest_paths = {
        **{f"training_seed{seed}": _relative(path, root) for seed, path in train_paths.items()},
        "evaluation": _relative(eval_path, root),
    }
    return training_samples, evaluation, manifest_paths


def _load_interpretable_subset_and_history_counts(
    path: Path,
    *,
    identifier_column: str,
    feature_order: list[str],
    wanted_ids: set[str],
    chunk_rows: int,
) -> tuple[pd.DataFrame, Counter[int]]:
    """Stream the full count distribution while retaining only manifest customers."""
    usecols = [identifier_column, *feature_order]
    selected_parts: list[pd.DataFrame] = []
    frequencies: Counter[int] = Counter()
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={identifier_column: "string"},
        chunksize=chunk_rows,
    ):
        counts = pd.to_numeric(chunk["Transaction_Count"], errors="raise").to_numpy(dtype=np.int64)
        frequencies.update(int(value) for value in counts)
        matched = chunk.loc[chunk[identifier_column].isin(wanted_ids), usecols].copy()
        if len(matched):
            selected_parts.append(matched)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected[identifier_column] = selected[identifier_column].astype("string")
    if selected[identifier_column].duplicated().any():
        raise ValueError("Interpretable matrix has duplicate IDs among the selected manifests.")
    observed = set(selected[identifier_column].astype(str))
    if observed != wanted_ids:
        raise ValueError(
            f"Interpretable subset mismatch: missing={len(wanted_ids - observed)}, "
            f"unexpected={len(observed - wanted_ids)}."
        )
    return selected.set_index(identifier_column, drop=True), frequencies


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _save_joblib_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Task 07 model: {path}")
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


def _load_assignment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"CustomerID": "string"})
    required = {"CustomerID", "split", "cluster"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Candidate assignment is missing columns: {sorted(required - set(frame.columns))}")
    frame["CustomerID"] = frame["CustomerID"].astype("string")
    if frame["CustomerID"].isna().any() or frame["CustomerID"].duplicated().any():
        raise ValueError(f"Candidate assignment has null or duplicate CustomerID values: {path}")
    return frame


def _audit_existing_candidates(
    root: Path,
    *,
    config: dict[str, Any],
    training_samples: dict[int, SampledCustomers],
    evaluation_sample: SampledCustomers,
) -> tuple[list[ExistingCandidate], dict[str, Any]]:
    board_path = root / config["execution"]["comparison_board"]
    with board_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if header != BOARD_FIELDS:
        raise ValueError("Task 05/06 comparison ledger schema has changed unexpectedly.")
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Comparison ledger contains duplicate run_id values.")

    errors: list[str] = []
    task06_rows = [row for row in rows if row["phase"].startswith("task06_")]
    task06_smoke_rows = [row for row in task06_rows if row["phase"] == "task06_real_smoke"]
    task06_comparison_rows = [row for row in task06_rows if row["phase"] == "task06_real_comparison"]
    candidate_metric_keys: set[tuple[str, str]] = set()
    task06_candidates: dict[str, dict[str, Any]] = {}
    autoencoder_history_audit: dict[str, dict[str, Any]] = {}
    for row in task06_rows:
        if row["run_status"] != "completed":
            errors.append(f"Task 06 run did not complete: {row['run_id']} ({row['run_status']})")
            continue
        if row["reload_predictions_identical"] != "True":
            errors.append(f"Task 06 serialized prediction check failed: {row['run_id']}")
        for artifact_column in ("model_path", "assignment_path", "run_metadata_path"):
            if not (root / row[artifact_column]).is_file():
                errors.append(f"Task 06 artifact path missing for {row['run_id']}: {artifact_column}")
        row_metadata_path = root / row["run_metadata_path"]
        metadata = json.loads(row_metadata_path.read_text(encoding="utf-8"))
        params = json.loads(row["parameters_json"] or "{}")
        metric_space = params.get("metric_space")
        if not metric_space or metadata.get("metric_space") != metric_space:
            errors.append(f"Task 06 metric-space metadata mismatch: {row['run_id']}")
            continue
        candidate_id = str(metadata["candidate_id"])
        key = (candidate_id, str(metric_space))
        if key in candidate_metric_keys:
            errors.append(f"Duplicate Task 06 candidate/metric-space record: {key}")
        candidate_metric_keys.add(key)
        candidate = task06_candidates.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "family": str(metadata["family"]),
                "seed": int(metadata["seed"]),
                "k": int(metadata["k"]),
                "algorithm": str(metadata["algorithm"]),
                "feature_order": list(metadata["input_feature_order"]),
                "assignment_path": str(metadata["assignment_path"]),
                "model_path": str(metadata["kmeans_model_path"]),
                "metadata": metadata,
                "metric_spaces": [],
            },
        )
        candidate["metric_spaces"].append(str(metric_space))
        if candidate["assignment_path"] != metadata["assignment_path"]:
            errors.append(f"Task 06 metric spaces disagree on candidate assignments: {candidate_id}")
        ae_artifacts = metadata.get("autoencoder_artifacts")
        if ae_artifacts and ae_artifacts["checkpoint_path"] not in autoencoder_history_audit:
            checkpoint_path = root / ae_artifacts["checkpoint_path"]
            history_path = root / ae_artifacts["history_path"]
            latent_scaler_path = root / ae_artifacts["latent_scaler_path"]
            if not all(path.is_file() for path in (checkpoint_path, history_path, latent_scaler_path)):
                errors.append(f"Missing Task 06 autoencoder history/checkpoint/scaler for {candidate_id}.")
            else:
                history = pd.read_csv(history_path)
                summary = ae_artifacts["training_summary"]
                best_epoch_from_history = int(history["validation_loss"].idxmin()) + 1
                finite_history = bool(
                    np.isfinite(history[["training_loss", "validation_loss"]].to_numpy()).all()
                )
                if not finite_history:
                    errors.append(f"Non-finite Task 06 autoencoder loss history: {candidate_id}.")
                if best_epoch_from_history != int(summary["best_epoch"]):
                    errors.append(f"Task 06 best checkpoint epoch mismatch: {candidate_id}.")
                scaler_payload = joblib.load(latent_scaler_path)
                if int(scaler_payload["fit_rows"]) != int(row["training_rows"]):
                    errors.append(f"Task 06 latent scaler fit-row scope mismatch: {candidate_id}.")
                restored = load_autoencoder_checkpoint(checkpoint_path)
                if restored["model"].architecture() != ae_artifacts["architecture"]:
                    errors.append(f"Task 06 checkpoint architecture mismatch: {candidate_id}.")
                autoencoder_history_audit[ae_artifacts["checkpoint_path"]] = {
                    "checkpoint_path": ae_artifacts["checkpoint_path"],
                    "history_path": ae_artifacts["history_path"],
                    "epochs": len(history),
                    "best_epoch": best_epoch_from_history,
                    "best_validation_loss": float(history["validation_loss"].min()),
                    "last_training_loss": float(history.iloc[-1]["training_loss"]),
                    "last_validation_loss": float(history.iloc[-1]["validation_loss"]),
                    "early_stopping_triggered": bool(summary["stopped_early"]),
                    "per_feature_training_reconstruction_errors": summary[
                        "training_reconstruction_errors"
                    ],
                    "per_feature_validation_reconstruction_errors": summary[
                        "internal_validation_reconstruction_errors"
                    ],
                }

    if len(task06_smoke_rows) != 2 or len(task06_comparison_rows) != 42:
        errors.append(
            "Task 06 ledger count differs from the recorded execution: "
            f"smoke={len(task06_smoke_rows)}, comparison={len(task06_comparison_rows)}."
        )
    if len(task06_candidates) != 19:
        errors.append(f"Expected 19 Task 06 candidate assignment sets, found {len(task06_candidates)}.")

    expected_features = list(config["data"]["features"])
    full_feature_index = {feature: index for index, feature in enumerate(expected_features)}
    latent_cache: dict[tuple[str, int, str], np.ndarray] = {}
    candidates: list[ExistingCandidate] = []

    for row in rows:
        phase = row["phase"]
        seed = int(row["seed"] or 0)
        k = int(row["k"] or 0)
        algorithm = str(row["algorithm"])
        is_task05 = phase == "comparison" and algorithm in {"kmeans", "gaussian_mixture"} and k in {2, 3}
        if is_task05:
            family = "task05_kmeans9" if algorithm == "kmeans" else "task05_gmm9"
            candidate_id = str(row["run_id"])
            metadata = {}
            feature_order = expected_features
        elif phase == "task06_real_comparison":
            params = json.loads(row["parameters_json"] or "{}")
            if k not in {2, 3} or params.get("family") not in {
                "ae_nine_feature",
                "ae_eight_feature",
                "kmeans_eight_feature",
            }:
                continue
            metadata = json.loads(
                (root / row["run_metadata_path"]).read_text(encoding="utf-8")
            )
            candidate_id = str(metadata["candidate_id"])
            family = str(metadata["family"])
            feature_order = list(metadata["input_feature_order"])
            # One assignment per candidate is intentionally referenced by its multiple metric-space rows.
            if candidate_id in {candidate.candidate_id for candidate in candidates}:
                continue
        else:
            continue

        training_sample = training_samples.get(seed)
        if training_sample is None:
            errors.append(f"No Task 05 sample manifest for candidate {candidate_id} seed {seed}.")
            continue
        assignment_path = root / str(row["assignment_path"])
        assignment = _load_assignment(assignment_path)
        train_assignments = assignment.loc[assignment["split"] == "train"]
        evaluation_assignments = assignment.loc[assignment["split"] == "evaluation"]
        try:
            validate_customer_order(training_sample.customer_ids, train_assignments["CustomerID"].to_numpy(dtype=str))
            validate_customer_order(
                evaluation_sample.customer_ids,
                evaluation_assignments["CustomerID"].to_numpy(dtype=str),
            )
        except ValueError as exc:
            errors.append(f"Assignment/manifest alignment failure for {candidate_id}: {exc}")
            continue
        if row["train_customer_id_sha256"] != ordered_id_sha256(training_sample.customer_ids):
            errors.append(f"Training manifest hash mismatch for {candidate_id}.")
        if row["evaluation_customer_id_sha256"] != ordered_id_sha256(evaluation_sample.customer_ids):
            errors.append(f"Evaluation manifest hash mismatch for {candidate_id}.")

        model_artifact = load_model_artifact(root / str(row["model_path"]))
        estimator = model_artifact["estimator"]
        if is_task05:
            train_features = training_sample.features
            eval_features = evaluation_sample.features
        elif metadata.get("autoencoder_artifacts"):
            ae_artifacts = metadata["autoencoder_artifacts"]
            checkpoint_path = str(ae_artifacts["checkpoint_path"])
            checkpoint = load_autoencoder_checkpoint(root / checkpoint_path)
            scaler_payload = joblib.load(root / ae_artifacts["latent_scaler_path"])
            scaler = scaler_payload["scaler"]

            def get_latent(
                sample: SampledCustomers,
                split: str,
                checkpoint_path: str = checkpoint_path,
                seed: int = seed,
                feature_order: list[str] = feature_order,
                scaler: Any = scaler,
                checkpoint: dict[str, Any] = checkpoint,
            ) -> np.ndarray:
                cache_key = (checkpoint_path, seed, split)
                if cache_key not in latent_cache:
                    indices = np.asarray(
                        [full_feature_index[name] for name in feature_order], dtype=np.int64
                    )
                    values = sample.features[:, indices]
                    latent_cache[cache_key] = scaler.transform(
                        encode_features(checkpoint["model"], values)
                    )
                return latent_cache[cache_key]

            train_features = get_latent(training_sample, "train")
            eval_features = get_latent(evaluation_sample, "evaluation")
        else:
            indices = np.asarray([full_feature_index[name] for name in feature_order], dtype=np.int64)
            train_features = training_sample.features[:, indices]
            eval_features = evaluation_sample.features[:, indices]
        fitted_eval_labels = estimator.predict(eval_features).astype(np.int32)
        saved_eval_labels = evaluation_assignments["cluster"].to_numpy(dtype=np.int32)
        if not np.array_equal(fitted_eval_labels, saved_eval_labels):
            errors.append(f"Reloaded model predictions do not match saved evaluation labels: {candidate_id}.")
        fitted_train_labels = estimator.predict(train_features).astype(np.int32)
        saved_train_labels = train_assignments["cluster"].to_numpy(dtype=np.int32)
        if not np.array_equal(fitted_train_labels, saved_train_labels):
            errors.append(f"Reloaded model predictions do not match saved training labels: {candidate_id}.")

        candidates.append(
            ExistingCandidate(
                candidate_id=candidate_id,
                family=family,
                source_task="task05" if is_task05 else "task06",
                algorithm=algorithm,
                seed=seed,
                k=k,
                feature_order=feature_order,
                assignment_path=str(row["assignment_path"]),
                model_path=str(row["model_path"]),
                train_ids=training_sample.customer_ids,
                train_labels=saved_train_labels,
                evaluation_ids=evaluation_sample.customer_ids,
                evaluation_labels=saved_eval_labels,
                metadata=metadata,
            )
        )

    audit = {
        "ledger_rows_before_task07": len(rows),
        "ledger_schema_columns": len(header),
        "ledger_schema_unchanged": header == BOARD_FIELDS,
        "task06_smoke_metric_rows": len(task06_smoke_rows),
        "task06_comparison_metric_rows": len(task06_comparison_rows),
        "task06_unique_candidate_ids": len(task06_candidates),
        "task06_metric_space_duplicate_count": sum(
            len(candidate["metric_spaces"]) - len(set(candidate["metric_spaces"]))
            for candidate in task06_candidates.values()
        ),
        "task06_candidate_metric_spaces": {
            family: sorted(
                {
                    len(candidate["metric_spaces"])
                    for candidate in task06_candidates.values()
                    if candidate["family"] == family
                }
            )
            for family in sorted({candidate["family"] for candidate in task06_candidates.values()})
        },
        "unique_task05_task06_candidates_reloaded": len(candidates),
        "autoencoder_history_checkpoints": list(autoencoder_history_audit.values()),
        "model_assignment_alignment_errors": errors,
        "candidate_metrics_and_assignment_rows": {
            "Task 06 rows": "multiple metric-space rows intentionally reference one candidate assignment per fitted model",
            "smoke": "one k=2 candidate × original_p1_9d and scaled_latent_4d = two rows",
            "comparison": "12 AE candidates × 2/3 spaces (30 rows) plus 6 K-Means-8 candidates × 2 spaces (12 rows) = 42 rows",
        },
    }
    return candidates, audit


def _profile_candidates(
    candidates: list[ExistingCandidate],
    *,
    root: Path,
    data_by_id: pd.DataFrame,
    global_count_frequencies: Counter[int],
    feature_order: list[str],
    quantiles: list[float],
) -> dict[str, pd.DataFrame]:
    profile_rows: list[dict[str, Any]] = []
    transaction_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    global_cohort_sizes = {
        "exactly_one": int(global_count_frequencies.get(1, 0)),
        "at_least_two": int(sum(count for value, count in global_count_frequencies.items() if value >= 2)),
    }

    for candidate in candidates:
        assignment = _load_assignment(root / candidate.assignment_path)
        for split, expected_ids in (
            ("train", candidate.train_ids),
            ("evaluation", candidate.evaluation_ids),
        ):
            part = assignment.loc[assignment["split"] == split].copy()
            observed_ids = part["CustomerID"].astype("string").to_numpy(dtype=str)
            validate_customer_order(expected_ids, observed_ids)
            labels = part["cluster"].to_numpy(dtype=np.int32)
            raw = data_by_id.loc[observed_ids, feature_order]
            counts = pd.to_numeric(raw["Transaction_Count"], errors="raise").to_numpy(dtype=np.int64)
            split_total = len(part)
            association = history_assignment_association(labels, counts)
            association_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_task": candidate.source_task,
                    "family": candidate.family,
                    "algorithm": candidate.algorithm,
                    "seed": candidate.seed,
                    "k": candidate.k,
                    "split": split,
                    "customers": split_total,
                    "singleton_percent": float(100 * np.mean(counts == 1)),
                    **association,
                }
            )

            for cluster in np.unique(labels):
                mask = labels == cluster
                group = raw.iloc[np.flatnonzero(mask)]
                group_counts = counts[mask]
                one_count = int(np.count_nonzero(group_counts == 1))
                row: dict[str, Any] = {
                    "candidate_id": candidate.candidate_id,
                    "source_task": candidate.source_task,
                    "family": candidate.family,
                    "algorithm": candidate.algorithm,
                    "seed": candidate.seed,
                    "k": candidate.k,
                    "split": split,
                    "cluster": int(cluster),
                    "customer_count": int(mask.sum()),
                    "percent_of_split": float(100 * np.sum(mask) / split_total),
                    "single_transaction_customers": one_count,
                    "single_transaction_percent": float(100 * one_count / np.sum(mask)),
                    "repeat_customer_count": int(np.count_nonzero(group_counts >= 2)),
                    "repeat_customer_percent": float(100 * np.count_nonzero(group_counts >= 2) / np.sum(mask)),
                }
                for quantile in quantiles:
                    suffix = f"p{round(100 * quantile):02d}"
                    values = group[feature_order].quantile(quantile)
                    for feature in feature_order:
                        row[f"{feature}_{suffix}"] = float(values[feature])
                profile_rows.append(row)

                for history_count, count in sorted(Counter(group_counts.tolist()).items()):
                    transaction_rows.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "source_task": candidate.source_task,
                            "family": candidate.family,
                            "seed": candidate.seed,
                            "k": candidate.k,
                            "split": split,
                            "cluster": int(cluster),
                            "transaction_count": int(history_count),
                            "customers": int(count),
                            "percent_of_segment": float(100 * count / np.sum(mask)),
                        }
                    )

            cohorts_df = cohort_segment_profiles(observed_ids, labels, counts)
            for record in cohorts_df.to_dict(orient="records"):
                cohort_total = global_cohort_sizes[record["cohort"]]
                cohort_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "source_task": candidate.source_task,
                        "family": candidate.family,
                        "seed": candidate.seed,
                        "k": candidate.k,
                        "split": split,
                        "global_cohort_customers": cohort_total,
                        "cohort_sample_coverage_percent": float(
                            100 * record["cohort_customers_in_sample"] / cohort_total
                        )
                        if cohort_total
                        else None,
                        **record,
                    }
                )

    return {
        "segment_feature_quantiles": pd.DataFrame(profile_rows),
        "segment_transaction_count_distribution": pd.DataFrame(transaction_rows),
        "cohort_segment_distribution": pd.DataFrame(cohort_rows),
        "history_assignment_association": pd.DataFrame(association_rows),
    }


def _ledger_metric_row(
    *,
    root: Path,
    run_id: str,
    phase: str,
    algorithm: str,
    candidate_family: str,
    metric_space: str,
    metric_feature_order: list[str],
    feature_order: list[str],
    seed: int,
    k: int,
    training_ids: np.ndarray,
    evaluation_ids: np.ndarray,
    training_hash: str | None,
    evaluation_hash: str,
    model_matrix_hash: str | None,
    preprocessor_hash: str | None,
    metrics: dict[str, Any],
    inertia: float,
    iterations: int,
    training_seconds: float,
    memory_before: int | None,
    memory_peak: int | None,
    memory_after: int | None,
    reload_identical: bool,
    model_path: Path,
    assignment_path: Path,
    metadata_path: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    row = {field: None for field in BOARD_FIELDS}
    row.update(
        {
            "run_id": run_id,
            "phase": phase,
            "algorithm": algorithm,
            "training_rows": len(training_ids),
            "evaluation_rows": len(evaluation_ids),
            "k": int(k),
            "seed": int(seed),
            "train_customer_id_sha256": training_hash,
            "evaluation_customer_id_sha256": evaluation_hash,
            "input_model_matrix_sha256": model_matrix_hash,
            "p1_preprocessor_sha256": preprocessor_hash,
            "feature_order_json": _json(feature_order),
            "parameters_json": _json(
                {
                    **parameters,
                    "candidate_family": candidate_family,
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
            "kmeans_inertia_train": float(inertia),
            "iterations": int(iterations),
            "training_seconds": float(training_seconds),
            "working_set_before_bytes": memory_before,
            "peak_working_set_during_fit_bytes": memory_peak,
            "working_set_after_bytes": memory_after,
            "reload_predictions_identical": bool(reload_identical),
            "stability_ari_across_seeds": None,
            "metric_status": metrics["metric_status"],
            "run_status": "completed" if metrics["metric_status"] == "ok" else metrics["metric_status"],
            "model_path": _relative(model_path, root),
            "assignment_path": _relative(assignment_path, root),
            "run_metadata_path": _relative(metadata_path, root),
            "mlflow_status": "pending",
            "error": metrics["metric_error"],
        }
    )
    return row


def _assignment_pair(
    training_ids: np.ndarray,
    training_labels: np.ndarray,
    evaluation_ids: np.ndarray,
    evaluation_labels: np.ndarray,
    candidate_id: str,
) -> pd.DataFrame:
    train = make_assignment_frame(
        training_ids, training_labels, run_id=candidate_id, split="train"
    )
    evaluation = make_assignment_frame(
        evaluation_ids,
        evaluation_labels,
        run_id=candidate_id,
        split="evaluation",
    )
    return pd.concat([train, evaluation], ignore_index=True)


def _candidate_manifest_row(candidate: ExistingCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "source_task": candidate.source_task,
        "algorithm": candidate.algorithm,
        "seed": candidate.seed,
        "k": candidate.k,
        "feature_order": candidate.feature_order,
        "assignment_path": candidate.assignment_path,
        "model_path": candidate.model_path,
    }


def _metric_payloads_from_candidate(candidate: ExistingCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_task": candidate.source_task,
        "family": candidate.family,
        "algorithm": candidate.algorithm,
        "seed": candidate.seed,
        "k": candidate.k,
        "assignment_path": candidate.assignment_path,
        "model_path": candidate.model_path,
    }


def _set_parameters_json(row: dict[str, Any], updates: dict[str, Any]) -> None:
    parameters = json.loads(row["parameters_json"] or "{}")
    parameters.update(updates)
    row["parameters_json"] = _json(parameters)


def _run_ratio_clip_sensitivity(
    *,
    root: Path,
    config: dict[str, Any],
    run_id: str,
    training_samples: dict[int, SampledCustomers],
    evaluation_sample: SampledCustomers,
    existing_candidates: list[ExistingCandidate],
    model_matrix_hash: str,
    preprocessor_hash: str,
    versions: dict[str, str],
) -> tuple[
    list[ExistingCandidate],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    data = config["data"]
    sensitivity = config["sensitivity"]
    feature_order = list(data["features"])
    ratio_index = feature_order.index(str(sensitivity["feature"]))
    lower_q = float(sensitivity["lower_quantile"])
    upper_q = float(sensitivity["upper_quantile"])
    metric_spaces = {
        str(sensitivity["metric_spaces"]["original"]): feature_order,
        str(sensitivity["metric_spaces"]["clipped"]): feature_order,
    }
    existing_by_key = {
        (candidate.family, candidate.seed, candidate.k): candidate
        for candidate in existing_candidates
    }
    task07_model_directory = root / config["execution"]["model_directory"] / run_id
    assignment_directory = root / config["execution"]["assignment_directory"]
    metric_rows: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    new_candidates: list[ExistingCandidate] = []
    threshold_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    membership_effect_rows: list[dict[str, Any]] = []
    candidate_models: dict[tuple[int, int], dict[str, Any]] = {}

    for seed in [int(value) for value in sensitivity["seeds"]]:
        training_sample = training_samples[seed]
        evaluation_ids = evaluation_sample.customer_ids
        training_features = training_sample.features.copy()
        evaluation_features = evaluation_sample.features.copy()
        bounds = fit_quantile_clip_bounds(
            training_features[:, ratio_index],
            feature_name=str(sensitivity["feature"]),
            lower_quantile=lower_q,
            upper_quantile=upper_q,
            method=str(sensitivity["quantile_method"]),
        )
        clipped_train_ratio, train_below, train_above = apply_quantile_clip(
            training_features[:, ratio_index], bounds
        )
        clipped_eval_ratio, eval_below, eval_above = apply_quantile_clip(
            evaluation_features[:, ratio_index], bounds
        )
        clipped_training = training_features.copy()
        clipped_evaluation = evaluation_features.copy()
        clipped_training[:, ratio_index] = clipped_train_ratio
        clipped_evaluation[:, ratio_index] = clipped_eval_ratio
        threshold_rows.append(
            {
                "sensitivity_run_id": run_id,
                "seed": seed,
                **bounds,
                "train_below_lower": int(train_below.sum()),
                "train_above_upper": int(train_above.sum()),
                "train_changed_rows": int(np.count_nonzero(train_below | train_above)),
                "train_changed_percent": float(100 * np.mean(train_below | train_above)),
                "evaluation_rows": len(evaluation_ids),
                "evaluation_below_lower": int(eval_below.sum()),
                "evaluation_above_upper": int(eval_above.sum()),
                "evaluation_changed_rows": int(np.count_nonzero(eval_below | eval_above)),
                "evaluation_changed_percent": float(100 * np.mean(eval_below | eval_above)),
                "transform_scope": "post-P1 P1-model-matrix sensitivity; cutoffs fit on training manifest only",
            }
        )
        for split, sample, original, clipped, below, above in (
            ("train", training_sample, training_features[:, ratio_index], clipped_train_ratio, train_below, train_above),
            ("evaluation", evaluation_sample, evaluation_features[:, ratio_index], clipped_eval_ratio, eval_below, eval_above),
        ):
            for index in np.flatnonzero(below | above):
                changed_rows.append(
                    {
                        "sensitivity_run_id": run_id,
                        "seed": seed,
                        "split": split,
                        "CustomerID": sample.customer_ids[index],
                        "original_p1_ratio": float(original[index]),
                        "clipped_p1_ratio": float(clipped[index]),
                        "clip_side": "lower" if below[index] else "upper",
                        "lower_bound": bounds["lower_bound"],
                        "upper_bound": bounds["upper_bound"],
                    }
                )

        for k in [int(value) for value in sensitivity["k_values"]]:
            candidate_id = f"{run_id}_ratio_clip_seed{seed}_k{k}"
            model_path = task07_model_directory / f"{candidate_id}_kmeans.joblib"
            assignment_path = assignment_directory / f"{candidate_id}_assignments.csv"
            memory = _WorkingSetSampler()
            started = time.perf_counter()
            with threadpool_limits(limits=1), memory:
                estimator = KMeans(
                    n_clusters=k,
                    random_state=seed,
                    n_init=int(sensitivity["n_init"]),
                    max_iter=int(sensitivity["max_iter"]),
                    algorithm=str(sensitivity["algorithm"]),
                ).fit(clipped_training)
            fit_seconds = time.perf_counter() - started
            train_labels = estimator.predict(clipped_training).astype(np.int32)
            evaluation_labels = estimator.predict(clipped_evaluation).astype(np.int32)
            model_metadata = {
                "run_id": candidate_id,
                "phase": "task07_ratio_clip",
                "family": "ratio_clip_kmeans9",
                "seed": seed,
                "k": k,
                "metric_spaces": list(metric_spaces),
                "metric_space_semantics": "post-P1 train-only quantile clipping; original-P1 and clipped-P1 evaluation spaces both recorded",
                "input_feature_order": feature_order,
                "ratio_feature": sensitivity["feature"],
                "clip_bounds": bounds,
                "train_changed_rows": int(np.count_nonzero(train_below | train_above)),
                "evaluation_changed_rows": int(np.count_nonzero(eval_below | eval_above)),
                "training_customer_id_sha256": ordered_id_sha256(training_sample.customer_ids),
                "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_ids),
                "p1_matrix_sha256": model_matrix_hash,
                "p1_preprocessor_sha256": preprocessor_hash,
                "versions": versions,
            }
            save_model_artifact(model_path, estimator, model_metadata)
            reloaded = load_model_artifact(model_path)["estimator"]
            reload_identical = bool(
                np.array_equal(evaluation_labels, reloaded.predict(clipped_evaluation))
            )
            if not reload_identical:
                raise RuntimeError(f"Clipped K-Means predictions changed after reload: {candidate_id}")
            _save_frame(
                assignment_path,
                pd.concat(
                    [
                        make_assignment_frame(
                            training_sample.customer_ids,
                            train_labels,
                            run_id=candidate_id,
                            split="train",
                        ),
                        make_assignment_frame(
                            evaluation_ids,
                            evaluation_labels,
                            run_id=candidate_id,
                            split="evaluation",
                        ),
                    ],
                    ignore_index=True,
                ),
            )

            original_baseline = existing_by_key[("task05_kmeans9", seed, k)]
            eight_baseline = existing_by_key[("kmeans_eight_feature", seed, k)]
            ari_vs_original_train = aligned_adjusted_rand_index(
                original_baseline.train_labels,
                training_sample.customer_ids,
                train_labels,
                training_sample.customer_ids,
            )
            ari_vs_original_eval = aligned_adjusted_rand_index(
                original_baseline.evaluation_labels,
                evaluation_ids,
                evaluation_labels,
                evaluation_ids,
            )
            ari_vs_eight_eval = aligned_adjusted_rand_index(
                eight_baseline.evaluation_labels,
                evaluation_ids,
                evaluation_labels,
                evaluation_ids,
            )
            assignment_effects = {
                "task05_kmeans9_train": match_cluster_membership(
                    training_sample.customer_ids,
                    original_baseline.train_labels,
                    training_sample.customer_ids,
                    train_labels,
                ),
                "task05_kmeans9_evaluation": match_cluster_membership(
                    evaluation_ids,
                    original_baseline.evaluation_labels,
                    evaluation_ids,
                    evaluation_labels,
                ),
                "task06_kmeans8_evaluation": match_cluster_membership(
                    evaluation_ids,
                    eight_baseline.evaluation_labels,
                    evaluation_ids,
                    evaluation_labels,
                ),
            }
            for comparison_name, effect in assignment_effects.items():
                membership_effect_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "k": k,
                        "comparison": comparison_name,
                        **effect,
                    }
                )
            candidate_models[(seed, k)] = {
                "candidate_id": candidate_id,
                "seed": seed,
                "k": k,
                "train_labels": train_labels,
                "evaluation_labels": evaluation_labels,
                "training_ids": training_sample.customer_ids,
                "evaluation_ids": evaluation_ids,
                "fit_seconds": fit_seconds,
                "memory": memory,
                "model_path": model_path,
                "assignment_path": assignment_path,
                "bounds": bounds,
                "ari_vs_original_train": ari_vs_original_train,
                "ari_vs_original_eval": ari_vs_original_eval,
                "ari_vs_eight_eval": ari_vs_eight_eval,
                "assignment_effects": assignment_effects,
                "estimator": estimator,
                "training_sample": training_sample,
            }

            for metric_space, metric_features in (
                (str(sensitivity["metric_spaces"]["original"]), evaluation_features),
                (str(sensitivity["metric_spaces"]["clipped"]), clipped_evaluation),
            ):
                metrics = evaluate_assignments(
                    metric_features,
                    evaluation_labels,
                    max_silhouette_rows=int(config["metrics"]["silhouette_max_rows"]),
                    random_state=int(config["metrics"]["silhouette_seed"]),
                )
                metric_run_id = f"{candidate_id}__{metric_space}"
                metadata_path = root / config["execution"]["run_metadata_directory"] / f"{metric_run_id}.json"
                parameters = {
                    "experiment": "task07_ratio_clip",
                    "candidate_family": "ratio_clip_kmeans9",
                    "post_p1_sensitivity": True,
                    "clip_feature": sensitivity["feature"],
                    "clip_lower_quantile": lower_q,
                    "clip_upper_quantile": upper_q,
                    "clip_lower_bound": bounds["lower_bound"],
                    "clip_upper_bound": bounds["upper_bound"],
                    "quantile_method": bounds["quantile_method"],
                    "metric_space": metric_space,
                    "original_kmeans_train_ari": ari_vs_original_train,
                    "original_kmeans_eval_ari": ari_vs_original_eval,
                    "eight_feature_kmeans_eval_ari": ari_vs_eight_eval,
                    "label_aligned_changes_vs_task05_kmeans_eval": assignment_effects[
                        "task05_kmeans9_evaluation"
                    ]["changed_customers_after_label_alignment"],
                    "label_aligned_changes_vs_task06_kmeans8_eval": assignment_effects[
                        "task06_kmeans8_evaluation"
                    ]["changed_customers_after_label_alignment"],
                    "kmeans": {
                        "n_init": int(sensitivity["n_init"]),
                        "max_iter": int(sensitivity["max_iter"]),
                        "algorithm": str(sensitivity["algorithm"]),
                    },
                }
                row = _ledger_metric_row(
                    root=root,
                    run_id=metric_run_id,
                    phase="task07_ratio_clip",
                    algorithm="kmeans_ratio_clip",
                    candidate_family="ratio_clip_kmeans9",
                    metric_space=metric_space,
                    metric_feature_order=feature_order,
                    feature_order=feature_order,
                    seed=seed,
                    k=k,
                    training_ids=training_sample.customer_ids,
                    evaluation_ids=evaluation_ids,
                    training_hash=ordered_id_sha256(training_sample.customer_ids),
                    evaluation_hash=ordered_id_sha256(evaluation_ids),
                    model_matrix_hash=model_matrix_hash,
                    preprocessor_hash=preprocessor_hash,
                    metrics=metrics,
                    inertia=float(estimator.inertia_),
                    iterations=int(estimator.n_iter_),
                    training_seconds=fit_seconds,
                    memory_before=memory.before_bytes,
                    memory_peak=memory.peak_bytes,
                    memory_after=memory.after_bytes,
                    reload_identical=reload_identical,
                    model_path=model_path,
                    assignment_path=assignment_path,
                    metadata_path=metadata_path,
                    parameters=parameters,
                )
                metric_rows.append(row)
                metadata_items.append(
                    {
                        "path": metadata_path,
                        "payload": {
                            **model_metadata,
                            "run_id": metric_run_id,
                            "candidate_id": candidate_id,
                            "metric_space": metric_space,
                            "metrics": metrics,
                            "parameters": parameters,
                            "assignment_path": _relative(assignment_path, root),
                            "model_path": _relative(model_path, root),
                            "changed_counts": {
                                "training": int(np.count_nonzero(train_below | train_above)),
                                "evaluation": int(np.count_nonzero(eval_below | eval_above)),
                            },
                            "label_aligned_assignment_effects": assignment_effects,
                        },
                    }
                )

            new_candidates.append(
                ExistingCandidate(
                    candidate_id=candidate_id,
                    family="task07_ratio_clip_kmeans9",
                    source_task="task07",
                    algorithm="kmeans_ratio_clip",
                    seed=seed,
                    k=k,
                    feature_order=feature_order,
                    assignment_path=_relative(assignment_path, root),
                    model_path=_relative(model_path, root),
                    train_ids=training_sample.customer_ids,
                    train_labels=train_labels,
                    evaluation_ids=evaluation_ids,
                    evaluation_labels=evaluation_labels,
                    metadata={
                        **model_metadata,
                        "same_seed_ari_vs_task05_kmeans_train": ari_vs_original_train,
                        "same_seed_ari_vs_task05_kmeans_evaluation": ari_vs_original_eval,
                        "same_seed_ari_vs_task06_kmeans8_evaluation": ari_vs_eight_eval,
                        "label_aligned_assignment_effects": assignment_effects,
                    },
                )
            )

    stability: dict[int, float] = {}
    for k in [int(value) for value in sensitivity["k_values"]]:
        left = candidate_models[(int(sensitivity["seeds"][0]), k)]
        right = candidate_models[(int(sensitivity["seeds"][1]), k)]
        ari = aligned_adjusted_rand_index(
            left["evaluation_labels"], evaluation_sample.customer_ids,
            right["evaluation_labels"], evaluation_sample.customer_ids,
        )
        stability[k] = ari
        for row in metric_rows:
            if int(row["k"]) == k:
                row["stability_ari_across_seeds"] = ari
                parameters = json.loads(row["parameters_json"])
                parameters["stability_ari_across_seeds"] = ari
                row["parameters_json"] = _json(parameters)
        for item in metadata_items:
            if int(item["payload"]["k"]) == k:
                item["payload"]["stability_ari_across_seeds"] = ari

    return (
        new_candidates,
        metric_rows,
        metadata_items,
        pd.DataFrame(threshold_rows),
        pd.DataFrame(changed_rows),
        pd.DataFrame(membership_effect_rows),
    )


def _transform_raw_repeat_features(
    frame: pd.DataFrame,
    feature_order: list[str],
    log1p_features: list[str],
) -> np.ndarray:
    values = frame.loc[:, feature_order].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Repeat-only raw feature matrix contains non-finite values.")
    for feature in log1p_features:
        index = feature_order.index(feature)
        if np.any(values[:, index] <= -1):
            raise ValueError(f"Cannot log1p repeat-only feature {feature}: values must be > -1.")
        values[:, index] = np.log1p(values[:, index])
    return values


def _run_repeat_only_kmeans(
    *,
    root: Path,
    config: dict[str, Any],
    run_id: str,
    training_samples: dict[int, SampledCustomers],
    evaluation_sample: SampledCustomers,
    raw_features_by_id: pd.DataFrame,
    raw_interpretable_hash: str,
    versions: dict[str, str],
) -> tuple[list[ExistingCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config["repeat_only"]
    feature_order = list(config["data"]["features"])
    log1p_features = list(settings["log1p_features"])
    evaluation_raw = raw_features_by_id.loc[evaluation_sample.customer_ids, feature_order]
    evaluation_counts = pd.to_numeric(evaluation_raw["Transaction_Count"], errors="raise").to_numpy(dtype=np.int64)
    evaluation_keep = evaluation_counts >= 2
    repeat_evaluation_ids = evaluation_sample.customer_ids[evaluation_keep]
    if len(repeat_evaluation_ids) < max(int(k) for k in settings["k_values"]) + 1:
        raise ValueError("Too few held-out repeat customers for the repeat-only K-Means evaluation.")

    candidates: list[ExistingCandidate] = []
    rows: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    repeat_eval_labels: dict[tuple[int, int], np.ndarray] = {}
    for seed in [int(value) for value in settings["seeds"]]:
        full_train_sample = training_samples[seed]
        train_raw = raw_features_by_id.loc[full_train_sample.customer_ids, feature_order]
        train_counts = pd.to_numeric(train_raw["Transaction_Count"], errors="raise").to_numpy(dtype=np.int64)
        train_keep = train_counts >= 2
        repeat_train_ids = full_train_sample.customer_ids[train_keep]
        repeat_train_positions = full_train_sample.row_positions[train_keep]
        if len(repeat_train_ids) > int(config["sampling"]["expected_training_rows"]):
            raise ValueError("Repeat-only training sample exceeded the configured 50,000-row bound.")
        validate_train_evaluation_ids(repeat_train_ids, repeat_evaluation_ids)
        train_matrix = _transform_raw_repeat_features(
            train_raw.loc[train_keep], feature_order, log1p_features
        )
        eval_matrix = _transform_raw_repeat_features(
            evaluation_raw.loc[evaluation_keep], feature_order, log1p_features
        )
        scaler = StandardScaler().fit(train_matrix)
        train_scaled = scaler.transform(train_matrix)
        eval_scaled = scaler.transform(eval_matrix)
        if not np.isfinite(train_scaled).all() or not np.isfinite(eval_scaled).all():
            raise ValueError("Repeat-only train-fitted scaler produced non-finite values.")
        for k in [int(value) for value in settings["k_values"]]:
            if len(repeat_train_ids) <= k:
                raise ValueError(f"Repeat-only training cohort is too small for k={k}.")
            candidate_id = f"{run_id}_repeat_only_seed{seed}_k{k}"
            model_path = root / config["execution"]["model_directory"] / run_id / f"{candidate_id}_kmeans.joblib"
            scaler_path = root / config["execution"]["model_directory"] / run_id / f"{candidate_id}_scaler.joblib"
            assignment_path = root / config["execution"]["assignment_directory"] / f"{candidate_id}_assignments.csv"
            train_manifest_path = root / config["execution"]["sample_directory"] / f"{candidate_id}_repeat_train.csv"
            evaluation_manifest_path = root / config["execution"]["sample_directory"] / f"{candidate_id}_repeat_evaluation.csv"
            _save_frame(
                train_manifest_path,
                pd.DataFrame({"row_position": repeat_train_positions, "CustomerID": repeat_train_ids}),
            )
            _save_frame(
                evaluation_manifest_path,
                pd.DataFrame(
                    {
                        "row_position": evaluation_sample.row_positions[evaluation_keep],
                        "CustomerID": repeat_evaluation_ids,
                    }
                ),
            )
            _save_joblib_new(
                scaler_path,
                {
                    "scaler": scaler,
                    "fit_scope": "repeat-customer training manifest only",
                    "fit_rows": len(repeat_train_ids),
                    "fit_customer_id_sha256": ordered_id_sha256(repeat_train_ids),
                    "feature_order": feature_order,
                    "log1p_features": log1p_features,
                    "population": "at_least_two_transactions",
                },
            )
            memory = _WorkingSetSampler()
            started = time.perf_counter()
            with threadpool_limits(limits=1), memory:
                estimator = KMeans(
                    n_clusters=k,
                    random_state=seed,
                    n_init=int(settings["n_init"]),
                    max_iter=int(settings["max_iter"]),
                    algorithm=str(settings["algorithm"]),
                ).fit(train_scaled)
            fit_seconds = time.perf_counter() - started
            train_labels = estimator.predict(train_scaled).astype(np.int32)
            eval_labels = estimator.predict(eval_scaled).astype(np.int32)
            model_metadata = {
                "candidate_id": candidate_id,
                "phase": "task07_repeat_only",
                "family": "repeat_only_kmeans_trainfit",
                "population": "at_least_two_transactions",
                "metric_space": str(settings["metric_space"]),
                "feature_order": feature_order,
                "seed": seed,
                "k": k,
                "training_rows": len(repeat_train_ids),
                "evaluation_rows": len(repeat_evaluation_ids),
                "training_customer_id_sha256": ordered_id_sha256(repeat_train_ids),
                "evaluation_customer_id_sha256": ordered_id_sha256(repeat_evaluation_ids),
                "training_manifest": _relative(train_manifest_path, root),
                "evaluation_manifest": _relative(evaluation_manifest_path, root),
                "latent_or_feature_scaler_path": _relative(scaler_path, root),
                "input_interpretable_sha256": raw_interpretable_hash,
                "transformations": {
                    "log1p_features": log1p_features,
                    "scaler": "StandardScaler fit on repeat-only training customers",
                    "evaluation_used_in_fit": False,
                },
                "versions": versions,
            }
            save_model_artifact(model_path, estimator, model_metadata)
            reloaded = load_model_artifact(model_path)["estimator"]
            reload_identical = bool(np.array_equal(eval_labels, reloaded.predict(eval_scaled)))
            if not reload_identical:
                raise RuntimeError(f"Repeat-only K-Means reload changed predictions: {candidate_id}")
            _save_frame(
                assignment_path,
                pd.concat(
                    [
                        make_assignment_frame(repeat_train_ids, train_labels, run_id=candidate_id, split="train"),
                        make_assignment_frame(repeat_evaluation_ids, eval_labels, run_id=candidate_id, split="evaluation"),
                    ],
                    ignore_index=True,
                ),
            )
            repeat_eval_labels[(seed, k)] = eval_labels
            metrics = evaluate_assignments(
                eval_scaled,
                eval_labels,
                max_silhouette_rows=int(config["metrics"]["silhouette_max_rows"]),
                random_state=int(config["metrics"]["silhouette_seed"]),
            )
            metric_run_id = f"{candidate_id}__{settings['metric_space']}"
            metadata_path = root / config["execution"]["run_metadata_directory"] / f"{metric_run_id}.json"
            parameters = {
                "experiment": "task07_repeat_only",
                "candidate_family": "repeat_only_kmeans_trainfit",
                "metric_space": str(settings["metric_space"]),
                "population": "at_least_two_transactions",
                "input_feature_order": feature_order,
                "log1p_features": log1p_features,
                "scaler_fit_scope": "repeat-only training IDs only",
                "n_init": int(settings["n_init"]),
                "max_iter": int(settings["max_iter"]),
                "random_state": seed,
                "stability_ari_across_seeds": None,
            }
            row = _ledger_metric_row(
                root=root,
                run_id=metric_run_id,
                phase="task07_repeat_only",
                algorithm="kmeans_repeat_only",
                candidate_family="repeat_only_kmeans_trainfit",
                metric_space=str(settings["metric_space"]),
                metric_feature_order=feature_order,
                feature_order=feature_order,
                seed=seed,
                k=k,
                training_ids=repeat_train_ids,
                evaluation_ids=repeat_evaluation_ids,
                training_hash=ordered_id_sha256(repeat_train_ids),
                evaluation_hash=ordered_id_sha256(repeat_evaluation_ids),
                model_matrix_hash=None,
                preprocessor_hash=None,
                metrics=metrics,
                inertia=float(estimator.inertia_),
                iterations=int(estimator.n_iter_),
                training_seconds=fit_seconds,
                memory_before=memory.before_bytes,
                memory_peak=memory.peak_bytes,
                memory_after=memory.after_bytes,
                reload_identical=reload_identical,
                model_path=model_path,
                assignment_path=assignment_path,
                metadata_path=metadata_path,
                parameters=parameters,
            )
            rows.append(row)
            metadata_items.append(
                {
                    "path": metadata_path,
                    "payload": {
                        **model_metadata,
                        "run_id": metric_run_id,
                        "metrics": metrics,
                        "parameters": parameters,
                        "assignment_path": _relative(assignment_path, root),
                        "model_path": _relative(model_path, root),
                        "reload_predictions_identical": reload_identical,
                    },
                }
            )
            candidates.append(
                ExistingCandidate(
                    candidate_id=candidate_id,
                    family="repeat_only_kmeans_trainfit",
                    source_task="task07",
                    algorithm="kmeans_repeat_only",
                    seed=seed,
                    k=k,
                    feature_order=feature_order,
                    assignment_path=_relative(assignment_path, root),
                    model_path=_relative(model_path, root),
                    train_ids=repeat_train_ids,
                    train_labels=train_labels,
                    evaluation_ids=repeat_evaluation_ids,
                    evaluation_labels=eval_labels,
                    metadata=model_metadata,
                )
            )

    for k in [int(value) for value in settings["k_values"]]:
        ari = aligned_adjusted_rand_index(
            repeat_eval_labels[(int(settings["seeds"][0]), k)],
            repeat_evaluation_ids,
            repeat_eval_labels[(int(settings["seeds"][1]), k)],
            repeat_evaluation_ids,
        )
        for row in rows:
            if int(row["k"]) == k:
                row["stability_ari_across_seeds"] = ari
                params = json.loads(row["parameters_json"])
                params["stability_ari_across_seeds"] = ari
                row["parameters_json"] = _json(params)
        for item in metadata_items:
            if int(item["payload"]["k"]) == k:
                item["payload"]["stability_ari_across_seeds"] = ari
    return candidates, rows, metadata_items


def _persist_task07_records(
    root: Path,
    board_path: Path,
    rows: list[dict[str, Any]],
    metadata_items: list[dict[str, Any]],
    *,
    experiment_name: str,
) -> None:
    metadata_by_path = {item["path"]: item["payload"] for item in metadata_items}
    for row in rows:
        metadata_path = root / row["run_metadata_path"]
        payload = metadata_by_path[metadata_path]
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
        payload["mlflow_status"] = mlflow_status
    for item in metadata_items:
        _atomic_write(item["path"], (_json(item["payload"]) + "\n").encode("utf-8"))
    _persist_comparison_rows(board_path, rows)


def _preprocessor_leakage_audit(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    artifact_path = root / config["data"]["preprocessor_artifact"]
    artifact = joblib.load(artifact_path)
    scaler = artifact["scaler"]
    fitted_rows = int(scaler.n_samples_seen_)
    model_rows = int(config["data"]["row_count"])
    return {
        "artifact_path": str(config["data"]["preprocessor_artifact"]),
        "artifact_sha256": _sha256_file(artifact_path),
        "p1_preprocessor_version": artifact.get("version"),
        "scaler_class": type(scaler).__name__,
        "scaler_n_samples_seen": fitted_rows,
        "processed_customer_rows": model_rows,
        "fitted_on_full_processed_population": fitted_rows == model_rows,
        "scaler_feature_order": list(scaler.feature_names_in_),
        "selected_train_eval_split_occurs_after_p1_matrix": True,
        "implication": (
            "Task 05/06 holdout metrics are transductive with respect to P1 StandardScaler means/scales; "
            "the P1 scaler was fitted before sample separation on the full customer matrix. "
            "No historical metric is modified by Task 07."
        ),
    }


def _task07_versions() -> dict[str, str]:
    versions = _versions()
    versions["scipy"] = __import__("scipy").__version__
    versions["threadpoolctl"] = __import__("threadpoolctl").__version__
    versions["torch"] = __import__("torch").__version__
    return versions


def run_task07(config_path: str | Path) -> int:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("track") != "track_a":
        raise ValueError("Robustness config must describe track_a.")
    root = config_path.parent.parent.parent
    data = config["data"]
    features = list(data["features"])
    if data.get("ratio_feature") not in features:
        raise ValueError("The configured spending-ratio feature is not in the Track A contract.")
    preflight = run_preflight(
        root / data["feature_contract_file"], selected_tracks=["track_a"]
    )
    if not preflight["ready"]:
        raise ValueError(f"Track A preflight failed: {preflight['tracks']['track_a']['errors']}")
    if int(preflight["tracks"]["track_a"]["rows"]["model"]) != int(data["row_count"]):
        raise ValueError("Track A row count differs from the existing robustness config.")

    with (root / data["feature_contract_file"]).open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)["tracks"]["track_a"]
    if list(contract["model_features"]) != features:
        raise ValueError("Task 07 feature order must exactly match the Track A feature contract.")
    validate_p1_preprocessor(
        root / data["preprocessor_artifact"],
        feature_order=features,
        identifier_column=str(data["identifier_column"]),
    )

    training_samples, evaluation_sample, manifest_paths = _load_manifest_samples(root, config)
    wanted_ids = set(evaluation_sample.customer_ids.astype(str))
    for sample in training_samples.values():
        wanted_ids.update(sample.customer_ids.astype(str))
    data_by_id, transaction_count_frequencies = _load_interpretable_subset_and_history_counts(
        root / data["interpretable_file"],
        identifier_column=str(data["identifier_column"]),
        feature_order=features,
        wanted_ids=wanted_ids,
        chunk_rows=int(data["chunk_rows"]),
    )

    run_id = _new_run_id("task07_track_a_robustness")
    diagnostic_directory = root / config["execution"]["diagnostic_root"] / run_id
    diagnostic_directory.mkdir(parents=True, exist_ok=False)
    model_matrix_path = root / data["model_file"]
    model_matrix_hash = _sha256_file(model_matrix_path)
    preprocessor_path = root / data["preprocessor_artifact"]
    preprocessor_hash = _sha256_file(preprocessor_path)
    leakage = _preprocessor_leakage_audit(root, config)
    versions = _task07_versions()

    existing_candidates, reproducibility_audit = _audit_existing_candidates(
        root,
        config=config,
        training_samples=training_samples,
        evaluation_sample=evaluation_sample,
    )
    if reproducibility_audit["model_assignment_alignment_errors"]:
        raise ValueError(
            "Historical candidate reproducibility audit failed: "
            f"{reproducibility_audit['model_assignment_alignment_errors'][:10]}"
        )

    (
        clipped_candidates,
        clipped_rows,
        clipped_metadata,
        clip_thresholds,
        changed_observations,
        clip_membership_effects,
    ) = (
        _run_ratio_clip_sensitivity(
            root=root,
            config=config,
            run_id=run_id,
            training_samples=training_samples,
            evaluation_sample=evaluation_sample,
            existing_candidates=existing_candidates,
            model_matrix_hash=model_matrix_hash,
            preprocessor_hash=preprocessor_hash,
            versions=versions,
        )
    )
    repeat_candidates, repeat_rows, repeat_metadata = _run_repeat_only_kmeans(
        root=root,
        config=config,
        run_id=run_id,
        training_samples=training_samples,
        evaluation_sample=evaluation_sample,
        raw_features_by_id=data_by_id,
        raw_interpretable_hash=_sha256_file(root / data["interpretable_file"]),
        versions=versions,
    )

    all_candidates = [*existing_candidates, *clipped_candidates, *repeat_candidates]
    tables = _profile_candidates(
        all_candidates,
        root=root,
        data_by_id=data_by_id,
        global_count_frequencies=transaction_count_frequencies,
        feature_order=features,
        quantiles=[float(q) for q in config["audit"]["feature_quantiles"]],
    )
    diagnostic_paths: dict[str, str] = {}
    for name, frame in tables.items():
        path = diagnostic_directory / f"{name}.csv"
        _save_frame(path, frame)
        diagnostic_paths[name] = _relative(path, root)
    frequency_frame = pd.DataFrame(
        [
            {
                "transaction_count": count,
                "customers": n,
                "percent_of_population": 100.0 * n / int(data["row_count"]),
            }
            for count, n in sorted(transaction_count_frequencies.items())
        ]
    )
    frequency_path = diagnostic_directory / "population_transaction_count_distribution.csv"
    _save_frame(frequency_path, frequency_frame)
    diagnostic_paths["population_transaction_count_distribution"] = _relative(frequency_path, root)

    threshold_path = diagnostic_directory / "ratio_clip_thresholds_and_changed_counts.csv"
    _save_frame(threshold_path, clip_thresholds)
    diagnostic_paths["ratio_clip_thresholds_and_changed_counts"] = _relative(threshold_path, root)
    changed_path = diagnostic_directory / "ratio_clip_changed_customer_values.csv"
    if len(changed_observations):
        _save_frame(changed_path, changed_observations)
    else:
        _save_frame(
            changed_path,
            pd.DataFrame(
                columns=[
                    "sensitivity_run_id", "seed", "split", "CustomerID", "original_p1_ratio",
                    "clipped_p1_ratio", "clip_side", "lower_bound", "upper_bound",
                ]
            ),
        )
    diagnostic_paths["ratio_clip_changed_customer_values"] = _relative(changed_path, root)
    membership_effect_path = diagnostic_directory / "ratio_clip_label_aligned_membership_effects.csv"
    _save_frame(membership_effect_path, clip_membership_effects)
    diagnostic_paths["ratio_clip_label_aligned_membership_effects"] = _relative(
        membership_effect_path, root
    )

    all_new_rows = [*clipped_rows, *repeat_rows]
    all_new_metadata = [*clipped_metadata, *repeat_metadata]
    board_path = root / config["execution"]["comparison_board"]
    _persist_task07_records(
        root,
        board_path,
        all_new_rows,
        all_new_metadata,
        experiment_name=str(config["execution"]["mlflow_experiment"]),
    )

    repeat_global = int(sum(n for count, n in transaction_count_frequencies.items() if count >= 2))
    singleton_global = int(transaction_count_frequencies.get(1, 0))
    summary = {
        "run_id": run_id,
        "track": "track_a",
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "versions": versions,
        "config_path": _relative(config_path, root),
        "config": config,
        "task05_task06_input_hashes": {
            "model_matrix_sha256": model_matrix_hash,
            "p1_preprocessor_sha256": preprocessor_hash,
            "interpretable_matrix_sha256": _sha256_file(root / data["interpretable_file"]),
        },
        "manifest_paths": manifest_paths,
        "manifest_ids": {
            "training": {
                str(seed): {
                    "rows": len(sample.customer_ids),
                    "ordered_id_sha256": ordered_id_sha256(sample.customer_ids),
                }
                for seed, sample in training_samples.items()
            },
            "evaluation": {
                "rows": len(evaluation_sample.customer_ids),
                "ordered_id_sha256": ordered_id_sha256(evaluation_sample.customer_ids),
            },
        },
        "historical_ledger_audit": reproducibility_audit,
        "preprocessing_leakage_audit": leakage,
        "population_transaction_history": {
            "rows": int(sum(transaction_count_frequencies.values())),
            "exactly_one_transaction_customers": singleton_global,
            "exactly_one_transaction_percent": 100.0 * singleton_global / int(data["row_count"]),
            "at_least_two_transaction_customers": repeat_global,
            "at_least_two_transaction_percent": 100.0 * repeat_global / int(data["row_count"]),
            "transaction_count_frequencies": {
                str(count): int(number) for count, number in sorted(transaction_count_frequencies.items())
            },
        },
        "ratio_sensitivity": {
            "type": "post-P1 clipping; no source/model file modified",
            "clip_quantile_config": {
                "lower": float(config["sensitivity"]["lower_quantile"]),
                "upper": float(config["sensitivity"]["upper_quantile"]),
                "method": str(config["sensitivity"]["quantile_method"]),
                "feature": str(config["sensitivity"]["feature"]),
            },
            "threshold_table": diagnostic_paths["ratio_clip_thresholds_and_changed_counts"],
            "changed_observation_table": diagnostic_paths["ratio_clip_changed_customer_values"],
            "membership_effect_table": diagnostic_paths[
                "ratio_clip_label_aligned_membership_effects"
            ],
            "candidate_count": len(clipped_candidates),
        },
        "repeat_only_experiment": {
            "population": "at_least_two_transactions only",
            "candidate_count": len(repeat_candidates),
            "metric_space": str(config["repeat_only"]["metric_space"]),
        },
        "diagnostic_tables": diagnostic_paths,
        "new_ledger_rows": len(all_new_rows),
        "new_run_metadata_paths": [_relative(item["path"], root) for item in all_new_metadata],
    }
    summary_path = diagnostic_directory / "run_audit_summary.json"
    _atomic_write(summary_path, (_json(summary) + "\n").encode("utf-8"))
    print(
        f"Task 07 {run_id}: existing candidates checked={len(existing_candidates)}, "
        f"profiles={len(tables['segment_feature_quantiles'])} segment rows, "
        f"clipped candidates={len(clipped_candidates)}, repeat-only candidates={len(repeat_candidates)}, "
        f"new ledger rows={len(all_new_rows)}"
    )
    print(f"Task 07 diagnostics: {diagnostic_directory}")
    print(f"Comparison board: {board_path}")
    if reproducibility_audit["model_assignment_alignment_errors"]:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/models/track_a_robustness.yaml",
        help="Track A robustness experiment YAML.",
    )
    args = parser.parse_args(argv)
    return run_task07(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
