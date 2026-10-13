"""Run bounded, reproducible Track A K-Means and GMM experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from customer_segmentation.models.clustering_baselines import (
    SampledCustomers,
    aligned_adjusted_rand_index,
    evaluate_assignments,
    load_model_artifact,
    load_sampled_rows,
    make_assignment_frame,
    make_deterministic_sample_indices,
    ordered_id_sha256,
    save_model_artifact,
    train_gaussian_mixture,
    train_kmeans,
    validate_feature_contract,
)
from customer_segmentation.models.preflight import run_preflight

BOARD_FIELDS = [
    "run_id",
    "phase",
    "algorithm",
    "training_rows",
    "evaluation_rows",
    "k",
    "seed",
    "train_customer_id_sha256",
    "evaluation_customer_id_sha256",
    "input_model_matrix_sha256",
    "p1_preprocessor_sha256",
    "feature_order_json",
    "parameters_json",
    "silhouette_score",
    "silhouette_sample_rows",
    "davies_bouldin_index",
    "calinski_harabasz_index",
    "evaluation_segment_counts_json",
    "evaluation_segment_percentages_json",
    "smallest_segment_pct",
    "kmeans_inertia_train",
    "gmm_bic_evaluation",
    "gmm_aic_evaluation",
    "gmm_converged",
    "iterations",
    "training_seconds",
    "working_set_before_bytes",
    "peak_working_set_during_fit_bytes",
    "working_set_after_bytes",
    "reload_predictions_identical",
    "stability_ari_across_seeds",
    "metric_status",
    "run_status",
    "model_path",
    "assignment_path",
    "run_metadata_path",
    "mlflow_status",
    "error",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def _json(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = temporary.name
        Path(temporary_path).write_bytes(content)
        if path.exists():
            if path.read_bytes() != content:
                raise FileExistsError(f"Refusing to overwrite a different existing artifact: {path}")
            return
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _process_memory_bytes() -> int | None:
    """Current resident working set, using platform APIs without adding a dependency."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
            kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
            return None
    except (AttributeError, OSError, TypeError):
        return None

    try:
        with Path("/proc/self/status").open(encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class _WorkingSetSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_bytes: int | None = None
        self.before_bytes: int | None = None
        self.after_bytes: int | None = None

    def _sample(self) -> None:
        value = _process_memory_bytes()
        if value is not None and (self.peak_bytes is None or value > self.peak_bytes):
            self.peak_bytes = value

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> Self:
        self.before_bytes = _process_memory_bytes()
        self.peak_bytes = self.before_bytes
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self._sample()
        self.after_bytes = _process_memory_bytes()


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("track") != "track_a":
        raise ValueError("Baseline config must describe track_a.")
    return config, config_path.parent.parent.parent


def validate_p1_preprocessor(
    artifact_path: Path,
    *,
    feature_order: list[str],
    identifier_column: str,
) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    if artifact.get("version") != 1 or artifact.get("track") != "track_a":
        raise ValueError("Unsupported or mismatched Track A preprocessing artifact.")
    if identifier_column not in artifact.get("identifier_columns", []):
        raise ValueError("P1 preprocessing artifact does not declare the Track A identifier.")
    if not set(feature_order).issubset(artifact.get("feature_order", [])):
        raise ValueError("P1 preprocessing artifact does not include all baseline features.")
    scaler_features = list(artifact["scaler"].feature_names_in_)
    transformations = artifact.get("transformations", {})
    scaled_features = list(transformations.get("scaled_features", []))
    if identifier_column in scaler_features:
        raise ValueError("CustomerID must never appear in scaler inputs.")
    if identifier_column in artifact.get("feature_order", []):
        raise ValueError("CustomerID must never appear in preprocessing feature metadata.")
    if not set(feature_order).issubset(scaled_features):
        raise ValueError("P1 scaler was not fitted on every contracted Track A feature.")
    if scaler_features != scaled_features:
        raise ValueError("P1 scaler feature order does not match its saved transformation metadata.")
    if transformations.get("scaler") != "sklearn.preprocessing.StandardScaler":
        raise ValueError("Track A baselines require the saved P1 StandardScaler metadata.")
    if transformations.get("input_representation") != "unscaled_interpretable":
        raise ValueError("Unexpected preprocessing input representation in the P1 artifact.")
    expected_log1p = [
        "AccountBalance",
        "Avg_Transaction_Amt",
        "Std_Transaction_Amt",
        "Total_Spend",
    ]
    if transformations.get("log1p_columns") != expected_log1p:
        raise ValueError("Unexpected Track A P1 log1p transformation metadata.")
    if transformations.get("clip_nonnegative_columns") != []:
        raise ValueError("Unexpected Track A P1 clipping metadata.")
    return artifact


def _versions() -> dict[str, str]:
    import sklearn

    return {
        "python": __import__("platform").python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _persist_sample_manifest(
    root: Path,
    *,
    sample_name: str,
    sample: SampledCustomers,
) -> tuple[Path, str]:
    manifest = pd.DataFrame(
        {
            "row_position": sample.row_positions,
            "CustomerID": sample.customer_ids,
        }
    )
    content = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    id_hash = ordered_id_sha256(sample.customer_ids)
    path = root / f"{sample_name}_{id_hash[:12]}.csv"
    _atomic_write(path, content)
    return path, id_hash


def _mlflow_log(
    *,
    experiment_name: str,
    run_id: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    model_path: Path,
    assignment_path: Path,
) -> str:
    try:
        import mlflow
    except ImportError:
        return "not_installed; CSV tracking used"
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({key: str(value) for key, value in parameters.items()})
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    mlflow.log_metric(key, float(value))
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(assignment_path))
        return "logged"
    except Exception as exc:  # noqa: BLE001 - optional tracking failure must not invalidate CSV evidence
        return f"failed:{type(exc).__name__}:{exc}"


def _persist_comparison_rows(board_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    board_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if board_path.exists():
        with board_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != BOARD_FIELDS:
                raise ValueError(f"Existing comparison board has an incompatible schema: {board_path}")
            existing_ids = {row["run_id"] for row in reader}
        duplicates = existing_ids.intersection(row["run_id"] for row in rows)
        if duplicates:
            raise FileExistsError(f"Run IDs already exist in comparison board: {sorted(duplicates)}")

    needs_header = not board_path.exists() or board_path.stat().st_size == 0
    with board_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=BOARD_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _run_one(
    *,
    root: Path,
    config: dict[str, Any],
    phase: str,
    algorithm: str,
    seed: int,
    k: int,
    train_sample: SampledCustomers,
    eval_sample: SampledCustomers,
    train_manifest: Path,
    eval_manifest: Path,
    train_hash: str,
    eval_hash: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"track_a_{phase}_{algorithm}_n{len(train_sample.customer_ids)}_k{k}_seed{seed}_{timestamp}_{uuid.uuid4().hex[:8]}"
    model_path = root / config["execution"]["model_directory"] / f"{run_id}.joblib"
    assignment_path = root / config["execution"]["assignment_directory"] / f"{run_id}.csv"
    metadata_path = root / config["execution"]["run_metadata_directory"] / f"{run_id}.json"
    row: dict[str, Any] = {field: None for field in BOARD_FIELDS}
    row.update(
        {
            "run_id": run_id,
            "phase": phase,
            "algorithm": algorithm,
            "training_rows": len(train_sample.customer_ids),
            "evaluation_rows": len(eval_sample.customer_ids),
            "k": int(k),
            "seed": int(seed),
            "train_customer_id_sha256": train_hash,
            "evaluation_customer_id_sha256": eval_hash,
            "input_model_matrix_sha256": config["_runtime_evidence"]["model_matrix_sha256"],
            "p1_preprocessor_sha256": config["_runtime_evidence"]["preprocessor_sha256"],
            "feature_order_json": _json(config["data"]["features"]),
            "train_sample_manifest": str(train_manifest.relative_to(root)),
            "evaluation_sample_manifest": str(eval_manifest.relative_to(root)),
            "model_path": str(model_path.relative_to(root)),
            "assignment_path": str(assignment_path.relative_to(root)),
            "run_metadata_path": str(metadata_path.relative_to(root)),
            "run_status": "failed",
        }
    )
    predictions_eval: np.ndarray | None = None
    parameters = {"phase": phase, "algorithm": algorithm, "k": k, "random_state": seed}
    warnings_seen: list[str] = []

    try:
        memory_sampler = _WorkingSetSampler()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(
            limits=int(config["execution"].get("thread_limit", 1))
        ), memory_sampler:
            warnings.simplefilter("always")
            if algorithm == "kmeans":
                settings = config["algorithms"]["kmeans"]
                parameters.update(settings)
                estimator = train_kmeans(
                    train_sample.features,
                    n_clusters=k,
                    random_state=seed,
                    n_init=int(settings["n_init"]),
                    max_iter=int(settings["max_iter"]),
                    algorithm=str(settings["algorithm"]),
                )
            elif algorithm == "gaussian_mixture":
                settings = config["algorithms"]["gaussian_mixture"]
                parameters.update(settings)
                estimator = train_gaussian_mixture(
                    train_sample.features,
                    n_components=k,
                    covariance_type=str(settings["covariance_type"]),
                    reg_covar=float(settings["reg_covar"]),
                    random_state=seed,
                    n_init=int(settings["n_init"]),
                    max_iter=int(settings["max_iter"]),
                    tol=float(settings["tol"]),
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        training_seconds = time.perf_counter() - start
        warnings_seen = [f"{type(item.message).__name__}: {item.message}" for item in caught]

        train_labels = estimator.predict(train_sample.features).astype(np.int32)
        predictions_eval = estimator.predict(eval_sample.features).astype(np.int32)
        eval_probabilities = None
        train_probabilities = None
        if algorithm == "gaussian_mixture":
            eval_probabilities = estimator.predict_proba(eval_sample.features).max(axis=1)
            train_probabilities = estimator.predict_proba(train_sample.features).max(axis=1)

        metrics = evaluate_assignments(
            eval_sample.features,
            predictions_eval,
            max_silhouette_rows=int(config["metrics"]["silhouette_max_rows"]),
            random_state=int(config["metrics"]["silhouette_seed"]),
        )
        if algorithm == "kmeans":
            metrics["kmeans_inertia_train"] = float(estimator.inertia_)
            metrics["gmm_bic_evaluation"] = None
            metrics["gmm_aic_evaluation"] = None
            metrics["converged"] = None
            metrics["iterations"] = int(estimator.n_iter_)
        else:
            metrics["kmeans_inertia_train"] = None
            metrics["gmm_bic_evaluation"] = float(estimator.bic(eval_sample.features))
            metrics["gmm_aic_evaluation"] = float(estimator.aic(eval_sample.features))
            metrics["converged"] = bool(estimator.converged_)
            metrics["iterations"] = int(estimator.n_iter_)

        train_assignments = make_assignment_frame(
            train_sample.customer_ids,
            train_labels,
            run_id=run_id,
            split="train",
            probabilities=train_probabilities,
        )
        eval_assignments = make_assignment_frame(
            eval_sample.customer_ids,
            predictions_eval,
            run_id=run_id,
            split="evaluation",
            probabilities=eval_probabilities,
        )
        assignments = pd.concat([train_assignments, eval_assignments], ignore_index=True)
        assignment_content = assignments.to_csv(index=False, lineterminator="\n").encode("utf-8")
        _atomic_write(assignment_path, assignment_content)

        artifact_metadata = {
            "run_id": run_id,
            "phase": phase,
            "track": "track_a",
            "algorithm": algorithm,
            "parameters": parameters,
            "feature_order": list(config["data"]["features"]),
            "identifier_column": config["data"]["identifier_column"],
            "input_model_matrix": config["data"]["model_file"],
            "preprocessor_artifact": config["data"]["preprocessor_artifact"],
            "p1_preprocessor_version": 1,
            "assumption": "Features are already transformed by P1; no additional scaler was fit or applied.",
            "experiment_config": config,
            "training_rows": len(train_sample.customer_ids),
            "evaluation_rows": len(eval_sample.customer_ids),
            "train_customer_id_sha256": train_hash,
            "evaluation_customer_id_sha256": eval_hash,
            "input_model_matrix_sha256": config["_runtime_evidence"]["model_matrix_sha256"],
            "p1_preprocessor_sha256": config["_runtime_evidence"]["preprocessor_sha256"],
            "train_sample_manifest": str(train_manifest.relative_to(root)),
            "evaluation_sample_manifest": str(eval_manifest.relative_to(root)),
            "assignment_path": str(assignment_path.relative_to(root)),
            "metrics": metrics,
            "warnings": warnings_seen,
            "versions": _versions(),
            "training_seconds": training_seconds,
        }
        save_model_artifact(model_path, estimator, artifact_metadata)
        reloaded = load_model_artifact(model_path)["estimator"]
        reload_predictions = reloaded.predict(eval_sample.features)
        reload_same = bool(np.array_equal(predictions_eval, reload_predictions))
        if not reload_same:
            raise RuntimeError("Reloaded model produced different held-out predictions.")
        artifact_metadata["reload_predictions_identical"] = reload_same
        _atomic_write(metadata_path, (_json(artifact_metadata) + "\n").encode("utf-8"))

        row.update(
            {
                "parameters_json": _json(parameters),
                "silhouette_score": metrics["silhouette_score"],
                "silhouette_sample_rows": metrics["silhouette_sample_rows"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "calinski_harabasz_index": metrics["calinski_harabasz_index"],
                "evaluation_segment_counts_json": _json(metrics["segment_counts"]),
                "evaluation_segment_percentages_json": _json(metrics["segment_percentages"]),
                "smallest_segment_pct": metrics["smallest_segment_pct"],
                "kmeans_inertia_train": metrics["kmeans_inertia_train"],
                "gmm_bic_evaluation": metrics["gmm_bic_evaluation"],
                "gmm_aic_evaluation": metrics["gmm_aic_evaluation"],
                "gmm_converged": metrics["converged"],
                "iterations": metrics["iterations"],
                "training_seconds": training_seconds,
                "working_set_before_bytes": memory_sampler.before_bytes,
                "peak_working_set_during_fit_bytes": memory_sampler.peak_bytes,
                "working_set_after_bytes": memory_sampler.after_bytes,
                "reload_predictions_identical": reload_same,
                "metric_status": metrics["metric_status"],
                "run_status": "completed" if metrics["metric_status"] == "ok" else metrics["metric_status"],
                "mlflow_status": _mlflow_log(
                    experiment_name=config["execution"]["mlflow_experiment"],
                    run_id=run_id,
                    parameters=parameters,
                    metrics=metrics,
                    model_path=model_path,
                    assignment_path=assignment_path,
                ),
                "error": metrics["metric_error"],
            }
        )
        return row, predictions_eval
    except Exception as exc:  # noqa: BLE001 - record a failed run and continue the configured experiment matrix
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["run_status"] = "failed"
        row["mlflow_status"] = "not_attempted"
        return row, None


def _run_phase(
    *,
    phase: str,
    root: Path,
    config: dict[str, Any],
    samples: dict[str, SampledCustomers],
    manifests: dict[str, tuple[Path, str]],
    expected_row_count: int,
) -> list[dict[str, Any]]:
    phase_settings = config["sampling"][phase]
    train_count = int(phase_settings["training_rows"])
    seeds = [int(seed) for seed in phase_settings["seeds"]]
    if phase == "smoke":
        k_values = [int(phase_settings["k"])]
    else:
        k_values = [int(k) for k in phase_settings["k_values"]]

    rows: list[dict[str, Any]] = []
    evaluation = samples["evaluation"]
    eval_manifest, eval_hash = manifests["evaluation"]
    prediction_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for seed in seeds:
        train_sample = samples[f"{phase}_train_seed{seed}"]
        train_manifest, train_hash = manifests[f"{phase}_train_seed{seed}"]
        if len(train_sample.customer_ids) != train_count:
            raise ValueError(f"Unexpected training sample size for {phase}/seed{seed}.")
        if set(train_sample.customer_ids).intersection(evaluation.customer_ids):
            raise ValueError("Training and evaluation customer IDs overlap.")
        for k in k_values:
            for algorithm in ("kmeans", "gaussian_mixture"):
                row, labels = _run_one(
                    root=root,
                    config=config,
                    phase=phase,
                    algorithm=algorithm,
                    seed=seed,
                    k=k,
                    train_sample=train_sample,
                    eval_sample=evaluation,
                    train_manifest=train_manifest,
                    eval_manifest=eval_manifest,
                    train_hash=train_hash,
                    eval_hash=eval_hash,
                )
                rows.append(row)
                if labels is not None:
                    prediction_cache[(algorithm, k, seed)] = labels

    if phase == "comparison" and len(seeds) >= 2:
        first_seed, *other_seeds = seeds
        for algorithm in ("kmeans", "gaussian_mixture"):
            for k in k_values:
                first = prediction_cache.get((algorithm, k, first_seed))
                for seed in other_seeds:
                    other = prediction_cache.get((algorithm, k, seed))
                    if first is None or other is None:
                        continue
                    ari = aligned_adjusted_rand_index(
                        first,
                        evaluation.customer_ids,
                        other,
                        evaluation.customer_ids,
                    )
                    for row in rows:
                        if row["algorithm"] == algorithm and row["k"] == k and row["seed"] in (first_seed, seed):
                            row["stability_ari_across_seeds"] = ari
    return rows


def _ensure_preflight(root: Path, track_spec: dict[str, Any]) -> int:
    config_path = root / "configs/models/feature_contracts.yaml"
    report = run_preflight(config_path, selected_tracks=["track_a"])
    if not report["ready"]:
        track_report = report.get("tracks", {}).get("track_a", {})
        errors = track_report.get("errors", report.get("errors", []))
        raise ValueError(f"Track A preflight failed: {errors}")
    return int(report["tracks"]["track_a"]["rows"]["model"])


def run_experiments(config_path: str | Path, phase: str = "all") -> int:
    if phase not in {"smoke", "comparison", "all"}:
        raise ValueError(f"Unsupported phase: {phase}")
    config, root = load_config(config_path)
    feature_order = list(config["data"]["features"])
    identifier = str(config["data"]["identifier_column"])
    with (root / config["data"]["feature_contract_file"]).open(encoding="utf-8") as stream:
        feature_contracts = yaml.safe_load(stream)
    track_spec = feature_contracts["tracks"]["track_a"]
    validate_feature_contract(track_spec, feature_order, feature_order)

    preprocessor_path = root / config["data"]["preprocessor_artifact"]
    validate_p1_preprocessor(
        preprocessor_path,
        feature_order=feature_order,
        identifier_column=identifier,
    )
    if config["data"].get("input_representation") != "p1_model_matrix_already_scaled":
        raise ValueError("Baseline runner must consume the existing P1-transformed model matrix.")

    model_path = root / config["data"]["model_file"]
    config["_runtime_evidence"] = {
        "model_matrix_sha256": _sha256_file(model_path),
        "preprocessor_sha256": _sha256_file(preprocessor_path),
    }
    preflight_rows = _ensure_preflight(root, track_spec)
    if phase == "comparison" and preflight_rows < (
        int(config["sampling"]["comparison"]["training_rows"])
        + int(config["sampling"]["evaluation_rows"])
    ):
        raise ValueError("Track A rows are insufficient for requested comparison samples.")

    evaluation_count = int(config["sampling"]["evaluation_rows"])
    eval_seed = int(config["sampling"]["evaluation_seed"])
    sample_specs: list[tuple[str, int, int]] = []
    if phase in {"smoke", "all"}:
        settings = config["sampling"]["smoke"]
        sample_specs.extend(("smoke", int(settings["training_rows"]), int(seed)) for seed in settings["seeds"])
    if phase in {"comparison", "all"}:
        settings = config["sampling"]["comparison"]
        if preflight_rows < int(settings["training_rows"]) + evaluation_count:
            raise ValueError("Track A rows are insufficient for requested comparison samples.")
        sample_specs.extend(
            ("comparison", int(settings["training_rows"]), int(seed))
            for seed in settings["seeds"]
        )

    requested: dict[str, np.ndarray] = {}
    for sample_phase, training_count, seed in sample_specs:
        train_positions, evaluation_positions = make_deterministic_sample_indices(
            preflight_rows,
            training_count=training_count,
            evaluation_count=evaluation_count,
            training_seed=seed,
            evaluation_seed=eval_seed,
        )
        requested[f"{sample_phase}_train_seed{seed}"] = train_positions
        if "evaluation" in requested and not np.array_equal(requested["evaluation"], evaluation_positions):
            raise RuntimeError("Evaluation sample changed across experiment configurations.")
        requested["evaluation"] = evaluation_positions

    samples = load_sampled_rows(
        model_path,
        identifier_column=identifier,
        feature_order=feature_order,
        requested_positions=requested,
        expected_row_count=preflight_rows,
        chunk_rows=int(config["data"].get("chunk_rows", 50_000)),
    )
    sample_directory = root / config["execution"]["sample_manifests"]
    manifests: dict[str, tuple[Path, str]] = {}
    for sample_name, sample in samples.items():
        manifest_name = f"track_a_{sample_name}_n{len(sample.customer_ids)}"
        manifests[sample_name] = _persist_sample_manifest(
            sample_directory,
            sample_name=manifest_name,
            sample=sample,
        )

    board_path = root / config["execution"]["comparison_board"]
    all_rows: list[dict[str, Any]] = []
    phases = [phase] if phase != "all" else ["smoke", "comparison"]
    for current_phase in phases:
        rows = _run_phase(
            phase=current_phase,
            root=root,
            config=config,
            samples=samples,
            manifests=manifests,
            expected_row_count=preflight_rows,
        )
        _persist_comparison_rows(board_path, rows)
        all_rows.extend(rows)
        print(
            f"Phase {current_phase}: {len(rows)} run(s); "
            f"completed={sum(row['run_status'] == 'completed' for row in rows)}, "
            f"failed={sum(row['run_status'] == 'failed' for row in rows)}, "
            f"degenerate={sum(row['run_status'] == 'degenerate' for row in rows)}"
        )
        if current_phase == "smoke" and any(row["run_status"] != "completed" for row in rows):
            print("Smoke phase was not fully successful; comparison phase was not started.")
            return 1

    successful = sum(row["run_status"] == "completed" for row in all_rows)
    print(f"Comparison board: {board_path}")
    print(f"Candidate models: {root / config['execution']['model_directory']}")
    print(f"Candidate assignments: {root / config['execution']['assignment_directory']}")
    return 0 if successful == len(all_rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/models/track_a/track_a_baselines.yaml",
        help="Track A baseline experiment YAML.",
    )
    parser.add_argument(
        "--phase",
        choices=("smoke", "comparison", "all"),
        default="all",
        help="Run the smoke sample, comparison matrix, or smoke followed by comparison.",
    )
    args = parser.parse_args(argv)
    return run_experiments(args.config, phase=args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
