"""Run leakage-safe, transaction-history-aware Track A P2 experiments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from customer_segmentation.models.clustering_baselines import (
    aligned_adjusted_rand_index,
    evaluate_assignments,
    load_model_artifact,
    make_assignment_frame,
    ordered_id_sha256,
    save_model_artifact,
)
from customer_segmentation.models.p2_behavioral_pipeline import (
    BehavioralSegmentationPipeline,
    P2TrainFittedPreprocessor,
)
from customer_segmentation.models.preflight import run_preflight
from customer_segmentation.models.robustness import (
    apply_quantile_clip,
    validate_disjoint_customer_ids,
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

RAW_COLUMNS = [
    "TransactionID",
    "CustomerID",
    "CustomerDOB",
    "CustGender",
    "CustLocation",
    "CustAccountBalance",
    "TransactionDate",
    "TransactionTime",
    "TransactionAmount (INR)",
]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _save_joblib_new(path: Path, artifact: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Task 08 artifact: {path}")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = temporary.name
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _population_eligibility(
    interpreted_path: Path,
    *,
    identifier_column: str,
    minimum_transactions: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    population = pd.read_csv(
        interpreted_path,
        usecols=[identifier_column, "Transaction_Count"],
        dtype={identifier_column: "string"},
    )
    if population[identifier_column].isna().any() or population[identifier_column].duplicated().any():
        raise ValueError("Interpretable population must contain unique, non-null CustomerIDs.")
    counts = pd.to_numeric(population["Transaction_Count"], errors="raise").to_numpy(dtype=np.int64)
    if np.any(counts < 1):
        raise ValueError("Interpretable Transaction_Count values must be positive.")
    population["row_position"] = np.arange(len(population), dtype=np.int64)
    population["Transaction_Count"] = counts
    eligible = population.loc[counts >= minimum_transactions].copy()
    return eligible, {str(int(value)): int(number) for value, number in Counter(counts).items()}


def _sample_cohort_manifests(
    root: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    cohort_name: str,
    eligible: pd.DataFrame,
) -> dict[str, Any]:
    sampling = config["sampling"]["cohorts"][cohort_name]
    train_rows = int(sampling["training_rows"])
    eval_rows = int(sampling["evaluation_rows"])
    seeds = [int(seed) for seed in sampling["seeds"]]
    if train_rows + eval_rows > len(eligible):
        raise ValueError(
            f"Cohort {cohort_name} has {len(eligible)} customers; cannot draw "
            f"{train_rows} training plus {eval_rows} evaluation IDs."
        )
    pool_positions = eligible["row_position"].to_numpy(dtype=np.int64)
    eval_seed = int(config["sampling"]["evaluation_seed"])
    eval_positions = np.sort(
        np.random.default_rng(eval_seed + int(sampling["minimum_transactions"])).choice(
            pool_positions, size=eval_rows, replace=False
        )
    )
    available = pool_positions[~np.isin(pool_positions, eval_positions)]
    manifests_root = root / config["execution"]["sample_directory"] / run_id
    manifests_root.mkdir(parents=True, exist_ok=False)
    output: dict[str, Any] = {
        "cohort": cohort_name,
        "minimum_transactions": int(sampling["minimum_transactions"]),
        "population_rows": len(eligible),
        "population_percent": float(100 * len(eligible) / int(config["data"]["row_count"])),
        "training_rows_requested": train_rows,
        "evaluation_rows_requested": eval_rows,
        "seeds": seeds,
        "train": {},
    }
    for seed in seeds:
        train_positions = np.sort(
            np.random.default_rng(seed + 100_003 * int(sampling["minimum_transactions"])).choice(
                available, size=train_rows, replace=False
            )
        )
        if np.intersect1d(train_positions, eval_positions).size:
            raise RuntimeError("Internal sampling error: training/evaluation positions overlap.")
        train_frame = eligible.set_index("row_position").loc[train_positions].reset_index()
        train_path = manifests_root / f"{cohort_name}_train_seed{seed}_n{train_rows}.csv"
        _save_frame(train_path, train_frame[["row_position", "CustomerID", "Transaction_Count"]])
        ids = train_frame["CustomerID"].astype("string").to_numpy(dtype=str)
        output["train"][str(seed)] = {
            "manifest_path": _relative(train_path, root),
            "row_positions": train_positions,
            "customer_ids": ids,
            "customer_id_sha256": ordered_id_sha256(ids),
        }
    eval_frame = eligible.set_index("row_position").loc[eval_positions].reset_index()
    eval_path = manifests_root / f"{cohort_name}_evaluation_n{eval_rows}.csv"
    _save_frame(eval_path, eval_frame[["row_position", "CustomerID", "Transaction_Count"]])
    eval_ids = eval_frame["CustomerID"].astype("string").to_numpy(dtype=str)
    output["evaluation"] = {
        "manifest_path": _relative(eval_path, root),
        "row_positions": eval_positions,
        "customer_ids": eval_ids,
        "customer_id_sha256": ordered_id_sha256(eval_ids),
    }
    for seed in seeds:
        validate_ids = output["train"][str(seed)]["customer_ids"]
        if np.intersect1d(validate_ids, eval_ids).size:
            raise RuntimeError(f"Cohort {cohort_name}/seed{seed} training overlaps evaluation.")
    return output


def _aggregate_raw_customer_features(
    raw_path: Path,
    *,
    wanted_ids: set[str],
    chunk_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the P1 transaction aggregates and last dates for selected IDs."""
    selected_parts: list[pd.DataFrame] = []
    raw_rows = 0
    complete_case_rows = 0
    invalid_complete_dates = 0
    full_as_of: pd.Timestamp | None = None

    for chunk in pd.read_csv(
        raw_path,
        usecols=RAW_COLUMNS,
        dtype={"CustomerID": "string"},
        chunksize=chunk_rows,
    ):
        raw_rows += len(chunk)
        complete = chunk.dropna().copy()
        complete_case_rows += len(complete)
        dates = pd.to_datetime(
            complete["TransactionDate"], format="%d/%m/%y", errors="coerce"
        )
        invalid_complete_dates += int(dates.isna().sum())
        max_date = dates.max()
        if pd.notna(max_date):
            full_as_of = max_date if full_as_of is None or max_date > full_as_of else full_as_of
        selected = complete.loc[complete["CustomerID"].isin(wanted_ids)].copy()
        if not len(selected):
            continue
        selected["TransactionDate"] = pd.to_datetime(
            selected["TransactionDate"], format="%d/%m/%y", errors="raise"
        )
        selected["TransactionAmount (INR)"] = pd.to_numeric(
            selected["TransactionAmount (INR)"], errors="raise"
        )
        selected["CustAccountBalance"] = pd.to_numeric(
            selected["CustAccountBalance"], errors="raise"
        )
        selected["HourOfDay"] = pd.to_numeric(selected["TransactionTime"], errors="raise") // 10000
        selected["Is_Weekend"] = selected["TransactionDate"].dt.dayofweek.isin([5, 6]).astype(np.int8)
        selected["Is_Night_Spender"] = (
            (selected["HourOfDay"] >= 20) | (selected["HourOfDay"] < 4)
        ).astype(np.int8)
        selected["Spend_Balance_Ratio"] = selected["TransactionAmount (INR)"] / (
            selected["CustAccountBalance"] + 1.0
        )
        selected_parts.append(
            selected[
                [
                    "CustomerID",
                    "TransactionID",
                    "CustAccountBalance",
                    "TransactionAmount (INR)",
                    "Spend_Balance_Ratio",
                    "TransactionDate",
                    "Is_Weekend",
                    "Is_Night_Spender",
                ]
            ]
        )

    if full_as_of is None:
        raise ValueError("Raw Track A transaction file contains no valid complete-case dates.")
    if invalid_complete_dates:
        raise ValueError(
            f"Found {invalid_complete_dates} unparsable dates after the P1 complete-case policy."
        )
    if not selected_parts:
        raise ValueError("No requested manifest customers were found in the raw transaction input.")
    selected_transactions = pd.concat(selected_parts, ignore_index=True)
    selected_transactions["CustomerID"] = selected_transactions["CustomerID"].astype("string")
    aggregate = selected_transactions.groupby("CustomerID", sort=True).agg(
        AccountBalance=("CustAccountBalance", "mean"),
        Transaction_Count=("TransactionID", "count"),
        Avg_Transaction_Amt=("TransactionAmount (INR)", "mean"),
        Std_Transaction_Amt=("TransactionAmount (INR)", "std"),
        Total_Spend=("TransactionAmount (INR)", "sum"),
        Avg_Spend_Ratio=("Spend_Balance_Ratio", "mean"),
        Weekend_Spend_Ratio=("Is_Weekend", "mean"),
        Night_Spend_Ratio=("Is_Night_Spender", "mean"),
        Last_Transaction=("TransactionDate", "max"),
    )
    aggregate["Std_Transaction_Amt"] = aggregate["Std_Transaction_Amt"].fillna(0.0)
    aggregate["Transaction_Count"] = aggregate["Transaction_Count"].astype(np.int64)
    if len(aggregate) != len(wanted_ids):
        missing_count = len(wanted_ids - set(aggregate.index.astype(str)))
        raise ValueError(f"Raw aggregate ID coverage mismatch; missing selected customers={missing_count}.")
    return aggregate, {
        "raw_transaction_rows": raw_rows,
        "complete_case_transaction_rows": complete_case_rows,
        "complete_case_invalid_dates": invalid_complete_dates,
        "p1_global_max_date": full_as_of,
        "p1_full_input_max_transaction_date": full_as_of.date().isoformat(),
        "selected_customer_ids": len(aggregate),
    }


def _load_interpretable_customer_features(
    path: Path,
    *,
    identifier_column: str,
    features: list[str],
    wanted_ids: set[str],
    chunk_rows: int,
) -> pd.DataFrame:
    selected_parts = []
    usecols = [identifier_column, *features]
    for frame in pd.read_csv(
        path,
        usecols=usecols,
        dtype={identifier_column: "string"},
        chunksize=chunk_rows,
    ):
        matched = frame.loc[frame[identifier_column].isin(wanted_ids), usecols]
        if len(matched):
            selected_parts.append(matched.copy())
    selected = pd.concat(selected_parts, ignore_index=True)
    selected[identifier_column] = selected[identifier_column].astype("string")
    if selected[identifier_column].duplicated().any():
        raise ValueError("Interpretable feature file has duplicate selected CustomerID values.")
    if set(selected[identifier_column].astype(str)) != wanted_ids:
        raise ValueError("Interpretable file does not cover every selected manifest ID.")
    return selected.set_index(identifier_column, drop=True)


def _verify_p1_aggregation_semantics(
    raw_aggregates: pd.DataFrame,
    interpreted: pd.DataFrame,
    *,
    feature_order: list[str],
    p1_global_max_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    errors: dict[str, float] = {}
    counts_match = np.array_equal(
        raw_aggregates["Transaction_Count"].to_numpy(dtype=np.int64),
        interpreted.loc[raw_aggregates.index, "Transaction_Count"].to_numpy(dtype=np.int64),
    )
    for feature in feature_order:
        if feature == "Recency_Days":
            rebuilt_p1_recency = (
                p1_global_max_date - raw_aggregates["Last_Transaction"]
            ).dt.days.to_numpy(dtype=np.float64)
            differences = np.abs(
                rebuilt_p1_recency
                - interpreted.loc[raw_aggregates.index, feature].to_numpy(dtype=np.float64)
            )
        else:
            differences = np.abs(
                raw_aggregates[feature].to_numpy(dtype=np.float64)
                - interpreted.loc[raw_aggregates.index, feature].to_numpy(dtype=np.float64)
            )
        errors[feature] = float(differences.max(initial=0.0))
    if not counts_match:
        raise ValueError("Raw complete-case Transaction_Count does not reproduce P1 interpretable counts.")
    bad = {feature: delta for feature, delta in errors.items() if delta > 1e-5}
    if bad:
        raise ValueError(f"Raw aggregate/P1 interpretable validation failed: {bad}")
    if as_of_date < p1_global_max_date:
        raise ValueError("Configured P2 as_of_date precedes a valid source transaction date.")
    return {
        "selected_customer_count": len(raw_aggregates),
        "p1_transaction_counts_exactly_match": counts_match,
        "maximum_absolute_differences_by_feature": errors,
        "p1_recency_rebuilt_from_max_complete_case_date": p1_global_max_date.date().isoformat(),
        "p2_fixed_as_of_date": as_of_date.date().isoformat(),
        "maximum_selected_last_transaction": raw_aggregates["Last_Transaction"].max().date().isoformat(),
        "all_selected_transactions_precede_as_of_date": bool(
            (raw_aggregates["Last_Transaction"] <= as_of_date).all()
        ),
    }


def _with_as_of_recency(frame: pd.DataFrame, *, as_of_date: str) -> pd.DataFrame:
    output = frame.copy()
    as_of = pd.Timestamp(as_of_date)
    last_date = pd.to_datetime(output["Last_Transaction"], errors="raise")
    if last_date.isna().any() or (last_date > as_of).any():
        raise ValueError("Selected customer transaction date is invalid for the configured as-of date.")
    output["Recency_Days"] = (as_of - last_date).dt.days.astype(np.int64)
    return output


def _evaluation_segment_profile(
    evaluation_frame: pd.DataFrame,
    labels: np.ndarray,
    feature_order: list[str],
    *,
    as_of_date: str,
    component_posterior: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = _with_as_of_recency(evaluation_frame, as_of_date=as_of_date)
    labels = np.asarray(labels, dtype=np.int32)
    if len(enriched) != len(labels):
        raise ValueError("Evaluation segment profiles require aligned labels and customers.")
    singleton = enriched["Transaction_Count"].to_numpy(dtype=np.int64) == 1
    rows: list[dict[str, Any]] = []
    transaction_rows: list[dict[str, Any]] = []
    if component_posterior is not None:
        component_posterior = np.asarray(component_posterior, dtype=np.float64)
        if component_posterior.shape != labels.shape or not np.isfinite(component_posterior).all():
            raise ValueError("GMM posterior values do not align with evaluation assignments.")
    for cluster in np.unique(labels):
        mask = labels == cluster
        segment = enriched.loc[mask, feature_order]
        row: dict[str, Any] = {
            "cluster": int(cluster),
            "customers": int(mask.sum()),
            "evaluation_percent": float(100 * mask.sum() / len(labels)),
            "single_transaction_customers": int(np.count_nonzero(singleton[mask])),
            "single_transaction_percent": float(100 * np.mean(singleton[mask])),
        }
        quantiles = segment.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
            suffix = f"p{round(100 * quantile):02d}"
            for feature in feature_order:
                row[f"{feature}_{suffix}"] = float(quantiles.loc[quantile, feature])
        if component_posterior is not None:
            row["gmm_max_posterior_mean"] = float(component_posterior[mask].mean())
            row["gmm_max_posterior_p10"] = float(np.quantile(component_posterior[mask], 0.1))
            row["gmm_max_posterior_p50"] = float(np.quantile(component_posterior[mask], 0.5))
        rows.append(row)
        count_distribution = Counter(segment["Transaction_Count"].astype(int).tolist())
        for transaction_count, n in sorted(count_distribution.items()):
            transaction_rows.append(
                {
                    "cluster": int(cluster),
                    "transaction_count": int(transaction_count),
                    "customers": int(n),
                    "percent_of_segment": float(100 * n / mask.sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(transaction_rows)


def _versions() -> dict[str, str]:
    return {
        "python": __import__("platform").python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "joblib": joblib.__version__,
        "scipy": __import__("scipy").__version__,
        "threadpoolctl": __import__("threadpoolctl").__version__,
    }


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _fit_one_candidate(
    *,
    root: Path,
    config: dict[str, Any],
    top_run_id: str,
    cohort_name: str,
    minimum_transactions: int,
    variant_name: str,
    preprocessor: P2TrainFittedPreprocessor,
    preprocessor_path: Path,
    preprocessor_fit_seconds: float,
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    seed: int,
    k: int,
    algorithm_name: str,
    raw_input_hash: str,
    aggregate_validation: dict[str, Any],
    cohort_population_rows: int,
) -> dict[str, Any]:
    feature_order = list(config["data"]["feature_order"])
    train_ids = training_frame["CustomerID"].astype("string").to_numpy(dtype=str)
    eval_ids = evaluation_frame["CustomerID"].astype("string").to_numpy(dtype=str)
    validate_disjoint_customer_ids(train_ids, eval_ids)
    training_features = preprocessor.transform(training_frame)
    evaluation_features = preprocessor.transform(evaluation_frame)
    clip_evaluation_changed_rows = 0
    if preprocessor.clip_bounds is not None and preprocessor.clip_feature is not None:
        _, eval_below, eval_above = apply_quantile_clip(
            evaluation_frame[preprocessor.clip_feature].to_numpy(dtype=np.float64),
            preprocessor.clip_bounds,
        )
        clip_evaluation_changed_rows = int(np.count_nonzero(eval_below | eval_above))
    if len(train_ids) > 50_000:
        raise ValueError("Task 08 candidate exceeds the 50,000 training-row resource limit.")

    candidate_id = f"{top_run_id}_{cohort_name}_{variant_name}_{algorithm_name}_seed{seed}_k{k}"
    model_path = root / config["execution"]["model_directory"] / top_run_id / f"{candidate_id}.joblib"
    assignment_path = root / config["execution"]["assignment_directory"] / f"{candidate_id}_assignments.csv"
    profile_path = root / config["execution"]["profile_directory"] / top_run_id / f"{candidate_id}_evaluation_profile.csv"
    metadata_path = root / config["execution"]["run_metadata_directory"] / f"{candidate_id}.json"

    memory = _WorkingSetSampler()
    warnings_seen: list[str] = []
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(
        limits=int(config["execution"]["thread_limit"])
    ), memory:
        warnings.simplefilter("always")
        if algorithm_name == "kmeans":
            settings = config["models"]["kmeans"]
            estimator = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=int(settings["n_init"]),
                max_iter=int(settings["max_iter"]),
                algorithm=str(settings["algorithm"]),
            ).fit(training_features)
        elif algorithm_name == "gaussian_mixture":
            settings = config["models"]["gaussian_mixture"]
            estimator = GaussianMixture(
                n_components=k,
                covariance_type=str(settings["covariance_type"]),
                reg_covar=float(settings["reg_covar"]),
                random_state=seed,
                n_init=int(settings["n_init"]),
                max_iter=int(settings["max_iter"]),
                tol=float(settings["tol"]),
            ).fit(training_features)
        else:
            raise ValueError(f"Unsupported Task 08 algorithm: {algorithm_name}")
    estimator_fit_seconds = time.perf_counter() - fit_started
    warnings_seen = [f"{type(item.message).__name__}: {item.message}" for item in caught]

    train_labels = estimator.predict(training_features).astype(np.int32)
    eval_labels = estimator.predict(evaluation_features).astype(np.int32)
    train_posterior = None
    eval_posterior = None
    gmm_converged = None
    if algorithm_name == "gaussian_mixture":
        train_posterior = estimator.predict_proba(training_features).max(axis=1)
        eval_posterior = estimator.predict_proba(evaluation_features).max(axis=1)
        gmm_converged = bool(estimator.converged_)

    pipeline = BehavioralSegmentationPipeline(
        preprocessor=preprocessor,
        estimator=estimator,
        minimum_transactions=minimum_transactions,
        cohort=cohort_name,
        metric_space=f"p2_{cohort_name}_{variant_name}_trainfit_scaled_9d",
        metadata={
            "top_run_id": top_run_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "k": k,
            "algorithm": algorithm_name,
            "preprocessor_path": _relative(preprocessor_path, root),
        },
    )
    inference_train = pipeline.predict_features(training_frame)
    inference_eval = pipeline.predict_features(evaluation_frame)
    if not np.array_equal(inference_train["cluster"].to_numpy(dtype=np.int32), train_labels):
        raise RuntimeError("Pipeline train inference did not reproduce fitted-model labels.")
    if not np.array_equal(inference_eval["cluster"].to_numpy(dtype=np.int32), eval_labels):
        raise RuntimeError("Pipeline evaluation inference did not reproduce fitted-model labels.")
    if algorithm_name == "gaussian_mixture" and not np.allclose(
        inference_eval["max_component_posterior"].to_numpy(dtype=np.float64),
        eval_posterior,
    ):
        raise RuntimeError("GMM uncertainty output does not match estimator posterior probabilities.")

    train_assignments = make_assignment_frame(
        train_ids,
        train_labels,
        run_id=candidate_id,
        split="train",
        probabilities=train_posterior,
    )
    eval_assignments = make_assignment_frame(
        eval_ids,
        eval_labels,
        run_id=candidate_id,
        split="evaluation",
        probabilities=eval_posterior,
    )
    _save_frame(
        assignment_path,
        pd.concat([train_assignments, eval_assignments], ignore_index=True),
    )
    eval_profiles, tx_distribution = _evaluation_segment_profile(
        evaluation_frame,
        eval_labels,
        feature_order,
        as_of_date=str(config["data"]["as_of_date"]),
        component_posterior=eval_posterior,
    )
    _save_frame(profile_path, eval_profiles)

    model_metadata = {
        "candidate_id": candidate_id,
        "top_run_id": top_run_id,
        "cohort": cohort_name,
        "minimum_transactions": minimum_transactions,
        "preprocessing_variant": variant_name,
        "metric_space": pipeline.metric_space,
        "algorithm": algorithm_name,
        "input_feature_order": feature_order,
        "seed": seed,
        "k": k,
        "as_of_date": config["data"]["as_of_date"],
        "preprocessor_path": _relative(preprocessor_path, root),
        "preprocessor_metadata": preprocessor.metadata(),
        "clip_evaluation_changed_rows": clip_evaluation_changed_rows,
        "preprocessor_fit_seconds": preprocessor_fit_seconds,
        "estimator_fit_seconds": estimator_fit_seconds,
        "training_customer_id_sha256": ordered_id_sha256(train_ids),
        "evaluation_customer_id_sha256": ordered_id_sha256(eval_ids),
        "assignment_path": _relative(assignment_path, root),
        "segment_profile_path": _relative(profile_path, root),
        "transaction_count_distribution": tx_distribution.to_dict(orient="records"),
        "raw_aggregate_validation": aggregate_validation,
        "raw_interpretable_input_sha256": raw_input_hash,
        "warnings": warnings_seen,
        "gmm_converged": gmm_converged,
        "versions": _versions(),
    }
    if algorithm_name == "gaussian_mixture":
        model_metadata["gmm_uncertainty"] = {
            "field": "max_component_posterior",
            "summary": {
                "mean": float(np.mean(eval_posterior)),
                "p10": float(np.quantile(eval_posterior, 0.1)),
                "p50": float(np.quantile(eval_posterior, 0.5)),
                "p90": float(np.quantile(eval_posterior, 0.9)),
            },
            "calibrated": False,
        }
    save_model_artifact(model_path, pipeline, model_metadata)
    reloaded_pipeline = load_model_artifact(model_path)["estimator"]
    reload_eval = reloaded_pipeline.predict_features(evaluation_frame)
    reload_predictions_identical = bool(
        np.array_equal(reload_eval["cluster"].to_numpy(dtype=np.int32), eval_labels)
    )
    if not reload_predictions_identical:
        raise RuntimeError("Reloaded P2 pipeline changed held-out cluster predictions.")

    metrics = evaluate_assignments(
        evaluation_features,
        eval_labels,
        max_silhouette_rows=int(config["metrics"]["silhouette_max_rows"]),
        random_state=int(config["metrics"]["silhouette_seed"]),
    )
    if algorithm_name == "gaussian_mixture":
        metrics["gmm_bic_evaluation"] = float(estimator.bic(evaluation_features))
        metrics["gmm_aic_evaluation"] = float(estimator.aic(evaluation_features))
        metrics["gmm_converged"] = bool(estimator.converged_)
    else:
        metrics["gmm_bic_evaluation"] = None
        metrics["gmm_aic_evaluation"] = None
        metrics["gmm_converged"] = None

    metadata_payload = {
        **model_metadata,
        "metrics": metrics,
        "metrics_metric_space": pipeline.metric_space,
        "silhouette_sample_rows": metrics["silhouette_sample_rows"],
        "evaluation_segment_counts": metrics["segment_counts"],
        "evaluation_segment_percentages": metrics["segment_percentages"],
        "smallest_segment_pct": metrics["smallest_segment_pct"],
        "kmeans_inertia_train": float(estimator.inertia_)
        if algorithm_name == "kmeans"
        else None,
        "gmm_aic_evaluation": metrics["gmm_aic_evaluation"],
        "gmm_bic_evaluation": metrics["gmm_bic_evaluation"],
        "reload_predictions_identical": reload_predictions_identical,
    }
    return {
        "candidate_id": candidate_id,
        "cohort": cohort_name,
        "variant": variant_name,
        "algorithm": algorithm_name,
        "seed": seed,
        "k": k,
        "metric_space": pipeline.metric_space,
        "evaluation_ids": eval_ids,
        "evaluation_labels": eval_labels,
        "training_ids": train_ids,
        "training_labels": train_labels,
        "row": None,
        "row_metadata": {
            "path": metadata_path,
            "payload": metadata_payload,
        },
        "metadata_payload": metadata_payload,
        "model_path": model_path,
        "assignment_path": assignment_path,
        "profile_path": profile_path,
        "preprocessor_path": preprocessor_path,
        "preprocessor_metadata": preprocessor.metadata(),
        "metadata_path": metadata_path,
        "metrics": metrics,
        "inertia": float(estimator.inertia_) if algorithm_name == "kmeans" else None,
        "iterations": int(estimator.n_iter_),
        "training_seconds": float(preprocessor_fit_seconds + estimator_fit_seconds),
        "estimator_fit_seconds": estimator_fit_seconds,
        "memory_before": memory.before_bytes,
        "memory_peak": memory.peak_bytes,
        "memory_after": memory.after_bytes,
        "gmm_converged": gmm_converged,
        "reload_predictions_identical": reload_predictions_identical,
        "warnings": warnings_seen,
    }


def _make_p2_ledger_row(
    *,
    root: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    cohort_metadata: dict[str, Any],
    variant_name: str,
    seed: int,
    k: int,
    data_hashes: dict[str, str],
) -> dict[str, Any]:
    metrics = result["metrics"]
    algorithm = result["algorithm"]
    metric_space = result["metric_space"]
    training_ids = result["training_ids"]
    evaluation_ids = result["evaluation_ids"]
    parameters = {
        "candidate_family": "p2_leakage_safe",
        "cohort": result["cohort"],
        "minimum_transactions": cohort_metadata["minimum_transactions"],
        "preprocessing_variant": variant_name,
        "metric_space": metric_space,
        "input_feature_order": list(config["data"]["feature_order"]),
        "as_of_date": config["data"]["as_of_date"],
        "training_manifest": cohort_metadata["training"][str(seed)]["manifest_path"],
        "evaluation_manifest": cohort_metadata["evaluation"]["manifest_path"],
        "preprocessor_path": _relative(result["preprocessor_path"], root),
        "preprocessor_fit_customer_id_sha256": result["preprocessor_metadata"][
            "fit_customer_id_sha256"
        ],
        "clip_bounds": result["preprocessor_metadata"].get("clip_bounds"),
        "clip_training_changed_rows": result["preprocessor_metadata"].get(
            "clip_training_changed_rows", 0
        ),
        "clip_evaluation_changed_rows": result["metadata_payload"][
            "clip_evaluation_changed_rows"
        ],
        "model_parameters": config["models"][algorithm],
        "random_state": seed,
        "k": k,
        "raw_input_sha256": data_hashes["raw_transactions_sha256"],
        "interpretable_input_sha256": data_hashes["interpretable_sha256"],
        "historical_p1_scaler_sha256_for_audit_only": data_hashes["p1_scaler_sha256"],
        "stability_pairwise_ari": None,
        "stability_ari_min": None,
    }
    row = {field: None for field in BOARD_FIELDS}
    row.update(
        {
            "run_id": result["candidate_id"],
            "phase": "task08_p2_leakage_safe",
            "algorithm": "kmeans_p2" if algorithm == "kmeans" else "gaussian_mixture_p2",
            "training_rows": len(training_ids),
            "evaluation_rows": len(evaluation_ids),
            "k": k,
            "seed": seed,
            "train_customer_id_sha256": ordered_id_sha256(training_ids),
            "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_ids),
            # These P2 runs use raw interpretable aggregates, not the full-fit P1 model matrix.
            "input_model_matrix_sha256": None,
            "p1_preprocessor_sha256": None,
            "feature_order_json": _json(config["data"]["feature_order"]),
            "parameters_json": _json(parameters),
            "silhouette_score": metrics["silhouette_score"],
            "silhouette_sample_rows": metrics["silhouette_sample_rows"],
            "davies_bouldin_index": metrics["davies_bouldin_index"],
            "calinski_harabasz_index": metrics["calinski_harabasz_index"],
            "evaluation_segment_counts_json": _json(metrics["segment_counts"]),
            "evaluation_segment_percentages_json": _json(metrics["segment_percentages"]),
            "smallest_segment_pct": metrics["smallest_segment_pct"],
            "kmeans_inertia_train": result["inertia"],
            "gmm_bic_evaluation": metrics["gmm_bic_evaluation"],
            "gmm_aic_evaluation": metrics["gmm_aic_evaluation"],
            "gmm_converged": metrics["gmm_converged"],
            "iterations": result["iterations"],
            "training_seconds": result["training_seconds"],
            "working_set_before_bytes": result["memory_before"],
            "peak_working_set_during_fit_bytes": result["memory_peak"],
            "working_set_after_bytes": result["memory_after"],
            "reload_predictions_identical": result["reload_predictions_identical"],
            "metric_status": metrics["metric_status"],
            "run_status": "completed" if metrics["metric_status"] == "ok" else metrics["metric_status"],
            "model_path": _relative(result["model_path"], root),
            "assignment_path": _relative(result["assignment_path"], root),
            "run_metadata_path": _relative(result["metadata_path"], root),
            "mlflow_status": "pending",
            "error": metrics["metric_error"],
        }
    )
    return row


def _attach_stability(
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    metadata_items: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(
            (result["cohort"], result["variant"], result["algorithm"], int(result["k"])), []
        ).append(result)
    stability_report: dict[str, Any] = {}
    for group_key, group in grouped.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda item: int(item["seed"]))
        pairs: dict[str, float] = {}
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                ari = aligned_adjusted_rand_index(
                    left["evaluation_labels"],
                    left["evaluation_ids"],
                    right["evaluation_labels"],
                    right["evaluation_ids"],
                )
                pair_key = f"seed{left['seed']}_vs_seed{right['seed']}"
                pairs[pair_key] = ari
        mean_ari = float(np.mean(list(pairs.values())))
        min_ari = float(np.min(list(pairs.values())))
        stable_key = "|".join(map(str, group_key))
        stability_report[stable_key] = {
            "pairwise_ari": pairs,
            "mean_pairwise_ari": mean_ari,
            "minimum_pairwise_ari": min_ari,
        }
        for result in group:
            result["stability_pairwise_ari"] = pairs
            result["stability_ari_mean"] = mean_ari
            result["stability_ari_min"] = min_ari
            for row in result["rows"]:
                row["stability_ari_across_seeds"] = mean_ari
                parameters = json.loads(row["parameters_json"])
                parameters["stability_pairwise_ari"] = pairs
                parameters["stability_ari_mean"] = mean_ari
                parameters["stability_ari_min"] = min_ari
                row["parameters_json"] = _json(parameters)
            result["metadata_payload"]["stability_pairwise_ari"] = pairs
            result["metadata_payload"]["stability_ari_mean"] = mean_ari
            result["metadata_payload"]["stability_ari_min"] = min_ari
    return stability_report


def _failed_p2_row(
    *,
    root: Path,
    config: dict[str, Any],
    run_id: str,
    cohort: dict[str, Any],
    variant_name: str,
    algorithm: str,
    seed: int,
    k: int,
    training_ids: np.ndarray,
    evaluation_ids: np.ndarray,
    error: str,
    data_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_space = f"p2_{cohort['name']}_{variant_name}_trainfit_scaled_9d"
    metadata_path = root / config["execution"]["run_metadata_directory"] / f"{run_id}.json"
    row = {field: None for field in BOARD_FIELDS}
    row.update(
        {
            "run_id": run_id,
            "phase": "task08_p2_leakage_safe",
            "algorithm": f"{algorithm}_p2",
            "training_rows": len(training_ids),
            "evaluation_rows": len(evaluation_ids),
            "k": k,
            "seed": seed,
            "train_customer_id_sha256": ordered_id_sha256(training_ids),
            "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_ids),
            "feature_order_json": _json(config["data"]["feature_order"]),
            "parameters_json": _json(
                {
                    "candidate_family": "p2_leakage_safe",
                    "cohort": cohort["name"],
                    "minimum_transactions": cohort["minimum_transactions"],
                    "preprocessing_variant": variant_name,
                    "metric_space": metric_space,
                    "error": error,
                }
            ),
            "metric_status": "failed",
            "run_status": "failed",
            "run_metadata_path": _relative(metadata_path, root),
            "mlflow_status": "not_attempted",
            "error": error,
        }
    )
    metadata = {
        "run_id": run_id,
        "cohort": cohort["name"],
        "preprocessing_variant": variant_name,
        "algorithm": algorithm,
        "seed": seed,
        "k": k,
        "metric_space": metric_space,
        "training_customer_id_sha256": ordered_id_sha256(training_ids),
        "evaluation_customer_id_sha256": ordered_id_sha256(evaluation_ids),
        "raw_input_sha256": data_hashes["raw_transactions_sha256"],
        "error": error,
        "versions": _versions(),
        "config": config,
    }
    return row, {"path": metadata_path, "payload": metadata}


def _persist_p2_records(
    root: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    metadata_items: list[dict[str, Any]],
) -> None:
    metadata_by_path = {item["path"]: item["payload"] for item in metadata_items}
    for row in rows:
        metadata_path = root / row["run_metadata_path"]
        metadata = metadata_by_path[metadata_path]
        if row["run_status"] == "failed":
            mlflow_status = "not_attempted"
        else:
            mlflow_status = _mlflow_log(
                experiment_name=str(config["execution"]["mlflow_experiment"]),
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
                        "gmm_aic_evaluation",
                        "gmm_bic_evaluation",
                    )
                    if row.get(key) is not None
                },
                model_path=root / row["model_path"],
                assignment_path=root / row["assignment_path"],
            )
        row["mlflow_status"] = mlflow_status
        metadata["mlflow_status"] = mlflow_status
    for item in metadata_items:
        _atomic_write(item["path"], (_json(item["payload"]) + "\n").encode("utf-8"))
    if rows:
        _persist_comparison_rows(root / config["execution"]["comparison_board"], rows)


def run_p2_experiments(config_path: str | Path) -> int:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("track") != "track_a":
        raise ValueError("P2 experiment config must describe Track A.")
    root = config_path.parent.parent.parent
    data = config["data"]
    features = list(data["feature_order"])
    as_of_date = pd.Timestamp(data["as_of_date"])
    preflight = run_preflight(
        root / data["feature_contract_file"], selected_tracks=["track_a"]
    )
    if not preflight["ready"]:
        raise ValueError(f"Track A preflight failed: {preflight['tracks']['track_a']['errors']}")
    if int(preflight["tracks"]["track_a"]["rows"]["model"]) != int(data["row_count"]):
        raise ValueError("Track A customer row count differs from the P2 config.")
    with (root / data["feature_contract_file"]).open(encoding="utf-8") as stream:
        feature_contract = yaml.safe_load(stream)["tracks"]["track_a"]
    if list(feature_contract["model_features"]) != features:
        raise ValueError("P2 feature order must match the approved Track A feature contract.")

    population, transaction_count_frequencies = _population_eligibility(
        root / data["interpretable_file"],
        identifier_column=str(data["identifier_column"]),
        minimum_transactions=1,
    )
    if len(population) != int(data["row_count"]):
        raise ValueError("Interpretable population row count differs from the validated P1 population.")
    run_id = _run_id("task08_track_a_p2")
    cohort_manifests = _make_population_manifests(
        root, run_id=run_id, config=config, population=population
    )

    wanted_ids: set[str] = set()
    for cohort in cohort_manifests.values():
        wanted_ids.update(cohort["evaluation"]["customer_ids"].astype(str))
        for sample in cohort["training"].values():
            wanted_ids.update(sample["customer_ids"].astype(str))
    raw_aggregates, raw_stats = _aggregate_raw_customer_features(
        root / data["raw_transactions_file"],
        wanted_ids=wanted_ids,
        chunk_rows=int(data["chunk_rows"]),
    )
    p1_global_max_date = pd.Timestamp(raw_stats["p1_global_max_date"])
    raw_stats["p1_max_date_iso"] = p1_global_max_date.date().isoformat()
    if raw_stats["p1_max_date_iso"] != str(data["max_raw_transaction_date"]):
        raise ValueError(
            f"Raw max date changed: config={data['max_raw_transaction_date']}, "
            f"observed={raw_stats['p1_max_date_iso']}."
        )
    if as_of_date < p1_global_max_date:
        raise ValueError("Configured P2 as-of date must be after the complete-case source maximum date.")
    interpreted_selected = _load_interpretable_customer_features(
        root / data["interpretable_file"],
        identifier_column=str(data["identifier_column"]),
        features=features,
        wanted_ids=wanted_ids,
        chunk_rows=int(data["chunk_rows"]),
    )
    aggregate_validation = _verify_p1_aggregation_semantics(
        raw_aggregates,
        interpreted_selected,
        feature_order=features,
        p1_global_max_date=p1_global_max_date,
        as_of_date=as_of_date,
    )
    raw_stats["p1_global_max_date"] = raw_stats["p1_max_date_iso"]

    raw_hash = _sha256_file(root / data["raw_transactions_file"])
    interpreted_hash = _sha256_file(root / data["interpretable_file"])
    p1_scaler_hash = _sha256_file(root / data["preprocessor_artifact_reference_only"])
    hashes = {
        "raw_transactions_sha256": raw_hash,
        "interpretable_sha256": interpreted_hash,
        "p1_scaler_sha256": p1_scaler_hash,
    }
    versions = _versions()
    versions["scipy"] = __import__("scipy").__version__
    versions["threadpoolctl"] = __import__("threadpoolctl").__version__

    p2_model_directory = root / config["execution"]["model_directory"] / run_id
    p2_model_directory.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    cohort_run_summaries: dict[str, Any] = {}
    aggregate_by_id = raw_aggregates

    for cohort_name, cohort in cohort_manifests.items():
        eval_ids = cohort["evaluation"]["customer_ids"]
        evaluation_frame = aggregate_by_id.loc[eval_ids].reset_index()
        cohort_candidate_count = 0
        for seed_key, manifest in cohort["training"].items():
            seed = int(seed_key)
            train_ids = manifest["customer_ids"]
            training_frame = aggregate_by_id.loc[train_ids].reset_index()
            validate_disjoint_customer_ids(train_ids, eval_ids)
            for variant_name, variant_config in config["preprocessing_variants"].items():
                clip_quantiles = (
                    tuple(float(value) for value in variant_config["clip_quantiles"])
                    if variant_config.get("clip_quantiles") is not None
                    else None
                )
                preprocessor = P2TrainFittedPreprocessor(
                    feature_order=features,
                    log1p_features=list(data["log1p_features"]),
                    as_of_date=str(data["as_of_date"]),
                    clip_feature=str(variant_config.get("clip_feature"))
                    if variant_config.get("clip_feature")
                    else None,
                    clip_quantiles=clip_quantiles,
                )
                prep_started = time.perf_counter()
                preprocessor.fit(training_frame)
                preprocessor_fit_seconds = time.perf_counter() - prep_started
                prep_id = f"{run_id}_{cohort_name}_{variant_name}_seed{seed}_preprocessor"
                preprocessor_path = p2_model_directory / f"{prep_id}.joblib"
                _save_joblib_new(
                    preprocessor_path,
                    {
                        "preprocessor": preprocessor,
                        "metadata": {
                            **preprocessor.metadata(),
                            "cohort": cohort_name,
                            "minimum_transactions": cohort["minimum_transactions"],
                            "variant": variant_name,
                            "training_manifest": manifest["manifest_path"],
                            "evaluation_manifest": cohort["evaluation"]["manifest_path"],
                            "evaluation_customer_id_sha256": cohort["evaluation"]["customer_id_sha256"],
                            "preprocessor_fit_seconds": preprocessor_fit_seconds,
                            "versions": versions,
                            "raw_aggregate_validation": aggregate_validation,
                        },
                    },
                )
                for algorithm_name in ("kmeans", "gaussian_mixture"):
                    for k in range(2, 6):
                        try:
                            result = _fit_one_candidate(
                                root=root,
                                config=config,
                                top_run_id=run_id,
                                cohort_name=cohort_name,
                                minimum_transactions=int(cohort["minimum_transactions"]),
                                variant_name=variant_name,
                                preprocessor=preprocessor,
                                preprocessor_path=preprocessor_path,
                                preprocessor_fit_seconds=preprocessor_fit_seconds,
                                training_frame=training_frame,
                                evaluation_frame=evaluation_frame,
                                seed=seed,
                                k=k,
                                algorithm_name=algorithm_name,
                                raw_input_hash=raw_hash,
                                aggregate_validation=aggregate_validation,
                                cohort_population_rows=int(cohort["eligible_customers"]),
                            )
                            row = _make_p2_ledger_row(
                                root=root,
                                config=config,
                                result=result,
                                cohort_metadata=cohort,
                                variant_name=variant_name,
                                seed=seed,
                                k=k,
                                data_hashes=hashes,
                            )
                            result["row"] = row
                            result["rows"] = [row]
                            results.append(result)
                            rows.append(row)
                            metadata_items.append(result["row_metadata"])
                            cohort_candidate_count += 1
                        except Exception as exc:  # noqa: BLE001 - retain candidate failure evidence
                            failed_run_id = f"{run_id}_{cohort_name}_{variant_name}_{algorithm_name}_seed{seed}_k{k}_failed"
                            row, metadata_item = _failed_p2_row(
                                root=root,
                                config=config,
                                run_id=failed_run_id,
                                cohort=cohort,
                                variant_name=variant_name,
                                algorithm=algorithm_name,
                                seed=seed,
                                k=k,
                                training_ids=train_ids,
                                evaluation_ids=eval_ids,
                                error=f"{type(exc).__name__}: {exc}",
                                data_hashes=hashes,
                            )
                            rows.append(row)
                            metadata_items.append(metadata_item)
        cohort_run_summaries[cohort_name] = {
            "eligible_customers": int(cohort["eligible_customers"]),
            "full_population_coverage_percent": float(cohort["full_population_coverage_percent"]),
            "training_rows_by_seed": {
                seed: len(manifest["customer_ids"])
                for seed, manifest in cohort["training"].items()
            },
            "evaluation_rows": len(cohort["evaluation"]["customer_ids"]),
            "evaluation_customer_id_sha256": cohort["evaluation"]["customer_id_sha256"],
            "successful_candidate_fits": cohort_candidate_count,
        }

    stability_summary = _attach_stability(results, rows, metadata_items)
    board_path = root / config["execution"]["comparison_board"]
    _persist_p2_records(root, config, rows, metadata_items)

    run_summary = {
        "run_id": run_id,
        "track": "track_a",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "versions": versions,
        "device": "cpu",
        "as_of_date": str(data["as_of_date"]),
        "raw_transaction_audit": raw_stats,
        "p1_aggregate_semantics_verification": aggregate_validation,
        "preprocessing_leakage_audit": {
            "historical_p1_scaler_fit_rows": int(
                joblib.load(root / data["preprocessor_artifact_reference_only"])["scaler"].n_samples_seen_
            ),
            "historical_p1_scaler_fit_scope": "entire processed customer population before split",
            "new_p2_scalers_fit_scope": "eligible training IDs only for each cohort/seed/variant",
            "p2_recency_source": "raw customer Last_Transaction and fixed configured as_of_date",
        },
        "cohorts": cohort_run_summaries,
        "transaction_count_frequencies": {
            str(count): int(number)
            for count, number in sorted(transaction_count_frequencies.items())
        },
        "preprocessing_variants": config["preprocessing_variants"],
        "stability_pairwise_ari": stability_summary,
        "successful_rows": sum(row["run_status"] == "completed" for row in rows),
        "failed_rows": sum(row["run_status"] == "failed" for row in rows),
        "ledger_rows_appended": len(rows),
        "comparison_board": _relative(board_path, root),
        "model_directory": _relative(p2_model_directory, root),
        "config": config,
    }
    summary_path = root / config["execution"]["diagnostic_root"] / run_id / "run_summary.json"
    _atomic_write(summary_path, (_json(run_summary) + "\n").encode("utf-8"))
    print(
        f"Task 08 run {run_id}: rows={len(rows)} completed={run_summary['successful_rows']} "
        f"failed={run_summary['failed_rows']} cohorts={cohort_run_summaries}"
    )
    print(f"Comparison board: {board_path}")
    print(f"P2 artifacts: {p2_model_directory}")
    print(f"P2 diagnostics: {summary_path.parent}")
    return 0 if run_summary["failed_rows"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/models/track_a/track_a_p2_leakage_safe.yaml",
        help="Leakage-safe Track A P2 experiment YAML.",
    )
    args = parser.parse_args(argv)
    return run_p2_experiments(args.config)


def _make_population_manifests(
    root: Path,
    *,
    run_id: str,
    config: dict[str, Any],
    population: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    manifest_root = root / config["execution"]["sample_directory"] / run_id
    manifest_root.mkdir(parents=True, exist_ok=False)
    population_size = int(config["data"]["row_count"])
    for cohort_name, settings in config["sampling"]["cohorts"].items():
        minimum = int(settings["minimum_transactions"])
        eligible = population.loc[population["Transaction_Count"] >= minimum]
        eligible_positions = eligible["row_position"].to_numpy(dtype=np.int64)
        train_size = int(settings["training_rows"])
        eval_size = int(settings["evaluation_rows"])
        if train_size + eval_size > len(eligible_positions):
            raise ValueError(
                f"Cohort {cohort_name} has {len(eligible_positions)} IDs; requested "
                f"{train_size} train + {eval_size} evaluation."
            )
        eval_positions = np.sort(
            np.random.default_rng(
                int(config["sampling"]["evaluation_seed"]) + minimum
            ).choice(eligible_positions, size=eval_size, replace=False)
        )
        remaining = eligible_positions[~np.isin(eligible_positions, eval_positions)]
        evaluation = population.set_index("row_position").loc[eval_positions].reset_index()
        eval_path = manifest_root / f"{cohort_name}_evaluation_n{eval_size}.csv"
        _save_frame(
            eval_path,
            evaluation[["row_position", "CustomerID", "Transaction_Count"]],
        )
        eval_ids = evaluation["CustomerID"].astype("string").to_numpy(dtype=str)
        cohort_result: dict[str, Any] = {
            "name": cohort_name,
            "minimum_transactions": minimum,
            "eligible_customers": len(eligible_positions),
            "full_population_coverage_percent": 100.0 * len(eligible_positions) / population_size,
            "training_rows_requested": train_size,
            "evaluation_rows": eval_size,
            "evaluation_manifest_path": _relative(eval_path, root),
            "evaluation_row_positions": eval_positions,
            "evaluation_customer_ids": eval_ids,
            "evaluation_customer_id_sha256": ordered_id_sha256(eval_ids),
            "evaluation": {
                "manifest_path": _relative(eval_path, root),
                "row_positions": eval_positions,
                "customer_ids": eval_ids,
                "customer_id_sha256": ordered_id_sha256(eval_ids),
            },
            "training": {},
        }
        for seed in [int(value) for value in settings["seeds"]]:
            train_positions = np.sort(
                np.random.default_rng(seed + 100_003 * minimum).choice(
                    remaining, size=train_size, replace=False
                )
            )
            if np.intersect1d(train_positions, eval_positions).size:
                raise RuntimeError(f"Train/evaluation split overlap in cohort {cohort_name}.")
            training = population.set_index("row_position").loc[train_positions].reset_index()
            train_path = manifest_root / f"{cohort_name}_train_seed{seed}_n{train_size}.csv"
            _save_frame(
                train_path,
                training[["row_position", "CustomerID", "Transaction_Count"]],
            )
            train_ids = training["CustomerID"].astype("string").to_numpy(dtype=str)
            cohort_result["training"][str(seed)] = {
                "manifest_path": _relative(train_path, root),
                "row_positions": train_positions,
                "customer_ids": train_ids,
                "customer_id_sha256": ordered_id_sha256(train_ids),
            }
        result[cohort_name] = cohort_result
    return result


def _validate_population_aggregate(
    raw_aggregate: pd.DataFrame,
    interpreted: pd.DataFrame,
    *,
    feature_order: list[str],
    raw_stats: dict[str, Any],
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    if len(raw_aggregate) != len(interpreted):
        raise ValueError("Raw aggregate and interpreted sample row counts differ.")
    missing = set(interpreted.index.astype(str)) - set(raw_aggregate.index.astype(str))
    if missing:
        raise ValueError(f"Raw aggregate is missing {len(missing)} selected CustomerIDs.")
    raw_aggregate = raw_aggregate.loc[interpreted.index]
    max_abs_differences: dict[str, float] = {}
    for feature in feature_order:
        if feature == "Recency_Days":
            expected = (raw_stats["p1_global_max_date"] - raw_aggregate["Last_Transaction"]).dt.days
        else:
            expected = raw_aggregate[feature]
        observed = interpreted[feature]
        differences = np.abs(
            pd.to_numeric(expected, errors="raise").to_numpy(dtype=np.float64)
            - pd.to_numeric(observed, errors="raise").to_numpy(dtype=np.float64)
        )
        maximum = float(differences.max(initial=0.0))
        max_abs_differences[feature] = maximum
        if feature == "Transaction_Count":
            if np.any(differences != 0):
                raise ValueError("Raw complete-case transaction counts do not match P1 interpretable features.")
        elif maximum > 1e-5:
            raise ValueError(f"Raw aggregate/P1 interpreted mismatch for {feature}: {maximum}.")
    if raw_stats["p1_global_max_date"] > as_of_date:
        raise ValueError("Configured as_of_date precedes the full raw source maximum date.")
    if raw_aggregate["Last_Transaction"].max() > as_of_date:
        raise ValueError("A sampled customer has a last transaction after configured as_of_date.")
    return {
        "sampled_customers_checked": len(raw_aggregate),
        "aggregate_match_max_absolute_difference": max_abs_differences,
        "transaction_count_exact_match": True,
        "p1_global_max_date": raw_stats["p1_global_max_date"].date().isoformat(),
        "configured_p2_as_of_date": as_of_date.date().isoformat(),
        "selected_last_transaction_max": raw_aggregate["Last_Transaction"].max().date().isoformat(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
