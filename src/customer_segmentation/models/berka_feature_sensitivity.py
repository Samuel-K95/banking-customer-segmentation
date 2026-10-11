"""Feature-ablation sensitivity experiments for the Berka k=2 baselines."""

from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from customer_segmentation.models.berka_behavioral_baselines import (
    BerkaTrainFittedPreprocessor,
    _sha256_ids,
    load_snapshot,
    select_behavioral_features,
)

DEFAULT_CONFIG = Path("configs/models/berka_feature_sensitivity_v1.yaml")
PROFILE_FEATURES = [
    "account_tenure_days",
    "balance_median",
    "balance_std",
    "credit_amount_total",
    "debit_amount_total",
    "net_cash_flow",
    "credit_frequency_per_day",
    "debit_frequency_per_day",
    "transaction_amount_median",
    "negative_balance_frequency",
    "interarrival_days_median",
    "interarrival_days_std",
    "recent_90d_transaction_frequency_per_day",
    "activity_change_count",
    "gross_money_movement",
]


def make_feature_configurations(
    base_feature_order: list[str], exclusions: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Apply declared ablations while retaining the original baseline order."""
    configurations: dict[str, list[str]] = {}
    for name, excluded in exclusions.items():
        unknown = set(excluded).difference(base_feature_order)
        if unknown:
            raise ValueError(f"Configuration {name} excludes unknown features: {sorted(unknown)}")
        selected = [feature for feature in base_feature_order if feature not in set(excluded)]
        if len(selected) < 2:
            raise ValueError(f"Configuration {name} has fewer than two features")
        configurations[name] = selected
    return configurations


def _preprocessor_for_features(
    baseline_features: dict[str, Any], feature_order: list[str]
) -> BerkaTrainFittedPreprocessor:
    selected = set(feature_order)
    return BerkaTrainFittedPreprocessor(
        feature_order,
        positive_log1p=[name for name in baseline_features["positive_log1p"] if name in selected],
        signed_asinh=[name for name in baseline_features["signed_asinh"] if name in selected],
        bounded_unchanged=[name for name in baseline_features["bounded_unchanged"] if name in selected],
        clipping_quantiles=tuple(float(value) for value in baseline_features["clipping_quantiles"]),
    )


def align_to_gross_movement(labels: np.ndarray, gross_movement: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """Give profile labels a consistent low/high gross-movement orientation.

    This changes label names for interpretation only; stored model labels and
    ARI calculations remain permutation-invariant raw predictions.
    """
    labels = np.asarray(labels)
    gross_movement = np.asarray(gross_movement, dtype=float)
    if labels.ndim != 1 or gross_movement.shape != labels.shape or not np.isfinite(gross_movement).all():
        raise ValueError("Labels and finite gross movement must be aligned one-dimensional arrays")
    clusters = np.unique(labels)
    if len(clusters) != 2:
        raise ValueError("Gross-movement orientation requires exactly two observed clusters")
    ordered = sorted(clusters, key=lambda cluster: float(np.median(gross_movement[labels == cluster])))
    mapping = {ordered[0]: "lower_gross_movement", ordered[1]: "higher_gross_movement"}
    profile_codes = np.asarray([0 if mapping[value] == "lower_gross_movement" else 1 for value in labels])
    return profile_codes, {int(label): name for label, name in mapping.items()}


def compare_held_out_assignments(
    labels_left: np.ndarray,
    ids_left: np.ndarray,
    labels_right: np.ndarray,
    ids_right: np.ndarray,
) -> float:
    """Compute ARI only when both assignments cover identical ordered accounts."""
    ids_left = np.asarray(ids_left, dtype=str)
    ids_right = np.asarray(ids_right, dtype=str)
    labels_left = np.asarray(labels_left)
    labels_right = np.asarray(labels_right)
    if not np.array_equal(ids_left, ids_right):
        raise ValueError("Assignment comparison requires identical ordered held-out account IDs")
    if labels_left.shape != ids_left.shape or labels_right.shape != ids_right.shape:
        raise ValueError("Assignment labels must align one-to-one with held-out account IDs")
    return float(adjusted_rand_score(labels_left, labels_right))


def _metrics(matrix: np.ndarray, labels: np.ndarray, *, tiny_threshold_percent: float) -> dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    counts_by_label = {str(int(label)): int(count) for label, count in zip(unique, counts)}
    proportions = {key: 100.0 * value / len(labels) for key, value in counts_by_label.items()}
    return {
        "held_out_rows": len(labels),
        "observed_clusters": len(unique),
        "cluster_counts": counts_by_label,
        "cluster_percentages": proportions,
        "minimum_segment_pct": min(proportions.values()),
        "tiny_cluster": min(proportions.values()) < tiny_threshold_percent,
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
    }


def _profiles(
    holdout: pd.DataFrame,
    labels: np.ndarray,
    *,
    configuration: str,
    algorithm: str,
    seed: int,
) -> list[dict[str, Any]]:
    raw_gross = (holdout["credit_amount_total"] + holdout["debit_amount_total"]).to_numpy(dtype=float)
    codes, mapping = align_to_gross_movement(labels, raw_gross)
    frame = holdout.copy()
    frame["gross_money_movement"] = raw_gross
    frame["profile_cluster_code"] = codes
    frame["profile_cluster"] = np.where(codes == 0, "lower_gross_movement", "higher_gross_movement")
    output: list[dict[str, Any]] = []
    for code, members in frame.groupby("profile_cluster_code", sort=True):
        raw_label = next(label for label, profile_name in mapping.items() if profile_name == members["profile_cluster"].iloc[0])
        row: dict[str, Any] = {
            "configuration": configuration,
            "algorithm": algorithm,
            "seed": seed,
            "profile_cluster": members["profile_cluster"].iloc[0],
            "raw_model_cluster": int(raw_label),
            "account_count": len(members),
            "segment_pct": 100.0 * len(members) / len(frame),
        }
        row.update({f"median_{column}": float(members[column].median()) for column in PROFILE_FEATURES})
        output.append(row)
    return output


def _save_new_joblib(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Berka sensitivity artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def _load_fixed_population(
    *, baseline_config: dict[str, Any], baseline_directory: Path, snapshot_directory: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_metadata_path = baseline_directory / "run_metadata.json"
    split_path = baseline_directory / "account_split.csv"
    baseline_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    snapshot, snapshot_metadata = load_snapshot(
        snapshot_directory, expected_as_of_date=baseline_config["data"]["as_of_date"]
    )
    if snapshot_metadata["as_of_date"] != "1997-12-31":
        raise ValueError("Berka feature sensitivity is restricted to the 1997-12-31 snapshot")
    eligible = snapshot.loc[
        snapshot["account_tenure_days"].ge(int(baseline_config["data"]["minimum_tenure_days"]))
    ].copy()
    split = pd.read_csv(split_path)
    if split["account_id"].duplicated().any() or split["account_id"].isna().any():
        raise ValueError("Saved baseline split must have unique, non-null account IDs")
    if len(eligible) != int(baseline_metadata["eligible_accounts"]):
        raise ValueError("Eligible population size differs from the baseline run")
    if set(split["account_id"]) != set(eligible["account_id"]):
        raise ValueError("Saved split account set differs from the eligible population")
    training_ids = split.loc[split["split"].eq("train"), "account_id"].astype(str).to_numpy()
    held_out_ids = split.loc[split["split"].eq("held_out"), "account_id"].astype(str).to_numpy()
    if _sha256_ids(training_ids) != baseline_metadata["training_account_ids_sha256"]:
        raise ValueError("Saved training split order/hash differs from baseline metadata")
    if _sha256_ids(held_out_ids) != baseline_metadata["held_out_account_ids_sha256"]:
        raise ValueError("Saved held-out split order/hash differs from baseline metadata")
    by_id = eligible.set_index("account_id")
    if not by_id.index.is_unique:
        raise ValueError("Eligible population account IDs must be unique")
    train = by_id.loc[split.loc[split["split"].eq("train"), "account_id"].to_numpy()].reset_index()
    holdout = by_id.loc[split.loc[split["split"].eq("held_out"), "account_id"].to_numpy()].reset_index()
    return train, holdout, snapshot, baseline_metadata


def run_feature_sensitivity(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline_config_path = Path(config["baseline"]["config"])
    baseline_config = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    baseline_directory = Path(config["baseline"]["artifacts"])
    snapshot_directory = Path(config["data"]["snapshot_directory"])
    output_root = Path(output_directory or config["data"]["output_directory"])
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Berka sensitivity directory: {output_root}")

    base_features = list(baseline_config["features"]["selected"])
    configurations = make_feature_configurations(base_features, config["features"]["exclusions"])
    if next(iter(configurations)) != config["features"]["original"]:
        raise ValueError("First sensitivity configuration must be the original baseline feature set")
    train, holdout, snapshot, _baseline_metadata = _load_fixed_population(
        baseline_config=baseline_config,
        baseline_directory=baseline_directory,
        snapshot_directory=snapshot_directory,
    )
    expected_counts = {
        "eligible": int(config["data"]["expected_eligible_accounts"]),
        "training": int(config["data"]["expected_training_accounts"]),
        "held_out": int(config["data"]["expected_held_out_accounts"]),
    }
    if (len(train), len(holdout)) != (expected_counts["training"], expected_counts["held_out"]):
        raise ValueError("Saved baseline split dimensions do not match the sensitivity configuration")
    if len(snapshot.loc[snapshot["account_tenure_days"].ge(365)]) != expected_counts["eligible"]:
        raise ValueError("Eligible account count differs from the sensitivity configuration")

    baseline_features_config = baseline_config["features"]
    model_config = config["models"]
    seeds = [int(seed) for seed in model_config["seeds"]]
    algorithms = list(model_config["algorithms"])
    k = int(model_config["k"])
    tiny_threshold = float(config["evaluation"]["tiny_cluster_threshold_percent"])
    train_ids = train["account_id"].astype(str).to_numpy()
    held_out_ids = holdout["account_id"].astype(str).to_numpy()

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_root / "feature_configurations.json").write_text(
        json.dumps(configurations, indent=2) + "\n", encoding="utf-8"
    )
    split_copy_path = output_root / "account_split.csv"
    pd.concat(
        [train[["account_id"]].assign(split="train"), holdout[["account_id"]].assign(split="held_out")],
        ignore_index=True,
    ).to_csv(split_copy_path, index=False)

    candidate_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    assignment_parts: list[pd.DataFrame] = []
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    preprocessors: dict[str, BerkaTrainFittedPreprocessor] = {}
    output_models = output_root / "models"
    output_models.mkdir()

    for configuration, feature_order in configurations.items():
        train_selected = select_behavioral_features(train, base_features).loc[:, feature_order]
        holdout_selected = select_behavioral_features(holdout, base_features).loc[:, feature_order]
        preprocessor = _preprocessor_for_features(baseline_features_config, feature_order).fit(
            train_selected, train["account_id"].to_numpy()
        )
        train_matrix = preprocessor.transform(train_selected)
        holdout_matrix = preprocessor.transform(holdout_selected)
        preprocessors[configuration] = preprocessor
        preprocessor_path = output_root / f"preprocessor_{configuration}.joblib"
        _save_new_joblib(
            preprocessor_path,
            {"preprocessor_state": preprocessor.state_dict(), "metadata": preprocessor.metadata()},
        )
        config_model_dir = output_models / configuration
        config_model_dir.mkdir()

        for algorithm in algorithms:
            for seed in seeds:
                params = model_config[algorithm]
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as warning_records:
                    warnings.simplefilter("always")
                    if algorithm == "kmeans":
                        with threadpool_limits(limits=1):
                            estimator = KMeans(
                                n_clusters=k,
                                random_state=seed,
                                n_init=int(params["n_init"]),
                                max_iter=int(params["max_iter"]),
                                algorithm=str(params["algorithm"]),
                            ).fit(train_matrix)
                        labels = estimator.predict(holdout_matrix)
                        converged = True
                        iterations = int(estimator.n_iter_)
                        bic = aic = None
                    elif algorithm == "gaussian_mixture":
                        with threadpool_limits(limits=1):
                            estimator = GaussianMixture(
                                n_components=k,
                                covariance_type=str(params["covariance_type"]),
                                n_init=int(params["n_init"]),
                                max_iter=int(params["max_iter"]),
                                tol=float(params["tolerance"]),
                                reg_covar=float(params["regularization"]),
                                random_state=seed,
                            ).fit(train_matrix)
                        labels = estimator.predict(holdout_matrix)
                        converged = bool(estimator.converged_)
                        iterations = int(estimator.n_iter_)
                        bic = float(estimator.bic(holdout_matrix))
                        aic = float(estimator.aic(holdout_matrix))
                    else:
                        raise ValueError(f"Unsupported algorithm: {algorithm}")
                training_seconds = time.perf_counter() - started
                metrics = _metrics(holdout_matrix, labels, tiny_threshold_percent=tiny_threshold)
                metrics.update(
                    {
                        "configuration": configuration,
                        "algorithm": algorithm,
                        "k": k,
                        "seed": seed,
                        "feature_count": len(feature_order),
                        "feature_order": json.dumps(feature_order),
                        "training_rows": len(train),
                        "training_seconds": training_seconds,
                        "converged": converged,
                        "iterations": iterations,
                        "held_out_bic": bic,
                        "held_out_aic": aic,
                        "warnings": " | ".join(
                            f"{warning.category.__name__}: {warning.message}" for warning in warning_records
                        ),
                    }
                )
                candidate_rows.append(metrics)
                predictions[(configuration, algorithm, seed)] = labels
                profile_rows.extend(
                    _profiles(holdout, labels, configuration=configuration, algorithm=algorithm, seed=seed)
                )
                assignment = pd.DataFrame(
                    {
                        "account_id": holdout["account_id"].to_numpy(),
                        "configuration": configuration,
                        "algorithm": algorithm,
                        "seed": seed,
                        "raw_cluster": labels.astype(int),
                    }
                )
                if algorithm == "gaussian_mixture":
                    assignment["max_component_probability"] = estimator.predict_proba(holdout_matrix).max(axis=1)
                assignment_parts.append(assignment)
                model_path = config_model_dir / f"{algorithm}_k{k}_seed{seed}.joblib"
                model_metadata = {
                    "experiment": config["experiment"],
                    "configuration": configuration,
                    "algorithm": algorithm,
                    "k": k,
                    "seed": seed,
                    "as_of_date": config["data"]["as_of_date"],
                    "feature_order": feature_order,
                    "training_account_ids_sha256": _sha256_ids(train_ids),
                    "held_out_account_ids_sha256": _sha256_ids(held_out_ids),
                    "preprocessor_path": str(preprocessor_path),
                    "fit_scope": "baseline training account IDs only",
                }
                _save_new_joblib(model_path, {"estimator": estimator, "metadata": model_metadata})

    candidate_frame = pd.DataFrame(candidate_rows)
    stability_rows: list[dict[str, Any]] = []
    for configuration in configurations:
        for algorithm in algorithms:
            pair_scores = [
                compare_held_out_assignments(
                    predictions[(configuration, algorithm, left)],
                    held_out_ids,
                    predictions[(configuration, algorithm, right)],
                    held_out_ids,
                )
                for left, right in itertools.combinations(seeds, 2)
            ]
            stability_rows.append(
                {
                    "configuration": configuration,
                    "algorithm": algorithm,
                    "pairwise_seed_ari_mean": float(np.mean(pair_scores)),
                    "pairwise_seed_ari_min": float(np.min(pair_scores)),
                    "pairwise_seed_ari_max": float(np.max(pair_scores)),
                }
            )
    stability_frame = pd.DataFrame(stability_rows)
    candidate_frame = candidate_frame.merge(stability_frame, on=["configuration", "algorithm"], validate="many_to_one")
    candidate_path = output_root / "candidate_metrics.csv"
    candidate_frame.to_csv(candidate_path, index=False)

    assignment_frame = pd.concat(assignment_parts, ignore_index=True)
    assignments_path = output_root / "held_out_assignments.csv"
    assignment_frame.to_csv(assignments_path, index=False)
    profiles_path = output_root / "held_out_segment_profiles.csv"
    pd.DataFrame(profile_rows).to_csv(profiles_path, index=False)

    comparison_rows: list[dict[str, Any]] = []
    comparison_summary_rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        for left_config, right_config in itertools.combinations(configurations, 2):
            scores: list[dict[str, Any]] = []
            for left_seed in seeds:
                for right_seed in seeds:
                    ari = compare_held_out_assignments(
                        predictions[(left_config, algorithm, left_seed)],
                        held_out_ids,
                        predictions[(right_config, algorithm, right_seed)],
                        held_out_ids,
                    )
                    record = {
                        "algorithm": algorithm,
                        "left_configuration": left_config,
                        "left_seed": left_seed,
                        "right_configuration": right_config,
                        "right_seed": right_seed,
                        "same_seed": left_seed == right_seed,
                        "adjusted_rand_index": ari,
                    }
                    comparison_rows.append(record)
                    scores.append(record)
            paired = [record["adjusted_rand_index"] for record in scores if record["same_seed"]]
            all_pairs = [record["adjusted_rand_index"] for record in scores]
            comparison_summary_rows.append(
                {
                    "algorithm": algorithm,
                    "left_configuration": left_config,
                    "right_configuration": right_config,
                    "paired_seed_ari_mean": float(np.mean(paired)),
                    "paired_seed_ari_min": float(np.min(paired)),
                    "paired_seed_ari_max": float(np.max(paired)),
                    "all_seed_pair_ari_mean": float(np.mean(all_pairs)),
                    "all_seed_pair_ari_min": float(np.min(all_pairs)),
                    "all_seed_pair_ari_max": float(np.max(all_pairs)),
                }
            )
    comparisons_path = output_root / "cross_configuration_assignment_ari.csv"
    pd.DataFrame(comparison_rows).to_csv(comparisons_path, index=False)
    comparison_summary_path = output_root / "cross_configuration_ari_summary.csv"
    pd.DataFrame(comparison_summary_rows).to_csv(comparison_summary_path, index=False)

    metadata = {
        "experiment": config["experiment"],
        "snapshot_directory": str(snapshot_directory),
        "snapshot_version": "berka_account_snapshot_v1",
        "as_of_date": config["data"]["as_of_date"],
        "eligible_accounts": len(snapshot.loc[snapshot["account_tenure_days"].ge(365)]),
        "training_accounts": len(train),
        "held_out_accounts": len(holdout),
        "training_account_ids_sha256": _sha256_ids(train_ids),
        "held_out_account_ids_sha256": _sha256_ids(held_out_ids),
        "split_seed": baseline_config["split"]["random_seed"],
        "baseline_split_metadata": str(baseline_directory / "run_metadata.json"),
        "feature_configurations": configurations,
        "algorithms": algorithms,
        "k": k,
        "seeds": seeds,
        "preprocessor_fit_scope": "training accounts only; separately fit per feature configuration",
        "1998_transactions": "not used",
        "outputs": {
            "candidate_metrics": str(candidate_path),
            "held_out_assignments": str(assignments_path),
            "profiles": str(profiles_path),
            "cross_configuration_ari": str(comparisons_path),
            "cross_configuration_ari_summary": str(comparison_summary_path),
        },
    }
    metadata_path = output_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "output_directory": output_root,
        "metrics": candidate_path,
        "assignments": assignments_path,
        "profiles": profiles_path,
        "comparisons": comparisons_path,
        "comparison_summary": comparison_summary_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Berka k=2 feature sensitivity experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=None)
    args = parser.parse_args()
    paths = run_feature_sensitivity(args.config, output_directory=args.output_directory)
    print(f"Completed Berka sensitivity analysis: {paths['output_directory']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
