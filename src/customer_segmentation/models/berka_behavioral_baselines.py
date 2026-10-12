"""Train-fitted Berka K-Means and diagonal-GMM baseline experiments."""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

DEFAULT_CONFIG = Path("configs/models/berka_behavioral_baselines_v1_1.yaml")
EXPECTED_FEATURES = [
    "account_tenure_days",
    "recency_days",
    "credit_frequency_per_day",
    "debit_frequency_per_day",
    "gross_money_movement",
    "net_cash_flow",
    "transaction_amount_median",
    "balance_std",
    "negative_balance_frequency",
    "interarrival_days_median",
    "interarrival_days_std",
    "recent_90d_transaction_frequency_per_day",
    "activity_change_count",
]
DIAGNOSTIC_FEATURES_FOR_PROFILES = [
    "balance_median",
    "credit_amount_total",
    "debit_amount_total",
    "net_cash_flow",
    "credit_frequency_per_day",
    "debit_frequency_per_day",
    "balance_std",
    "activity_change_count",
    "account_tenure_days",
    "gross_money_movement",
]
MODEL_NAMES = ("kmeans", "gaussian_mixture")


def _sha256_ids(ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for account_id in np.asarray(ids, dtype=str):
        digest.update(account_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_snapshot(snapshot_directory: str | Path, *, expected_as_of_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = Path(snapshot_directory)
    metadata_path = directory / "berka_account_snapshot_metadata.json"
    features_path = directory / "berka_account_snapshot_features.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("as_of_date") != expected_as_of_date:
        raise ValueError(
            f"Snapshot as_of_date mismatch: expected {expected_as_of_date}, got {metadata.get('as_of_date')}"
        )
    if metadata.get("feature_version") != "berka_account_snapshot_v1":
        raise ValueError(f"Unsupported Berka snapshot version: {metadata.get('feature_version')}")
    frame = pd.read_csv(features_path)
    if "account_id" not in frame or frame["account_id"].isna().any() or frame["account_id"].duplicated().any():
        raise ValueError("Snapshot must contain one non-null row per account_id")
    if int(metadata.get("summary", {}).get("rows", -1)) != len(frame):
        raise ValueError("Snapshot row count does not match metadata")
    documented = set(metadata.get("feature_definitions", {}))
    observed = set(frame.columns) - {"account_id"}
    if documented != observed:
        raise ValueError("Snapshot feature definitions do not match the CSV columns")
    if not pd.api.types.is_numeric_dtype(frame["account_tenure_days"]):
        raise TypeError("account_tenure_days must be numeric")
    if not np.isfinite(frame["account_tenure_days"].to_numpy(dtype=float)).all():
        raise ValueError("Snapshot tenure must be finite")
    return frame, metadata


def make_shared_split(
    eligible: pd.DataFrame, *, training_fraction: float = 0.75, random_seed: int = 8675309
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one stable, disjoint account split independent of model seeds."""
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training_fraction must be strictly between zero and one")
    ordered = eligible.sort_values("account_id", kind="mergesort").reset_index(drop=True)
    if ordered["account_id"].isna().any() or ordered["account_id"].duplicated().any():
        raise ValueError("Eligible account IDs must be unique and non-null")
    permutation = np.random.default_rng(random_seed).permutation(len(ordered))
    train_count = int(len(ordered) * training_fraction)
    train = ordered.iloc[permutation[:train_count]].copy().reset_index(drop=True)
    holdout = ordered.iloc[permutation[train_count:]].copy().reset_index(drop=True)
    if set(train["account_id"]).intersection(holdout["account_id"]):
        raise RuntimeError("Shared split produced overlapping train and held-out accounts")
    return train, holdout


def select_behavioral_features(frame: pd.DataFrame, feature_order: list[str] | None = None) -> pd.DataFrame:
    """Select the documented behavioral inputs and derive gross money movement."""
    features = EXPECTED_FEATURES if feature_order is None else feature_order
    if features != EXPECTED_FEATURES:
        raise ValueError("Feature names/order differ from the reviewed Berka v1 feature contract")
    required = set(features) - {"gross_money_movement"}
    required.update({"credit_amount_total", "debit_amount_total"})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Snapshot is missing selected or derived source features: {sorted(missing)}")
    selected = frame.copy()
    selected["gross_money_movement"] = selected["credit_amount_total"] + selected["debit_amount_total"]
    selected = selected.loc[:, features]
    for column in features:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


class BerkaTrainFittedPreprocessor:
    """Train-only imputation, signed/skew transforms, clipping, and scaling."""

    def __init__(
        self,
        feature_order: list[str] | None = None,
        *,
        positive_log1p: list[str] | None = None,
        signed_asinh: list[str] | None = None,
        bounded_unchanged: list[str] | None = None,
        clipping_quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> None:
        self.feature_order = list(EXPECTED_FEATURES if feature_order is None else feature_order)
        self.positive_log1p = list(positive_log1p or [])
        self.signed_asinh = list(signed_asinh or [])
        self.bounded_unchanged = list(bounded_unchanged or [])
        self.clipping_quantiles = clipping_quantiles
        transform_features = self.positive_log1p + self.signed_asinh + self.bounded_unchanged
        if len(transform_features) != len(set(transform_features)):
            raise ValueError("Each feature must have exactly one transformation policy")
        if set(transform_features) != set(self.feature_order):
            raise ValueError("Transformation policies must cover the selected feature set exactly")
        self.imputation_values_: dict[str, float] = {}
        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        self.scaler_: StandardScaler | None = None
        self.fit_rows_: int | None = None
        self.fit_ids_sha256_: str | None = None

    def _transformed(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_order).difference(frame.columns)
        if missing:
            raise ValueError(f"Selected frame missing preprocessing features: {sorted(missing)}")
        output = frame.loc[:, self.feature_order].apply(pd.to_numeric, errors="coerce").astype(float)
        for column in self.positive_log1p:
            values = output[column]
            if values.dropna().lt(0).any():
                raise ValueError(f"log1p feature {column} contains negative values")
            output[column] = np.log1p(values)
        for column in self.signed_asinh:
            output[column] = np.arcsinh(output[column])
        for column in self.bounded_unchanged:
            values = output[column].dropna()
            if values.lt(0).any() or values.gt(1).any():
                raise ValueError(f"Bounded feature {column} must be within [0, 1]")
        return output

    def fit(self, frame: pd.DataFrame, account_ids: np.ndarray) -> BerkaTrainFittedPreprocessor:
        if len(frame) != len(account_ids) or len(frame) < 2:
            raise ValueError("Training features and account IDs must align and contain at least two rows")
        if len(set(np.asarray(account_ids, dtype=str))) != len(account_ids):
            raise ValueError("Training account IDs must be unique")
        low_q, high_q = self.clipping_quantiles
        if not 0 <= low_q < high_q <= 1:
            raise ValueError("clipping_quantiles must satisfy 0 <= low < high <= 1")
        transformed = self._transformed(frame)
        self.imputation_values_ = {}
        for column in self.feature_order:
            median = transformed[column].median()
            if pd.isna(median):
                raise ValueError(f"Training feature {column} is entirely missing")
            self.imputation_values_[column] = float(median)
            transformed[column] = transformed[column].fillna(median)
        self.clip_bounds_ = {
            column: (float(transformed[column].quantile(low_q)), float(transformed[column].quantile(high_q)))
            for column in self.feature_order
        }
        clipped = self._clip(transformed)
        self.scaler_ = StandardScaler().fit(clipped.loc[:, self.feature_order])
        self.fit_rows_ = len(frame)
        self.fit_ids_sha256_ = _sha256_ids(np.asarray(account_ids, dtype=str))
        return self

    def _clip(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for column, (lower, upper) in self.clip_bounds_.items():
            output[column] = output[column].clip(lower=lower, upper=upper)
        return output

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor must be fit on training accounts before transform")
        transformed = self._transformed(frame)
        for column, median in self.imputation_values_.items():
            transformed[column] = transformed[column].fillna(median)
        clipped = self._clip(transformed)
        matrix = self.scaler_.transform(clipped.loc[:, self.feature_order])
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessing produced non-finite values")
        return np.ascontiguousarray(matrix, dtype=np.float64)

    def metadata(self) -> dict[str, Any]:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        return {
            "feature_order": self.feature_order,
            "imputation_values_after_transforms": self.imputation_values_,
            "positive_log1p_features": self.positive_log1p,
            "signed_asinh_features": self.signed_asinh,
            "bounded_unchanged_features": self.bounded_unchanged,
            "training_fitted_clip_bounds": {key: list(value) for key, value in self.clip_bounds_.items()},
            "training_fitted_scaler_mean": self.scaler_.mean_.tolist(),
            "training_fitted_scaler_scale": self.scaler_.scale_.tolist(),
            "clip_quantiles": list(self.clipping_quantiles),
            "fit_rows": self.fit_rows_,
            "fit_account_ids_sha256": self.fit_ids_sha256_,
            "fit_scope": "training accounts only",
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a portable state payload without pickling this source class."""
        return {
            "feature_order": self.feature_order,
            "positive_log1p": self.positive_log1p,
            "signed_asinh": self.signed_asinh,
            "bounded_unchanged": self.bounded_unchanged,
            "clipping_quantiles": self.clipping_quantiles,
            "imputation_values": self.imputation_values_,
            "clip_bounds": self.clip_bounds_,
            "scaler": self.scaler_,
            "fit_rows": self.fit_rows_,
            "fit_ids_sha256": self.fit_ids_sha256_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> BerkaTrainFittedPreprocessor:
        processor = cls(
            feature_order=list(state["feature_order"]),
            positive_log1p=list(state["positive_log1p"]),
            signed_asinh=list(state["signed_asinh"]),
            bounded_unchanged=list(state["bounded_unchanged"]),
            clipping_quantiles=tuple(state["clipping_quantiles"]),
        )
        processor.imputation_values_ = dict(state["imputation_values"])
        processor.clip_bounds_ = {key: tuple(value) for key, value in state["clip_bounds"].items()}
        processor.scaler_ = state["scaler"]
        processor.fit_rows_ = state["fit_rows"]
        processor.fit_ids_sha256_ = state["fit_ids_sha256"]
        return processor


def diagnose_training_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe all numeric snapshot inputs on the training population only."""
    numeric = frame.select_dtypes(include=["number"]).drop(columns=["account_id"], errors="ignore")
    rows: list[dict[str, Any]] = []
    for column in numeric.columns:
        values = numeric[column].dropna().astype(float)
        if not len(values):
            rows.append({"feature": column, "missing_count": int(numeric[column].isna().sum()), "missing_pct": 100.0, "skew": None})
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        rows.append(
            {
                "feature": column,
                "missing_count": int(numeric[column].isna().sum()),
                "missing_pct": float(100 * numeric[column].isna().mean()),
                "skew": float(values.skew()),
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p25": float(q1),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "p75": float(q3),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
                "iqr_outlier_count": int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()),
                "constant": bool(values.nunique() <= 1),
            }
        )
    diagnostics = pd.DataFrame(rows)
    correlations = numeric.corr(method="spearman", min_periods=3)
    correlations.index.name = "feature"
    return diagnostics, correlations


def _cluster_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    requested_k: int,
    tiny_threshold_pct: float,
    severe_threshold_pct: float,
) -> dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    count_map = {str(int(label)): int(count) for label, count in zip(unique, counts)}
    pct_map = {key: 100.0 * value / len(labels) for key, value in count_map.items()}
    observed_k = len(unique)
    degenerate = observed_k != requested_k or observed_k < 2
    metrics: dict[str, Any] = {
        "observed_clusters": int(observed_k),
        "cluster_counts": count_map,
        "cluster_percentages": pct_map,
        "minimum_segment_pct": min(pct_map.values()) if pct_map else 0.0,
        "degenerate": bool(degenerate),
        "tiny_cluster": bool(pct_map and min(pct_map.values()) < tiny_threshold_pct),
        "severe_tiny_cluster": bool(pct_map and min(pct_map.values()) < severe_threshold_pct),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if not degenerate:
        metrics["silhouette"] = float(silhouette_score(features, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(features, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(features, labels))
    return metrics


def _eta_squared(values: pd.Series, labels: np.ndarray) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric)
    if valid.sum() < 2 or np.unique(labels[valid]).size < 2:
        return None
    values_valid = numeric[valid]
    labels_valid = labels[valid]
    grand_mean = values_valid.mean()
    total = float(np.square(values_valid - grand_mean).sum())
    if total == 0:
        return 0.0
    between = 0.0
    for label in np.unique(labels_valid):
        members = values_valid[labels_valid == label]
        between += len(members) * (members.mean() - grand_mean) ** 2
    return float(between / total)


def _segment_profiles(
    frame: pd.DataFrame, labels: np.ndarray, *, algorithm: str, k: int, seed: int
) -> list[dict[str, Any]]:
    data = frame.copy()
    data["gross_money_movement"] = data["credit_amount_total"] + data["debit_amount_total"]
    data["cluster"] = labels
    output: list[dict[str, Any]] = []
    for cluster, members in data.groupby("cluster", sort=True):
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "k": k,
            "seed": seed,
            "cluster": int(cluster),
            "account_count": len(members),
            "segment_pct": 100.0 * len(members) / len(data),
        }
        for feature in DIAGNOSTIC_FEATURES_FOR_PROFILES:
            row[f"median_{feature}"] = float(members[feature].median())
        for feature in DIAGNOSTIC_FEATURES_FOR_PROFILES:
            eta = _eta_squared(data[feature], labels)
            row[f"cluster_eta_squared_{feature}"] = eta
        output.append(row)
    return output


def _save_new_joblib(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Berka artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def run_berka_baselines(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    snapshot_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    snapshot_path = Path(snapshot_directory or data_config["snapshot_directory"])
    output_root = Path(output_directory or data_config["output_directory"])
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Berka baseline directory: {output_root}")
    feature_order = list(config["features"]["selected"])
    if feature_order != EXPECTED_FEATURES:
        raise ValueError("Configured feature order differs from reviewed Berka baseline v1 contract")
    snapshot, snapshot_metadata = load_snapshot(snapshot_path, expected_as_of_date=data_config["as_of_date"])
    if len(snapshot) != int(data_config["expected_snapshot_rows"]):
        raise ValueError("Berka snapshot row count differs from configured expectation")
    if snapshot_metadata["as_of_date"] != "1997-12-31":
        raise ValueError("This baseline is restricted to the 1997-12-31 historical snapshot")

    eligible = snapshot.loc[snapshot["account_tenure_days"].ge(int(data_config["minimum_tenure_days"]))].copy()
    train, holdout = make_shared_split(
        eligible,
        training_fraction=float(config["split"]["training_fraction"]),
        random_seed=int(config["split"]["random_seed"]),
    )
    train_selected = select_behavioral_features(train, feature_order)
    holdout_selected = select_behavioral_features(holdout, feature_order)
    diagnostics, correlations = diagnose_training_features(train)

    prep_config = config["features"]
    preprocessor = BerkaTrainFittedPreprocessor(
        feature_order,
        positive_log1p=list(prep_config["positive_log1p"]),
        signed_asinh=list(prep_config["signed_asinh"]),
        bounded_unchanged=list(prep_config["bounded_unchanged"]),
        clipping_quantiles=tuple(float(value) for value in prep_config["clipping_quantiles"]),
    ).fit(train_selected, train["account_id"].to_numpy())
    train_matrix = preprocessor.transform(train_selected)
    holdout_matrix = preprocessor.transform(holdout_selected)

    # Allocate the run directory only after all snapshot and contract checks pass.
    output_root.mkdir(parents=True, exist_ok=False)
    diagnostic_path = output_root / "feature_diagnostics_training_only.csv"
    correlation_path = output_root / "spearman_correlations_training_only.csv"
    diagnostics.to_csv(diagnostic_path, index=False)
    correlations.to_csv(correlation_path)
    split_frame = pd.concat(
        [
            train[["account_id"]].assign(split="train"),
            holdout[["account_id"]].assign(split="held_out"),
        ],
        ignore_index=True,
    )
    split_path = output_root / "account_split.csv"
    split_frame.to_csv(split_path, index=False)
    train_ids = train["account_id"].astype(str).to_numpy()
    holdout_ids = holdout["account_id"].astype(str).to_numpy()
    preprocessor_path = output_root / "preprocessor_train_fitted.joblib"
    _save_new_joblib(
        preprocessor_path,
        {"preprocessor_state": preprocessor.state_dict(), "metadata": preprocessor.metadata()},
    )

    model_config = config["models"]
    seeds = [int(value) for value in model_config["seeds"]]
    cluster_counts = [int(value) for value in model_config["cluster_counts"]]
    metrics_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    assignment_parts: list[pd.DataFrame] = []
    predictions: dict[tuple[str, int, int], np.ndarray] = {}
    model_directory = output_root / "models"
    model_directory.mkdir()
    with threadpool_limits(limits=1):
        for algorithm in MODEL_NAMES:
            for k in cluster_counts:
                for seed in seeds:
                    estimator_config = model_config[algorithm]
                    started = time.perf_counter()
                    captured_warnings: list[str] = []
                    with warnings.catch_warnings(record=True) as warning_records:
                        warnings.simplefilter("always")
                        if algorithm == "kmeans":
                            estimator = KMeans(
                                n_clusters=k,
                                random_state=seed,
                                n_init=int(estimator_config["n_init"]),
                                max_iter=int(estimator_config["max_iter"]),
                                algorithm=str(estimator_config["algorithm"]),
                            ).fit(train_matrix)
                            labels = estimator.predict(holdout_matrix)
                            converged = True
                            iterations = int(estimator.n_iter_)
                            train_bic = train_aic = holdout_bic = holdout_aic = None
                        else:
                            estimator = GaussianMixture(
                                n_components=k,
                                covariance_type=str(estimator_config["covariance_type"]),
                                n_init=int(estimator_config["n_init"]),
                                max_iter=int(estimator_config["max_iter"]),
                                tol=float(estimator_config["tolerance"]),
                                reg_covar=float(estimator_config["regularization"]),
                                random_state=seed,
                            ).fit(train_matrix)
                            labels = estimator.predict(holdout_matrix)
                            converged = bool(estimator.converged_)
                            iterations = int(estimator.n_iter_)
                            train_bic = float(estimator.bic(train_matrix))
                            train_aic = float(estimator.aic(train_matrix))
                            holdout_bic = float(estimator.bic(holdout_matrix))
                            holdout_aic = float(estimator.aic(holdout_matrix))
                    elapsed = time.perf_counter() - started
                    captured_warnings = [f"{record.category.__name__}: {record.message}" for record in warning_records]
                    score = _cluster_metrics(
                        holdout_matrix,
                        labels,
                        requested_k=k,
                        tiny_threshold_pct=float(config["evaluation"]["tiny_cluster_threshold_percent"]),
                        severe_threshold_pct=float(config["evaluation"]["severe_cluster_threshold_percent"]),
                    )
                    score.update(
                        {
                            "algorithm": algorithm,
                            "k": k,
                            "seed": seed,
                            "training_rows": len(train),
                            "held_out_rows": len(holdout),
                            "training_seconds": elapsed,
                            "converged": converged,
                            "iterations": iterations,
                            "train_bic": train_bic,
                            "train_aic": train_aic,
                            "held_out_bic": holdout_bic,
                            "held_out_aic": holdout_aic,
                            "warnings": " | ".join(captured_warnings),
                        }
                    )
                    metrics_rows.append(score)
                    predictions[(algorithm, k, seed)] = labels
                    profiles.extend(_segment_profiles(holdout, labels, algorithm=algorithm, k=k, seed=seed))
                    run_id = f"{algorithm}_k{k}_seed{seed}"
                    assignment = pd.DataFrame(
                        {
                            "account_id": holdout_ids,
                            "algorithm": algorithm,
                            "k": k,
                            "seed": seed,
                            "cluster": labels.astype(int),
                        }
                    )
                    if algorithm == "gaussian_mixture":
                        assignment["max_component_probability"] = estimator.predict_proba(holdout_matrix).max(axis=1)
                    assignment_parts.append(assignment)
                    model_metadata = {
                        "experiment": config["experiment"],
                        "algorithm": algorithm,
                        "k": k,
                        "seed": seed,
                        "as_of_date": data_config["as_of_date"],
                        "feature_order": feature_order,
                        "training_account_ids_sha256": _sha256_ids(train_ids),
                        "held_out_account_ids_sha256": _sha256_ids(holdout_ids),
                        "preprocessor_path": str(preprocessor_path),
                        "preprocessor_fit_scope": "training accounts only",
                        "training_rows": len(train),
                        "held_out_rows": len(holdout),
                    }
                    _save_new_joblib(
                        model_directory / f"{run_id}.joblib",
                        {"estimator": estimator, "metadata": model_metadata},
                    )

    metric_frame = pd.DataFrame(metrics_rows)
    stability_rows: list[dict[str, Any]] = []
    for algorithm in MODEL_NAMES:
        for k in cluster_counts:
            seed_pairs = [
                adjusted_rand_score(predictions[(algorithm, k, left)], predictions[(algorithm, k, right)])
                for left, right in itertools.combinations(seeds, 2)
            ]
            stability_rows.append(
                {
                    "algorithm": algorithm,
                    "k": k,
                    "pairwise_seed_ari_mean": float(np.mean(seed_pairs)),
                    "pairwise_seed_ari_min": float(np.min(seed_pairs)),
                    "pairwise_seed_ari_max": float(np.max(seed_pairs)),
                }
            )
    stability_frame = pd.DataFrame(stability_rows)
    metric_frame = metric_frame.merge(stability_frame, on=["algorithm", "k"], validate="many_to_one")
    metric_frame["unstable"] = metric_frame["pairwise_seed_ari_mean"].lt(
        float(config["evaluation"]["unstable_pairwise_mean_ari_threshold"])
    )
    metrics_path = output_root / "candidate_metrics.csv"
    metric_frame.to_csv(metrics_path, index=False)
    assignments_path = output_root / "held_out_assignments.csv"
    pd.concat(assignment_parts, ignore_index=True).to_csv(assignments_path, index=False)
    profiles_path = output_root / "held_out_segment_profiles.csv"
    pd.DataFrame(profiles).to_csv(profiles_path, index=False)

    run_metadata = {
        "experiment": config["experiment"],
        "config_path": str(config_path),
        "snapshot_directory": str(snapshot_path),
        "snapshot_version": snapshot_metadata["feature_version"],
        "as_of_date": data_config["as_of_date"],
        "eligible_accounts": len(eligible),
        "training_accounts": len(train),
        "held_out_accounts": len(holdout),
        "minimum_tenure_days": int(data_config["minimum_tenure_days"]),
        "split_seed": int(config["split"]["random_seed"]),
        "training_account_ids_sha256": _sha256_ids(train_ids),
        "held_out_account_ids_sha256": _sha256_ids(holdout_ids),
        "feature_order": feature_order,
        "preprocessor_metadata": preprocessor.metadata(),
        "models": MODEL_NAMES,
        "cluster_counts": cluster_counts,
        "model_seeds": seeds,
        "all_feature_diagnostics_training_only": str(diagnostic_path),
        "spearman_correlations_training_only": str(correlation_path),
        "preprocessor_path": str(preprocessor_path),
        "metrics_path": str(metrics_path),
        "held_out_assignments_path": str(assignments_path),
        "profiles_path": str(profiles_path),
        "use_of_1998_transactions": "none; only the validated 1997-12-31 snapshot is read",
    }
    metadata_path = output_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config_copy_path = output_root / "config.yaml"
    config_copy_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "output_directory": output_root,
        "metrics": metrics_path,
        "assignments": assignments_path,
        "profiles": profiles_path,
        "metadata": metadata_path,
        "preprocessor": preprocessor_path,
        "diagnostics": diagnostic_path,
        "correlations": correlation_path,
        "split": split_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Berka behavioral K-Means and diagonal-GMM baselines.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--snapshot-directory", default=None)
    parser.add_argument("--output-directory", default=None)
    args = parser.parse_args()
    paths = run_berka_baselines(
        args.config, snapshot_directory=args.snapshot_directory, output_directory=args.output_directory
    )
    print(f"Completed Berka baseline experiment: {paths['output_directory']}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
