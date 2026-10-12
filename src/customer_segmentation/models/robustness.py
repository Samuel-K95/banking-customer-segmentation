"""Pure helpers for clipping sensitivity and transaction-history audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def fit_quantile_clip_bounds(
    training_values: np.ndarray,
    *,
    feature_name: str,
    lower_quantile: float,
    upper_quantile: float,
    method: str = "linear",
) -> dict[str, Any]:
    """Fit winsorization cutoffs from a training vector only."""
    values = np.asarray(training_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("training_values must be a finite vector with at least two rows.")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
    if not feature_name:
        raise ValueError("feature_name must be non-empty.")
    lower, upper = np.quantile(
        values, [lower_quantile, upper_quantile], method=method
    )
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Fitted clipping bounds are invalid.")
    return {
        "feature": feature_name,
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "quantile_method": method,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "fit_rows": len(values),
        "fit_scope": "training customers only",
    }


def apply_quantile_clip(
    values: np.ndarray,
    bounds: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply saved cutoffs to a vector without mutating the input."""
    original = np.asarray(values, dtype=np.float64)
    if original.ndim != 1 or not np.isfinite(original).all():
        raise ValueError("values must be a finite vector.")
    lower = float(bounds["lower_bound"])
    upper = float(bounds["upper_bound"])
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Clip bounds must be finite and ordered.")
    below = original < lower
    above = original > upper
    transformed = np.clip(original, lower, upper)
    return transformed, below, above


def validate_customer_order(expected_ids: np.ndarray, observed_ids: np.ndarray) -> None:
    """Require unique, non-empty IDs in the exact expected order."""
    expected = np.asarray(expected_ids, dtype=str)
    observed = np.asarray(observed_ids, dtype=str)
    for name, values in (("expected", expected), ("observed", observed)):
        if values.ndim != 1 or np.any(values == "") or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} CustomerIDs must be unique, non-empty one-dimensional arrays.")
    if not np.array_equal(expected, observed):
        raise ValueError("CustomerID ordering does not match the requested sample manifest.")


def validate_disjoint_customer_ids(training_ids: np.ndarray, evaluation_ids: np.ndarray) -> None:
    """Require disjoint train/evaluation IDs with unique, non-empty values."""
    training = np.asarray(training_ids, dtype=str)
    evaluation = np.asarray(evaluation_ids, dtype=str)
    for name, values in (("training", training), ("evaluation", evaluation)):
        if values.ndim != 1 or np.any(values == "") or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} IDs must be unique, non-empty one-dimensional arrays.")
    if np.intersect1d(training, evaluation).size:
        raise ValueError("Training and held-out evaluation IDs must be disjoint.")


def transaction_history_cohort(transaction_counts: np.ndarray) -> np.ndarray:
    """Partition customers into exactly-one and at-least-two-transaction cohorts."""
    counts = np.asarray(transaction_counts)
    if counts.ndim != 1 or len(counts) < 1:
        raise ValueError("transaction_counts must be a non-empty vector.")
    if not np.isfinite(counts).all() or np.any(counts < 1):
        raise ValueError("transaction_counts must contain finite positive counts.")
    cohorts = np.where(counts == 1, "exactly_one", "at_least_two")
    return cohorts.astype(str, copy=False)


def cohort_segment_profiles(
    customer_ids: np.ndarray,
    cluster_labels: np.ndarray,
    transaction_counts: np.ndarray,
) -> pd.DataFrame:
    """Summarize segment membership and coverage separately by history cohort."""
    ids = np.asarray(customer_ids, dtype=str)
    labels = np.asarray(cluster_labels)
    counts = np.asarray(transaction_counts)
    if ids.ndim != 1 or labels.ndim != 1 or counts.ndim != 1:
        raise ValueError("CustomerIDs, cluster labels, and transaction counts must be vectors.")
    if not (len(ids) == len(labels) == len(counts)):
        raise ValueError("CustomerIDs, labels, and transaction counts must align row for row.")
    if np.any(ids == "") or len(np.unique(ids)) != len(ids):
        raise ValueError("CustomerIDs must be unique and non-empty.")
    cohorts = transaction_history_cohort(counts)
    rows: list[dict[str, Any]] = []
    for cohort in ("exactly_one", "at_least_two"):
        cohort_mask = cohorts == cohort
        cohort_total = int(cohort_mask.sum())
        if cohort_total == 0:
            continue
        for label in np.unique(labels[cohort_mask]):
            segment_mask = cohort_mask & (labels == label)
            segment_count = int(segment_mask.sum())
            rows.append(
                {
                    "cohort": cohort,
                    "cluster": int(label),
                    "cohort_customers_in_sample": cohort_total,
                    "customers_in_cluster": segment_count,
                    "percent_of_cohort_sample": 100.0 * segment_count / cohort_total,
                    "percent_of_all_assigned": 100.0 * segment_count / len(ids),
                }
            )
    return pd.DataFrame(rows)


def history_assignment_association(
    cluster_labels: np.ndarray,
    transaction_counts: np.ndarray,
) -> dict[str, float]:
    """Quantify association with singleton status and exact transaction depth."""
    labels = np.asarray(cluster_labels)
    counts = np.asarray(transaction_counts)
    if labels.ndim != 1 or counts.ndim != 1 or len(labels) != len(counts):
        raise ValueError("Cluster labels and transaction counts must align.")
    cohorts = transaction_history_cohort(counts)
    return {
        "ari_vs_singleton_status": float(
            adjusted_rand_score(labels, cohorts == "exactly_one")
        ),
        "nmi_vs_singleton_status": float(
            normalized_mutual_info_score(labels, cohorts == "exactly_one")
        ),
        "ari_vs_transaction_count": float(adjusted_rand_score(labels, counts)),
        "nmi_vs_transaction_count": float(normalized_mutual_info_score(labels, counts)),
    }


def match_cluster_membership(
    reference_ids: np.ndarray,
    reference_labels: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_labels: np.ndarray,
) -> dict[str, Any]:
    """Align arbitrary cluster IDs and count changed assignments on shared customers."""
    reference_ids = np.asarray(reference_ids, dtype=str)
    candidate_ids = np.asarray(candidate_ids, dtype=str)
    reference_labels = np.asarray(reference_labels)
    candidate_labels = np.asarray(candidate_labels)
    if reference_labels.ndim != 1 or candidate_labels.ndim != 1:
        raise ValueError("Cluster labels must be one-dimensional.")
    if reference_labels.shape != candidate_labels.shape:
        raise ValueError("Reference and candidate labels must have the same length.")
    if not np.array_equal(reference_ids, candidate_ids):
        raise ValueError("Membership comparison requires identical ordered CustomerIDs.")
    if len(reference_ids) != len(reference_labels):
        raise ValueError("CustomerIDs and cluster labels must align.")
    reference_clusters = np.unique(reference_labels)
    candidate_clusters = np.unique(candidate_labels)
    contingency = np.zeros(
        (len(reference_clusters), len(candidate_clusters)), dtype=np.int64
    )
    reference_indices = {int(label): index for index, label in enumerate(reference_clusters)}
    candidate_indices = {int(label): index for index, label in enumerate(candidate_clusters)}
    for ref_label, candidate_label in zip(reference_labels, candidate_labels, strict=True):
        contingency[reference_indices[int(ref_label)], candidate_indices[int(candidate_label)]] += 1
    from scipy.optimize import linear_sum_assignment

    ref_rows, candidate_columns = linear_sum_assignment(-contingency)
    label_map = {
        int(candidate_clusters[candidate_index]): int(reference_clusters[reference_index])
        for reference_index, candidate_index in zip(ref_rows, candidate_columns, strict=True)
    }
    aligned_candidate = np.asarray(
        [label_map.get(int(label), -1) for label in candidate_labels], dtype=np.int64
    )
    changed = reference_labels.astype(np.int64) != aligned_candidate
    from sklearn.metrics import adjusted_rand_score

    reference_counts = {
        str(int(label)): int(np.count_nonzero(reference_labels == label))
        for label in reference_clusters
    }
    candidate_counts_aligned = {
        str(label_map[int(candidate_label)]): int(np.count_nonzero(candidate_labels == candidate_label))
        for candidate_label in candidate_clusters
        if int(candidate_label) in label_map
    }
    return {
        "label_map_candidate_to_reference": {
            str(candidate): reference for candidate, reference in label_map.items()
        },
        "reference_cluster_counts": reference_counts,
        "candidate_cluster_counts_aligned": candidate_counts_aligned,
        "customer_count": len(reference_ids),
        "changed_customers_after_label_alignment": int(changed.sum()),
        "changed_percent_after_label_alignment": float(100 * changed.mean()),
        "adjusted_rand_index": float(
            adjusted_rand_score(reference_labels, candidate_labels)
        ),
    }
