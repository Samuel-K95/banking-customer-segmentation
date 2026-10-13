"""Read-only Task 10A finalist, manifest, fairness, and OOT feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from customer_segmentation.models.clustering_baselines import (
    load_model_artifact,
    ordered_id_sha256,
    validate_feature_contract,
)
from customer_segmentation.models.p2_segmentation_experiments import (
    RAW_COLUMNS,
    _aggregate_raw_customer_features,
    _load_interpretable_customer_features,
    _verify_p1_aggregation_semantics,
)

TASK08_RUN = "task08_track_a_p2_20261011T135839451060Z_f80fa838"
TASK08_FAILED_RUN = "task08_track_a_p2_20261011T135659725728Z_47cae105"
TASK09_RUN = "task09_track_a_final_ae_20261011T160711218914Z_97ecb040"
TASK09_PARTIAL_RUN = "task09_track_a_final_ae_20261011T160333143960Z_8cca89c2"
AS_OF_DATE = "2016-11-01"
AGE_BINS = [18, 25, 35, 45, 55, 65, 101]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-100"]
DEMOGRAPHIC_COLUMNS = ["Age", "Gender_M", "Is_Tier_1_City"]
PROFILE_COLUMNS = [
    "Transaction_Count",
    "AccountBalance",
    "Avg_Transaction_Amt",
    "Total_Spend",
    "Avg_Spend_Ratio",
    "Recency_Days",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Task 10A audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Task 10A audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def demographic_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent profile-only categories; never used by a model."""
    output = frame.copy()
    age = pd.to_numeric(output["Age"], errors="coerce")
    output["Age_Group"] = pd.cut(
        age,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    ).astype("string")
    output["Age_Group"] = output["Age_Group"].fillna("Missing_or_out_of_range")

    gender = pd.to_numeric(output["Gender_M"], errors="coerce")
    output["Gender_Group"] = gender.map({0: "Gender_M_indicator_0_non_M", 1: "Gender_M_indicator_1_Male"})
    output["Gender_Group"] = output["Gender_Group"].fillna("Missing_or_unknown")

    city = pd.to_numeric(output["Is_Tier_1_City"], errors="coerce")
    output["City_Tier_Group"] = city.map({0: "Non_tier_1_city", 1: "Tier_1_city"})
    output["City_Tier_Group"] = output["City_Tier_Group"].fillna("Missing_or_unknown")
    return output


def _task_label(path: Path) -> tuple[str, str]:
    name = path.name
    normalized = path.as_posix()
    if TASK08_FAILED_RUN in normalized:
        return "Task08", "failed_manifest_only"
    if TASK09_PARTIAL_RUN in normalized:
        return "Task09", "partial_fit_attempt"
    if TASK09_RUN in normalized:
        return "Task09", "completed"
    if TASK08_RUN in normalized:
        return "Task08", "completed"
    if "task07_track_a_robustness" in normalized:
        return "Task07", "completed"
    if "track_a_ae_" in name:
        return "Task06", "completed"
    if name.startswith("track_a_"):
        return "Task05", "completed"
    return "Unknown", "unclassified"


def _manifest_roles(frame: pd.DataFrame, path: Path) -> list[str]:
    if "internal_split" in frame.columns:
        return frame["internal_split"].astype(str).tolist()
    name = path.name.lower()
    if "repeat_train" in name:
        return ["repeat_training"] * len(frame)
    if "repeat_evaluation" in name:
        return ["repeat_evaluation"] * len(frame)
    if "evaluation" in name:
        return ["evaluation"] * len(frame)
    if "train" in name:
        return ["training"] * len(frame)
    return ["sampled_manifest"] * len(frame)


def _manifest_audit(samples_root: Path, population_ids: np.ndarray) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    inventory: list[dict[str, Any]] = []
    manifest_sets: dict[str, set[str]] = {}
    usage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "manifest_occurrences": 0,
            "training_manifest_occurrences": 0,
            "evaluation_manifest_occurrences": 0,
            "internal_validation_occurrences": 0,
            "internal_train_occurrences": 0,
            "task05_seen": False,
            "task06_seen": False,
            "task07_seen": False,
            "task08_seen": False,
            "task09_seen": False,
            "in_incomplete_manifest": False,
            "in_partial_fit_attempt": False,
            "in_partial_model_selection_attempt": False,
            "in_completed_training_or_fit": False,
            "in_completed_evaluation_or_selection": False,
        }
    )
    for path in sorted(samples_root.rglob("*.csv")):
        frame = pd.read_csv(path, dtype={"CustomerID": "string"})
        if "CustomerID" not in frame:
            continue
        ids = frame["CustomerID"].astype("string").to_numpy(dtype=str)
        ids_set = set(ids)
        relative = path.as_posix()
        task, status = _task_label(path)
        roles = _manifest_roles(frame, path)
        if len(roles) != len(ids):
            raise ValueError(f"Manifest role alignment failure: {path}")
        if len(ids) != len(ids_set):
            raise ValueError(f"Duplicate CustomerIDs in manifest: {path}")
        manifest_sets[relative] = ids_set
        role_counts = pd.Series(roles, dtype="string").value_counts().to_dict()
        inventory.append(
            {
                "manifest_path": relative,
                "task": task,
                "status": status,
                "rows": len(ids),
                "unique_customer_ids": len(ids_set),
                "ordered_id_sha256": ordered_id_sha256(ids),
                "row_position_min": int(frame["row_position"].min())
                if "row_position" in frame.columns and len(frame)
                else None,
                "row_position_max": int(frame["row_position"].max())
                if "row_position" in frame.columns and len(frame)
                else None,
                "role_counts_json": json.dumps(role_counts, sort_keys=True),
            }
        )
        for customer_id, role in zip(ids, roles, strict=True):
            record = usage[customer_id]
            record["manifest_occurrences"] += 1
            task_key = f"task{task[-2:]}_seen" if task.startswith("Task") else ""
            if task_key in record:
                record[task_key] = True
            if status != "completed":
                record["in_incomplete_manifest"] = True
            if status == "completed":
                if role in {"training", "repeat_training", "internal_train"}:
                    record["training_manifest_occurrences"] += 1
                    record["in_completed_training_or_fit"] = True
                elif role in {"evaluation", "repeat_evaluation", "internal_validation"}:
                    record["evaluation_manifest_occurrences"] += 1
                    record["in_completed_evaluation_or_selection"] = True
                if role == "internal_train":
                    record["internal_train_occurrences"] += 1
                if role == "internal_validation":
                    record["internal_validation_occurrences"] += 1
            elif status == "partial_fit_attempt":
                if role == "internal_train":
                    record["in_partial_fit_attempt"] = True
                elif role == "internal_validation":
                    record["in_partial_model_selection_attempt"] = True

    pairwise: list[dict[str, Any]] = []
    paths = list(manifest_sets)
    for left_path, right_path in itertools.combinations(paths, 2):
        left, right = manifest_sets[left_path], manifest_sets[right_path]
        intersection = len(left & right)
        union = len(left | right)
        pairwise.append(
            {
                "left_manifest": left_path,
                "right_manifest": right_path,
                "left_customers": len(left),
                "right_customers": len(right),
                "overlap_customers": intersection,
                "overlap_percent_of_left": 100 * intersection / len(left) if left else 0.0,
                "overlap_percent_of_right": 100 * intersection / len(right) if right else 0.0,
                "jaccard_percent": 100 * intersection / union if union else 0.0,
            }
        )
    usage_rows = [
        {"CustomerID": customer_id, **record}
        for customer_id, record in sorted(usage.items())
    ]
    usage_frame = pd.DataFrame(usage_rows)
    population_set = set(np.asarray(population_ids, dtype=str))
    manifest_union = set(usage) & population_set
    completed_used = set(
        usage_frame.loc[
            usage_frame["in_completed_training_or_fit"]
            | usage_frame["in_completed_evaluation_or_selection"],
            "CustomerID",
        ].astype(str)
    )
    completed_fit = set(
        usage_frame.loc[usage_frame["in_completed_training_or_fit"], "CustomerID"].astype(str)
    )
    completed_eval = set(
        usage_frame.loc[
            usage_frame["in_completed_evaluation_or_selection"], "CustomerID"
        ].astype(str)
    )
    partial_fit = set(
        usage_frame.loc[usage_frame["in_partial_fit_attempt"], "CustomerID"].astype(str)
    )
    partial_selection = set(
        usage_frame.loc[usage_frame["in_partial_model_selection_attempt"], "CustomerID"].astype(str)
    )
    attempted_used = completed_used | partial_fit | partial_selection
    all_saved_unseen = population_set - manifest_union
    completed_manifest_unseen = population_set - completed_used
    summary = {
        "manifest_files_audited": len(inventory),
        "population_customers": len(population_set),
        "unique_ids_in_any_saved_manifest": len(manifest_union),
        "unique_ids_in_completed_training_or_evaluation_or_validation": len(completed_used),
        "unique_ids_in_completed_training_or_fit_manifests": len(completed_fit),
        "unique_ids_in_completed_evaluation_or_model_selection_manifests": len(completed_eval),
        "unique_ids_in_partial_fit_attempts": len(partial_fit),
        "unique_ids_in_partial_model_selection_attempts": len(partial_selection),
        "unique_ids_in_any_completed_or_partial_fit_or_selection_manifest": len(attempted_used),
        "customer_ids_not_in_any_saved_manifest": len(all_saved_unseen),
        "customer_ids_not_in_completed_fit_or_evaluation_manifests": len(completed_manifest_unseen),
        "customer_ids_not_in_completed_or_partial_fit_or_selection_manifests": len(
            population_set - attempted_used
        ),
        "customer_ids_strictly_unexposed_to_prior_P1_fit": 0,
        "reason_strict_unexposed_zero": "The historical P1 StandardScaler was fitted on all 837,951 validated customer rows before Task 05/06 split creation.",
        "pairwise_manifest_comparisons": len(pairwise),
    }
    return pd.DataFrame(inventory), pd.DataFrame(pairwise), usage_frame, summary


def _candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "display_name": "2+ K-Means k=2, nine-feature train-fit",
            "algorithm": "kmeans_p2",
            "variant": "nine_feature_trainfit",
            "k": 2,
            "geometry": "unclipped nine-feature P2 train-fit",
        },
        {
            "display_name": "2+ K-Means k=3, nine-feature train-fit",
            "algorithm": "kmeans_p2",
            "variant": "nine_feature_trainfit",
            "k": 3,
            "geometry": "unclipped nine-feature P2 train-fit",
        },
        {
            "display_name": "2+ diagonal GMM k=2, ratio-clipped train-fit",
            "algorithm": "gaussian_mixture_p2",
            "variant": "ratio_clip_trainfit",
            "k": 2,
            "geometry": "raw Avg_Spend_Ratio train-q0.5/q99.5 clipping, then train-fit StandardScaler",
        },
    ]


def _load_demographics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["CustomerID", "Age", "Transaction_Count", *DEMOGRAPHIC_COLUMNS[1:]],
        dtype={"CustomerID": "string"},
    )
    if frame["CustomerID"].isna().any() or frame["CustomerID"].duplicated().any():
        raise ValueError("P1 interpretable customer IDs must be unique and non-null.")
    for column in ["Age", "Transaction_Count", *DEMOGRAPHIC_COLUMNS[1:]]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Age_Group"] = pd.cut(
        frame["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True
    ).astype("string").fillna("Missing_or_out_of_range")
    frame["Gender_Group"] = frame["Gender_M"].map(
        {0: "Female_indicator_0", 1: "Male_indicator_1"}
    ).fillna("Missing_or_unknown")
    frame["City_Tier_Group"] = frame["Is_Tier_1_City"].map(
        {0: "Non_tier_1_city", 1: "Tier_1_city"}
    ).fillna("Missing_or_unknown")
    frame["CustomerID"] = frame["CustomerID"].astype("string")
    return frame


def _population_demographics(demographics: pd.DataFrame, evaluation_ids: np.ndarray) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    eval_set = set(np.asarray(evaluation_ids, dtype=str))
    scopes = {
        "validated_population": demographics,
        "two_plus_population": demographics.loc[demographics["Transaction_Count"] >= 2],
        "task08_evaluation": demographics.loc[demographics["CustomerID"].isin(eval_set)],
    }
    dimensions = {
        "Age": "Age_Group",
        "Gender_M": "Gender_Group",
        "Is_Tier_1_City": "City_Tier_Group",
    }
    distribution_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for scope, frame in scopes.items():
        for raw_field, group_column in dimensions.items():
            total = len(frame)
            missing = int(frame[raw_field].isna().sum())
            missing_rows.append(
                {
                    "scope": scope,
                    "demographic_field": raw_field,
                    "rows": total,
                    "missing_or_null": missing,
                    "missing_percent": 100 * missing / total if total else 0.0,
                    "observed": total - missing,
                }
            )
            counts = frame[group_column].value_counts(dropna=False)
            for group, count in counts.items():
                distribution_rows.append(
                    {
                        "scope": scope,
                        "demographic_field": raw_field,
                        "group": str(group),
                        "customers": int(count),
                        "percent": float(100 * count / total) if total else 0.0,
                    }
                )
    return pd.DataFrame(distribution_rows), pd.DataFrame(missing_rows), {
        name: len(frame) for name, frame in scopes.items()
    }


def _profile_candidate(
    *,
    candidate: dict[str, Any],
    labels: np.ndarray,
    profile_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = np.asarray(labels, dtype=np.int32)
    total = len(labels)
    min_share = min(float(np.mean(labels == cluster)) for cluster in np.unique(labels))
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    for cluster in np.unique(labels):
        segment = profile_frame.loc[labels == cluster]
        base = {
            "candidate_id": candidate["candidate_id"],
            "display_name": candidate["display_name"],
            "seed": candidate["seed"],
            "k": candidate["k"],
            "algorithm": candidate["algorithm"],
            "preprocessing_variant": candidate["variant"],
            "geometry": candidate["geometry"],
            "cluster": int(cluster),
            "customers": len(segment),
            "segment_percent": 100 * len(segment) / total,
            "minimum_segment_percent_for_candidate": 100 * min_share,
        }
        for feature in PROFILE_COLUMNS:
            values = pd.to_numeric(segment[feature], errors="coerce")
            for q in quantiles:
                base[f"{feature}_p{int(q * 100):02d}"] = float(values.quantile(q))
        rows.append(base)
    return pd.DataFrame(rows)


def _demographic_segment_rows(
    candidate: dict[str, Any], labels: np.ndarray, demographic_eval: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = demographic_eval.reset_index(drop=True).copy()
    frame["cluster"] = np.asarray(labels, dtype=np.int32)
    dimensions = ["Age_Group", "Gender_Group", "City_Tier_Group"]
    for dimension in dimensions:
        base_counts = frame[dimension].value_counts(dropna=False).to_dict()
        base_n = len(frame)
        for cluster, segment in frame.groupby("cluster", sort=True):
            segment_counts = segment[dimension].value_counts(dropna=False).to_dict()
            for group in sorted(base_counts, key=str):
                group_n = int(base_counts[group])
                cell_n = int(segment_counts.get(group, 0))
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "display_name": candidate["display_name"],
                        "seed": candidate["seed"],
                        "k": candidate["k"],
                        "cluster": int(cluster),
                        "demographic_dimension": dimension,
                        "group": str(group),
                        "customers_in_segment_group": cell_n,
                        "percent_of_segment": 100 * cell_n / len(segment) if len(segment) else 0.0,
                        "customers_in_evaluation_group": group_n,
                        "percent_of_evaluation_group_assigned_to_segment": 100 * cell_n / group_n
                        if group_n
                        else 0.0,
                        "group_share_in_evaluation_percent": 100 * group_n / base_n if base_n else 0.0,
                        "segment_vs_evaluation_group_share_delta_pp": (
                            100 * cell_n / len(segment) - 100 * group_n / base_n
                            if len(segment) and base_n
                            else 0.0
                        ),
                        "small_cell_lt_30": cell_n < 30 or group_n < 30,
                    }
                )
    return rows


def _candidate_overlaps(
    by_seed: dict[int, list[tuple[dict[str, Any], np.ndarray]]],
    eval_ids: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, candidates in by_seed.items():
        for (left, left_labels), (right, right_labels) in itertools.combinations(candidates, 2):
            if len(left_labels) != len(eval_ids) or len(right_labels) != len(eval_ids):
                raise ValueError("Finalist overlap labels do not cover the shared evaluation IDs.")
            matrix = pd.crosstab(
                pd.Series(left_labels, name="left_cluster"),
                pd.Series(right_labels, name="right_cluster"),
            )
            rows.append(
                {
                    "seed": seed,
                    "left_candidate_id": left["candidate_id"],
                    "left_display_name": left["display_name"],
                    "right_candidate_id": right["candidate_id"],
                    "right_display_name": right["display_name"],
                    "evaluation_rows": len(eval_ids),
                    "evaluation_customer_id_sha256": ordered_id_sha256(eval_ids),
                    "adjusted_rand_index": float(adjusted_rand_score(left_labels, right_labels)),
                    "contingency_matrix_json": matrix.to_json(orient="split"),
                }
            )
    return pd.DataFrame(rows)


def _load_raw_eval_and_events(
    root: Path,
    *,
    valid_ids: set[str],
    eval_ids: np.ndarray,
    chunk_rows: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_path = root / "data/raw/bank_transactions.csv"
    selected_parts: list[pd.DataFrame] = []
    daily_rows: Counter[str] = Counter()
    raw_rows = 0
    complete_rows = 0
    invalid_dates = 0
    for chunk in pd.read_csv(
        raw_path,
        usecols=RAW_COLUMNS,
        dtype={"CustomerID": "string"},
        chunksize=chunk_rows,
    ):
        raw_rows += len(chunk)
        complete = chunk.dropna().copy()
        complete_rows += len(complete)
        dates = pd.to_datetime(complete["TransactionDate"], format="%d/%m/%y", errors="coerce")
        invalid_dates += int(dates.isna().sum())
        if dates.notna().any():
            daily = dates.dt.strftime("%Y-%m-%d").value_counts()
            daily_rows.update({str(day): int(count) for day, count in daily.items()})
        selected = complete.loc[complete["CustomerID"].astype(str).isin(valid_ids)].copy()
        if len(selected):
            selected["TransactionDate"] = pd.to_datetime(
                selected["TransactionDate"], format="%d/%m/%y", errors="raise"
            )
            for column in ["CustAccountBalance", "TransactionAmount (INR)", "TransactionTime"]:
                selected[column] = pd.to_numeric(selected[column], errors="raise")
            selected["HourOfDay"] = selected["TransactionTime"] // 10000
            selected["Spend_Balance_Ratio"] = selected["TransactionAmount (INR)"] / (
                selected["CustAccountBalance"] + 1.0
            )
            selected["Is_Weekend"] = selected["TransactionDate"].dt.dayofweek.isin([5, 6]).astype(np.int8)
            selected["Is_Night"] = (
                (selected["HourOfDay"] >= 20) | (selected["HourOfDay"] < 4)
            ).astype(np.int8)
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
                        "Is_Night",
                    ]
                ]
            )
    if invalid_dates:
        raise ValueError(f"Raw source has {invalid_dates} invalid complete-case transaction dates.")
    events = pd.concat(selected_parts, ignore_index=True)
    events["CustomerID"] = events["CustomerID"].astype("string")
    if set(events["CustomerID"].astype(str).unique()) != valid_ids:
        raise ValueError("P1-validated population IDs are not fully covered by complete-case raw events.")
    evaluation_frame, _ = _aggregate_raw_customer_features(
        raw_path,
        wanted_ids=set(np.asarray(eval_ids, dtype=str)),
        chunk_rows=50_000,
    )
    evaluation_frame = evaluation_frame.loc[np.asarray(eval_ids, dtype=str)].reset_index()
    return events, evaluation_frame, {
        "raw_rows": raw_rows,
        "complete_case_rows": complete_rows,
        "complete_case_invalid_dates": invalid_dates,
        "source_min_date": min(daily_rows),
        "source_max_date": max(daily_rows),
        "daily_complete_case_rows": dict(sorted(daily_rows.items())),
    }


def _snapshot_aggregates(events: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    history = events.loc[events["TransactionDate"] <= cutoff].copy()
    if not len(history):
        return pd.DataFrame()
    aggregate = history.groupby("CustomerID", sort=True).agg(
        Transaction_Count=("TransactionID", "count"),
        AccountBalance=("CustAccountBalance", "mean"),
        Avg_Transaction_Amt=("TransactionAmount (INR)", "mean"),
        Std_Transaction_Amt=("TransactionAmount (INR)", "std"),
        Total_Spend=("TransactionAmount (INR)", "sum"),
        Avg_Spend_Ratio=("Spend_Balance_Ratio", "mean"),
        Weekend_Spend_Ratio=("Is_Weekend", "mean"),
        Night_Spend_Ratio=("Is_Night", "mean"),
        Last_Transaction=("TransactionDate", "max"),
    )
    aggregate["Std_Transaction_Amt"] = aggregate["Std_Transaction_Amt"].fillna(0.0)
    aggregate["Recency_Days"] = (cutoff - aggregate["Last_Transaction"]).dt.days.astype(np.int64)
    return aggregate


def _raw_demographic_source_audit(root: Path, *, chunk_rows: int = 100_000) -> pd.DataFrame:
    """Summarize transaction-row demographic missingness before P1 filtering."""
    fields = ["CustomerDOB", "CustGender", "CustLocation"]
    totals: Counter[str] = Counter()
    raw_rows = 0
    tier_1 = {"MUMBAI", "DELHI", "BANGALORE", "CHENNAI", "HYDERABAD", "KOLKATA", "PUNE", "AHMEDABAD"}
    for chunk in pd.read_csv(
        root / "data/raw/bank_transactions.csv",
        usecols=["CustomerID", *fields],
        dtype={"CustomerID": "string"},
        chunksize=chunk_rows,
    ):
        raw_rows += len(chunk)
        totals["CustomerDOB: missing"] += int(chunk["CustomerDOB"].isna().sum())
        totals["CustGender: missing"] += int(chunk["CustGender"].isna().sum())
        totals["CustLocation: missing"] += int(chunk["CustLocation"].isna().sum())
        dob = pd.to_datetime(chunk["CustomerDOB"], format="%d/%m/%y", errors="coerce")
        dob.loc[dob.dt.year > 2005] -= pd.DateOffset(years=100)
        age = 2016 - dob.dt.year
        totals["Age_from_DOB: unparsable_or_outside_18_100"] += int(
            (dob.isna() | (age < 18) | (age > 100)).sum()
        )
        gender = chunk["CustGender"].astype("string").str.upper()
        totals["CustGender: M"] += int((gender == "M").sum())
        totals["CustGender: F"] += int((gender == "F").sum())
        totals["CustGender: other_nonmissing"] += int(
            (gender.notna() & ~gender.isin(["M", "F"])).sum()
        )
        location = chunk["CustLocation"].astype("string").str.upper()
        totals["Is_Tier_1_City: tier1_location"] += int(location.isin(tier_1).sum())
        totals["Is_Tier_1_City: non_tier1_location"] += int(
            (location.notna() & ~location.isin(tier_1)).sum()
        )
    return pd.DataFrame(
        [
            {
                "source_field_or_derived_group": category,
                "transaction_rows": count,
                "percent_of_raw_transaction_rows": 100 * count / raw_rows if raw_rows else 0.0,
                "denominator_raw_rows": raw_rows,
                "unit": "transaction row; repeat customers contribute multiple rows",
            }
            for category, count in sorted(totals.items())
        ]
    )


def _oot_feasibility(
    *,
    events: pd.DataFrame,
    demographics: pd.DataFrame,
    source_min_date: str,
    source_max_date: str,
    full_current_2plus: pd.DataFrame,
    previously_manifested_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff_dates = [pd.Timestamp(value) for value in ("2016-09-01", "2016-09-15", "2016-09-30")]
    valid_ids = demographics["CustomerID"].astype(str).to_numpy()
    summary_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    max_date = pd.Timestamp(source_max_date)
    for cutoff in cutoff_dates:
        before = events.loc[events["TransactionDate"] <= cutoff]
        after = events.loc[events["TransactionDate"] > cutoff]
        pre_counts = before.groupby("CustomerID").size().reindex(valid_ids, fill_value=0).astype(np.int64)
        future_counts = after.groupby("CustomerID").size().reindex(valid_ids, fill_value=0).astype(np.int64)
        pre2 = pre_counts >= 2
        pre3 = pre_counts >= 3
        new_ids = (pre_counts == 0) & (future_counts > 0)
        new_repeaters = new_ids & (future_counts >= 2)
        new_repeat_ids = pre_counts.index[new_repeaters.to_numpy()].astype(str)
        unseen_new_repeaters = set(new_repeat_ids) - previously_manifested_ids
        snapshot_repeat_ids = pre_counts.index[pre2.to_numpy()].astype(str)
        unseen_snapshot_repeaters = set(snapshot_repeat_ids) - previously_manifested_ids
        post_activity = future_counts > 0
        snapshot = _snapshot_aggregates(events, cutoff)
        snapshot2 = snapshot.loc[snapshot["Transaction_Count"] >= 2]
        for count_label, count in [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4+", 4)]:
            n = int((pre_counts == count).sum()) if count_label != "4+" else int((pre_counts >= 4).sum())
            history_rows.append(
                {
                    "snapshot_as_of_date": cutoff.date().isoformat(),
                    "history_transaction_count": count_label,
                    "customers": n,
                    "percent_of_validated_population": 100 * n / len(valid_ids),
                }
            )
        n_eligible = int(pre2.sum())
        followup = int((pre2 & post_activity).sum())
        followup_2plus = int((pre2 & (future_counts >= 2)).sum())
        new_2plus = int(new_repeaters.sum())
        future_days = int((max_date - cutoff).days)
        enough_for_50k_10k = n_eligible >= 60_000
        summary_rows.append(
            {
                "snapshot_as_of_date": cutoff.date().isoformat(),
                "history_start_date": source_min_date,
                "future_window_start_inclusive": (cutoff + pd.Timedelta(days=1)).date().isoformat(),
                "future_window_end": source_max_date,
                "future_window_days": future_days,
                "history_complete_case_rows": len(before),
                "future_complete_case_rows": len(after),
                "validated_population": len(valid_ids),
                "customers_with_any_history": int((pre_counts > 0).sum()),
                "customers_2plus_by_snapshot": n_eligible,
                "customers_3plus_by_snapshot": int(pre3.sum()),
                "snapshot_2plus_ids_in_any_saved_manifest": len(
                    set(snapshot_repeat_ids) & previously_manifested_ids
                ),
                "snapshot_2plus_ids_not_in_any_saved_manifest": len(unseen_snapshot_repeaters),
                "snapshot_2plus_manifest_unseen_can_support_10000_eval": len(
                    unseen_snapshot_repeaters
                )
                >= 10_000,
                "exactly_two_share_of_snapshot_2plus_percent": 100
                * int((pre_counts == 2).sum())
                / n_eligible
                if n_eligible
                else 0.0,
                "snapshot_2plus_percent_of_validated_population": 100 * n_eligible / len(valid_ids),
                "snapshot_2plus_customers_with_future_activity": followup,
                "snapshot_2plus_customers_with_future_2plus_events": followup_2plus,
                "new_after_cutoff_customers": int(new_ids.sum()),
                "new_after_cutoff_customers_with_2plus_future_events": new_2plus,
                "new_2plus_future_ids_in_prior_manifests": len(set(new_repeat_ids) & previously_manifested_ids),
                "new_2plus_future_ids_not_in_any_saved_manifest": len(unseen_new_repeaters),
                "manifest_unseen_new_repeaters_can_support_10000_eval": len(unseen_new_repeaters) >= 10_000,
                "supports_50000_train_plus_10000_holdout": enough_for_50k_10k,
                "eligible_customers_after_60000_reserved": max(0, n_eligible - 60_000),
                "snapshot_2plus_median_transaction_count": float(snapshot2["Transaction_Count"].median())
                if len(snapshot2)
                else None,
                "snapshot_2plus_median_balance": float(snapshot2["AccountBalance"].median())
                if len(snapshot2)
                else None,
                "snapshot_2plus_median_total_spend": float(snapshot2["Total_Spend"].median())
                if len(snapshot2)
                else None,
                "snapshot_2plus_median_recency_days": float(snapshot2["Recency_Days"].median())
                if len(snapshot2)
                else None,
            }
        )

        cohort_frames = {
            "current_full_period_2plus": full_current_2plus,
            "snapshot_2plus": snapshot2.reset_index().merge(
                demographics, on="CustomerID", how="left", validate="one_to_one"
            ),
            "new_repeaters_after_cutoff": demographics.loc[new_repeaters.to_numpy()].copy(),
        }
        for cohort_name, frame in cohort_frames.items():
            grouped = demographic_groups(frame)
            for dimension in ["Age_Group", "Gender_Group", "City_Tier_Group"]:
                base_counts = grouped[dimension].value_counts(dropna=False)
                for group, n in base_counts.items():
                    shift_rows.append(
                        {
                            "snapshot_as_of_date": cutoff.date().isoformat(),
                            "cohort": cohort_name,
                            "demographic_dimension": dimension,
                            "group": str(group),
                            "customers": int(n),
                            "percent": 100 * n / len(grouped) if len(grouped) else 0.0,
                            "cohort_rows": len(grouped),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(shift_rows), pd.DataFrame(history_rows)


def run_audit(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = root / "configs/models/track_a/track_a_p2_leakage_safe.yaml"
    with config_path.open(encoding="utf-8") as stream:
        p2_config = yaml.safe_load(stream)
    feature_order = list(p2_config["data"]["feature_order"])
    with (root / p2_config["data"]["feature_contract_file"]).open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)["tracks"]["track_a"]
    validate_feature_contract(contract, feature_order, feature_order)

    interpretable_path = root / p2_config["data"]["interpretable_file"]
    demographics = _load_demographics(interpretable_path)
    if len(demographics) != int(p2_config["data"]["row_count"]):
        raise ValueError("P1 validated population row count does not match the Task 08 contract.")
    population_ids = demographics["CustomerID"].astype(str).to_numpy()
    if demographics["CustomerID"].duplicated().any():
        raise ValueError("P1 validated population contains duplicate CustomerIDs.")
    population_count = len(population_ids)
    two_plus_population_count = int((demographics["Transaction_Count"] >= 2).sum())
    eval_manifest_path = root / f"artifacts/experiments/samples/{TASK08_RUN}/two_plus_evaluation_n10000.csv"
    eval_manifest = pd.read_csv(eval_manifest_path, dtype={"CustomerID": "string"})
    eval_ids = eval_manifest["CustomerID"].astype(str).to_numpy()

    # Load and verify the nine exact Task 08 candidate artifacts in the P2 shortlist.
    board = pd.read_csv(root / "artifacts/experiments/comparison_board.csv", dtype={"run_id": "string"})
    specs = _candidate_specs()
    selected_rows: list[tuple[dict[str, Any], pd.Series, dict[str, Any]]] = []
    for spec in specs:
        matches = []
        for _, row in board.iterrows():
            if (
                row["phase"] != "task08_p2_leakage_safe"
                or row["algorithm"] != spec["algorithm"]
                or int(row["k"]) != spec["k"]
                or row["run_status"] != "completed"
            ):
                continue
            parameters = json.loads(row["parameters_json"] or "{}")
            if (
                parameters.get("cohort") == "two_plus"
                and parameters.get("preprocessing_variant") == spec["variant"]
            ):
                matches.append((row, parameters))
        if len(matches) != 3:
            raise ValueError(f"Expected three completed Task 08 candidates for {spec['display_name']}.")
        for row, parameters in matches:
            selected_rows.append((spec, row, parameters))

    raw_aggregates, raw_stats = _aggregate_raw_customer_features(
        root / p2_config["data"]["raw_transactions_file"],
        wanted_ids=set(eval_ids),
        chunk_rows=int(p2_config["data"]["chunk_rows"]),
    )
    p1_interpretable_eval = _load_interpretable_customer_features(
        interpretable_path,
        identifier_column="CustomerID",
        features=feature_order,
        wanted_ids=set(eval_ids),
        chunk_rows=50_000,
    )
    aggregate_validation = _verify_p1_aggregation_semantics(
        raw_aggregates,
        p1_interpretable_eval,
        feature_order=feature_order,
        p1_global_max_date=pd.Timestamp(raw_stats["p1_global_max_date"]),
        as_of_date=pd.Timestamp(AS_OF_DATE),
    )
    evaluation_frame = raw_aggregates.loc[eval_ids].reset_index()
    if not np.array_equal(evaluation_frame["CustomerID"].astype(str).to_numpy(), eval_ids):
        raise RuntimeError("Raw Task 08 evaluation features do not preserve manifest order.")
    evaluation_frame["Recency_Days"] = (
        pd.Timestamp(AS_OF_DATE) - pd.to_datetime(evaluation_frame["Last_Transaction"])
    ).dt.days.astype(np.int64)
    demographic_eval = demographics.set_index("CustomerID").loc[eval_ids].reset_index()

    candidate_inventory: list[dict[str, Any]] = []
    profile_frames: list[pd.DataFrame] = []
    demographic_segment_rows: list[dict[str, Any]] = []
    labels_by_seed: dict[int, list[tuple[dict[str, Any], np.ndarray]]] = defaultdict(list)
    audit_counts = pd.read_csv(
        root
        / f"artifacts/diagnostics/track_a/{TASK08_RUN}/ratio_clip_audit_20261011/ratio_clip_transform_audit.csv"
    )
    expected_clip_count = {42: 101, 2024: 106, 123: 101}
    for spec, row, parameters in selected_rows:
        seed = int(row["seed"])
        train_manifest = pd.read_csv(root / parameters["training_manifest"], dtype={"CustomerID": "string"})
        train_ids = train_manifest["CustomerID"].astype(str).to_numpy()
        eval_rows = pd.read_csv(root / row["assignment_path"], dtype={"CustomerID": "string"})
        eval_assignments = eval_rows.loc[eval_rows["split"] == "evaluation"]
        train_assignments = eval_rows.loc[eval_rows["split"] == "train"]
        if not np.array_equal(train_assignments["CustomerID"].astype(str).to_numpy(), train_ids):
            raise ValueError(f"Task 08 assignment training IDs misaligned for {row['run_id']}.")
        if not np.array_equal(eval_assignments["CustomerID"].astype(str).to_numpy(), eval_ids):
            raise ValueError(f"Task 08 assignment evaluation IDs misaligned for {row['run_id']}.")
        expected_train_hash = ordered_id_sha256(train_ids)
        if expected_train_hash != str(row["train_customer_id_sha256"]):
            raise ValueError(f"Task 08 training-ID hash mismatch for {row['run_id']}.")
        if ordered_id_sha256(eval_ids) != str(row["evaluation_customer_id_sha256"]):
            raise ValueError(f"Task 08 evaluation-ID hash mismatch for {row['run_id']}.")

        artifact_path = root / row["model_path"]
        model_artifact = load_model_artifact(artifact_path)
        pipeline = model_artifact["estimator"]
        processor = pipeline.preprocessor
        if processor.feature_order != feature_order:
            raise ValueError(f"Task 08 preprocessor feature order mismatch for {row['run_id']}.")
        if processor.fit_rows != len(train_ids) or processor.fit_customer_id_sha256 != expected_train_hash:
            raise ValueError(f"Task 08 preprocessor fit scope mismatch for {row['run_id']}.")
        predictions = pipeline.predict_features(evaluation_frame)
        prediction_ids = predictions["CustomerID"].astype(str).to_numpy()
        prediction_labels = predictions["cluster"].astype(np.int32).to_numpy()
        saved_labels = eval_assignments["cluster"].to_numpy(dtype=np.int32)
        if not np.array_equal(prediction_ids, eval_ids) or not np.array_equal(prediction_labels, saved_labels):
            raise ValueError(f"Task 08 saved model reload does not reproduce assignments for {row['run_id']}.")

        run_metadata_path = root / row["run_metadata_path"]
        with run_metadata_path.open(encoding="utf-8") as stream:
            run_metadata = json.load(stream)
        if run_metadata.get("candidate_id") != row["run_id"]:
            raise ValueError(f"Task 08 run metadata candidate mismatch for {row['run_id']}.")
        profile_artifact_path = root / run_metadata["segment_profile_path"]
        if not profile_artifact_path.is_file():
            raise FileNotFoundError(f"Task 08 segment-profile artifact missing: {profile_artifact_path}")
        ledger_clip_count = parameters.get("clip_evaluation_changed_rows")
        if spec["variant"] == "ratio_clip_trainfit":
            audit_row = audit_counts.loc[
                (audit_counts["cohort"] == "two_plus")
                & (audit_counts["variant"] == "ratio_clip_trainfit")
                & (audit_counts["seed"] == seed)
            ]
            if len(audit_row) != 1:
                raise ValueError(f"Task 08 clipping audit row missing for seed={seed}.")
            actual_clip_count = int(audit_row.iloc[0]["evaluation_feature_values_changed"])
            metadata_clip_count = int(run_metadata["clip_evaluation_changed_rows"])
            if actual_clip_count != expected_clip_count[seed] or metadata_clip_count != actual_clip_count:
                raise ValueError(f"Corrected Task 08 clipping metadata mismatch for seed={seed}.")
        else:
            actual_clip_count = 0
        candidate_id = str(row["run_id"])
        candidate = {
            **spec,
            "candidate_id": candidate_id,
            "seed": seed,
            "k": int(row["k"]),
            "algorithm": str(row["algorithm"]),
            "variant": str(parameters["preprocessing_variant"]),
            "training_rows": int(row["training_rows"]),
            "evaluation_rows": int(row["evaluation_rows"]),
                "training_customer_id_sha256": expected_train_hash,
                "evaluation_customer_id_sha256": ordered_id_sha256(eval_ids),
                "metric_space": str(parameters["metric_space"]),
                "feature_order_json": json.dumps(feature_order),
                "training_manifest_path": parameters["training_manifest"],
                "evaluation_manifest_path": parameters["evaluation_manifest"],
        }
        profiles = _profile_candidate(
            candidate=candidate,
            labels=prediction_labels,
            profile_frame=evaluation_frame,
        )
        profile_frames.append(profiles)
        demographic_joined = demographic_eval.copy()
        demographic_segment_rows.extend(
            _demographic_segment_rows(candidate, prediction_labels, demographic_joined)
        )
        labels_by_seed[seed].append((candidate, prediction_labels))
        candidate_inventory.append(
            {
                **candidate,
                "silhouette_score": float(row["silhouette_score"]),
                "davies_bouldin_index": float(row["davies_bouldin_index"]),
                "calinski_harabasz_index": float(row["calinski_harabasz_index"]),
                "smallest_segment_pct": float(row["smallest_segment_pct"]),
                "segment_counts_json": str(row["evaluation_segment_counts_json"]),
                "segment_percentages_json": str(row["evaluation_segment_percentages_json"]),
                "p2_preprocessor_path": parameters["preprocessor_path"],
                "p2_preprocessor_sha256": _sha256(root / parameters["preprocessor_path"]),
                "model_path": str(row["model_path"]),
                "model_sha256": _sha256(artifact_path),
                "assignment_path": str(row["assignment_path"]),
                "assignment_sha256": _sha256(root / row["assignment_path"]),
                "profile_path": run_metadata["segment_profile_path"],
                "run_metadata_path": str(row["run_metadata_path"]),
                "run_metadata_sha256": _sha256(run_metadata_path),
                "profile_artifact_exists": True,
                "run_metadata_eval_clip_count": int(run_metadata["clip_evaluation_changed_rows"]),
                "historical_ledger_eval_clip_count": ledger_clip_count,
                "recomputed_eval_clip_count": actual_clip_count,
                "verified_saved_prediction": True,
                "verified_ordered_ids": True,
                "verified_preprocessor_fit_ids": True,
                "population_coverage_percent": 100 * two_plus_population_count / population_count,
                "cohort_customers": two_plus_population_count,
                "cohort_note": "2+ transactions; approximately 87.1% exactly-two-transaction customers",
            }
        )

    inventory_frame = pd.DataFrame(candidate_inventory)
    profile_frame = pd.concat(profile_frames, ignore_index=True)
    demographic_segment_frame = pd.DataFrame(demographic_segment_rows)
    overlap_frame = _candidate_overlaps(labels_by_seed, eval_ids)
    stability_rows: list[dict[str, Any]] = []
    for family_candidates in zip(*(labels_by_seed[seed] for seed in (42, 2024, 123)), strict=True):
        # Candidate list order follows the three shortlist specifications for each seed.
        left = family_candidates[0][0]
        labels = [item[1] for item in family_candidates]
        for left_index, right_index in itertools.combinations(range(3), 2):
            stability_rows.append(
                {
                    "candidate_family": left["display_name"],
                    "left_seed": (42, 2024, 123)[left_index],
                    "right_seed": (42, 2024, 123)[right_index],
                    "k": left["k"],
                    "left_candidate_id": family_candidates[left_index][0]["candidate_id"],
                    "right_candidate_id": family_candidates[right_index][0]["candidate_id"],
                    "evaluation_customer_id_sha256": ordered_id_sha256(eval_ids),
                    "adjusted_rand_index": float(adjusted_rand_score(labels[left_index], labels[right_index])),
                }
            )

    population_distribution, population_missing, scope_sizes = _population_demographics(
        demographics, eval_ids
    )
    full_2plus = demographics.loc[demographics["Transaction_Count"] >= 2].copy()
    manifest_inventory, manifest_pairwise, manifest_usage, manifest_summary = _manifest_audit(
        root / "artifacts/experiments/samples", population_ids
    )
    p1 = joblib.load(root / "artifacts/preprocessors/track_a.joblib")
    p1_n_samples = int(np.asarray(p1["scaler"].n_samples_seen_).max())
    if p1_n_samples != len(population_ids):
        raise ValueError("P1 fit exposure does not match the P1-validated population size.")
    manifest_summary["p1_scaler_fit_rows"] = p1_n_samples
    manifest_summary["strictly_unexposed_customer_ids_after_p1_scaler"] = 0
    manifest_id_set = set(manifest_usage["CustomerID"].astype(str))
    two_plus_ids = set(
        demographics.loc[demographics["Transaction_Count"] >= 2, "CustomerID"].astype(str)
    )
    manifest_summary["two_plus_population_customers"] = len(two_plus_ids)
    manifest_summary["two_plus_ids_in_any_saved_manifest"] = len(two_plus_ids & manifest_id_set)
    manifest_summary["two_plus_ids_not_in_any_saved_manifest"] = len(two_plus_ids - manifest_id_set)

    events, _, source_meta = _load_raw_eval_and_events(
        root,
        valid_ids=set(population_ids),
        eval_ids=eval_ids,
        chunk_rows=100_000,
    )
    raw_transaction_counts = events.groupby("CustomerID").size().reindex(
        demographics["CustomerID"].astype(str).to_numpy(), fill_value=0
    )
    if not np.array_equal(
        raw_transaction_counts.to_numpy(dtype=np.int64),
        demographics["Transaction_Count"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Complete-case raw transaction counts differ from the P1 validated population.")
    daily_rows = source_meta.pop("daily_complete_case_rows")
    daily_coverage = pd.DataFrame(
        [
            {"transaction_date": day, "complete_case_transaction_rows": rows}
            for day, rows in daily_rows.items()
        ]
    )
    raw_demographic_missingness = _raw_demographic_source_audit(root)
    oot_summary, oot_demographic_shift, oot_history = _oot_feasibility(
        events=events,
        demographics=demographics,
        source_min_date=source_meta["source_min_date"],
        source_max_date=source_meta["source_max_date"],
        full_current_2plus=full_2plus,
        previously_manifested_ids=set(manifest_usage["CustomerID"].astype(str)),
    )

    outputs = {
        "candidate_inventory.csv": inventory_frame,
        "finalist_segment_profiles.csv": profile_frame,
        "finalist_demographic_distribution.csv": demographic_segment_frame,
        "demographic_population_distribution.csv": population_distribution,
        "demographic_missingness.csv": population_missing,
        "raw_source_demographic_missingness.csv": raw_demographic_missingness,
        "candidate_segment_overlap.csv": overlap_frame,
        "candidate_cross_seed_stability.csv": pd.DataFrame(stability_rows),
        "manifest_inventory.csv": manifest_inventory,
        "manifest_pairwise_overlaps.csv": manifest_pairwise,
        "manifest_customer_usage.csv": manifest_usage,
        "raw_source_daily_coverage.csv": daily_coverage,
        "oot_snapshot_feasibility.csv": oot_summary,
        "oot_snapshot_history_sufficiency.csv": oot_history,
        "oot_demographic_population_shift.csv": oot_demographic_shift,
    }
    for filename, frame in outputs.items():
        _write_csv(output_dir / filename, frame)
    summary = {
        "audit_id": output_dir.name,
        "candidate_count": len(inventory_frame),
        "candidate_run_ids": inventory_frame["candidate_id"].tolist(),
        "task08_run_id": TASK08_RUN,
        "task09_run_id": TASK09_RUN,
        "task09_metric_seed_pair_stability": {
            "K-Means k=2": 0.9908202416759926,
            "K-Means k=3": 0.9922902806067037,
        },
        "task07_repeat_only_k2_ari": 0.9434,
        "task07_repeat_only_evaluation_rows": 489,
        "task09_matched_baseline_evaluation_rows": len(eval_ids),
        "population_customers": population_count,
        "two_plus_population_customers": two_plus_population_count,
        "p1_scaler_fit_rows": p1_n_samples,
        "manifest_audit": manifest_summary,
        "task08_evaluation_raw_aggregate_validation": aggregate_validation,
        "demographic_scope_sizes": scope_sizes,
        "raw_source_coverage": source_meta,
        "oot_cutoff_candidates": ["2016-09-01", "2016-09-15", "2016-09-30"],
        "oot_protocol_status": "descriptive feasibility only; no model fit or validation execution",
        "raw_source_demographic_missingness_unit": (
            "Source missingness is transaction-row based; the validated P1 customer profile has a separate customer-row audit."
        ),
        "outputs": {name: str(output_dir / name) for name in outputs},
        "input_hashes": {
            "comparison_board_sha256": _sha256(root / "artifacts/experiments/comparison_board.csv"),
            "interpretable_sha256": _sha256(root / "data/processed/track_a/track_a_features_interpretable.csv"),
            "raw_transactions_sha256": _sha256(root / "data/raw/bank_transactions.csv"),
            "p1_preprocessor_sha256": _sha256(root / "artifacts/preprocessors/track_a.joblib"),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(output_dir / "audit_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/diagnostics/track_a/model_readiness_20261011",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = run_audit(root, root / args.output_dir)
    print(
        f"Task 10A audit {summary['audit_id']}: verified {summary['candidate_count']} finalist candidates; "
        f"manifest audit covers {summary['manifest_audit']['manifest_files_audited']} files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
