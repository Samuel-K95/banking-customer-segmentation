"""Frozen-assignment out-of-period validation for Berka account segments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from customer_segmentation.models.berka_behavioral_baselines import (
    BerkaTrainFittedPreprocessor,
    _sha256_ids,
    load_snapshot,
    select_behavioral_features,
)

DEFAULT_CONFIG = Path("configs/models/berka/berka_temporal_validation_v1.yaml")
WINDOWS = {
    "prior_365d": (pd.Timestamp("1997-01-01"), pd.Timestamp("1997-12-31")),
    "outcome_1998": (pd.Timestamp("1998-01-01"), pd.Timestamp("1998-12-31")),
}
ALGORITHMS = ("kmeans", "gaussian_mixture")
BOOTSTRAP_FEATURES = [
    "transaction_count",
    "active_months",
    "credit_transaction_count",
    "debit_transaction_count",
    "credit_amount_total",
    "debit_amount_total",
    "net_cash_flow",
    "avg_monthly_transaction_frequency",
    "avg_transaction_frequency_per_active_month",
    "median_transaction_amount",
    "median_balance",
    "balance_volatility",
    "negative_balance_frequency",
    "interarrival_median_days",
    "interarrival_std_days",
]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_berka_dates(values: pd.Series) -> pd.Series:
    """Parse Berka YYMMDD strings, with ISO-date fallback for synthetic inputs."""
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce")
    else:
        text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(text, format="%y%m%d", errors="coerce")
        fallback = parsed.isna() & text.notna()
        if fallback.any():
            parsed.loc[fallback] = pd.to_datetime(text.loc[fallback], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Found {int(parsed.isna().sum())} missing or invalid Berka transaction dates")
    return parsed.dt.normalize()


def filter_comparison_windows(transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only the two prespecified, adjacent 365-day outcome windows."""
    required = {"account_id", "date", "type", "amount", "balance"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Transaction input is missing required columns: {sorted(missing)}")
    frame = transactions.copy()
    frame["_date"] = parse_berka_dates(frame["date"])
    frame["_window"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    counts: dict[str, int] = {}
    for name, (start, end) in WINDOWS.items():
        mask = frame["_date"].between(start, end, inclusive="both")
        frame.loc[mask, "_window"] = name
        counts[name] = int(mask.sum())
    return frame.loc[frame["_window"].notna()].copy(), counts


def aggregate_period_outcomes(
    transactions: pd.DataFrame,
    account_ids: np.ndarray | list[Any],
    *,
    period_start: str | pd.Timestamp,
    period_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return one outcome row per requested account, retaining inactive IDs.

    Counts and amount totals are zero for accounts with no transactions. Statistics
    that require transactions or balances remain NaN when undefined.
    """
    ids = np.asarray(account_ids)
    if ids.ndim != 1 or pd.isna(ids).any() or len(pd.unique(ids)) != len(ids):
        raise ValueError("account_ids must be a unique, non-null one-dimensional sequence")
    required = {"account_id", "date", "type", "amount", "balance"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Transaction input is missing required columns: {sorted(missing)}")
    start = pd.Timestamp(period_start).normalize()
    end = pd.Timestamp(period_end).normalize()
    if start > end:
        raise ValueError("period_start must not follow period_end")

    frame = transactions.copy()
    frame["_date"] = frame["_date"] if "_date" in frame else parse_berka_dates(frame["date"])
    frame["_amount"] = pd.to_numeric(frame["amount"], errors="coerce").abs()
    frame["_balance"] = pd.to_numeric(frame["balance"], errors="coerce")
    if frame["_amount"].isna().any():
        raise ValueError("Transaction amounts must be numeric and non-null")
    frame["_type"] = frame["type"].astype("string").str.strip().str.upper()
    frame["_credit"] = frame["_type"].isin({"PRIJEM", "CREDIT"})
    frame["_debit"] = frame["_type"].isin({"VYDAJ", "VYBER", "WITHDRAWAL"})
    frame["_unknown_direction"] = ~(frame["_credit"] | frame["_debit"])
    mask = (
        frame["_date"].between(start, end, inclusive="both")
        & frame["account_id"].isin(ids)
    )
    period = frame.loc[mask].copy()
    sort_columns = ["account_id", "_date"]
    if "trans_id" in period.columns:
        sort_columns.append("trans_id")
    period = period.sort_values(sort_columns, kind="mergesort")
    by_account = {account_id: rows for account_id, rows in period.groupby("account_id", sort=False)}

    rows: list[dict[str, Any]] = []
    for account_id in ids:
        account_tx = by_account.get(account_id)
        if account_tx is None:
            rows.append(
                {
                    "account_id": account_id,
                    "transaction_count": 0,
                    "active_months": 0,
                    "credit_transaction_count": 0,
                    "debit_transaction_count": 0,
                    "unknown_direction_transaction_count": 0,
                    "credit_amount_total": 0.0,
                    "debit_amount_total": 0.0,
                    "net_cash_flow": 0.0,
                    "avg_monthly_transaction_frequency": 0.0,
                    "avg_transaction_frequency_per_active_month": np.nan,
                    "median_transaction_amount": np.nan,
                    "median_balance": np.nan,
                    "balance_volatility": np.nan,
                    "negative_balance_count": 0,
                    "negative_balance_frequency": np.nan,
                    "valid_balance_count": 0,
                    "interarrival_gap_count": 0,
                    "interarrival_median_days": np.nan,
                    "interarrival_std_days": np.nan,
                }
            )
            continue

        amount = account_tx["_amount"]
        credits = account_tx["_credit"]
        debits = account_tx["_debit"]
        balances = account_tx["_balance"]
        valid_balances = balances.dropna()
        gap_days = account_tx["_date"].diff().dt.days.dropna().astype(float)
        count = len(account_tx)
        credit_total = float(amount.loc[credits].sum())
        debit_total = float(amount.loc[debits].sum())
        active_months = int(account_tx["_date"].dt.to_period("M").nunique())
        negative_count = int(valid_balances.lt(0).sum())
        rows.append(
            {
                "account_id": account_id,
                "transaction_count": count,
                "active_months": active_months,
                "credit_transaction_count": int(credits.sum()),
                "debit_transaction_count": int(debits.sum()),
                "unknown_direction_transaction_count": int(account_tx["_unknown_direction"].sum()),
                "credit_amount_total": credit_total,
                "debit_amount_total": debit_total,
                "net_cash_flow": credit_total - debit_total,
                "avg_monthly_transaction_frequency": count / 12.0,
                "avg_transaction_frequency_per_active_month": count / active_months if active_months else np.nan,
                "median_transaction_amount": float(amount.median()) if amount.notna().any() else np.nan,
                "median_balance": float(valid_balances.median()) if len(valid_balances) else np.nan,
                "balance_volatility": float(valid_balances.std(ddof=1)) if len(valid_balances) > 1 else np.nan,
                "negative_balance_count": negative_count,
                "negative_balance_frequency": float(negative_count / len(valid_balances)) if len(valid_balances) else np.nan,
                "valid_balance_count": len(valid_balances),
                "interarrival_gap_count": len(gap_days),
                "interarrival_median_days": float(gap_days.median()) if len(gap_days) else np.nan,
                "interarrival_std_days": float(gap_days.std(ddof=1)) if len(gap_days) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def frozen_assignments_for_population(
    assignments: pd.DataFrame,
    account_ids: np.ndarray | list[Any],
    *,
    algorithm: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Select and order saved held-out cluster labels without refitting/relabeling."""
    required = {"account_id", "algorithm", "k", "seed", "cluster"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Saved assignments are missing columns: {sorted(missing)}")
    ids = np.asarray(account_ids)
    selected = assignments.loc[
        assignments["algorithm"].eq(algorithm)
        & assignments["k"].eq(2)
        & assignments["seed"].eq(seed),
        ["account_id", "cluster"],
    ].copy()
    if selected["account_id"].isna().any() or selected["account_id"].duplicated().any():
        raise ValueError("Frozen assignments must have unique, non-null account IDs")
    if set(selected["account_id"]) != set(ids):
        raise ValueError(f"Frozen {algorithm}/seed{seed} assignments do not match held-out account IDs")
    ordered = selected.set_index("account_id").loc[ids].reset_index()
    if ordered["cluster"].isna().any():
        raise ValueError("Frozen cluster assignments cannot be missing")
    return ordered


def _bootstrap_median_interval(
    values: np.ndarray | pd.Series,
    *,
    seed: int,
    repetitions: int,
    confidence_level: float,
) -> tuple[int, float | None, float | None]:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    n = len(observed)
    if n == 0:
        return 0, None, None
    if not 0 < confidence_level < 1 or repetitions < 1:
        raise ValueError("Bootstrap confidence_level and repetitions are invalid")
    if n == 1:
        return 1, float(observed[0]), float(observed[0])
    rng = np.random.default_rng(seed)
    medians = np.median(observed[rng.integers(0, n, size=(repetitions, n))], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(medians, [alpha, 1.0 - alpha])
    return n, float(lower), float(upper)


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def _summary_tables(
    outcomes: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome_features = [
        name
        for name in BOOTSTRAP_FEATURES
        if f"prior_365d_{name}" in outcomes and f"outcome_1998_{name}" in outcomes
    ]
    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    yoy_rows: list[dict[str, Any]] = []
    id_year_change_columns = {
        feature: (f"prior_365d_{feature}", f"outcome_1998_{feature}")
        for feature in outcome_features
    }
    for algorithm in ALGORITHMS:
        cluster_column = f"{algorithm}_cluster"
        for cluster in sorted(outcomes[cluster_column].dropna().unique()):
            members = outcomes.loc[outcomes[cluster_column].eq(cluster)]
            for period_name, prefix in (("prior_365d", "prior_365d_"), ("outcome_1998", "outcome_1998_")):
                for feature in outcome_features:
                    values = pd.to_numeric(members[f"{prefix}{feature}"], errors="coerce")
                    valid = values.dropna()
                    seed = _stable_seed(bootstrap_seed, "segment_median", algorithm, str(cluster), period_name, feature)
                    _n_boot, ci_low, ci_high = _bootstrap_median_interval(
                        valid.to_numpy(), seed=seed, repetitions=bootstrap_repetitions,
                        confidence_level=confidence_level,
                    )
                    summary_rows.append(
                        {
                            "algorithm": algorithm,
                            "cluster": int(cluster),
                            "period": period_name,
                            "outcome": feature,
                            "accounts_in_segment": len(members),
                            "accounts_with_transactions": int(members[f"{prefix}transaction_count"].gt(0).sum()),
                            "accounts_defined": len(valid),
                            "accounts_missing": int(values.isna().sum()),
                            "mean": float(valid.mean()) if len(valid) else np.nan,
                            "std": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                            "p10": float(valid.quantile(0.10)) if len(valid) else np.nan,
                            "p25": float(valid.quantile(0.25)) if len(valid) else np.nan,
                            "median": float(valid.median()) if len(valid) else np.nan,
                            "p75": float(valid.quantile(0.75)) if len(valid) else np.nan,
                            "p90": float(valid.quantile(0.90)) if len(valid) else np.nan,
                            "median_ci95_low": ci_low,
                            "median_ci95_high": ci_high,
                        }
                    )

            for feature, (prior_col, future_col) in id_year_change_columns.items():
                prior = pd.to_numeric(members[prior_col], errors="coerce")
                future = pd.to_numeric(members[future_col], errors="coerce")
                paired = prior.notna() & future.notna()
                paired_delta = future.loc[paired] - prior.loc[paired]
                seed = _stable_seed(bootstrap_seed, "paired_delta", algorithm, str(cluster), feature)
                _n_boot, ci_low, ci_high = _bootstrap_median_interval(
                    paired_delta.to_numpy(), seed=seed, repetitions=bootstrap_repetitions,
                    confidence_level=confidence_level,
                )
                pct_change = pd.Series(np.nan, index=members.index, dtype=float)
                if feature != "net_cash_flow":
                    pct_valid = paired & prior.ne(0)
                    pct_change.loc[pct_valid] = (
                        100.0 * (future.loc[pct_valid] - prior.loc[pct_valid]) / prior.loc[pct_valid].abs()
                    )
                yoy_rows.append(
                    {
                        "algorithm": algorithm,
                        "cluster": int(cluster),
                        "outcome": feature,
                        "accounts_in_segment": len(members),
                        "accounts_paired": int(paired.sum()),
                        "accounts_with_defined_pct_change": int(pct_change.notna().sum()),
                        "median_prior_365d": float(prior.dropna().median()) if prior.notna().any() else np.nan,
                        "median_outcome_1998": float(future.dropna().median()) if future.notna().any() else np.nan,
                        "median_paired_change": float(paired_delta.median()) if len(paired_delta) else np.nan,
                        "median_paired_change_ci95_low": ci_low,
                        "median_paired_change_ci95_high": ci_high,
                        "median_account_pct_change": float(pct_change.median()) if pct_change.notna().any() else np.nan,
                    }
                )

        # Segment 0 minus segment 1 contrasts, without changing their frozen labels.
        cluster0 = outcomes.loc[outcomes[cluster_column].eq(0)]
        cluster1 = outcomes.loc[outcomes[cluster_column].eq(1)]
        for period_name, prefix in (("prior_365d", "prior_365d_"), ("outcome_1998", "outcome_1998_")):
            for feature in outcome_features:
                left = pd.to_numeric(cluster0[f"{prefix}{feature}"], errors="coerce").dropna().to_numpy()
                right = pd.to_numeric(cluster1[f"{prefix}{feature}"], errors="coerce").dropna().to_numpy()
                contrast = float(np.median(left) - np.median(right)) if len(left) and len(right) else np.nan
                if len(left) and len(right) and len(left) > 1 and len(right) > 1:
                    seed = _stable_seed(bootstrap_seed, "segment_contrast", algorithm, period_name, feature)
                    rng = np.random.default_rng(seed)
                    boot_left = np.median(left[rng.integers(0, len(left), size=(bootstrap_repetitions, len(left)))], axis=1)
                    boot_right = np.median(right[rng.integers(0, len(right), size=(bootstrap_repetitions, len(right)))], axis=1)
                    alpha = (1.0 - confidence_level) / 2.0
                    ci_low, ci_high = np.quantile(boot_left - boot_right, [alpha, 1.0 - alpha])
                elif len(left) and len(right):
                    ci_low = ci_high = contrast
                else:
                    ci_low = ci_high = np.nan
                contrast_rows.append(
                    {
                        "algorithm": algorithm,
                        "period": period_name,
                        "outcome": feature,
                        "cluster0_accounts_defined": len(left),
                        "cluster1_accounts_defined": len(right),
                        "cluster0_median": float(np.median(left)) if len(left) else np.nan,
                        "cluster1_median": float(np.median(right)) if len(right) else np.nan,
                        "cluster0_minus_cluster1_median": contrast,
                        "difference_ci95_low": float(ci_low),
                        "difference_ci95_high": float(ci_high),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows), pd.DataFrame(yoy_rows)


def _read_held_out_transactions(raw_path: Path, account_ids: np.ndarray, chunk_rows: int = 100_000) -> tuple[pd.DataFrame, dict[str, int]]:
    selected_ids = set(account_ids.tolist())
    parts: list[pd.DataFrame] = []
    invalid_dates = 0
    rows_by_period = {name: 0 for name in WINDOWS}
    usecols = ["trans_id", "account_id", "date", "type", "amount", "balance"]
    for chunk in pd.read_csv(
        raw_path,
        sep=";",
        usecols=usecols,
        dtype={"date": "string", "type": "string"},
        chunksize=chunk_rows,
        low_memory=False,
    ):
        parsed = parse_berka_dates(chunk["date"])
        chunk["_date"] = parsed
        in_window = parsed.between(WINDOWS["prior_365d"][0], WINDOWS["outcome_1998"][1], inclusive="both")
        held_out = chunk["account_id"].isin(selected_ids)
        chosen = chunk.loc[in_window & held_out].copy()
        if chosen.empty:
            continue
        for name, (start, end) in WINDOWS.items():
            rows_by_period[name] += int(chosen["_date"].between(start, end, inclusive="both").sum())
        parts.append(chosen[usecols + ["_date"]])
    if not parts:
        return pd.DataFrame(columns=usecols + ["_date"]), {
            **rows_by_period,
            "invalid_dates": invalid_dates,
        }
    transactions = pd.concat(parts, ignore_index=True)
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce")
    transactions["balance"] = pd.to_numeric(transactions["balance"], errors="coerce")
    if transactions["amount"].isna().any():
        raise ValueError("Held-out period source contains nonnumeric or missing transaction amounts")
    return transactions, {**rows_by_period, "invalid_dates": invalid_dates}


def _verify_frozen_baseline(
    *,
    baseline_directory: Path,
    baseline_config_path: Path,
    snapshot_directory: Path,
    held_out_ids: np.ndarray,
    training_ids: np.ndarray,
    assignments: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    baseline_config = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    run_metadata = json.loads((baseline_directory / "run_metadata.json").read_text(encoding="utf-8"))
    if run_metadata["as_of_date"] != "1997-12-31" or baseline_config["data"]["as_of_date"] != "1997-12-31":
        raise ValueError("Baseline model is not tied to the 1997-12-31 snapshot")
    if _sha256_ids(training_ids) != run_metadata["training_account_ids_sha256"]:
        raise ValueError("Training IDs do not match the frozen baseline training hash")
    if _sha256_ids(held_out_ids) != run_metadata["held_out_account_ids_sha256"]:
        raise ValueError("Held-out IDs do not match the frozen baseline evaluation hash")
    if len(training_ids) != 2701 or len(held_out_ids) != 901:
        raise ValueError("Frozen split dimensions differ from the validated baseline")

    preprocessor_path = baseline_directory / "preprocessor_train_fitted.joblib"
    preprocessor_artifact = joblib.load(preprocessor_path)
    preprocessor = BerkaTrainFittedPreprocessor.from_state_dict(preprocessor_artifact["preprocessor_state"])
    if preprocessor.metadata() != run_metadata["preprocessor_metadata"]:
        raise ValueError("Loaded preprocessing state does not match the baseline run metadata")
    prep_metadata = preprocessor.metadata()
    if prep_metadata["fit_rows"] != len(training_ids) or prep_metadata["fit_account_ids_sha256"] != _sha256_ids(training_ids):
        raise ValueError("Preprocessor training provenance does not match the frozen training IDs")

    snapshot, snapshot_metadata = load_snapshot(snapshot_directory, expected_as_of_date="1997-12-31")
    if snapshot_metadata["as_of_date"] != "1997-12-31":
        raise ValueError("Snapshot metadata date mismatch")
    features = list(baseline_config["features"]["selected"])
    ordered_held = snapshot.set_index("account_id").loc[held_out_ids].reset_index()
    model_hashes: dict[str, Any] = {
        "preprocessor": {
            "path": str(preprocessor_path),
            "sha256": _sha256_file(preprocessor_path),
            "feature_order": prep_metadata["feature_order"],
            "fit_rows": prep_metadata["fit_rows"],
            "fit_account_ids_sha256": prep_metadata["fit_account_ids_sha256"],
            "fit_scope": prep_metadata["fit_scope"],
        }
    }
    frozen: dict[str, pd.DataFrame] = {}
    for algorithm in ALGORITHMS:
        model_path = baseline_directory / "models" / f"{algorithm}_k2_seed42.joblib"
        model_artifact = joblib.load(model_path)
        estimator = model_artifact["estimator"]
        model_metadata = model_artifact["metadata"]
        if model_metadata["algorithm"] != algorithm or model_metadata["k"] != 2 or model_metadata["seed"] != 42:
            raise ValueError(f"Unexpected frozen model metadata for {algorithm}")
        if model_metadata["feature_order"] != features:
            raise ValueError(f"Frozen model feature order mismatch for {algorithm}")
        if model_metadata["training_account_ids_sha256"] != _sha256_ids(training_ids):
            raise ValueError(f"Frozen {algorithm} model training hash mismatch")
        if model_metadata["held_out_account_ids_sha256"] != _sha256_ids(held_out_ids):
            raise ValueError(f"Frozen {algorithm} model held-out hash mismatch")
        if algorithm == "kmeans" and (estimator.n_clusters != 2 or estimator.random_state != 42):
            raise ValueError("Frozen K-Means estimator parameters do not match seed 42/k=2")
        if algorithm == "gaussian_mixture" and (
            estimator.n_components != 2 or estimator.covariance_type != "diag" or estimator.random_state != 42
        ):
            raise ValueError("Frozen GMM estimator parameters do not match seed 42/k=2/diag")
        selected = select_behavioral_features(ordered_held, features)
        transformed = preprocessor.transform(selected)
        replay = estimator.predict(transformed)
        frozen_assignment = frozen_assignments_for_population(
            assignments, held_out_ids, algorithm=algorithm, seed=42
        )
        if not np.array_equal(replay, frozen_assignment["cluster"].to_numpy()):
            raise ValueError(f"Frozen model replay differs from saved held-out {algorithm} assignments")
        frozen[algorithm] = frozen_assignment
        model_hashes[algorithm] = {
            "path": str(model_path),
            "sha256": _sha256_file(model_path),
            "algorithm": algorithm,
            "k": 2,
            "seed": 42,
            "training_account_ids_sha256": model_metadata["training_account_ids_sha256"],
            "held_out_account_ids_sha256": model_metadata["held_out_account_ids_sha256"],
            "replay_matches_frozen_assignments": True,
        }
    model_hashes["baseline_assignment_file"] = {
        "path": str(baseline_directory / "held_out_assignments.csv"),
        "sha256": _sha256_file(baseline_directory / "held_out_assignments.csv"),
    }
    model_hashes["input_files"] = {
        "snapshot_features_sha256": _sha256_file(snapshot_directory / "berka_account_snapshot_features.csv"),
        "snapshot_metadata_sha256": _sha256_file(snapshot_directory / "berka_account_snapshot_metadata.json"),
        "baseline_run_metadata_sha256": _sha256_file(baseline_directory / "run_metadata.json"),
        "baseline_config_sha256": _sha256_file(baseline_config_path),
    }
    model_hashes["cross_algorithm_held_out_ari"] = float(
        adjusted_rand_score(frozen["kmeans"]["cluster"], frozen["gaussian_mixture"]["cluster"])
    )
    return frozen, model_hashes


def run_temporal_validation(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Path]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["artifacts"]["output_directory"])
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Berka temporal validation output: {output_root}")
    snapshot_directory = Path(config["snapshot"]["feature_directory"])
    baseline_config_path = Path(config["baseline"]["configuration"])
    baseline_directory = Path(config["baseline"]["artifact_directory"])
    snapshot, _ = load_snapshot(snapshot_directory, expected_as_of_date=config["snapshot"]["as_of_date"])
    split = pd.read_csv(baseline_directory / config["population"]["split_manifest"])
    if split["account_id"].duplicated().any() or split["account_id"].isna().any():
        raise ValueError("Saved baseline split contains duplicate or missing account IDs")
    eligible = snapshot.loc[
        snapshot["account_tenure_days"].ge(int(config["population"]["minimum_tenure_days"]))
    ]
    if set(split["account_id"]) != set(eligible["account_id"]):
        raise ValueError("Saved training/held-out manifest does not cover the eligible account population")
    training_ids = split.loc[split["split"].eq("train"), "account_id"].to_numpy()
    held_out_ids = split.loc[split["split"].eq("held_out"), "account_id"].to_numpy()
    if len(eligible) != 3602 or len(training_ids) != 2701 or len(held_out_ids) != 901:
        raise ValueError("Population or split dimensions do not match the requested Berka baseline cohort")

    baseline_assignments_path = baseline_directory / "held_out_assignments.csv"
    saved_assignments = pd.read_csv(baseline_assignments_path)
    frozen_assignments, model_hashes = _verify_frozen_baseline(
        baseline_directory=baseline_directory,
        baseline_config_path=baseline_config_path,
        snapshot_directory=snapshot_directory,
        held_out_ids=held_out_ids,
        training_ids=training_ids,
        assignments=saved_assignments,
    )

    raw_path = Path("data/raw/berka/trans.csv")
    period_transactions, period_counts = _read_held_out_transactions(raw_path, held_out_ids)
    prior_start, prior_end = WINDOWS["prior_365d"]
    outcome_start, outcome_end = WINDOWS["outcome_1998"]
    prior = aggregate_period_outcomes(
        period_transactions, held_out_ids, period_start=prior_start, period_end=prior_end
    ).add_prefix("prior_365d_").rename(columns={"prior_365d_account_id": "account_id"})
    future = aggregate_period_outcomes(
        period_transactions, held_out_ids, period_start=outcome_start, period_end=outcome_end
    ).add_prefix("outcome_1998_").rename(columns={"outcome_1998_account_id": "account_id"})
    outcomes = pd.DataFrame({"account_id": held_out_ids})
    for algorithm in ALGORITHMS:
        frozen = frozen_assignments[algorithm].set_index("account_id").loc[held_out_ids]
        outcomes[f"{algorithm}_cluster"] = frozen["cluster"].to_numpy(dtype=int)
    outcomes = outcomes.merge(prior, on="account_id", validate="one_to_one")
    outcomes = outcomes.merge(future, on="account_id", validate="one_to_one")
    for feature in BOOTSTRAP_FEATURES:
        prior_col = f"prior_365d_{feature}"
        future_col = f"outcome_1998_{feature}"
        if prior_col not in outcomes or future_col not in outcomes:
            continue
        outcomes[f"change_1998_minus_prior_{feature}"] = outcomes[future_col] - outcomes[prior_col]
        if feature != "net_cash_flow":
            valid = outcomes[prior_col].notna() & outcomes[prior_col].ne(0) & outcomes[future_col].notna()
            pct_column = f"yoy_pct_change_{feature}"
            outcomes[pct_column] = np.nan
            outcomes.loc[valid, pct_column] = (
                100.0
                * (outcomes.loc[valid, future_col] - outcomes.loc[valid, prior_col])
                / outcomes.loc[valid, prior_col].abs()
            )

    outcome_cols = [column for column in outcomes if column not in {"account_id", "kmeans_cluster", "gaussian_mixture_cluster"}]
    numeric = outcomes[outcome_cols].select_dtypes(include=["number"]).to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Outcome aggregation produced infinite values")
    summary, contrasts, yoy = _summary_tables(
        outcomes,
        bootstrap_repetitions=int(config["uncertainty"]["repetitions"]),
        bootstrap_seed=int(config["uncertainty"]["seed"]),
        confidence_level=float(config["uncertainty"]["confidence_level"]),
    )

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    account_outcomes_path = output_root / "held_out_account_outcomes.csv"
    outcomes.to_csv(account_outcomes_path, index=False)
    frozen_path = output_root / "frozen_seed42_assignments.csv"
    frozen_rows = []
    for algorithm, frame in frozen_assignments.items():
        frozen_rows.append(frame.assign(algorithm=algorithm, seed=42, k=2))
    pd.concat(frozen_rows, ignore_index=True).to_csv(frozen_path, index=False)
    summary_path = output_root / "segment_period_outcome_summary.csv"
    summary.to_csv(summary_path, index=False)
    contrast_path = output_root / "segment_contrast_bootstrap_ci.csv"
    contrasts.to_csv(contrast_path, index=False)
    yoy_path = output_root / "segment_year_over_year_changes.csv"
    yoy.to_csv(yoy_path, index=False)
    model_hashes_path = output_root / "frozen_model_integrity.json"
    model_hashes_path.write_text(json.dumps(model_hashes, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    metadata = {
        "experiment": config["experiment"],
        "snapshot_as_of_date": config["snapshot"]["as_of_date"],
        "prior_window_inclusive": [prior_start.date().isoformat(), prior_end.date().isoformat()],
        "outcome_window_inclusive": [outcome_start.date().isoformat(), outcome_end.date().isoformat()],
        "eligible_accounts": len(eligible),
        "training_accounts": len(training_ids),
        "held_out_accounts": len(held_out_ids),
        "training_ids_sha256": _sha256_ids(training_ids.astype(str)),
        "held_out_ids_sha256": _sha256_ids(held_out_ids.astype(str)),
        "frozen_assignment_seed": 42,
        "frozen_algorithms": list(ALGORITHMS),
        "frozen_k": 2,
        "transaction_source": str(raw_path),
        "transaction_source_sha256": _sha256_file(raw_path),
        "transaction_window_rows_for_held_out_accounts": period_counts,
        "inactive_1998_accounts": int(outcomes["outcome_1998_transaction_count"].eq(0).sum()),
        "bootstrap_repetitions": int(config["uncertainty"]["repetitions"]),
        "bootstrap_seed": int(config["uncertainty"]["seed"]),
        "bootstrap_confidence_level": float(config["uncertainty"]["confidence_level"]),
        "use_of_1998_for_model_fitting_or_assignment": "none",
        "model_integrity_path": str(model_hashes_path),
        "output_files": {
            "account_outcomes": str(account_outcomes_path),
            "frozen_assignments": str(frozen_path),
            "segment_period_summary": str(summary_path),
            "segment_contrasts": str(contrast_path),
            "year_over_year_changes": str(yoy_path),
        },
    }
    metadata_path = output_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "output_directory": output_root,
        "account_outcomes": account_outcomes_path,
        "frozen_assignments": frozen_path,
        "segment_summary": summary_path,
        "segment_contrasts": contrast_path,
        "year_over_year": yoy_path,
        "model_integrity": model_hashes_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen Berka segments against 1998 behavior.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    paths = run_temporal_validation(args.config)
    print(f"Completed Berka temporal validation: {paths['output_directory']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
