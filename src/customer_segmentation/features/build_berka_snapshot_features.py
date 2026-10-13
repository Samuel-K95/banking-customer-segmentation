"""Leakage-safe, account-grain Berka snapshot features for exploratory clustering.

This module is intentionally independent of the established Track C builder. It
does not fit transforms or models and never reads transactions after ``as_of_date``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_VERSION = "berka_account_snapshot_v1"
DEFAULT_AS_OF_DATE = "1997-12-31"
RECENT_DAYS = 90
PRIOR_DAYS = 90

FEATURE_DEFINITIONS: dict[str, str] = {
    "account_tenure_days": "Elapsed days from account opening through as_of_date; accounts are present only if opened by the cutoff.",
    "observation_days": "Calendar days available from account opening through and including as_of_date.",
    "has_90d_history": "1 when at least 90 elapsed days of account tenure are available, otherwise 0.",
    "has_365d_history": "1 when at least 365 elapsed days of account tenure are available, otherwise 0.",
    "transaction_count": "Number of dated transactions on or before as_of_date.",
    "credit_transaction_count": "Count of PRIJEM/CREDIT transactions on or before as_of_date.",
    "debit_transaction_count": "Count of VYDAJ/VYBER/WITHDRAWAL transactions on or before as_of_date.",
    "unclassified_transaction_count": "Count of transactions whose type is not recognized as credit or debit.",
    "credit_frequency_per_day": "Credit count divided by observation_days.",
    "debit_frequency_per_day": "Debit count divided by observation_days.",
    "credit_transaction_share": "Credit count divided by all dated transaction rows; zero when denominator is zero.",
    "debit_transaction_share": "Debit count divided by all dated transaction rows; zero when denominator is zero.",
    "credit_amount_total": "Sum of absolute source amounts for credit-direction transactions.",
    "debit_amount_total": "Sum of absolute source amounts for debit-direction transactions.",
    "net_cash_flow": "Credit amount total minus debit amount total; unclassified types are excluded.",
    "transaction_amount_mean": "Mean absolute source transaction amount across all dated transactions.",
    "transaction_amount_median": "Median absolute source transaction amount across all dated transactions.",
    "transaction_amount_std": "Sample standard deviation of absolute source transaction amounts; missing for fewer than two transactions.",
    "transaction_amount_p90": "90th percentile of absolute source transaction amounts.",
    "transaction_amount_max": "Maximum absolute source transaction amount.",
    "balance_mean": "Mean source balance across dated transactions; negative balances are retained.",
    "balance_median": "Median source balance across dated transactions; negative balances are retained.",
    "balance_std": "Sample standard deviation of source balances; missing for fewer than two transactions.",
    "negative_balance_count": "Number of dated transaction balances below zero.",
    "negative_balance_frequency": "Negative balance count divided by valid balance count; zero when denominator is zero.",
    "interarrival_days_mean": "Mean calendar-day gap between consecutive transaction dates; missing for fewer than two rows.",
    "interarrival_days_median": "Median calendar-day gap between consecutive transaction dates; missing for fewer than two rows.",
    "interarrival_days_std": "Sample standard deviation of calendar-day gaps; missing for fewer than two gaps.",
    "recent_90d_transaction_count": "Transactions in (as_of_date - 90 days, as_of_date], bounded by account opening.",
    "recent_90d_credit_count": "Credit transactions in the recent 90-day window.",
    "recent_90d_debit_count": "Debit transactions in the recent 90-day window.",
    "recent_90d_credit_total": "Absolute credit amount in the recent 90-day window.",
    "recent_90d_debit_total": "Absolute debit amount in the recent 90-day window.",
    "recent_90d_net_cash_flow": "Recent 90-day credit total minus debit total.",
    "recent_window_days_observed": "Number of calendar days available within the recent 90-day window.",
    "recent_90d_transaction_frequency_per_day": "Recent transaction count divided by recent_window_days_observed.",
    "prior_90d_transaction_count": "Transactions in (as_of_date - 180 days, as_of_date - 90 days]; missing unless the full prior 90 days were observable.",
    "prior_90d_credit_total": "Absolute credit amount in the prior 90-day window; missing unless fully observable.",
    "prior_90d_debit_total": "Absolute debit amount in the prior 90-day window; missing unless fully observable.",
    "activity_change_count": "Recent 90-day transaction count minus prior 90-day count; missing unless both windows are observable.",
    "activity_change_ratio": "(Recent count - prior count) / prior count; missing when the prior count is zero or the prior window is incomplete.",
    "recency_days": "Elapsed calendar days from the most recent observed transaction to as_of_date; missing when no transaction is observed.",
}

CREDIT_TYPES = frozenset({"PRIJEM", "CREDIT"})
DEBIT_TYPES = frozenset({"VYDAJ", "VYBER", "WITHDRAWAL"})
REQUIRED_TRANSACTION_COLUMNS = {"account_id", "date", "type", "amount", "balance"}
REQUIRED_ACCOUNT_COLUMNS = {"account_id", "date"}


def _parse_berka_dates(values: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce")
    else:
        text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(text, format="%y%m%d", errors="coerce")
        fallback = parsed.isna() & text.notna()
        if fallback.any():
            parsed.loc[fallback] = pd.to_datetime(text.loc[fallback], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{column_name} contains {int(parsed.isna().sum())} invalid or missing dates")
    return parsed.dt.normalize()


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return zero for a zero denominator; the corresponding count remains available."""
    return float(numerator / denominator) if denominator else 0.0


def build_berka_snapshot_features(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp = DEFAULT_AS_OF_DATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one interpretable row per account that existed by ``as_of_date``.

    Optional source fields such as ``operation``, ``k_symbol``, ``bank`` and
    counterparty ``account`` are deliberately ignored. No ``disp`` join is made:
    account rows remain unique and the source OWNER/DISPONENT role is not expanded.
    """
    missing_tx = REQUIRED_TRANSACTION_COLUMNS.difference(transactions.columns)
    missing_accounts = REQUIRED_ACCOUNT_COLUMNS.difference(accounts.columns)
    if missing_tx or missing_accounts:
        raise ValueError(
            f"Missing required columns: transactions={sorted(missing_tx)}, accounts={sorted(missing_accounts)}"
        )

    cutoff = pd.Timestamp(as_of_date).normalize()
    account_frame = accounts[["account_id", "date"]].copy()
    if account_frame["account_id"].isna().any() or account_frame["account_id"].duplicated().any():
        raise ValueError("accounts.account_id must be non-null and unique")
    account_frame["opened"] = _parse_berka_dates(account_frame["date"], "accounts.date")
    opened_by_cutoff = account_frame.loc[account_frame["opened"].le(cutoff)].copy()
    opened_by_cutoff["account_tenure_days"] = (cutoff - opened_by_cutoff["opened"]).dt.days
    opened_by_cutoff["observation_days"] = opened_by_cutoff["account_tenure_days"] + 1
    opened_by_cutoff["has_90d_history"] = opened_by_cutoff["account_tenure_days"].ge(90).astype("int8")
    opened_by_cutoff["has_365d_history"] = opened_by_cutoff["account_tenure_days"].ge(365).astype("int8")

    tx = transactions.copy()
    tx["_date"] = _parse_berka_dates(tx["date"], "transactions.date")
    transaction_rows_after_cutoff = int(tx["_date"].gt(cutoff).sum())
    tx["_amount"] = pd.to_numeric(tx["amount"], errors="coerce").abs()
    tx["_balance"] = pd.to_numeric(tx["balance"], errors="coerce")
    if tx["_amount"].isna().any() or tx["_balance"].isna().any():
        raise ValueError("transaction amount and balance must be numeric and non-null")
    tx["_type"] = tx["type"].astype("string").str.strip().str.upper()
    tx["_credit"] = tx["_type"].isin(CREDIT_TYPES)
    tx["_debit"] = tx["_type"].isin(DEBIT_TYPES)
    tx["_unclassified"] = ~(tx["_credit"] | tx["_debit"])

    known_accounts = set(account_frame["account_id"])
    unknown_account_rows = int((~tx["account_id"].isin(known_accounts)).sum())
    tx = tx.loc[tx["_date"].le(cutoff)].copy()
    tx = tx.loc[tx["account_id"].isin(opened_by_cutoff["account_id"])].copy()
    opened_map = opened_by_cutoff.set_index("account_id")["opened"]
    if len(tx) and (tx["_date"].lt(tx["account_id"].map(opened_map))).any():
        raise ValueError("transaction date precedes its account opening date")

    tx = tx.sort_values(["account_id", "_date"], kind="mergesort")
    grouped = {account_id: frame for account_id, frame in tx.groupby("account_id", sort=False)}
    rows: list[dict[str, Any]] = []
    recent_start = cutoff - pd.Timedelta(days=RECENT_DAYS)
    prior_start = cutoff - pd.Timedelta(days=RECENT_DAYS + PRIOR_DAYS)
    prior_end = recent_start

    for account in opened_by_cutoff.sort_values("account_id", kind="mergesort").itertuples(index=False):
        account_id = account.account_id
        account_tx = grouped.get(account_id)
        if account_tx is None:
            account_tx = tx.iloc[0:0]
        dates = account_tx["_date"]
        amounts = account_tx["_amount"]
        balances = account_tx["_balance"]
        credits = account_tx["_credit"]
        debits = account_tx["_debit"]
        unknown = account_tx["_unclassified"]
        count = len(account_tx)
        credit_total = float(amounts.loc[credits].sum())
        debit_total = float(amounts.loc[debits].sum())
        recent = account_tx.loc[account_tx["_date"].gt(recent_start)]
        prior = account_tx.loc[account_tx["_date"].gt(prior_start) & account_tx["_date"].le(prior_end)]
        tenure_days = int(account.account_tenure_days)
        prior_complete = tenure_days >= RECENT_DAYS + PRIOR_DAYS
        prior_count: float = float(len(prior)) if prior_complete else np.nan
        prior_credit = float(prior.loc[prior["_credit"], "_amount"].sum()) if prior_complete else np.nan
        prior_debit = float(prior.loc[prior["_debit"], "_amount"].sum()) if prior_complete else np.nan
        recent_days = min(RECENT_DAYS, int(account.observation_days))
        gap_days = dates.diff().dt.days.dropna().astype(float)
        valid_balance_count = int(balances.notna().sum())
        negative_count = int(balances.lt(0).sum())

        row: dict[str, Any] = {
            "account_id": account_id,
            "account_tenure_days": tenure_days,
            "observation_days": int(account.observation_days),
            "has_90d_history": int(account.has_90d_history),
            "has_365d_history": int(account.has_365d_history),
            "transaction_count": count,
            "credit_transaction_count": int(credits.sum()),
            "debit_transaction_count": int(debits.sum()),
            "unclassified_transaction_count": int(unknown.sum()),
            "credit_frequency_per_day": _safe_ratio(int(credits.sum()), int(account.observation_days)),
            "debit_frequency_per_day": _safe_ratio(int(debits.sum()), int(account.observation_days)),
            "credit_transaction_share": _safe_ratio(int(credits.sum()), count),
            "debit_transaction_share": _safe_ratio(int(debits.sum()), count),
            "credit_amount_total": credit_total,
            "debit_amount_total": debit_total,
            "net_cash_flow": credit_total - debit_total,
            "transaction_amount_mean": float(amounts.mean()) if count else np.nan,
            "transaction_amount_median": float(amounts.median()) if count else np.nan,
            "transaction_amount_std": float(amounts.std(ddof=1)) if count > 1 else np.nan,
            "transaction_amount_p90": float(amounts.quantile(0.90)) if count else np.nan,
            "transaction_amount_max": float(amounts.max()) if count else np.nan,
            "balance_mean": float(balances.mean()) if valid_balance_count else np.nan,
            "balance_median": float(balances.median()) if valid_balance_count else np.nan,
            "balance_std": float(balances.std(ddof=1)) if valid_balance_count > 1 else np.nan,
            "negative_balance_count": negative_count,
            "negative_balance_frequency": _safe_ratio(negative_count, valid_balance_count),
            "interarrival_days_mean": float(gap_days.mean()) if len(gap_days) else np.nan,
            "interarrival_days_median": float(gap_days.median()) if len(gap_days) else np.nan,
            "interarrival_days_std": float(gap_days.std(ddof=1)) if len(gap_days) > 1 else np.nan,
            "recent_90d_transaction_count": len(recent),
            "recent_90d_credit_count": int(recent["_credit"].sum()),
            "recent_90d_debit_count": int(recent["_debit"].sum()),
            "recent_90d_credit_total": float(recent.loc[recent["_credit"], "_amount"].sum()),
            "recent_90d_debit_total": float(recent.loc[recent["_debit"], "_amount"].sum()),
            "recent_90d_net_cash_flow": float(recent.loc[recent["_credit"], "_amount"].sum() - recent.loc[recent["_debit"], "_amount"].sum()),
            "recent_window_days_observed": recent_days,
            "recent_90d_transaction_frequency_per_day": _safe_ratio(len(recent), recent_days),
            "prior_90d_transaction_count": prior_count,
            "prior_90d_credit_total": prior_credit,
            "prior_90d_debit_total": prior_debit,
            "activity_change_count": float(len(recent) - prior_count) if prior_complete else np.nan,
            "activity_change_ratio": (float((len(recent) - prior_count) / prior_count) if prior_complete and prior_count else np.nan),
            "recency_days": int((cutoff - dates.max()).days) if count else np.nan,
        }
        rows.append(row)

    features = pd.DataFrame(rows)
    metadata: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "as_of_date": cutoff.date().isoformat(),
        "recent_window": f"({(cutoff - pd.Timedelta(days=RECENT_DAYS)).date().isoformat()}, {cutoff.date().isoformat()}]",
        "prior_window": f"({prior_start.date().isoformat()}, {prior_end.date().isoformat()}]",
        "account_rows_input": len(accounts),
        "accounts_opened_by_cutoff": len(opened_by_cutoff),
        "accounts_opened_after_cutoff_excluded": int(len(account_frame) - len(opened_by_cutoff)),
        "transaction_rows_input": len(transactions),
        "transaction_rows_after_cutoff_excluded": transaction_rows_after_cutoff,
        "transaction_rows_used": len(tx),
        "transaction_rows_for_unknown_accounts": unknown_account_rows,
        "unclassified_transactions_used": int(tx["_unclassified"].sum()),
        "transactions_for_accounts_opened_after_cutoff_excluded": int((transactions["account_id"].isin(account_frame.loc[account_frame["opened"].gt(cutoff), "account_id"])).sum()),
        "amount_direction_policy": "absolute source amount; direction determined by normalized type, not sign",
        "balance_policy": "source sign preserved, including negative balances",
        "zero_denominator_policy": "frequency/share ratios with denominator zero are set to 0; activity_change_ratio is missing when prior count is zero",
        "empty_window_policy": "count/total features are 0; distribution, balance, inter-arrival, and recency features are missing when undefined",
        "optional_transaction_attributes_ignored": [column for column in ("operation", "k_symbol", "bank", "account") if column in transactions.columns],
        "relationship_policy": "account grain from account.csv; disp.csv is not joined. The source has one OWNER per account and some DISPONENT roles; no client-level row expansion is performed.",
        "feature_definitions": FEATURE_DEFINITIONS,
    }
    return features, metadata


def _summary(features: pd.DataFrame) -> dict[str, Any]:
    numeric = features.drop(columns=["account_id"]).select_dtypes(include=["number"])
    return {
        "rows": len(features),
        "columns": len(features.columns),
        "unique_accounts": int(features["account_id"].nunique()),
        "duplicate_account_rows": int(features["account_id"].duplicated().sum()),
        "feature_missingness": {column: int(count) for column, count in numeric.isna().sum().items() if count},
        "feature_distributions": {
            column: {
                "min": None if numeric[column].dropna().empty else float(numeric[column].min()),
                "mean": None if numeric[column].dropna().empty else float(numeric[column].mean()),
                "median": None if numeric[column].dropna().empty else float(numeric[column].median()),
                "max": None if numeric[column].dropna().empty else float(numeric[column].max()),
            }
            for column in numeric.columns
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def generate_berka_snapshot(
    *,
    data_dir: str | Path = "data/raw/berka",
    output_dir: str | Path | None = None,
    as_of_date: str | pd.Timestamp = DEFAULT_AS_OF_DATE,
) -> tuple[Path, Path]:
    """Generate a new, non-overwriting CSV and JSON metadata pair from raw Berka tables."""
    source = Path(data_dir)
    cutoff = pd.Timestamp(as_of_date).normalize()
    output = Path(output_dir) if output_dir is not None else (
        Path("data/processed/berka_snapshot/v1") / f"asof_{cutoff.date().isoformat()}"
    )
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "berka_account_snapshot_features.csv"
    metadata_path = output / "berka_account_snapshot_metadata.json"
    if feature_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Berka snapshot output in {output}")
    transactions = pd.read_csv(source / "trans.csv", sep=";", dtype={"date": "string"}, low_memory=False)
    accounts = pd.read_csv(source / "account.csv", sep=";", dtype={"date": "string"}, low_memory=False)
    features, metadata = build_berka_snapshot_features(transactions, accounts, as_of_date=as_of_date)
    metadata["source_tables"] = {"transactions": str(source / "trans.csv"), "accounts": str(source / "account.csv")}
    metadata["summary"] = _summary(features)
    features.to_csv(feature_path, index=False, float_format="%.10g")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    return feature_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe account-grain Berka snapshot features.")
    parser.add_argument("--data-dir", default="data/raw/berka")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    args = parser.parse_args()
    feature_path, metadata_path = generate_berka_snapshot(
        data_dir=args.data_dir, output_dir=args.output_dir, as_of_date=args.as_of_date
    )
    print(f"Saved interpretable features: {feature_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
