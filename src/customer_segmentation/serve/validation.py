"""Strict inference validation: feature schema + eligibility gates."""

from __future__ import annotations

import math
from typing import Any

from fastapi import HTTPException

from customer_segmentation.serve.schemas import FEATURE_FIELDS


ALLOWED_EXTRA = frozenset({"account_id"})


def _as_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field} must be numeric") from exc
    if not math.isfinite(out):
        raise HTTPException(status_code=400, detail=f"{field} must be finite")
    return out


def validate_feature_payload(
    row: dict[str, Any],
    *,
    required_features: tuple[str, ...] | list[str] = FEATURE_FIELDS,
    minimum_tenure_days: int = 365,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Validate one interpretable feature row; return cleaned dict.

    Raises HTTPException(400) on schema or eligibility failures.
    """
    prefix = f"Row {row_index}: " if row_index is not None else ""
    if not isinstance(row, dict):
        raise HTTPException(status_code=400, detail=f"{prefix}expected object")

    unknown = [k for k in row.keys() if k not in required_features and k not in ALLOWED_EXTRA]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"{prefix}unknown feature keys not allowed: {unknown}",
        )

    cleaned: dict[str, Any] = {}
    if "account_id" in row and row["account_id"] not in (None, ""):
        cleaned["account_id"] = row["account_id"]

    for col in required_features:
        if col not in row or row[col] in (None, ""):
            raise HTTPException(status_code=400, detail=f"{prefix}missing required feature: {col}")
        cleaned[col] = _as_float(row[col], field=f"{prefix}{col}")

    for flag in ("has_loan", "has_card"):
        val = cleaned[flag]
        if val not in (0.0, 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"{prefix}{flag} must be 0 or 1 (got {val})",
            )

    if cleaned["Recency_Days"] < 0:
        raise HTTPException(status_code=400, detail=f"{prefix}Recency_Days must be >= 0")

    if cleaned["Account_Age_Days"] < float(minimum_tenure_days):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{prefix}ineligible: Account_Age_Days={cleaned['Account_Age_Days']} "
                f"< minimum_tenure_days={minimum_tenure_days}"
            ),
        )

    return cleaned


def validate_feature_rows(
    rows: list[dict[str, Any]],
    *,
    required_features: tuple[str, ...] | list[str] = FEATURE_FIELDS,
    minimum_tenure_days: int = 365,
) -> list[dict[str, Any]]:
    return [
        validate_feature_payload(
            row,
            required_features=required_features,
            minimum_tenure_days=minimum_tenure_days,
            row_index=i,
        )
        for i, row in enumerate(rows)
    ]
