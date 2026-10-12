"""Shared scoring helpers for inference endpoints."""

from __future__ import annotations

from typing import Any

import pandas as pd

from customer_segmentation.decision.applications import APPLICATION_PORTFOLIO
from customer_segmentation.serve.deps import serving_contract, validate_contract
from customer_segmentation.serve.schemas import FeatureRow


def score_feature_rows(
    champ: Any,
    customers: list[FeatureRow],
    *,
    model_version: str,
    as_of_date: str,
    include_applications: bool = False,
) -> dict[str, Any]:
    contract = validate_contract(model_version, as_of_date)
    exception_ids = {int(x) for x in (contract.get("exception_segment_ids") or [])}

    rows = [c.model_dump() for c in customers]
    df = pd.DataFrame(rows)
    scored = champ.score_interpretable(df, id_col="account_id")
    scored["model_version"] = champ.model_version
    scored["as_of_date"] = as_of_date
    scored["guardrail"] = scored["segment_id"].astype(int).isin(exception_ids)
    scored["guardrail_reason"] = scored["guardrail"].map(
        lambda g: "exception_queue_human_review" if g else ""
    )

    if include_applications:
        apps = []
        for name in scored["segment_name"]:
            apps.append(APPLICATION_PORTFOLIO.get(str(name), []))
        scored["applications"] = apps

    batch_mix = (
        scored["segment_id"].astype(int).value_counts(normalize=True).sort_index().to_dict()
        if len(scored)
        else {}
    )
    baseline = {int(k): float(v) for k, v in (contract.get("baseline_segment_mix") or {}).items()}
    drift = {
        str(sid): round(float(batch_mix.get(sid, 0.0) - baseline.get(sid, 0.0)), 4)
        for sid in sorted(set(baseline) | set(batch_mix))
    }

    return {
        "model": champ.run_id,
        "model_version": champ.model_version,
        "as_of_date": as_of_date,
        "n": int(len(scored)),
        "assignments": scored.to_dict(orient="records"),
        "monitoring": {
            "batch_segment_mix": {str(k): float(v) for k, v in batch_mix.items()},
            "drift_vs_baseline": drift,
            "baseline_segment_mix": {str(k): float(v) for k, v in baseline.items()},
        },
        "serving_contract": {
            "model_version": serving_contract().get("model_version"),
            "as_of_date": serving_contract().get("as_of_date"),
        },
    }
