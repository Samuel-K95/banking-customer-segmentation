"""Single-row and batch inference + monitoring."""

from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from customer_segmentation.serve.deps import require_champion, serving_contract
from customer_segmentation.serve.schemas import (
    FEATURE_FIELDS,
    FeatureRow,
    InferenceRequest,
    SingleInferenceRequest,
)
from customer_segmentation.serve.scoring import score_feature_rows

router = APIRouter(tags=["inference"])


@router.get("/inference/contract")
def inference_contract() -> dict[str, Any]:
    contract = serving_contract()
    return {
        "model_version": contract.get("model_version"),
        "as_of_date": contract.get("as_of_date"),
        "required_features": list(FEATURE_FIELDS),
        "exception_segment_ids": contract.get("exception_segment_ids") or [],
        "baseline_segment_mix": contract.get("baseline_segment_mix") or {},
        "notes": (
            "Every inference call must send model_version + as_of_date matching this contract. "
            "Features are interpretable-scale; the API applies frozen log1p + StandardScaler."
        ),
    }


@router.post("/inference")
def inference_single(body: SingleInferenceRequest) -> dict[str, Any]:
    """Score one customer (feature vector + serving contract)."""
    champ = require_champion()
    return score_feature_rows(
        champ,
        [body.to_feature_row()],
        model_version=body.model_version,
        as_of_date=body.as_of_date,
        include_applications=body.include_applications,
    )


@router.post("/inference/batch")
def inference_batch(body: InferenceRequest) -> dict[str, Any]:
    """Score a JSON batch of customers (max enforced by payload size)."""
    champ = require_champion()
    if len(body.customers) > 500:
        raise HTTPException(status_code=400, detail="Batch limited to 500 rows; use /inference/batch/csv")
    return score_feature_rows(
        champ,
        body.customers,
        model_version=body.model_version,
        as_of_date=body.as_of_date,
        include_applications=body.include_applications,
    )


@router.post("/inference/batch/csv")
async def inference_batch_csv(
    file: UploadFile = File(..., description="CSV with frozen feature columns"),
    model_version: str = Form(...),
    as_of_date: str = Form(...),
    include_applications: bool = Form(False),
    return_csv: bool = Form(False),
):
    """Upload a CSV of feature rows for batch scoring."""
    champ = require_champion()
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"CSV must be UTF-8: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV has no header row")

    customers: list[FeatureRow] = []
    for i, row in enumerate(reader):
        if i >= 500:
            raise HTTPException(status_code=400, detail="CSV batch limited to 500 rows")
        payload: dict[str, Any] = {}
        if row.get("account_id") not in (None, ""):
            payload["account_id"] = row["account_id"]
        for col in FEATURE_FIELDS:
            if col not in row or row[col] in (None, ""):
                raise HTTPException(status_code=400, detail=f"Row {i}: missing column {col}")
            payload[col] = float(row[col])
        try:
            customers.append(FeatureRow(**payload))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Row {i}: {exc}") from exc

    if not customers:
        raise HTTPException(status_code=400, detail="CSV contained no data rows")

    result = score_feature_rows(
        champ,
        customers,
        model_version=model_version,
        as_of_date=as_of_date,
        include_applications=include_applications,
    )

    if return_csv:
        buf = io.StringIO()
        assignments = result["assignments"]
        if not assignments:
            return PlainTextResponse("", media_type="text/csv")
        fieldnames = list(assignments[0].keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in assignments:
            flat = {k: (json_dumps_safe(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            writer.writerow(flat)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")

    return result


def json_dumps_safe(value: Any) -> str:
    import json

    return json.dumps(value, default=str)


@router.post("/customers/segment")
def customers_segment(body: InferenceRequest) -> dict[str, Any]:
    """Backward-compatible alias for batch inference."""
    return inference_batch(body)


@router.get("/monitoring/segment-mix")
def monitoring_segment_mix() -> dict[str, Any]:
    contract = serving_contract()
    return {
        "baseline_segment_mix": contract.get("baseline_segment_mix") or {},
        "exception_segment_ids": contract.get("exception_segment_ids") or [],
        "model_version": contract.get("model_version"),
        "as_of_date": contract.get("as_of_date"),
    }


@router.get("/monitoring/snapshot")
def monitoring_snapshot() -> dict[str, Any]:
    from customer_segmentation.serve.deps import (
        CHAMPION_REL,
        DECISION_REL,
        REPORTS_REL,
        resolve_path,
    )

    champ = require_champion()
    contract = serving_contract()
    checks = {
        "catalog": resolve_path(CHAMPION_REL / "segment_catalog.json").exists(),
        "scaler": resolve_path(CHAMPION_REL / "scaler.joblib").exists(),
        "kmeans": resolve_path(CHAMPION_REL / "kmeans.joblib").exists(),
        "encoder": resolve_path(CHAMPION_REL / "encoder.pt").exists(),
        "serving_contract": resolve_path(CHAMPION_REL / "serving_contract.json").exists(),
        "decision_summary": resolve_path(DECISION_REL / "run_summary.json").exists(),
        "decision_map": resolve_path(DECISION_REL / "decision_map_full.csv").exists(),
        "reports_dir": resolve_path(REPORTS_REL).is_dir(),
    }
    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "model_version": champ.model_version,
        "as_of_date": contract.get("as_of_date"),
        "n_clusters": champ.n_clusters,
        "checks": checks,
    }
