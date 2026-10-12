"""Champion model introspection endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from customer_segmentation.serve.deps import (
    CHAMPION_REL,
    champion_dir,
    read_json_artifact,
    require_champion,
    serving_contract,
)
from customer_segmentation.serve.schemas import FEATURE_FIELDS

router = APIRouter(tags=["model"])


@router.get("/model")
def model_overview() -> dict[str, Any]:
    champ = require_champion()
    contract = serving_contract()
    summary = champ.summary()
    summary["freeze_status"] = (champ.manifest or {}).get("freeze_status") or (
        champ.metrics or {}
    ).get("freeze_status")
    summary["as_of_date"] = contract.get("as_of_date")
    summary["serving_contract"] = {
        "model_version": contract.get("model_version"),
        "as_of_date": contract.get("as_of_date"),
        "exception_segment_ids": contract.get("exception_segment_ids"),
        "minimum_tenure_days": contract.get("minimum_tenure_days"),
    }
    summary["feature_columns"] = list(champ.feature_columns)
    return summary


@router.get("/model/manifest")
def model_manifest() -> dict[str, Any]:
    return read_json_artifact(CHAMPION_REL / "manifest.json")


@router.get("/model/metrics")
def model_metrics() -> dict[str, Any]:
    return read_json_artifact(CHAMPION_REL / "metrics.json")


@router.get("/model/features")
def model_features() -> dict[str, Any]:
    champ = require_champion()
    feat_meta = {}
    path = champion_dir() / "feature_columns.json"
    if path.exists():
        feat_meta = json.loads(path.read_text(encoding="utf-8"))
    return {
        "feature_version": champ.feature_version,
        "columns": list(champ.feature_columns or FEATURE_FIELDS),
        "required_for_inference": list(FEATURE_FIELDS),
        "meta": feat_meta,
        "notes": (
            "Send interpretable-scale features. The API applies the frozen log1p + StandardScaler."
        ),
    }


@router.get("/model/freeze")
def model_freeze() -> dict[str, Any]:
    path = champion_dir() / "freeze_readiness.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="freeze_readiness.json missing")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/model/serving-contract")
def model_serving_contract() -> dict[str, Any]:
    return serving_contract()


@router.get("/champion")
def champion_alias() -> dict[str, Any]:
    """Backward-compatible alias for /v1/model."""
    return model_overview()


@router.get("/segments")
def list_segments() -> dict[str, Any]:
    champ = require_champion()
    segments = list(champ.catalog.get("segments", {}).values())
    return {"n": len(segments), "segments": segments}


@router.get("/segments/{segment_id}")
def get_segment(segment_id: int) -> dict[str, Any]:
    champ = require_champion()
    meta = champ.catalog.get("segments", {}).get(str(segment_id))
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown segment_id={segment_id}")
    return meta
