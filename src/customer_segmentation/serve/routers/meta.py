"""Health + API catalog."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from customer_segmentation import __version__
from customer_segmentation.serve.deps import (
    champion_integrity,
    require_champion,
    serving_contract,
)

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict[str, Any]:
    integrity = champion_integrity()
    if not integrity.get("ok"):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "failed",
                "reason": "champion_integrity",
                "errors": integrity.get("errors"),
            },
        )
    champ = require_champion()
    contract = serving_contract()
    return {
        "status": "ok",
        "api_version": __version__,
        "run_id": champ.run_id,
        "model_version": champ.model_version,
        "as_of_date": contract.get("as_of_date"),
        "has_encoder": champ.encoder is not None,
        "freeze_status": (champ.manifest or {}).get("freeze_status")
        or (champ.metrics or {}).get("freeze_status"),
        "artifact_sha256": contract.get("artifact_sha256") or {},
        "integrity": "ok",
    }


@router.get("/catalog")
def api_catalog() -> dict[str, Any]:
    """Machine-readable index of available endpoints."""
    return {
        "api_version": __version__,
        "groups": {
            "meta": ["/v1/health", "/v1/catalog"],
            "model": [
                "/v1/model",
                "/v1/champion",
                "/v1/model/manifest",
                "/v1/model/metrics",
                "/v1/model/features",
                "/v1/model/freeze",
                "/v1/model/serving-contract",
                "/v1/segments",
                "/v1/segments/{segment_id}",
            ],
            "decisions": [
                "/v1/decisions/summary",
                "/v1/decision/summary",
                "/v1/decisions/map",
                "/v1/decision-map",
                "/v1/decisions/map/full",
                "/v1/decisions/domains",
                "/v1/decisions/profiles",
                "/v1/decisions/fairness",
                "/v1/decisions/explainability",
                "/v1/decisions/domain-coverage",
                "/v1/segments/{segment_id}/applications",
            ],
            "reports": [
                "/v1/reports",
                "/v1/reports?scope=track_c",
                "/v1/reports/tracks/berka",
                "/v1/reports/{slug}",
                "/v1/reports/{slug}/raw",
            ],
            "comparison": ["/v1/comparison/board", "/v1/comparison/board?board=explore"],
            "inference": [
                "/v1/inference",
                "/v1/inference/batch",
                "/v1/inference/batch/csv",
                "/v1/inference/contract",
                "/v1/customers/segment",
            ],
            "monitoring": [
                "/v1/monitoring/segment-mix",
                "/v1/monitoring/snapshot",
            ],
        },
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
