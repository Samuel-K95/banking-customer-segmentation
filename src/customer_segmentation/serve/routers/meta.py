"""Health + API catalog."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from customer_segmentation import __version__
from customer_segmentation.serve.deps import require_champion, serving_contract

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict[str, Any]:
    champ = require_champion()
    contract = serving_contract()
    return {
        "status": "ok",
        "api_version": __version__,
        "run_id": champ.run_id,
        "model_version": champ.model_version,
        "as_of_date": contract.get("as_of_date"),
        "freeze_status": (champ.manifest or {}).get("freeze_status")
        or (champ.metrics or {}).get("freeze_status"),
        "artifact_sha256": contract.get("artifact_sha256") or {},
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
                "/v1/decisions/map",
                "/v1/decisions/map/full",
                "/v1/decisions/domains",
                "/v1/decisions/profiles",
                "/v1/decisions/fairness",
                "/v1/decisions/explainability",
                "/v1/decisions/domain-coverage",
                "/v1/segments/{segment_id}/applications",
            ],
            "reports": ["/v1/reports", "/v1/reports/{slug}", "/v1/reports/{slug}/raw"],
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
    }
