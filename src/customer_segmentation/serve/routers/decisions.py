"""Decision-layer API: maps, profiles, fairness, explainability."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from customer_segmentation.decision.applications import (
    APPLICATION_PORTFOLIO,
    expand_decision_map,
)
from customer_segmentation.serve.deps import decision_dir, require_champion

router = APIRouter(tags=["decisions"])


def _load_decision_json(name: str) -> dict[str, Any] | list[Any]:
    path = decision_dir() / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing decision artifact: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/decisions/summary")
def decisions_summary() -> dict[str, Any]:
    path = decision_dir() / "run_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run_summary.json missing — run bcs decision-berka")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/decisions/map")
def decisions_map(domain: str | None = Query(default=None)) -> dict[str, Any]:
    champ = require_champion()
    frame = expand_decision_map(champ.catalog)
    if domain:
        frame = frame[frame["domain"] == domain]
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"No plays for domain={domain!r}")
    return {"n": len(frame), "domain": domain, "rows": frame.to_dict(orient="records")}


@router.get("/decisions/map/full")
def decisions_map_full(domain: str | None = Query(default=None)) -> dict[str, Any]:
    path = decision_dir() / "decision_map_full.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="decision_map_full.csv missing")
    frame = pd.read_csv(path)
    if domain:
        frame = frame[frame["domain"] == domain]
    return {"n": len(frame), "domain": domain, "rows": frame.to_dict(orient="records")}


@router.get("/decisions/domains")
def decisions_domains() -> dict[str, Any]:
    path = decision_dir() / "domain_coverage.csv"
    if path.exists():
        frame = pd.read_csv(path)
        return {"n": len(frame), "rows": frame.to_dict(orient="records")}
    summary = decisions_summary()
    return {"n": len(summary.get("domains") or []), "domains": summary.get("domains"), "rows": summary.get("domain_coverage")}


@router.get("/decisions/profiles")
def decisions_profiles() -> dict[str, Any]:
    return _load_decision_json("profile_packs.json")  # type: ignore[return-value]


@router.get("/decisions/fairness")
def decisions_fairness() -> dict[str, Any]:
    return _load_decision_json("fairness.json")  # type: ignore[return-value]


@router.get("/decisions/explainability")
def decisions_explainability() -> dict[str, Any]:
    return _load_decision_json("explainability.json")  # type: ignore[return-value]


@router.get("/decisions/domain-coverage")
def decisions_domain_coverage() -> dict[str, Any]:
    return decisions_domains()


@router.get("/decision-map")
def decision_map_alias(domain: str | None = None) -> dict[str, Any]:
    return decisions_map(domain=domain)


@router.get("/decision/summary")
def decision_summary_alias() -> dict[str, Any]:
    return decisions_summary()


@router.get("/segments/{segment_id}/applications")
def segment_applications(segment_id: int) -> dict[str, Any]:
    champ = require_champion()
    meta = champ.catalog.get("segments", {}).get(str(segment_id))
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown segment_id={segment_id}")
    plays = APPLICATION_PORTFOLIO.get(meta.get("name", ""), [])
    return {
        "segment_id": segment_id,
        "name": meta.get("name"),
        "one_liner": meta.get("one_liner"),
        "default_application": meta.get("decision"),
        "applications": plays,
    }
