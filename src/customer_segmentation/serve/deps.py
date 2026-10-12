"""Shared dependencies for the champion API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException


CHAMPION_REL = Path("artifacts") / "segmentation" / "champion"
DECISION_REL = Path("artifacts") / "decision" / "champion"
REPORTS_REL = Path("reports") / "champion"
CURRENT_CHAMPION_MD = Path("CURRENT_CHAMPION.md")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(*parts: str | Path) -> Path:
    """Prefer cwd-relative path; fall back to package repo root."""
    rel = Path(*parts) if not isinstance(parts[0], Path) else Path(parts[0]).joinpath(*parts[1:])
    for base in (Path("."), repo_root()):
        cand = base / rel if not rel.is_absolute() else rel
        if cand.exists():
            return cand
    return Path(".") / rel


@lru_cache(maxsize=1)
def get_champion():
    from customer_segmentation.models.champion import ChampionModel

    for cand in (ChampionModel.default_dir("."), repo_root() / CHAMPION_REL):
        if cand.exists():
            return ChampionModel.load(cand)
    raise FileNotFoundError(
        "Champion artifacts not found at artifacts/segmentation/champion/. "
        "Run `bcs restore-champion` or `bcs promote-production` first."
    )


def require_champion():
    try:
        return get_champion()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def champion_integrity() -> dict[str, Any]:
    from customer_segmentation.models.artifact_integrity import integrity_report

    path = resolve_path(CHAMPION_REL)
    return integrity_report(path)


def serving_contract() -> dict[str, Any]:
    """Return the on-disk serving contract. Fail closed — no manifest fallback."""
    path = resolve_path(CHAMPION_REL / "serving_contract.json")
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "serving_contract.json missing — refuse to serve without an "
                "integrity contract (fail closed)"
            ),
        )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(model_version: str, as_of_date: str) -> dict[str, Any]:
    contract = serving_contract()
    expected_version = str(contract.get("model_version") or "")
    expected_asof = str(contract.get("as_of_date") or "")
    if model_version != expected_version:
        raise HTTPException(
            status_code=400,
            detail=f"model_version mismatch: got {model_version!r}, expected {expected_version!r}",
        )
    if as_of_date != expected_asof:
        raise HTTPException(
            status_code=400,
            detail=f"as_of_date mismatch: got {as_of_date!r}, expected {expected_asof!r}",
        )
    return contract


def read_json_artifact(*rel_parts: str | Path) -> dict[str, Any]:
    path = resolve_path(*rel_parts)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def decision_dir() -> Path:
    path = resolve_path(DECISION_REL)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Decision artifacts missing. Run `bcs decision-berka`.",
        )
    return path


def champion_dir() -> Path:
    path = resolve_path(CHAMPION_REL)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Champion bundle missing.")
    return path


def reports_dir() -> Path:
    path = resolve_path(REPORTS_REL)
    if not path.exists():
        raise HTTPException(status_code=404, detail="reports/champion/ not found")
    return path
