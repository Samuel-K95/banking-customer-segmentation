"""Expose champion markdown reports via the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from customer_segmentation.serve.deps import CURRENT_CHAMPION_MD, reports_dir, resolve_path

router = APIRouter(tags=["reports"])


def _slug_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    rdir = reports_dir()
    for path in sorted(rdir.glob("*.md")):
        stem = path.stem
        mapping[stem] = path
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            mapping[parts[1]] = path
    one = resolve_path(CURRENT_CHAMPION_MD)
    if one.exists():
        mapping["current"] = one
        mapping["CURRENT_CHAMPION"] = one
    return mapping


@router.get("/reports")
def list_reports() -> dict[str, Any]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for path in sorted({p for p in _slug_map().values()}, key=lambda p: p.name):
        if path.name in seen:
            continue
        seen.add(path.name)
        slug = path.stem
        unique.append(
            {
                "slug": slug,
                "filename": path.name,
                "path": str(path).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "endpoints": {
                    "meta": f"/v1/reports/{slug}",
                    "raw": f"/v1/reports/{slug}/raw",
                },
            }
        )
    return {"n": len(unique), "reports": unique}


@router.get("/reports/{slug}")
def report_meta(slug: str) -> dict[str, Any]:
    mapping = _slug_map()
    path = mapping.get(slug)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown report slug={slug!r}. See GET /v1/reports")
    text = path.read_text(encoding="utf-8")
    title = next(
        (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")),
        path.stem,
    )
    return {
        "slug": slug,
        "filename": path.name,
        "title": title,
        "chars": len(text),
        "preview": "\n".join(text.splitlines()[:40]),
        "raw_url": f"/v1/reports/{slug}/raw",
    }


@router.get("/reports/{slug}/raw")
def report_raw(slug: str, download: bool = False):
    mapping = _slug_map()
    path = mapping.get(slug)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown report slug={slug!r}")
    if download:
        return FileResponse(path, filename=path.name, media_type="text/markdown")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")
