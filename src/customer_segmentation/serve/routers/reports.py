"""Expose champion markdown reports via the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

from customer_segmentation.serve.deps import (
    CURRENT_CHAMPION_MD,
    reports_dir,
    resolve_path,
)

router = APIRouter(tags=["reports"])

TRACK_C_REL = Path("reports") / "track_c_berka"


def _slug_map(*, scope: str = "champion") -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if scope in ("champion", "all"):
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
    if scope in ("track_c", "berka", "all"):
        tdir = resolve_path(TRACK_C_REL)
        if tdir.exists():
            for path in sorted(tdir.glob("*.md")):
                stem = path.stem
                mapping[stem] = path
                mapping[f"track_c/{stem}"] = path
    return mapping


def _list_from_dir(rdir: Path, *, prefix: str = "") -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    if not rdir.exists():
        return unique
    for path in sorted(rdir.glob("*.md"), key=lambda p: p.name):
        slug = f"{prefix}{path.stem}" if prefix else path.stem
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
    return unique


@router.get("/reports")
def list_reports(
    scope: str = Query("champion", description="champion | track_c | all"),
) -> dict[str, Any]:
    scope_n = scope.strip().lower()
    if scope_n not in ("champion", "track_c", "berka", "all"):
        raise HTTPException(status_code=400, detail="scope must be champion|track_c|all")
    if scope_n in ("track_c", "berka"):
        rows = _list_from_dir(resolve_path(TRACK_C_REL))
        return {"n": len(rows), "scope": "track_c", "reports": rows}
    if scope_n == "all":
        champ = _list_from_dir(reports_dir())
        track = _list_from_dir(resolve_path(TRACK_C_REL))
        for row in track:
            row["scope"] = "track_c"
        for row in champ:
            row["scope"] = "champion"
        rows = champ + track
        return {"n": len(rows), "scope": "all", "reports": rows}
    rows = _list_from_dir(reports_dir())
    for row in rows:
        row["scope"] = "champion"
    return {"n": len(rows), "scope": "champion", "reports": rows}


@router.get("/reports/tracks/berka")
def list_track_c_reports() -> dict[str, Any]:
    rows = _list_from_dir(resolve_path(TRACK_C_REL))
    return {"n": len(rows), "scope": "track_c", "reports": rows}


@router.get("/reports/{slug}/raw")
def report_raw(slug: str, download: bool = False):
    mapping = _slug_map(scope="all")
    path = mapping.get(slug)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown report slug={slug!r}")
    if download:
        return FileResponse(path, filename=path.name, media_type="text/markdown")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


@router.get("/reports/{slug}")
def report_meta(slug: str) -> dict[str, Any]:
    mapping = _slug_map(scope="all")
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
