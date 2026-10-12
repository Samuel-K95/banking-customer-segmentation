"""Promote leakage-safe research twin → production champion slot."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRODUCTION_DIR = Path("artifacts/segmentation/champion")
RESEARCH_DIR = Path("artifacts/segmentation/research/leakage_safe_source")
ARCHIVE_DIR = Path("artifacts/segmentation/archive/fullbook_demo")
MODEL_VERSION = "production_candidate_v1"
AS_OF_DATE = "1997-12-31"
CONTRACT_AS_OF = AS_OF_DATE


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def promote_production(
    *,
    source_dir: str | Path | None = None,
    dest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Copy research leakage-safe bundle into the production slot and stamp metadata."""
    src = Path(source_dir) if source_dir else RESEARCH_DIR
    dst = Path(dest_dir) if dest_dir else PRODUCTION_DIR
    if not src.exists():
        raise FileNotFoundError(f"Research twin not found at {src}")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    manifest = _read_json(dst / "manifest.json") if (dst / "manifest.json").exists() else {}
    metrics = _read_json(dst / "metrics.json") if (dst / "metrics.json").exists() else {}

    manifest.update(
        {
            "run_id": metrics.get("run_id", manifest.get("run_id", "c_ae_lat4_k6_leakage_safe")),
            "model_version": MODEL_VERSION,
            "feature_version": manifest.get("feature_version", "berka_rfm_product_v1_leakage_safe"),
            "as_of_date": AS_OF_DATE,
            "freeze_status": "PROVISIONAL",
            "tag": f"prod_segmentation_{MODEL_VERSION}",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "source_research_dir": str(src),
            "archive_demo_dir": str(ARCHIVE_DIR),
            "notes": (
                "Production candidate promoted from leakage-safe twin. "
                "Freeze PROVISIONAL — compliance sign-off and true prospective vintage still required."
            ),
            "serving_contract": {
                "as_of_date": CONTRACT_AS_OF,
                "model_version": MODEL_VERSION,
                "minimum_tenure_days": 365,
                "feature_columns": metrics.get("feature_columns")
                or _read_json(dst / "feature_columns.json").get("columns"),
            },
            "rollback": {
                "previous_demo_archive": str(ARCHIVE_DIR),
                "instruction": "Point CLI/API back to archive or re-promote a prior hashed bundle",
            },
        }
    )
    metrics["model_version"] = MODEL_VERSION
    metrics["as_of_date"] = AS_OF_DATE
    metrics["freeze_status"] = "PROVISIONAL"
    _write_json(dst / "manifest.json", manifest)
    _write_json(dst / "metrics.json", metrics)

    hashes = {
        name: _sha256(dst / name)
        for name in ("scaler.joblib", "kmeans.joblib", "encoder.pt", "account_split.csv", "assignments.csv")
    }
    contract = {
        "model_version": MODEL_VERSION,
        "as_of_date": CONTRACT_AS_OF,
        "minimum_tenure_days": 365,
        "artifact_sha256": hashes,
        "baseline_segment_mix": _baseline_mix(dst / "assignments.csv"),
        "exception_segment_ids": _exception_segments(dst),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(dst / "serving_contract.json", contract)

    return {
        "status": "ok",
        "production_dir": str(dst),
        "model_version": MODEL_VERSION,
        "as_of_date": AS_OF_DATE,
        "freeze_status": "PROVISIONAL",
        "artifact_sha256": hashes,
    }


def _baseline_mix(assignments_path: Path) -> dict[str, float]:
    if not assignments_path.exists():
        return {}
    import pandas as pd

    frame = pd.read_csv(assignments_path)
    counts = frame["segment_id"].value_counts(normalize=True).sort_index()
    return {str(int(k)): float(v) for k, v in counts.items()}


def _exception_segments(dst: Path) -> list[int]:
    """Tiny / care segments treated as exception queues (from size fractions)."""
    metrics = _read_json(dst / "metrics.json") if (dst / "metrics.json").exists() else {}
    size = metrics.get("size") or {}
    fracs = size.get("fractions") or []
    out: list[int] = []
    for i, frac in enumerate(fracs):
        if float(frac) < 0.03:
            out.append(int(i))
    # Fallback known stressed/dormant from catalog names
    catalog = _read_json(dst / "segment_catalog.json") if (dst / "segment_catalog.json").exists() else {}
    for sid, meta in (catalog.get("segments") or {}).items():
        name = str(meta.get("name", "")).lower()
        if any(k in name for k in ("stress", "dormant", "overdraft")):
            out.append(int(sid))
    return sorted(set(out))


def load_freeze_status(production_dir: str | Path | None = None) -> dict[str, Any]:
    dst = Path(production_dir) if production_dir else PRODUCTION_DIR
    path = dst / "freeze_readiness.json"
    if not path.exists():
        return {"overall_verdict": "MISSING", "production_dir": str(dst)}
    return _read_json(path)


def write_freeze_readiness(payload: dict[str, Any], production_dir: str | Path | None = None) -> Path:
    dst = Path(production_dir) if production_dir else PRODUCTION_DIR
    path = dst / "freeze_readiness.json"
    _write_json(path, payload)
    return path


if __name__ == "__main__":
    print(json.dumps(promote_production(), indent=2))
