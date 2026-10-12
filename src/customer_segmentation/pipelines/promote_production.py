"""Promote leakage-safe research twin → production champion slot (atomic + gated)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from customer_segmentation.models.artifact_integrity import (
    REQUIRED_ARTIFACTS,
    integrity_report,
    preflight_bundle,
    sync_serving_contract_hashes,
)

PRODUCTION_DIR = Path("artifacts/segmentation/champion")
RESEARCH_DIR = Path("artifacts/segmentation/research/leakage_safe_source")
ARCHIVE_DIR = Path("artifacts/segmentation/archive/fullbook_demo")
MODEL_VERSION = "production_candidate_v1"
AS_OF_DATE = "1997-12-31"
CONTRACT_AS_OF = AS_OF_DATE


class PromotionError(RuntimeError):
    """Raised when promotion is refused (incomplete source or integrity failure)."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _stamp_bundle(dst: Path, *, source_dir: Path) -> dict[str, Any]:
    """Write production metadata + serving contract; refuse if binary hashes are null."""
    manifest = _read_json(dst / "manifest.json") if (dst / "manifest.json").exists() else {}
    metrics = _read_json(dst / "metrics.json") if (dst / "metrics.json").exists() else {}
    feat_meta = _read_json(dst / "feature_columns.json") if (dst / "feature_columns.json").exists() else {}

    feature_columns = (
        metrics.get("feature_columns")
        or feat_meta.get("columns")
        or []
    )

    manifest.update(
        {
            "run_id": metrics.get("run_id", manifest.get("run_id", "c_ae_lat4_k6_leakage_safe")),
            "model_version": MODEL_VERSION,
            "feature_version": manifest.get("feature_version", "berka_rfm_product_v1_leakage_safe"),
            "as_of_date": AS_OF_DATE,
            "freeze_status": "PROVISIONAL",
            "tag": f"prod_segmentation_{MODEL_VERSION}",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "source_research_dir": str(source_dir),
            "archive_demo_dir": str(ARCHIVE_DIR),
            "notes": (
                "Production candidate promoted from leakage-safe twin. "
                "Freeze PROVISIONAL — compliance sign-off and true prospective vintage still required."
            ),
            "serving_contract": {
                "as_of_date": CONTRACT_AS_OF,
                "model_version": MODEL_VERSION,
                "minimum_tenure_days": 365,
                "feature_columns": feature_columns,
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

    try:
        # Normalize CSV newlines to LF before hashing so Linux/Windows checkouts match.
        contract = sync_serving_contract_hashes(dst, normalize_text=True)
    except FileNotFoundError as exc:
        raise PromotionError(str(exc)) from exc
    contract.update(
        {
            "model_version": MODEL_VERSION,
            "as_of_date": CONTRACT_AS_OF,
            "minimum_tenure_days": 365,
            "baseline_segment_mix": _baseline_mix(dst / "assignments.csv"),
            "exception_segment_ids": _exception_segments(dst),
            "feature_columns": list(feature_columns),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(dst / "serving_contract.json", contract)
    return contract


def promote_production(
    *,
    source_dir: str | Path | None = None,
    dest_dir: str | Path | None = None,
    backup: bool = True,
) -> dict[str, Any]:
    """Copy research leakage-safe bundle into the production slot (atomic + gated).

    Never deletes ``dest`` until the source passes preflight and the staged
    copy validates. Incomplete sources raise ``PromotionError`` and leave dest
    untouched.
    """
    src = Path(source_dir) if source_dir else RESEARCH_DIR
    dst = Path(dest_dir) if dest_dir else PRODUCTION_DIR

    if not src.exists():
        raise FileNotFoundError(f"Research twin not found at {src}")

    preflight_errors = preflight_bundle(src, load_binaries=True)
    if preflight_errors:
        raise PromotionError(
            "Source bundle failed preflight; destination not modified:\n- "
            + "\n- ".join(preflight_errors)
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = dst.parent / f".champion_promote_tmp_{stamp}"
    if staging.exists():
        shutil.rmtree(staging)

    try:
        shutil.copytree(src, staging)
        # Re-check staged copy before touching live dest
        staged_errors = preflight_bundle(staging, load_binaries=True)
        if staged_errors:
            raise PromotionError(
                "Staged copy failed preflight; destination not modified:\n- "
                + "\n- ".join(staged_errors)
            )

        contract = _stamp_bundle(staging, source_dir=src)
        # Stamp can rewrite JSON; re-validate integrity including hashes
        report = integrity_report(staging)
        if not report["ok"]:
            raise PromotionError(
                "Staged bundle failed integrity after stamp; destination not modified:\n- "
                + "\n- ".join(report["errors"])
            )

        if dst.exists():
            if backup:
                backup_root = ARCHIVE_DIR.parent / f"pre_promote_{stamp}"
                backup_root.parent.mkdir(parents=True, exist_ok=True)
                if backup_root.exists():
                    shutil.rmtree(backup_root)
                shutil.move(str(dst), str(backup_root))
            else:
                shutil.rmtree(dst)

        shutil.move(str(staging), str(dst))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    final = integrity_report(dst)
    if not final["ok"]:
        raise PromotionError(
            "Promotion completed but final integrity failed:\n- " + "\n- ".join(final["errors"])
        )

    return {
        "status": "ok",
        "production_dir": str(dst),
        "model_version": MODEL_VERSION,
        "as_of_date": AS_OF_DATE,
        "freeze_status": "PROVISIONAL",
        "artifact_sha256": contract.get("artifact_sha256"),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "integrity": final,
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
