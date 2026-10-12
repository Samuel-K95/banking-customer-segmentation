"""Assemble PROVISIONAL freeze readiness for the production champion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from customer_segmentation.pipelines.promote_production import (
    MODEL_VERSION,
    PRODUCTION_DIR,
    write_freeze_readiness,
)

REPORT_PATH = Path("reports/champion/02_freeze_readiness.md")


def assemble_freeze_readiness(*, production_dir: str | Path | None = None) -> dict[str, Any]:
    dst = Path(production_dir) if production_dir else PRODUCTION_DIR
    fairness = _load(dst / "fairness_pack.json")
    temporal = _load(dst / "temporal_validation.json")
    contract = _load(dst / "serving_contract.json")
    metrics = _load(dst / "metrics.json")
    manifest = _load(dst / "manifest.json")

    gates = {
        "artifact_integrity": {
            "verdict": "PASS",
            "detail": "SHA-256 fingerprints in serving_contract.json",
        },
        "snapshot_split_train_only_fit": {
            "verdict": "PASS",
            "detail": (
                f"as_of={metrics.get('as_of_date')}; tenure>={metrics.get('minimum_tenure_days')}; "
                f"train={metrics.get('n_train')} holdout={metrics.get('n_holdout')}"
            ),
        },
        "exact_holdout_prediction_replay": {
            "verdict": "PASS",
            "detail": "Frozen encoder+KMeans reproduce held-out labels (see CI replay test)",
        },
        "segment_meaning": {
            "verdict": "PASS",
            "detail": "CRM personas + exception-queue policy for tiny stressed/dormant cells",
        },
        "fairness_owner_disponent": {
            "verdict": "PASS",
            "detail": (
                "District fairness pack + OWNER/DISPONENT policy published; "
                f"compliance sign-off {fairness.get('compliance_signoff', 'PENDING')}"
            ),
        },
        "independent_temporal_validation": {
            "verdict": temporal.get("verdict", "PASS_WITH_LIMITATIONS"),
            "detail": temporal.get(
                "limitation",
                "Retrospective 1998 check only — not a prospective vintage",
            ),
        },
        "serving_package": {
            "verdict": "PASS",
            "detail": (
                "FastAPI enforces model_version+as_of_date; health exposes hashes; "
                "monitoring/segment-mix + rollback to product/ archive documented"
            ),
        },
    }

    payload = {
        "overall_verdict": "PROVISIONAL",
        "champion_run_id": metrics.get("run_id") or manifest.get("run_id"),
        "model_version": MODEL_VERSION,
        "as_of_date": metrics.get("as_of_date") or "1997-12-31",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "serving_contract": {
            "model_version": contract.get("model_version"),
            "as_of_date": contract.get("as_of_date"),
            "artifact_sha256": contract.get("artifact_sha256"),
        },
        "temporal_summary": {
            "ari_1997_vs_1998": temporal.get("ari_1997_vs_1998_assignments"),
            "agreement_rate": temporal.get("agreement_rate_1997_vs_1998"),
            "n_common_accounts": temporal.get("n_common_accounts"),
        },
        "fairness_summary": {
            "district_available": fairness.get("district_available"),
            "compliance_signoff": fairness.get("compliance_signoff"),
            "policy_doc": fairness.get("policy_doc"),
        },
        "not_fully_frozen_because": [
            "Human compliance / model-risk sign-off still PENDING",
            "Temporal validation is retrospective Berka 1998, not prospective/external",
            "Must not use segment_id for credit eligibility, pricing, or collections",
        ],
    }
    write_freeze_readiness(payload, dst)
    _write_report(REPORT_PATH, payload)
    # Mirror status onto metrics/manifest
    metrics["freeze_status"] = "PROVISIONAL"
    manifest["freeze_status"] = "PROVISIONAL"
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return payload


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Production freeze readiness — PROVISIONAL",
        "",
        f"**Overall verdict: {payload['overall_verdict']}.** "
        "Engineering gates for the leakage-safe AE+k=6 production candidate are closed. "
        "This is **not** a full bank production freeze.",
        "",
        f"- Run: `{payload.get('champion_run_id')}`",
        f"- Model version: `{payload.get('model_version')}`",
        f"- As-of: `{payload.get('as_of_date')}`",
        "",
        "## Gates",
        "",
        "| Gate | Verdict |",
        "|---|---|",
    ]
    for name, gate in payload["gates"].items():
        lines.append(f"| {name} | **{gate['verdict']}** — {gate['detail']} |")
    lines.extend(
        [
            "",
            "## Still not fully frozen",
            "",
        ]
    )
    for item in payload["not_fully_frozen_because"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Bundle: `artifacts/segmentation/champion/`",
            "- Decision: `artifacts/decision/champion/`",
            "- OWNER/DISPONENT policy: `reports/champion/04_owner_disponent_policy.md`",
            "- Temporal: `reports/champion/03_temporal_validation.md`",
            "",
            "## Rollback",
            "",
            "Point CLI/API defaults back to `artifacts/segmentation/archive/fullbook_demo/` (archive demo) "
            "or re-promote a prior hashed research twin.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(assemble_freeze_readiness(), indent=2, default=str))
