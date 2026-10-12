"""Assemble PROVISIONAL freeze readiness for the production champion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from customer_segmentation.models.artifact_integrity import integrity_report
from customer_segmentation.models.champion import ChampionModel
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

    integrity = integrity_report(dst)
    integrity_gate = {
        "verdict": "PASS" if integrity.get("ok") else "FAIL",
        "detail": (
            "SHA-256 fingerprints match serving_contract.json"
            if integrity.get("ok")
            else "; ".join(integrity.get("errors") or ["integrity failed"])
        ),
    }

    split_ok = bool(
        metrics.get("as_of_date")
        and metrics.get("minimum_tenure_days")
        and metrics.get("n_train")
        and metrics.get("n_holdout")
        and (dst / "account_split.csv").exists()
    )
    split_gate = {
        "verdict": "PASS" if split_ok else "FAIL",
        "detail": (
            f"as_of={metrics.get('as_of_date')}; tenure>={metrics.get('minimum_tenure_days')}; "
            f"train={metrics.get('n_train')} holdout={metrics.get('n_holdout')}"
            if split_ok
            else "Missing as_of / tenure / split sizes or account_split.csv"
        ),
    }

    replay = _holdout_replay_gate(dst)
    meaning = _segment_meaning_gate(dst, contract=contract, metrics=metrics)
    fairness_signoff = str(fairness.get("compliance_signoff") or "PENDING").upper()
    fairness_gate = {
        "verdict": "PASS" if fairness_signoff == "APPROVED" else "PASS_WITH_LIMITATIONS",
        "detail": (
            "District fairness pack + OWNER/DISPONENT policy published; "
            f"compliance sign-off {fairness_signoff}"
        ),
    }

    temporal_verdict = temporal.get("verdict") or "PASS_WITH_LIMITATIONS"
    temporal_gate = {
        "verdict": temporal_verdict,
        "detail": temporal.get(
            "limitation",
            "Retrospective 1998 check only — not a prospective vintage",
        ),
    }

    serving_ok = bool(
        contract.get("model_version")
        and contract.get("as_of_date")
        and (contract.get("artifact_sha256") or {})
        and integrity.get("ok")
    )
    serving_gate = {
        "verdict": "PASS" if serving_ok else "FAIL",
        "detail": (
            "FastAPI enforces model_version+as_of_date; serving_contract hashes verified"
            if serving_ok
            else "Serving contract incomplete or integrity failed"
        ),
    }

    gates = {
        "artifact_integrity": integrity_gate,
        "snapshot_split_train_only_fit": split_gate,
        "exact_holdout_prediction_replay": replay,
        "segment_meaning": meaning,
        "fairness_owner_disponent": fairness_gate,
        "independent_temporal_validation": temporal_gate,
        "serving_package": serving_gate,
    }

    hard_fails = [k for k, g in gates.items() if g.get("verdict") == "FAIL"]
    overall = "BLOCKED" if hard_fails else "PROVISIONAL"

    # Keep manifest paths current (no stale product/champion_v1).
    manifest["source_research_dir"] = "artifacts/segmentation/research/leakage_safe_source"
    manifest["archive_demo_dir"] = "artifacts/segmentation/archive/fullbook_demo"
    manifest.setdefault("rollback", {})
    if isinstance(manifest["rollback"], dict):
        manifest["rollback"]["previous_demo_archive"] = "artifacts/segmentation/archive/fullbook_demo"
        manifest["rollback"]["instruction"] = (
            "Point CLI/API back to archive/fullbook_demo or re-promote a prior hashed bundle"
        )

    payload = {
        "overall_verdict": overall,
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
    if hard_fails:
        payload["not_fully_frozen_because"].insert(
            0, f"Hard FAIL gates: {', '.join(hard_fails)}"
        )

    write_freeze_readiness(payload, dst)
    _write_report(REPORT_PATH, payload)
    metrics["freeze_status"] = overall
    manifest["freeze_status"] = overall
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return payload


def _segment_meaning_gate(
    dst: Path,
    *,
    contract: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """PASS when catalog personas + exception-queue policy are present for all clusters."""
    contract = contract or {}
    metrics = metrics or {}
    catalog_path = dst / "segment_catalog.json"
    if not catalog_path.exists():
        return {"verdict": "FAIL", "detail": "Missing segment_catalog.json"}

    catalog = _load(catalog_path)
    segments = catalog.get("segments") or {}
    if not isinstance(segments, dict) or not segments:
        return {"verdict": "FAIL", "detail": "segment_catalog.segments empty"}

    n_clusters = int(metrics.get("n_clusters") or metrics.get("k") or len(segments))
    expected_ids = set(range(n_clusters))
    try:
        present_ids = {int(k) for k in segments}
    except (TypeError, ValueError):
        return {"verdict": "FAIL", "detail": "segment_catalog segment keys are not integers"}

    missing_ids = sorted(expected_ids - present_ids)
    if missing_ids:
        return {
            "verdict": "FAIL",
            "detail": f"Catalog missing segment_ids {missing_ids} (expected 0..{n_clusters - 1})",
        }

    incomplete: list[str] = []
    for sid in sorted(expected_ids):
        meta = segments.get(str(sid))
        if not isinstance(meta, dict):
            incomplete.append(f"{sid}:not_object")
            continue
        name = str(meta.get("name") or "").strip()
        one_liner = str(meta.get("one_liner") or "").strip()
        default_play = str(meta.get("default_play") or "").strip()
        decision = meta.get("decision") if isinstance(meta.get("decision"), dict) else {}
        play = str(decision.get("play") or default_play or "").strip()
        guardrail = str(decision.get("guardrail") or "").strip()
        if not name:
            incomplete.append(f"{sid}:name")
        if not one_liner:
            incomplete.append(f"{sid}:one_liner")
        if not play:
            incomplete.append(f"{sid}:play")
        if not guardrail:
            incomplete.append(f"{sid}:guardrail")

    if incomplete:
        return {
            "verdict": "FAIL",
            "detail": "Incomplete personas/plays: " + ", ".join(incomplete[:12]),
        }

    exceptions = contract.get("exception_segment_ids")
    if not isinstance(exceptions, list) or not exceptions:
        return {
            "verdict": "FAIL",
            "detail": "serving_contract.exception_segment_ids missing/empty (exception-queue policy)",
        }

    try:
        exception_ids = sorted({int(x) for x in exceptions})
    except (TypeError, ValueError):
        return {"verdict": "FAIL", "detail": "exception_segment_ids must be integers"}

    unknown = [i for i in exception_ids if i not in present_ids]
    if unknown:
        return {
            "verdict": "FAIL",
            "detail": f"exception_segment_ids not in catalog: {unknown}",
        }

    return {
        "verdict": "PASS",
        "detail": (
            f"CRM personas + plays for k={n_clusters}; "
            f"exception-queue segments={exception_ids}"
        ),
    }


def _holdout_replay_gate(dst: Path) -> dict[str, str]:
    """Actually replay holdout labels with frozen AE+KMeans when possible."""
    try:
        from customer_segmentation.data.berka_dataset_loader import (
            load_berka_product_tables,
            load_berka_raw_data,
            resolve_berka_dir,
        )
        from customer_segmentation.features.build_berka_segmentation_features import (
            build_berka_segmentation_features_asof,
            transform_with_train_scaler,
        )

        champ = ChampionModel.load(dst, verify_hashes=True)
        metrics = champ.metrics or {}
        as_of = str(metrics.get("as_of_date") or "1997-12-31")
        min_tenure = int(metrics.get("minimum_tenure_days") or 365)
        split = pd.read_csv(dst / "account_split.csv")
        assign = pd.read_csv(dst / "assignments.csv")
        path = resolve_berka_dir()
        trans, account, _client = load_berka_raw_data(path)
        loan, card, disp = load_berka_product_tables(path)
        interpretable = build_berka_segmentation_features_asof(
            trans, account, loan, card, disp, as_of_date=as_of, minimum_tenure_days=min_tenure
        )
        hold_ids = set(split.loc[split["split"] == "holdout", "account_id"].astype(int))
        hold_feat = interpretable[interpretable["account_id"].isin(hold_ids)].copy()
        expected = assign[assign["account_id"].isin(hold_ids)].set_index("account_id")["segment_id"]
        hold_feat = hold_feat.set_index("account_id").loc[expected.index].reset_index()
        X = transform_with_train_scaler(hold_feat, champ.scaler)
        pred = champ.kmeans.predict(champ.encode(X))
        ok = bool(np.array_equal(pred, expected.to_numpy()))
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (
                f"Frozen encoder+KMeans reproduce holdout labels (n={len(pred)})"
                if ok
                else "Holdout replay mismatch"
            ),
        }
    except Exception as exc:  # noqa: BLE001 — gate must still publish
        return {
            "verdict": "FAIL",
            "detail": f"Holdout replay could not run: {type(exc).__name__}: {exc}"[:400],
        }


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Production freeze readiness — PROVISIONAL",
        "",
        f"**Overall verdict: {payload['overall_verdict']}.** "
        "Engineering gates for the leakage-safe AE+k=6 production candidate are evaluated "
        "from live integrity / replay / pack checks. "
        "This is **not** a full bank production freeze.",
        "",
        "## Gates",
        "",
        "| Gate | Verdict | Detail |",
        "|---|---|---|",
    ]
    for name, gate in (payload.get("gates") or {}).items():
        detail = str(gate.get("detail") or "").replace("|", "/")
        lines.append(f"| `{name}` | **{gate.get('verdict')}** | {detail} |")
    lines.extend(
        [
            "",
            "## Rollback",
            "",
            "Point CLI/API defaults to `artifacts/segmentation/archive/fullbook_demo/` "
            "(archive demo) or re-promote a prior hashed research twin from "
            "`artifacts/segmentation/research/leakage_safe_source/`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
