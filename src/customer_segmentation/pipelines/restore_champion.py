"""Restore a complete production champion bundle on a clean checkout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from customer_segmentation.models.artifact_integrity import integrity_report, preflight_bundle
from customer_segmentation.pipelines.promote_production import (
    PRODUCTION_DIR,
    RESEARCH_DIR,
    PromotionError,
    promote_production,
)


def _berka_available() -> bool:
    try:
        from customer_segmentation.data.berka_dataset_loader import resolve_berka_dir

        resolve_berka_dir(None)
        return True
    except Exception:
        return False


def restore_champion(
    *,
    champion_dir: str | Path | None = None,
    research_dir: str | Path | None = None,
    run_decision: bool = True,
) -> dict[str, Any]:
    """Ensure ``artifacts/segmentation/champion`` is complete and integrity-clean.

    Order:
    1. If champion already passes integrity → no-op.
    2. Else if research twin is complete → promote.
    3. Else if Berka raw is available → train leakage-safe twin, promote, optional decision.
    4. Else fail with instructions.
    """
    dest = Path(champion_dir) if champion_dir else PRODUCTION_DIR
    research = Path(research_dir) if research_dir else RESEARCH_DIR
    steps: list[str] = []

    if dest.exists():
        report = integrity_report(dest)
        if report["ok"]:
            return {
                "status": "ok",
                "action": "noop",
                "champion_dir": str(dest),
                "integrity": report,
                "steps": ["champion already complete"],
            }

    if research.exists() and not preflight_bundle(research):
        steps.append(f"promote from {research}")
        try:
            promo = promote_production(source_dir=research, dest_dir=dest)
        except PromotionError as exc:
            return {"status": "error", "action": "promote_failed", "detail": str(exc), "steps": steps}
        steps.append("promoted")
        if run_decision:
            try:
                from customer_segmentation.decision.pipeline import run_decision_layer

                run_decision_layer(
                    champion_dir=str(dest),
                    out_dir="artifacts/decision/champion",
                )
                steps.append("decision-berka")
            except Exception as exc:  # noqa: BLE001
                steps.append(f"decision-berka skipped: {exc}")
        return {
            "status": "ok",
            "action": "promoted_from_research",
            "champion_dir": str(dest),
            "promote": promo,
            "integrity": integrity_report(dest),
            "steps": steps,
        }

    if _berka_available():
        steps.append("segment-berka-leakage-safe")
        from customer_segmentation.pipelines.run_berka_segmentation_leakage_safe import (
            run_berka_segmentation_leakage_safe,
        )

        train = run_berka_segmentation_leakage_safe(output_directory=str(research))
        steps.append("promote-production")
        promo = promote_production(source_dir=research, dest_dir=dest)
        if run_decision:
            from customer_segmentation.decision.pipeline import run_decision_layer

            run_decision_layer(champion_dir=str(dest), out_dir="artifacts/decision/champion")
            steps.append("decision-berka")
        return {
            "status": "ok",
            "action": "trained_and_promoted",
            "champion_dir": str(dest),
            "train": {"champion_dir": train.get("champion_dir"), "freeze_status": train.get("freeze_status")},
            "promote": promo,
            "integrity": integrity_report(dest),
            "steps": steps,
        }

    return {
        "status": "error",
        "action": "cannot_restore",
        "champion_dir": str(dest),
        "steps": steps,
        "detail": (
            "Champion binaries missing and no research twin / Berka raw available. "
            "Run: docker compose run --rm data-layer  then  bcs segment-berka-leakage-safe "
            "&& bcs promote-production  (or mount a complete champion bundle)."
        ),
    }


def main() -> int:
    result = restore_champion()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
