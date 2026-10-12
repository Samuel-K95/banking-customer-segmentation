"""End-to-end segmentation pipelines.

Public entrypoints (also available via ``bcs`` CLI):

- ``run_berka_segmentation`` — Track C bake-off → archive full-book demo
- ``run_berka_segmentation_leakage_safe`` — leakage-safe research twin
- ``promote_production`` — research twin → ``artifacts/segmentation/champion``
- ``restore_champion`` / ``verify_champion`` — clean-checkout restore + integrity gate
- ``run_decision_layer`` — prefer ``customer_segmentation.decision``
"""

from __future__ import annotations

__all__ = [
    "run_berka_segmentation",
    "run_berka_segmentation_leakage_safe",
    "promote_production",
    "restore_champion",
    "verify_champion",
    "assemble_freeze_readiness",
    "run_production_fairness",
    "run_production_temporal",
]


def __getattr__(name: str):
    if name == "run_berka_segmentation":
        from customer_segmentation.pipelines.run_berka_segmentation import run_berka_segmentation

        return run_berka_segmentation
    if name == "run_berka_segmentation_leakage_safe":
        from customer_segmentation.pipelines.run_berka_segmentation_leakage_safe import (
            run_berka_segmentation_leakage_safe,
        )

        return run_berka_segmentation_leakage_safe
    if name == "promote_production":
        from customer_segmentation.pipelines.promote_production import promote_production

        return promote_production
    if name == "restore_champion":
        from customer_segmentation.pipelines.restore_champion import restore_champion

        return restore_champion
    if name == "verify_champion":
        from customer_segmentation.pipelines.verify_champion import verify_champion

        return verify_champion
    if name == "assemble_freeze_readiness":
        from customer_segmentation.pipelines.production_freeze import assemble_freeze_readiness

        return assemble_freeze_readiness
    if name == "run_production_fairness":
        from customer_segmentation.pipelines.production_fairness import run_production_fairness

        return run_production_fairness
    if name == "run_production_temporal":
        from customer_segmentation.pipelines.production_temporal import run_production_temporal

        return run_production_temporal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
