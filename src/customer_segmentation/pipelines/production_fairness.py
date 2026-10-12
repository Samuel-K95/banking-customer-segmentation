"""District fairness pack + OWNER/DISPONENT policy artifacts for production champion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from customer_segmentation.data.berka_dataset_loader import resolve_berka_dir
from customer_segmentation.decision.fairness import (
    build_fairness_pack,
    representation_tables,
    try_join_district,
)

PRODUCTION_DIR = Path("artifacts/segmentation/champion")
POLICY_REPORT = Path("reports/champion/04_owner_disponent_policy.md")


def run_production_fairness(
    *,
    production_dir: str | Path | None = None,
    berka_dir: str | Path | None = None,
) -> dict[str, Any]:
    dst = Path(production_dir) if production_dir else PRODUCTION_DIR
    assign = pd.read_csv(dst / "assignments.csv")
    # Prefer as-of interpretable rebuilt; fall back to assignment-only frame
    from customer_segmentation.data.berka_dataset_loader import (
        load_berka_product_tables,
        load_berka_raw_data,
    )
    from customer_segmentation.features.build_berka_segmentation_features import (
        build_berka_segmentation_features_asof,
    )

    metrics = json.loads((dst / "metrics.json").read_text(encoding="utf-8"))
    as_of = str(metrics.get("as_of_date", "1997-12-31"))
    min_tenure = int(metrics.get("minimum_tenure_days", 365))

    path = resolve_berka_dir(berka_dir)
    trans, account, _client = load_berka_raw_data(path)
    loan, card, disp = load_berka_product_tables(path)
    interpretable = build_berka_segmentation_features_asof(
        trans, account, loan, card, disp, as_of_date=as_of, minimum_tenure_days=min_tenure
    )
    df = interpretable.merge(assign[["account_id", "segment_id"]], on="account_id", how="inner")

    catalog = json.loads((dst / "segment_catalog.json").read_text(encoding="utf-8"))
    name_map = {int(k): v.get("name", "") for k, v in (catalog.get("segments") or {}).items()}
    df["segment_name"] = df["segment_id"].map(name_map)

    # build_fairness_pack joins district itself — do not pre-join (avoids district_id_x/y).
    pack = build_fairness_pack(df, berka_dir=path, catalog=catalog)
    pack["owner_disponent"] = _disp_summary(disp)
    pack["policy_doc"] = str(POLICY_REPORT)
    pack["generated_at"] = datetime.now(timezone.utc).isoformat()
    pack["model_version"] = metrics.get("model_version") or "production_candidate_v1"
    pack["compliance_signoff"] = "PENDING"
    pack["district_available"] = bool((pack.get("representation") or {}).get("available")) or bool(
        pack.get("district_concentration")
    )

    out = dst / "fairness_pack.json"
    out.write_text(json.dumps(pack, indent=2, default=str), encoding="utf-8")
    _write_policy_doc(POLICY_REPORT, pack)

    joined = try_join_district(df, berka_dir=path)
    rep = representation_tables(joined if joined is not None else df)
    (dst / "fairness_representation.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8"
    )

    district_ok = bool((pack.get("representation") or {}).get("available")) or bool(
        pack.get("district_concentration")
    )
    return {
        "fairness_pack": str(out),
        "policy_doc": str(POLICY_REPORT),
        "district_available": district_ok,
        "n_accounts": int(len(df)),
    }


def _disp_summary(disp: pd.DataFrame | None) -> dict[str, Any]:
    if disp is None or disp.empty or "type" not in disp.columns:
        return {"available": False}
    counts = disp["type"].astype(str).str.upper().value_counts().to_dict()
    n_accounts = int(disp["account_id"].nunique()) if "account_id" in disp.columns else None
    multi = 0
    if "account_id" in disp.columns:
        multi = int((disp.groupby("account_id").size() > 1).sum())
    return {
        "available": True,
        "type_counts": {str(k): int(v) for k, v in counts.items()},
        "n_accounts_in_disp": n_accounts,
        "n_accounts_with_multiple_disp": multi,
        "scoring_grain": "account_id",
        "policy": (
            "Score and treat at account grain only. Do not attribute segment membership "
            "to a specific OWNER or DISPONENT without an explicit joint-account policy."
        ),
    }


def _write_policy_doc(path: Path, pack: dict[str, Any]) -> None:
    od = pack.get("owner_disponent") or {}
    lines = [
        "# Production OWNER / DISPONENT policy",
        "",
        "**Status:** published for engineering freeze · **Compliance sign-off:** PENDING",
        "",
        "## Grain",
        "",
        "The production champion scores **accounts**, not clients. Berka `disp` can attach",
        "OWNER and DISPONENT rows to the same `account_id`. Segment membership must not be",
        "interpreted as a person-level label without an approved mapping policy.",
        "",
        "## Rules",
        "",
        "1. Features and `segment_id` are account-grain only.",
        "2. Do not use `segment_id` for credit eligibility, pricing, or collections aggression.",
        "3. Joint accounts (`n_accounts_with_multiple_disp` > 0 possible): suppress person-targeted",
        "   offers derived solely from the account segment until an OWNER-primary policy is approved.",
        "4. Exception-queue segments (stressed / dormant / tiny cells) require human review.",
        "5. Geography: monitor district concentration; do not target by district from segment alone.",
        "",
        "## Snapshot from fairness pack",
        "",
        f"- District join available: `{pack.get('district_available')}`",
        f"- DISP type counts: `{od.get('type_counts')}`",
        f"- Accounts with multiple disp rows: `{od.get('n_accounts_with_multiple_disp')}`",
        "",
        "## Rollback / misuse",
        "",
        "If segment outputs are used for adverse decisions, revoke serving access and roll back",
        "to the previous hashed production bundle.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_production_fairness(), indent=2, default=str))
