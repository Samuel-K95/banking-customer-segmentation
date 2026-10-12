"""Fairness diagnostics pack (04 §4) — concentrations + risk register.

Champion features are behavioral + product only (no age/gender in the model
matrix). We still measure geographic concentration when district is joinable,
document the behavior-vs-demographic design, and ship an explicit risk register.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from customer_segmentation.data.berka_dataset_loader import resolve_berka_dir, _read_table


RISK_REGISTER: list[dict[str, str]] = [
    {
        "risk": "Using segments for credit eligibility or pricing",
        "why": "Unsupervised clusters are not probability-of-default models",
        "mitigation": "Forbid eligibility/pricing use; require supervised risk models + policy",
        "severity": "critical",
    },
    {
        "risk": "Over-contacting dormant or stressed customers with sales",
        "why": "Wrong play drives complaints and harm",
        "mitigation": "Decision map guardrails + hard suppress lists by segment",
        "severity": "high",
    },
    {
        "risk": "Treating stressed segment as undeserving of inclusion products",
        "why": "Negative balance is a care signal, not a moral label",
        "mitigation": "Hardship / budgeting pathways; ban investment pitches",
        "severity": "high",
    },
    {
        "risk": "Geographic proxy discrimination via district concentration",
        "why": "District may correlate with socioeconomic status",
        "mitigation": "Monitor district concentration; do not target by region from segment ID alone",
        "severity": "medium",
    },
    {
        "risk": "Owner vs disponent confusion on joint accounts",
        "why": "Berka is account-grain; disp can attach multiple people",
        "mitigation": "Account-level CRM only until explicit owner policy exists",
        "severity": "medium",
    },
    {
        "risk": "Tiny segments (stressed / dormant) driving unstable ops policy",
        "why": "Small n → noisy centroids and overfit plays",
        "mitigation": "Human review queue; do not automate high-stakes actions on tiny cells",
        "severity": "medium",
    },
    {
        "risk": "Historical Czech 1990s context generalized as modern policy",
        "why": "Currency levels and era do not transfer",
        "mitigation": "Treat segments as behavioral types; revalidate on current book",
        "severity": "medium",
    },
]


def try_join_district(
    assignments: pd.DataFrame,
    *,
    id_col: str = "account_id",
    berka_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Left-join district_id from account table when raw Berka is available."""
    try:
        path = resolve_berka_dir(berka_dir)
        account = _read_table(path, "account")
    except Exception:
        return None
    if "district_id" not in account.columns:
        return None
    out = assignments.merge(
        account[[id_col if id_col in account.columns else "account_id", "district_id"]],
        left_on=id_col,
        right_on="account_id" if "account_id" in account.columns else id_col,
        how="left",
    )
    return out


def district_concentration(df: pd.DataFrame, *, segment_col: str = "segment_id") -> pd.DataFrame:
    """Max single-district share within each segment + top district id."""
    if "district_id" not in df.columns:
        return pd.DataFrame()
    rows = []
    for sid, g in df.groupby(segment_col):
        vc = g["district_id"].value_counts(normalize=True, dropna=True)
        if vc.empty:
            continue
        rows.append(
            {
                "segment_id": int(sid),
                "n": int(len(g)),
                "n_districts": int(g["district_id"].nunique(dropna=True)),
                "max_district_share": round(float(vc.iloc[0]), 4),
                "top_district_id": vc.index[0],
            }
        )
    return pd.DataFrame(rows).sort_values("max_district_share", ascending=False)


def representation_tables(df: pd.DataFrame, *, segment_col: str = "segment_id") -> dict[str, Any]:
    """P(segment|district top) style summaries when district exists."""
    if "district_id" not in df.columns:
        return {"available": False, "reason": "district_id not joined"}
    # Keep top 10 districts by volume for a readable crosstab
    top = df["district_id"].value_counts().head(10).index
    sub = df[df["district_id"].isin(top)].copy()
    p_seg_given_dist = pd.crosstab(sub["district_id"], sub[segment_col], normalize="index")
    p_dist_given_seg = pd.crosstab(sub[segment_col], sub["district_id"], normalize="index")
    return {
        "available": True,
        "top_districts": [str(x) for x in top.tolist()],
        "p_segment_given_district": p_seg_given_dist.round(3).reset_index().to_dict(orient="records"),
        "p_district_given_segment": p_dist_given_seg.round(3).reset_index().to_dict(orient="records"),
    }


def behavior_vs_demographic_note() -> dict[str, Any]:
    """Fallback narrative when ablation cannot run (no Berka / features)."""
    return {
        "model_features": [
            "Transaction_Count",
            "Account_Balance",
            "Total_Credit",
            "Total_Withdrawal",
            "Recency_Days",
            "Account_Age_Days",
            "has_loan",
            "has_card",
        ],
        "explicit_demographics_in_model": False,
        "ablation_status": "narrative_only_fallback",
        "driver_audit": (
            "Primary drivers are recency, balance sign/level, activity, and product flags "
            "(loan/card). District is not a model feature; any concentration is a usage risk."
        ),
    }


def build_fairness_pack(
    assignments: pd.DataFrame,
    *,
    segment_col: str = "segment_id",
    id_col: str = "account_id",
    berka_dir: str | Path | None = None,
    catalog: dict[str, Any] | None = None,
    feature_df: pd.DataFrame | None = None,
    n_clusters: int | None = None,
    as_of_year: int = 1997,
) -> dict[str, Any]:
    joined = try_join_district(assignments, id_col=id_col, berka_dir=berka_dir)
    conc = pd.DataFrame()
    rep: dict[str, Any] = {"available": False, "reason": "raw Berka account/district not available"}
    if joined is not None and "district_id" in joined.columns:
        conc = district_concentration(joined, segment_col=segment_col)
        rep = representation_tables(joined, segment_col=segment_col)

    # Attach names if catalog provided
    if catalog and not conc.empty:
        name_map = {int(k): v.get("name") for k, v in catalog.get("segments", {}).items()}
        conc.insert(1, "segment_name", conc["segment_id"].map(name_map))

    behavior_vs_demo: dict[str, Any] = behavior_vs_demographic_note()
    if feature_df is not None and id_col in feature_df.columns:
        try:
            from customer_segmentation.decision.demographics import (
                run_behavior_vs_demographic_ablation,
            )

            k = n_clusters
            if k is None and segment_col in feature_df.columns:
                k = int(feature_df[segment_col].nunique())
            behavior_vs_demo = run_behavior_vs_demographic_ablation(
                feature_df,
                segment_col=segment_col,
                id_col=id_col,
                n_clusters=int(k or 6),
                berka_dir=berka_dir,
                as_of_year=as_of_year,
            )
        except Exception as exc:  # pragma: no cover - soft fail keeps pack publishable
            behavior_vs_demo = {
                **behavior_vs_demographic_note(),
                "ablation_status": f"failed:{type(exc).__name__}",
                "ablation_error": str(exc)[:300],
            }

    return {
        "behavior_vs_demographic": behavior_vs_demo,
        "district_concentration": conc.to_dict(orient="records"),
        "representation": rep,
        "risk_register": RISK_REGISTER,
        "usage_constraints": [
            "Allowed: CRM content targeting, journey design, service prioritization with guardrails",
            "Forbidden: credit eligibility, pricing, collections aggression from segment ID alone",
            "Required: human review for stressed + low-confidence members",
        ],
        "limitations": [
            "Age/gender are ablation features only — not in the champion clustering matrix",
            "Berka 1993–1998 Czech context; revalidate on current book before policy use",
            "Tiny segments (stressed/dormant) need caution — small-n instability",
        ],
    }
