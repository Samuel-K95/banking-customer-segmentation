"""Segment naming, personas, and decision-map plays (decision layer).

Names are derived from centroid signatures so they stay meaningful even if
K-Means relabels IDs between re-trains. At champion registration we freeze
the resolved catalog into `segment_catalog.json`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

PERSONAS: dict[str, dict[str, str]] = {
    "New basic accounts": {
        "one_liner": "Newer accounts with light product depth; still active — onboarding window is open.",
        "default_play": "First card / digital activation; watch early dormancy.",
    },
    "Long-tenure high-activity basic": {
        "one_liner": "Loyal, high-activity, almost no loan/card. Trust exists; monetization does not.",
        "default_play": "Highest-value cross-sell: pre-approved card / personal loan from txn history.",
    },
    "Loan-centric high-volume": {
        "one_liner": "Credit relationship already established; high flows around loan servicing.",
        "default_play": "Card attach, PPI/protection, refinance / top-up retention.",
    },
    "Dormant long-term accounts": {
        "one_liner": "Long recency gap — relationship effectively elsewhere.",
        "default_play": "Win-back then close; do not push sales hard.",
    },
    "Card-centric active savers": {
        "one_liner": "High card penetration, healthy balances, no loan — solvent, engaged spenders.",
        "default_play": "Premium card upgrade + careful first-loan offer; savings/invest.",
    },
    "Financially stressed / overdraft risk": {
        "one_liner": "Negative mean balance; withdrawals dominate credits.",
        "default_play": "Service + risk review, budgeting tools — not investment pitches.",
    },
    "Mixed / review": {
        "one_liner": "No single dominant driver — inspect lift table before CRM use.",
        "default_play": "Hold for analyst review / low-confidence queue.",
    },
}

DECISION_PLAYS: dict[str, dict[str, str]] = {
    "New basic accounts": {
        "play": "Onboarding + first card",
        "evidence": "Short tenure, low loan/card penetration, still recent activity",
        "owner": "CRM / Digital",
        "kpi": "Card attach rate; 90-day activation",
        "guardrail": "Suitability checks; no aggressive credit",
    },
    "Long-tenure high-activity basic": {
        "play": "Priority cross-sell (card -> personal loan)",
        "evidence": "Highest activity + long tenure + ~0 credit products",
        "owner": "Product / CRM",
        "kpi": "Product/household lift; offer acceptance",
        "guardrail": "Use txn history for pre-approve; consent + eligibility",
    },
    "Loan-centric high-volume": {
        "play": "Card attach + loan protection / refinance",
        "evidence": "~100% loan holding, high credit/withdrawal flows",
        "owner": "Lending + Cards",
        "kpi": "Card penetration among borrowers; retention at refinance",
        "guardrail": "Avoid over-leverage; PPI disclosure rules",
    },
    "Dormant long-term accounts": {
        "play": "Win-back then closure path",
        "evidence": "Recency >> other segments (months-years inactive)",
        "owner": "Retention / Ops",
        "kpi": "Reactivation rate; dormant balance recovered / closed",
        "guardrail": "Not a churn model — strategy from dormancy proxy only",
    },
    "Card-centric active savers": {
        "play": "Premium card + first loan to solvent customers",
        "evidence": "High card penetration + highest balances + no loan",
        "owner": "Cards / Wealth",
        "kpi": "Interchange / premium upgrade; responsible first-loan take-up",
        "guardrail": "Credit assessment still required",
    },
    "Financially stressed / overdraft risk": {
        "play": "Service & risk review; budgeting tools",
        "evidence": "Negative mean balance; withdrawals >= credits",
        "owner": "Risk / Customer Care",
        "kpi": "Overdraft cure rate; complaint reduction",
        "guardrail": "Do NOT pitch investments; limit new unsecured credit",
    },
}


def name_from_centroid(row: pd.Series, population: pd.Series) -> str:
    """Behavior-first name from a segment centroid row (original scale)."""
    if float(row.get("Recency_Days", 0)) >= 90:
        return "Dormant long-term accounts"
    if float(row.get("Account_Balance", 0)) < 0:
        return "Financially stressed / overdraft risk"
    if float(row.get("has_loan", 0)) >= 0.85:
        return "Loan-centric high-volume"
    if float(row.get("has_card", 0)) >= 0.85:
        return "Card-centric active savers"
    age_ref = float(population.get("Account_Age_Days", 0)) or 1.0
    txn_ref = float(population.get("Transaction_Count", 0)) or 0.0
    if (
        float(row.get("Account_Age_Days", 0)) <= age_ref * 0.75
        and float(row.get("has_loan", 0)) < 0.1
        and float(row.get("has_card", 0)) < 0.15
    ):
        return "New basic accounts"
    if (
        float(row.get("Transaction_Count", 0)) >= txn_ref
        and float(row.get("has_loan", 0)) < 0.1
        and float(row.get("has_card", 0)) < 0.15
    ):
        return "Long-tenure high-activity basic"
    return "Mixed / review"


def build_segment_catalog(
    centroids: pd.DataFrame,
    *,
    population: pd.Series | None = None,
    sizes: dict[int, int] | None = None,
    total_n: int | None = None,
) -> dict[str, Any]:
    """Freeze segment_id → name / persona / play from centroids."""
    if population is None:
        if sizes and total_n:
            # weighted approx not available — use unweighted mean of centroids
            population = centroids.mean()
        else:
            population = centroids.mean()

    segments: dict[str, Any] = {}
    for sid, row in centroids.iterrows():
        name = name_from_centroid(row, population)
        persona = PERSONAS.get(name, PERSONAS["Mixed / review"])
        play = DECISION_PLAYS.get(name, {})
        n = int(sizes.get(int(sid), 0)) if sizes else None
        pct = round(100.0 * n / total_n, 2) if n is not None and total_n else None
        segments[str(int(sid))] = {
            "segment_id": int(sid),
            "name": name,
            "n": n,
            "pct": pct,
            "one_liner": persona["one_liner"],
            "default_play": persona["default_play"],
            "decision": play,
            "centroid": {c: float(row[c]) for c in centroids.columns},
        }
    return {
        "naming_policy": "behavior_rules_from_centroid",
        "segments": segments,
    }


def decision_map_frame(catalog: dict[str, Any]) -> pd.DataFrame:
    """Flat decision-map table from a frozen catalog."""
    rows = []
    for sid, meta in catalog.get("segments", {}).items():
        d = meta.get("decision") or {}
        if not d:
            continue
        rows.append(
            {
                "segment_id": int(sid),
                "segment": meta["name"],
                "pct_book": meta.get("pct"),
                "evidence": d.get("evidence"),
                "play": d.get("play"),
                "owner": d.get("owner"),
                "kpi": d.get("kpi"),
                "guardrail": d.get("guardrail"),
            }
        )
    return pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
