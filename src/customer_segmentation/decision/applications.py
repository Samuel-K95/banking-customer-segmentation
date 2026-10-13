"""Banking application portfolio — all domains from 04-analysis-and-decision-layer.

Each play cites segment evidence and stays strategy/design, not a supervised
churn/default/propensity predictor unless a separate model exists.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Domains A–E from the brief, plus Risk/Collections as an explicit ops domain.
DOMAINS = (
    "product_recommendations",
    "retention_strategies",
    "financial_inclusion",
    "targeted_engagement",
    "service_improvement",
    "risk_and_care",
)

# Multi-play catalog keyed by champion segment name.
APPLICATION_PORTFOLIO: dict[str, list[dict[str, str]]] = {
    "New basic accounts": [
        {
            "domain": "product_recommendations",
            "play": "First card / debit attach during onboarding",
            "evidence": "Short tenure, ~0 loan/card, still recently active",
            "action": "Day-7 and Day-30 digital card offers with light KYC friction",
            "expected_value": "Higher first-product attach; lower early mono-product attrition",
            "owner": "CRM / Digital",
            "kpi": "Card attach within 90 days; activated card rate",
            "guardrail": "Suitability only; no aggressive unsecured credit in first 90 days",
        },
        {
            "domain": "financial_inclusion",
            "play": "Basic-account literacy + micro-savings nudge",
            "evidence": "Newer, product-thin accounts with moderate balances",
            "action": "In-app savings goals and fee transparency for new-to-bank customers",
            "expected_value": "Ethical growth + stickier early relationship",
            "owner": "Inclusion / Digital",
            "kpi": "Savings-goal adoption; 90-day retention of new accounts",
            "guardrail": "No predatory cross-sell; avoid high-fee products",
        },
        {
            "domain": "targeted_engagement",
            "play": "Onboarding cadence (high-touch early, then taper)",
            "evidence": "Active recency but shallow product depth",
            "action": "3-message onboarding journey then suppress unless dormancy signal",
            "expected_value": "Higher activation without spam complaints",
            "owner": "CRM",
            "kpi": "Journey completion; opt-out rate",
            "guardrail": "Contact-policy caps; consent required",
        },
        {
            "domain": "retention_strategies",
            "play": "Early-dormancy watchlist",
            "evidence": "New accounts are highest risk of silent abandonment",
            "action": "If recency drifts >30d in first year, trigger win-back (not cold sales)",
            "expected_value": "Fewer zombie accounts from the new-book cohort",
            "owner": "Retention",
            "kpi": "Share of new book still active at day 180",
            "guardrail": "Strategy from dormancy proxy — not a labeled churn model",
        },
    ],
    "Long-tenure high-activity basic": [
        {
            "domain": "product_recommendations",
            "play": "Priority cross-sell: card then personal loan",
            "evidence": "Highest txn count, long tenure, ~0 loan and ~0 card",
            "action": "Pre-approved card limit from txn history; sequenced loan offer later",
            "expected_value": "Largest monetization gap closed with lowest acquisition cost",
            "owner": "Product / CRM",
            "kpi": "Products per household; offer acceptance",
            "guardrail": "Eligibility + consent; do not treat activity as credit approval",
        },
        {
            "domain": "targeted_engagement",
            "play": "Relationship-manager / personalized data-driven outreach",
            "evidence": "Loyal, high-activity; cold blast unlikely to convert",
            "action": "Personalized 'you qualify' style offers tied to observed flows",
            "expected_value": "Higher conversion vs mass campaigns",
            "owner": "CRM",
            "kpi": "Response rate; complaint rate",
            "guardrail": "Explainability of offer logic; no dark patterns",
        },
        {
            "domain": "service_improvement",
            "play": "Protect high-activity experience (frictionless digital)",
            "evidence": "Heavy usage implies cost-to-serve and loyalty sensitivity",
            "action": "Priority digital support queue; fee-waiver pilots for service failures",
            "expected_value": "NPS lift; reduced silent attrition of best users",
            "owner": "Service",
            "kpi": "NPS / CSAT among high-activity basics",
            "guardrail": "Service investment must not become unfair preferential pricing",
        },
    ],
    "Loan-centric high-volume": [
        {
            "domain": "product_recommendations",
            "play": "Card attach + payment-protection / refinance retention",
            "evidence": "~100% loan holding; high credit/withdrawal flows",
            "action": "Card for borrowers; PPI disclosure; proactive refinance at rate triggers",
            "expected_value": "Higher wallet share; lower refinance leakage to competitors",
            "owner": "Lending + Cards",
            "kpi": "Card penetration among borrowers; refinance retention",
            "guardrail": "Avoid over-leverage; full PPI disclosure",
        },
        {
            "domain": "targeted_engagement",
            "play": "Loan-lifecycle messaging (disbursement → servicing → maturity)",
            "evidence": "Credit relationship already established",
            "action": "Stage-based comms; suppress generic investment pitches",
            "expected_value": "Higher relevance; lower unsubscribe",
            "owner": "Lending CRM",
            "kpi": "Engagement rate by loan stage",
            "guardrail": "Suitability for add-on credit products",
        },
        {
            "domain": "service_improvement",
            "play": "Assisted servicing for high-flow loan accounts",
            "evidence": "High volumes around loan servicing",
            "action": "Dedicated repayment support and payment-plan UX",
            "expected_value": "Fewer arrears escalations from process friction",
            "owner": "Lending Ops",
            "kpi": "Missed-payment contacts resolved without collections",
            "guardrail": "Not a default predictor — servicing design only",
        },
    ],
    "Card-centric active savers": [
        {
            "domain": "product_recommendations",
            "play": "Premium card upgrade + careful first loan",
            "evidence": "~100% card, highest balances, no loan",
            "action": "Rewards/premium upgrade; solvent first-loan invite with underwriting",
            "expected_value": "Interchange + responsible lending growth",
            "owner": "Cards / Wealth",
            "kpi": "Premium upgrade rate; first-loan take-up among eligible",
            "guardrail": "Credit assessment still required; balance ≠ approval",
        },
        {
            "domain": "targeted_engagement",
            "play": "Wealth / savings content with low contact fatigue",
            "evidence": "Solvent, engaged spenders — high response potential if relevant",
            "action": "Monthly wealth digest; suppress after two non-opens",
            "expected_value": "AUM growth without over-contact",
            "owner": "Wealth / CRM",
            "kpi": "Open rate; savings product take-up",
            "guardrail": "Investment advice suitability rules",
        },
        {
            "domain": "service_improvement",
            "play": "Digital-first premium service",
            "evidence": "Card-heavy, active — likely digital-native usage",
            "action": "Invest in card controls UX and dispute speed",
            "expected_value": "Lower cost-to-serve; higher card loyalty",
            "owner": "Cards / Digital",
            "kpi": "Dispute resolution time; digital login frequency",
            "guardrail": "Do not infer demographics from card use",
        },
    ],
    "Dormant long-term accounts": [
        {
            "domain": "retention_strategies",
            "play": "Win-back then orderly closure",
            "evidence": "Recency hundreds of days; very low lifetime txn count",
            "action": "Single win-back offer → if no response, closure / dormancy fee path",
            "expected_value": "Recovered balances or clean book; regulatory dormancy compliance",
            "owner": "Retention / Ops",
            "kpi": "Reactivation rate; dormant accounts closed cleanly",
            "guardrail": "Not a churn classifier — dormancy-proxy strategy only",
        },
        {
            "domain": "targeted_engagement",
            "play": "Suppress sales; one reactivation channel only",
            "evidence": "Relationship effectively elsewhere",
            "action": "Remove from cross-sell audiences; email/SMS reactivation once",
            "expected_value": "Lower complaint/spam cost",
            "owner": "CRM Ops",
            "kpi": "Contact volume on dormant book; complaint rate",
            "guardrail": "Hard suppress after one attempt",
        },
        {
            "domain": "service_improvement",
            "play": "Low-cost ops handling for zombie accounts",
            "evidence": "Tiny share of book but long-lived inactive stock",
            "action": "Automated dormancy workflow; human only on dispute",
            "expected_value": "Cost-to-serve reduction",
            "owner": "Ops",
            "kpi": "Cost per dormant account; SLA on closure requests",
            "guardrail": "Customer-request access must remain easy",
        },
    ],
    "Financially stressed / overdraft risk": [
        {
            "domain": "risk_and_care",
            "play": "Service & risk review; budgeting tools — no investment pitch",
            "evidence": "Negative mean balance; withdrawals ≥ credits",
            "action": "Flag for care outreach; offer budgeting tools / debt consolidation review",
            "expected_value": "Cure overdrafts; fewer complaints; lower credit losses",
            "owner": "Risk / Customer Care",
            "kpi": "Overdraft cure rate; new unsecured credit blocked rate",
            "guardrail": "Do NOT pitch investments or premium products; limit new unsecured credit",
        },
        {
            "domain": "financial_inclusion",
            "play": "Responsible restructuring / hardship pathways",
            "evidence": "Financial stress signals without assuming protected-class status",
            "action": "Hardship form + counseling referral; secured short-bridge only if eligible",
            "expected_value": "ESG/regulatory alignment; retained relationship when cured",
            "owner": "Care / Inclusion",
            "kpi": "Hardship uptake; cured vs closed outcomes",
            "guardrail": "No predatory refinance; demographic labels forbidden in naming/targeting copy",
        },
        {
            "domain": "service_improvement",
            "play": "Empathy-first contact scripts",
            "evidence": "High sensitivity to fees and collections tone",
            "action": "Train care agents; fee-transparency checklist before any sales",
            "expected_value": "Complaint reduction; trust repair",
            "owner": "Customer Care",
            "kpi": "Complaint rate; FCR on stress cases",
            "guardrail": "Segment must not drive collections aggression alone",
        },
        {
            "domain": "retention_strategies",
            "play": "Retain via care, not discounts-for-spend",
            "evidence": "Stress segment; spend incentives may worsen balances",
            "action": "Fee relief pilots tied to repayment plans — not cashback-on-spend",
            "expected_value": "Stabilized balances; fewer forced closures",
            "owner": "Care / Risk",
            "kpi": "Plan completion rate",
            "guardrail": "Not a default model; human review for high-risk actions",
        },
    ],
}


def expand_decision_map(catalog: dict[str, Any]) -> pd.DataFrame:
    """Flatten catalog segment names into a multi-domain decision map."""
    rows: list[dict[str, Any]] = []
    for sid, meta in catalog.get("segments", {}).items():
        name = meta.get("name")
        plays = APPLICATION_PORTFOLIO.get(name, [])
        for play in plays:
            rows.append(
                {
                    "segment_id": int(sid),
                    "segment": name,
                    "pct_book": meta.get("pct"),
                    "n": meta.get("n"),
                    "domain": play["domain"],
                    "play": play["play"],
                    "evidence": play["evidence"],
                    "action": play["action"],
                    "expected_value": play["expected_value"],
                    "owner": play["owner"],
                    "kpi": play["kpi"],
                    "guardrail": play["guardrail"],
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "segment_id"]).reset_index(drop=True)


def domain_coverage_summary(decision_map: pd.DataFrame) -> pd.DataFrame:
    """Count plays and segments touched per domain."""
    if decision_map.empty:
        return pd.DataFrame(columns=["domain", "n_plays", "n_segments"])
    return (
        decision_map.groupby("domain")
        .agg(n_plays=("play", "count"), n_segments=("segment_id", "nunique"))
        .reindex(DOMAINS)
        .fillna(0)
        .astype({"n_plays": int, "n_segments": int})
        .reset_index()
    )
