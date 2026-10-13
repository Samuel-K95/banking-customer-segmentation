"""Decision-layer unit smoke tests (no raw Berka required)."""

from __future__ import annotations

import pandas as pd

from customer_segmentation.decision.applications import (
    APPLICATION_PORTFOLIO,
    DOMAINS,
    domain_coverage_summary,
    expand_decision_map,
)
from customer_segmentation.decision.profiles import compute_centroids_and_lift, nearest_segment_contrast


def test_application_portfolio_covers_all_champion_names():
    names = {
        "New basic accounts",
        "Long-tenure high-activity basic",
        "Loan-centric high-volume",
        "Dormant long-term accounts",
        "Card-centric active savers",
        "Financially stressed / overdraft risk",
    }
    assert names.issubset(APPLICATION_PORTFOLIO.keys())
    for name in names:
        assert len(APPLICATION_PORTFOLIO[name]) >= 1


def test_expand_decision_map_multi_domain():
    catalog = {
        "segments": {
            "0": {"segment_id": 0, "name": "Card-centric active savers", "n": 10, "pct": 1.0},
            "1": {"segment_id": 1, "name": "New basic accounts", "n": 20, "pct": 2.0},
        }
    }
    frame = expand_decision_map(catalog)
    assert len(frame) >= 2
    assert set(frame["domain"]).issubset(set(DOMAINS))
    cov = domain_coverage_summary(frame)
    assert cov["n_plays"].sum() == len(frame)


def test_centroids_and_contrast():
    df = pd.DataFrame(
        {
            "segment_id": [0, 0, 1, 1],
            "Transaction_Count": [10, 12, 100, 110],
            "Account_Balance": [1000, 1200, -50, -80],
            "Total_Credit": [100, 120, 500, 520],
            "Total_Withdrawal": [80, 90, 600, 610],
            "Recency_Days": [1, 2, 5, 6],
            "Account_Age_Days": [100, 120, 800, 820],
            "has_loan": [0, 0, 0, 0],
            "has_card": [0, 0, 0, 0],
        }
    )
    centroids, lift, _ = compute_centroids_and_lift(df)
    assert list(centroids.index) == [0, 1]
    assert "Transaction_Count" in lift.columns
    contrast = nearest_segment_contrast(centroids)
    assert len(contrast) == 2
