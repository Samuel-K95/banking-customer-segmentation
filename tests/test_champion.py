"""ChampionModel load + score smoke tests (uses registered artifacts if present)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.models.champion import ChampionModel
from customer_segmentation.models.segment_intel import (
    build_segment_catalog,
    name_from_centroid,
)

ROOT = Path(__file__).resolve().parents[1]
CHAMP_DIR = ROOT / "artifacts" / "segmentation" / "champion"


def test_name_from_centroid_rules():
    pop = pd.Series(
        {
            "Transaction_Count": 200,
            "Account_Balance": 40000,
            "Total_Credit": 500000,
            "Total_Withdrawal": 500000,
            "Recency_Days": 5,
            "Account_Age_Days": 1200,
            "has_loan": 0.2,
            "has_card": 0.2,
        }
    )
    dormant = pop.copy()
    dormant["Recency_Days"] = 500
    assert name_from_centroid(dormant, pop) == "Dormant long-term accounts"

    stressed = pop.copy()
    stressed["Account_Balance"] = -1000
    assert name_from_centroid(stressed, pop) == "Financially stressed / overdraft risk"

    loan = pop.copy()
    loan["has_loan"] = 1.0
    assert name_from_centroid(loan, pop) == "Loan-centric high-volume"


def test_build_segment_catalog_keys():
    centroids = pd.DataFrame(
        {
            "Transaction_Count": [100, 20],
            "Account_Balance": [10000, -500],
            "Total_Credit": [1e5, 1e5],
            "Total_Withdrawal": [1e5, 1.1e5],
            "Recency_Days": [2, 10],
            "Account_Age_Days": [800, 1300],
            "has_loan": [0.0, 0.2],
            "has_card": [0.0, 0.0],
        },
        index=[0, 1],
    )
    cat = build_segment_catalog(centroids, sizes={0: 10, 1: 5}, total_n=15)
    assert "0" in cat["segments"] and "1" in cat["segments"]
    assert cat["segments"]["1"]["name"] == "Financially stressed / overdraft risk"


@pytest.mark.requires_torch
@pytest.mark.skipif(not (CHAMP_DIR / "kmeans.joblib").exists(), reason="champion bundle not registered")
def test_champion_load_and_score():
    champ = ChampionModel.load(CHAMP_DIR)
    summary = champ.summary()
    assert summary["run_id"]
    assert summary["has_scaler"]
    assert summary["n_clusters"] == 6
    assert "leakage_safe" in str(summary["run_id"]) or champ.model_version

    # score a tiny synthetic batch in interpretable scale
    rows = pd.DataFrame(
        [
            {
                "account_id": 999001,
                "Transaction_Count": 50,
                "Account_Balance": 20000,
                "Total_Credit": 100000,
                "Total_Withdrawal": 80000,
                "Recency_Days": 3,
                "Account_Age_Days": 600,
                "has_loan": 0,
                "has_card": 0,
            },
            {
                "account_id": 999002,
                "Transaction_Count": 30,
                "Account_Balance": -2000,
                "Total_Credit": 50000,
                "Total_Withdrawal": 60000,
                "Recency_Days": 15,
                "Account_Age_Days": 1000,
                "has_loan": 0,
                "has_card": 0,
            },
        ]
    )
    scored = champ.score_interpretable(rows)
    assert len(scored) == 2
    assert set(scored.columns) >= {"account_id", "segment_id", "confidence", "segment_name"}
    assert scored["segment_id"].between(0, 5).all()
    assert np.isfinite(scored["confidence"]).all()
