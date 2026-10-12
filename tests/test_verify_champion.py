"""Verifier + integrity gates for the production champion."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from customer_segmentation.models.artifact_integrity import (
    integrity_report,
    missing_required_artifacts,
    verify_serving_contract,
)
from customer_segmentation.models.champion import ChampionModel
from customer_segmentation.pipelines.verify_champion import verify_champion

REPO = Path(__file__).resolve().parents[1]
CHAMPION = REPO / "artifacts" / "segmentation" / "champion"


@pytest.mark.skipif(not CHAMPION.exists(), reason="champion missing")
def test_integrity_report_requires_binaries():
    missing = missing_required_artifacts(CHAMPION)
    # If binaries are present locally / in git, report must be clean
    if (CHAMPION / "encoder.pt").exists() and (CHAMPION / "scaler.joblib").exists():
        assert missing == []
        report = integrity_report(CHAMPION)
        assert report["ok"], report["errors"]
        assert verify_serving_contract(CHAMPION) == []


@pytest.mark.skipif(
    not (CHAMPION / "encoder.pt").exists(),
    reason="encoder.pt not present",
)
def test_champion_load_requires_encoder_and_hashes():
    champ = ChampionModel.load(CHAMPION)
    assert champ.encoder is not None
    assert champ.scaler is not None
    assert champ.kmeans is not None
    summary = champ.summary()
    assert summary["has_encoder"] is True


@pytest.mark.skipif(
    not (CHAMPION / "encoder.pt").exists(),
    reason="encoder.pt not present",
)
def test_production_candidate_fails_closed_without_serving_contract(tmp_path: Path):
    dest = tmp_path / "champion"
    shutil.copytree(CHAMPION, dest)
    contract = dest / "serving_contract.json"
    assert contract.exists()
    contract.unlink()

    with pytest.raises(FileNotFoundError, match="serving_contract.json missing"):
        ChampionModel.load(dest, verify_hashes=True)

    # Explicit opt-out still loads (research / offline tooling only).
    champ = ChampionModel.load(dest, verify_hashes=False)
    assert champ.encoder is not None


@pytest.mark.skipif(
    not (CHAMPION / "encoder.pt").exists(),
    reason="encoder.pt not present",
)
def test_verify_champion_core_checks(monkeypatch):
    import os

    os.chdir(REPO)
    report = verify_champion(include_api=True, include_replay=False, include_path_scan=True)
    assert report["ok"], report
    names = {c["name"]: c for c in report["checks"]}
    assert names["bundle_integrity"]["ok"]
    assert names["loadability"]["ok"]
    assert names["decision_layer"]["ok"]
    assert names["stale_path_scan"]["ok"], names["stale_path_scan"].get("errors")
    assert names["api_smoke"]["ok"], names["api_smoke"].get("errors")
