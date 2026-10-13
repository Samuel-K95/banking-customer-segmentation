"""Comparison board + Track C report listing endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
BOARD = REPO / "artifacts" / "segmentation" / "champion" / "comparison_board.csv"
TRACK = REPO / "reports" / "track_c_berka"


@pytest.fixture
def client(monkeypatch):
    import os

    os.chdir(REPO)
    from customer_segmentation.serve.app import app
    from customer_segmentation.serve.deps import get_champion

    get_champion.cache_clear()
    with TestClient(app) as c:
        yield c
    get_champion.cache_clear()


@pytest.mark.skipif(not BOARD.exists(), reason="comparison_board.csv missing")
def test_comparison_board_champion(client):
    r = client.get("/v1/comparison/board")
    assert r.status_code == 200
    body = r.json()
    assert body["board"] == "champion"
    assert body["n"] >= 1
    assert any("product_champion" in str(row.get("role", "")) for row in body["rows"])


def test_comparison_board_rejects_unknown(client):
    r = client.get("/v1/comparison/board", params={"board": "nope"})
    assert r.status_code == 400


@pytest.mark.skipif(not TRACK.exists(), reason="track_c reports missing")
def test_track_c_reports_list(client):
    r = client.get("/v1/reports/tracks/berka")
    assert r.status_code == 200
    body = r.json()
    assert body["scope"] == "track_c"
    assert body["n"] >= 1
    slug = body["reports"][0]["slug"]
    meta = client.get(f"/v1/reports/{slug}")
    assert meta.status_code == 200
    raw = client.get(f"/v1/reports/{slug}/raw")
    assert raw.status_code == 200
    assert len(raw.text) > 20


def test_catalog_includes_comparison(client):
    r = client.get("/v1/catalog")
    assert r.status_code == 200
    groups = r.json()["groups"]
    assert "comparison" in groups
    assert any("comparison/board" in x for x in groups["comparison"])
