"""Smoke tests for the champion FastAPI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
CHAMPION = REPO / "artifacts" / "segmentation" / "champion"
CONTRACT = CHAMPION / "serving_contract.json"


@pytest.fixture(scope="module")
def client():
    if not CHAMPION.exists():
        pytest.skip("champion artifacts missing")
    import os

    os.chdir(REPO)
    from customer_segmentation.serve.app import app
    from customer_segmentation.serve.deps import get_champion

    get_champion.cache_clear()
    with TestClient(app) as c:
        yield c
    get_champion.cache_clear()


@pytest.fixture(scope="module")
def contract():
    if not CONTRACT.exists():
        pytest.skip("serving_contract.json missing")
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "catalog" in r.json()
    h = client.get("/v1/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_catalog(client):
    r = client.get("/v1/catalog")
    assert r.status_code == 200
    groups = r.json()["groups"]
    assert "model" in groups and "inference" in groups and "decisions" in groups


def test_model_and_segments(client):
    r = client.get("/v1/model")
    assert r.status_code == 200
    assert r.json()["n_clusters"] == 6
    segs = client.get("/v1/segments")
    assert segs.status_code == 200
    assert segs.json()["n"] == 6


def test_decisions(client):
    s = client.get("/v1/decisions/summary")
    assert s.status_code == 200
    m = client.get("/v1/decisions/map")
    assert m.status_code == 200
    assert m.json()["n"] >= 1
    apps = client.get("/v1/segments/0/applications")
    assert apps.status_code == 200
    assert "applications" in apps.json()


def test_reports(client):
    r = client.get("/v1/reports")
    assert r.status_code == 200
    assert r.json()["n"] >= 1
    slug = r.json()["reports"][0]["slug"]
    meta = client.get(f"/v1/reports/{slug}")
    assert meta.status_code == 200
    raw = client.get(f"/v1/reports/{slug}/raw")
    assert raw.status_code == 200
    assert len(raw.text) > 10


def test_inference_single_and_batch(client, contract):
    mv = contract["model_version"]
    asof = contract["as_of_date"]
    features = {
        "model_version": mv,
        "as_of_date": asof,
        "Transaction_Count": 120,
        "Account_Balance": 45000,
        "Total_Credit": 80000,
        "Total_Withdrawal": 35000,
        "Recency_Days": 5,
        "Account_Age_Days": 900,
        "has_loan": 0,
        "has_card": 1,
        "include_applications": True,
    }
    single = client.post("/v1/inference", json=features)
    assert single.status_code == 200, single.text
    body = single.json()
    assert body["n"] == 1
    assert "segment_id" in body["assignments"][0]

    batch = client.post(
        "/v1/inference/batch",
        json={
            "model_version": mv,
            "as_of_date": asof,
            "customers": [
                {
                    "account_id": 99,
                    "Transaction_Count": 120,
                    "Account_Balance": 45000,
                    "Total_Credit": 80000,
                    "Total_Withdrawal": 35000,
                    "Recency_Days": 5,
                    "Account_Age_Days": 900,
                    "has_loan": 0,
                    "has_card": 1,
                }
            ],
        },
    )
    assert batch.status_code == 200, batch.text
    assert batch.json()["n"] == 1


def test_inference_rejects_bad_contract(client):
    bad = client.post(
        "/v1/inference",
        json={
            "model_version": "wrong",
            "as_of_date": "1997-12-31",
            "Transaction_Count": 1,
            "Account_Balance": 1,
            "Total_Credit": 1,
            "Total_Withdrawal": 1,
            "Recency_Days": 1,
            "Account_Age_Days": 400,
            "has_loan": 0,
            "has_card": 0,
        },
    )
    assert bad.status_code == 400


def test_monitoring(client):
    r = client.get("/v1/monitoring/segment-mix")
    assert r.status_code == 200
    snap = client.get("/v1/monitoring/snapshot")
    assert snap.status_code == 200
    assert snap.json()["status"] in {"ok", "degraded"}


def test_legacy_customers_segment(client, contract):
    r = client.post(
        "/v1/customers/segment",
        json={
            "model_version": contract["model_version"],
            "as_of_date": contract["as_of_date"],
            "customers": [
                {
                    "Transaction_Count": 10,
                    "Account_Balance": 1000,
                    "Total_Credit": 2000,
                    "Total_Withdrawal": 500,
                    "Recency_Days": 30,
                    "Account_Age_Days": 500,
                    "has_loan": 0,
                    "has_card": 0,
                }
            ],
        },
    )
    assert r.status_code == 200
