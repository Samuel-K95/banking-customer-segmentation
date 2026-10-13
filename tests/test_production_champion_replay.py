"""CI replay: frozen production encoder+KMeans reproduce held-out labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.features.build_berka_segmentation_features import (
    transform_with_train_scaler,
)
from customer_segmentation.models.champion import ChampionModel

PROD = Path("artifacts/segmentation/champion")


def _berka_raw_available() -> bool:
    from customer_segmentation.data.berka_dataset_loader import resolve_berka_dir

    try:
        resolve_berka_dir()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.requires_torch
@pytest.mark.skipif(not PROD.exists(), reason="production champion artifacts missing")
@pytest.mark.skipif(
    not _berka_raw_available(),
    reason="raw Berka data is not downloaded (bash scripts/run_data_layer.sh fetches it)",
)
def test_production_holdout_replay_matches_stored_labels():
    champ = ChampionModel.load(PROD)
    split = pd.read_csv(PROD / "account_split.csv")
    assign = pd.read_csv(PROD / "assignments.csv")
    split["account_id"] = split["account_id"].astype(np.int64)
    assign["account_id"] = assign["account_id"].astype(np.int64)
    hold_ids = set(split.loc[split["split"] == "holdout", "account_id"].tolist())
    hold = assign[assign["account_id"].isin(hold_ids)].sort_values("account_id")

    # Rebuild as-of features for holdout IDs only
    from customer_segmentation.data.berka_dataset_loader import (
        load_berka_product_tables,
        load_berka_raw_data,
        resolve_berka_dir,
    )
    from customer_segmentation.features.build_berka_segmentation_features import (
        build_berka_segmentation_features_asof,
    )

    as_of = str((champ.metrics or {}).get("as_of_date") or "1997-12-31")
    min_tenure = int((champ.metrics or {}).get("minimum_tenure_days") or 365)
    path = resolve_berka_dir(None)
    trans, account, _ = load_berka_raw_data(path)
    loan, card, disp = load_berka_product_tables(path)
    feats = build_berka_segmentation_features_asof(
        trans, account, loan, card, disp, as_of_date=as_of, minimum_tenure_days=min_tenure
    )
    feats["account_id"] = feats["account_id"].astype(np.int64)
    feats = feats[feats["account_id"].isin(hold_ids)].sort_values("account_id")
    hold = hold[hold["account_id"].isin(set(feats["account_id"]))].sort_values("account_id")
    feats = feats.set_index("account_id").loc[hold["account_id"].to_numpy()].reset_index()

    assert champ.scaler is not None
    X = transform_with_train_scaler(feats, champ.scaler)
    pred = champ.predict(X)
    stored = hold["segment_id"].astype(int).to_numpy()
    assert np.array_equal(pred, stored)


@pytest.mark.skipif(not (PROD / "serving_contract.json").exists(), reason="serving contract missing")
def test_serving_contract_has_version_and_asof():
    import json

    from customer_segmentation.models.artifact_integrity import verify_serving_contract

    contract = json.loads((PROD / "serving_contract.json").read_text(encoding="utf-8"))
    assert contract.get("model_version") == "production_candidate_v1"
    assert contract.get("as_of_date") == "1997-12-31"
    assert contract.get("artifact_sha256")
    assert contract["artifact_sha256"].get("encoder.pt")
    assert contract["artifact_sha256"].get("scaler.joblib")
    assert contract["artifact_sha256"].get("kmeans.joblib")
    if (PROD / "encoder.pt").exists():
        assert verify_serving_contract(PROD) == []


@pytest.mark.requires_torch
@pytest.mark.skipif(not (PROD / "encoder.pt").exists(), reason="encoder missing")
def test_production_load_has_encoder():
    champ = ChampionModel.load(PROD)
    assert champ.encoder is not None
    assert champ.summary()["has_encoder"] is True
