"""Leakage-safe champion feature builder smoke tests."""

from __future__ import annotations

import pandas as pd

from customer_segmentation.features.build_berka_segmentation_features import (
    MODEL_COLUMNS,
    build_berka_segmentation_features_asof,
    fit_train_scaler,
    transform_with_train_scaler,
)
from customer_segmentation.models.berka_behavioral_baselines import make_shared_split


def _toy_berka():
    account = pd.DataFrame(
        {
            "account_id": [1, 2, 3],
            "date": ["930101", "950101", "980101"],  # third opens after as_of
        }
    )
    # Dates as YYMMDD; as_of 1997-12-31 → post-cutoff txn on account 1 must be dropped
    trans = pd.DataFrame(
        {
            "account_id": [1, 1, 1, 2, 2],
            "date": ["940101", "960601", "980601", "960101", "970601"],
            "type": ["PRIJEM", "VYDAJ", "PRIJEM", "PRIJEM", "VYDAJ"],
            "amount": [1000.0, 200.0, 9999.0, 500.0, 100.0],
            "balance": [1000.0, 800.0, 10799.0, 500.0, 400.0],
            "trans_id": [1, 2, 3, 4, 5],
        }
    )
    loan = pd.DataFrame({"account_id": [2], "date": ["960701"]})
    card = pd.DataFrame({"disp_id": [10], "issued": ["970101"]})
    disp = pd.DataFrame({"disp_id": [10], "account_id": [1], "client_id": [1], "type": ["OWNER"]})
    return account, trans, loan, card, disp


def test_asof_excludes_future_txns_and_accounts():
    account, trans, loan, card, disp = _toy_berka()
    frame = build_berka_segmentation_features_asof(
        trans,
        account,
        loan,
        card,
        disp,
        as_of_date="1997-12-31",
        minimum_tenure_days=365,
    )
    assert set(frame["account_id"]) == {1, 2}
    # Account 1: only two pre-cutoff txns; post-cutoff credit must not inflate totals
    row1 = frame.set_index("account_id").loc[1]
    assert row1["Transaction_Count"] == 2
    assert row1["Total_Credit"] == 1000.0
    assert row1["has_card"] == 1
    assert row1["has_loan"] == 0
    row2 = frame.set_index("account_id").loc[2]
    assert row2["has_loan"] == 1
    assert list(MODEL_COLUMNS) == [c for c in MODEL_COLUMNS if c in frame.columns]


def test_train_only_scaler_transform_shape():
    account, trans, loan, card, disp = _toy_berka()
    frame = build_berka_segmentation_features_asof(
        trans, account, loan, card, disp, as_of_date="1997-12-31", minimum_tenure_days=0
    )
    train, holdout = make_shared_split(frame, training_fraction=0.5, random_seed=8675309)
    scaler = fit_train_scaler(train)
    Xh = transform_with_train_scaler(holdout, scaler)
    assert Xh.shape == (len(holdout), len(MODEL_COLUMNS))
