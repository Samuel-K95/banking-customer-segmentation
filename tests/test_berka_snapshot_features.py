import numpy as np
import pandas as pd

from customer_segmentation.features.build_berka_snapshot_features import (
    build_berka_snapshot_features,
)


def _tables():
    accounts = pd.DataFrame(
        [
            (10, "930101"),
            (20, "971231"),
            (30, "980101"),
        ],
        columns=["account_id", "date"],
    )
    transactions = pd.DataFrame(
        [
            (1, 10, "971001", "PRIJEM", 100.0, 50.0),
            (2, 10, "971101", "VYDAJ", 40.0, -10.0),
            (3, 10, "971201", "VYBER", -5.0, -15.0),
            (4, 20, "971231", "CREDIT", 25.0, 25.0),
            (5, 10, "980101", "PRIJEM", 1_000_000.0, 1_000_000.0),
            (6, 30, "980101", "PRIJEM", 200.0, 200.0),
        ],
        columns=["trans_id", "account_id", "date", "type", "amount", "balance"],
    )
    return transactions, accounts


def test_post_cutoff_transactions_cannot_affect_features():
    transactions, accounts = _tables()
    before, _ = build_berka_snapshot_features(transactions, accounts, as_of_date="1997-12-31")
    changed = transactions.copy()
    changed.loc[changed["date"].eq("980101"), "amount"] = 9_999_999.0
    changed.loc[changed["date"].eq("980101"), "balance"] = -9_999_999.0
    after, metadata = build_berka_snapshot_features(changed, accounts, as_of_date="1997-12-31")

    pd.testing.assert_frame_equal(before, after)
    assert metadata["transaction_rows_after_cutoff_excluded"] == 2


def test_output_is_unique_account_grain_and_excludes_accounts_opened_after_cutoff():
    transactions, accounts = _tables()
    features, metadata = build_berka_snapshot_features(transactions, accounts, as_of_date="1997-12-31")

    assert features["account_id"].tolist() == [10, 20]
    assert features["account_id"].is_unique
    assert metadata["accounts_opened_by_cutoff"] == 2
    assert metadata["accounts_opened_after_cutoff_excluded"] == 1


def test_credit_and_debit_direction_uses_type_and_absolute_source_amount():
    transactions, accounts = _tables()
    features, _ = build_berka_snapshot_features(transactions, accounts, as_of_date="1997-12-31")
    account = features.set_index("account_id").loc[10]

    assert account["credit_transaction_count"] == 1
    assert account["debit_transaction_count"] == 2
    assert account["credit_amount_total"] == 100.0
    assert account["debit_amount_total"] == 45.0
    assert account["net_cash_flow"] == 55.0


def test_negative_balances_are_retained_in_statistics_and_frequency():
    transactions, accounts = _tables()
    features, _ = build_berka_snapshot_features(transactions, accounts, as_of_date="1997-12-31")
    account = features.set_index("account_id").loc[10]

    assert account["negative_balance_count"] == 2
    assert account["negative_balance_frequency"] == 2 / 3
    assert account["balance_mean"] == (50.0 - 10.0 - 15.0) / 3


def test_output_is_deterministic_under_input_row_permutations():
    transactions, accounts = _tables()
    expected, _ = build_berka_snapshot_features(transactions, accounts, as_of_date="1997-12-31")
    shuffled, _ = build_berka_snapshot_features(
        transactions.sample(frac=1, random_state=3),
        accounts.sample(frac=1, random_state=7),
        as_of_date="1997-12-31",
    )

    pd.testing.assert_frame_equal(expected, shuffled)


def test_empty_and_sparse_windows_have_defined_counts_and_missing_statistics():
    transactions, accounts = _tables()
    features, _ = build_berka_snapshot_features(transactions, accounts, as_of_date="1997-12-31")
    account = features.set_index("account_id").loc[20]

    assert account["transaction_count"] == 1
    assert account["credit_transaction_share"] == 1.0
    assert account["debit_transaction_share"] == 0.0
    assert np.isnan(account["interarrival_days_mean"])
    assert np.isnan(account["prior_90d_transaction_count"])

    empty_transactions = transactions.iloc[0:0]
    empty, _ = build_berka_snapshot_features(empty_transactions, accounts, as_of_date="1997-12-31")
    empty_account = empty.set_index("account_id").loc[10]
    assert empty_account["transaction_count"] == 0
    assert empty_account["credit_transaction_share"] == 0.0
    assert empty_account["negative_balance_frequency"] == 0.0
    assert np.isnan(empty_account["transaction_amount_median"])
    assert np.isnan(empty_account["balance_median"])
    assert np.isnan(empty_account["recency_days"])
