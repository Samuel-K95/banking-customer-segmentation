import numpy as np
import pandas as pd
import pytest

from customer_segmentation.models.berka_temporal_validation import (
    _summary_tables,
    aggregate_period_outcomes,
    filter_comparison_windows,
    frozen_assignments_for_population,
)


def _transactions(rows):
    return pd.DataFrame(
        rows,
        columns=["trans_id", "account_id", "date", "type", "amount", "balance"],
    )


def test_comparison_windows_include_1997_and_1998_boundaries_only():
    transactions = _transactions(
        [
            (1, 1, "961231", "PRIJEM", 10.0, 10.0),
            (2, 1, "970101", "PRIJEM", 20.0, 30.0),
            (3, 1, "971231", "VYDAJ", 5.0, 25.0),
            (4, 1, "980101", "VYBER", 2.0, 23.0),
            (5, 1, "981231", "PRIJEM", 3.0, 26.0),
            (6, 1, "990101", "VYDAJ", 1.0, 25.0),
        ]
    )

    filtered, counts = filter_comparison_windows(transactions)

    assert counts == {"prior_365d": 2, "outcome_1998": 2}
    assert filtered.trans_id.tolist() == [2, 3, 4, 5]
    assert set(filtered["_window"]) == {"prior_365d", "outcome_1998"}


def test_credits_debits_use_type_direction_and_absolute_source_amounts():
    transactions = _transactions(
        [
            (1, 10, "980101", "PRIJEM", -100.0, 80.0),
            (2, 10, "980102", "VYBER", 20.0, -5.0),
            (3, 10, "980103", "VYDAJ", -5.0, 2.0),
            (4, 10, "980104", "OTHER", 77.0, 79.0),
        ]
    )

    outcomes = aggregate_period_outcomes(
        transactions, [10], period_start="1998-01-01", period_end="1998-12-31"
    ).iloc[0]

    assert outcomes.transaction_count == 4
    assert outcomes.credit_transaction_count == 1
    assert outcomes.debit_transaction_count == 2
    assert outcomes.unknown_direction_transaction_count == 1
    assert outcomes.credit_amount_total == 100.0
    assert outcomes.debit_amount_total == 25.0
    assert outcomes.net_cash_flow == 75.0
    assert outcomes.negative_balance_count == 1
    assert outcomes.negative_balance_frequency == 0.25


def test_inactive_accounts_are_retained_with_undefined_statistics_missing():
    transactions = _transactions([(1, 10, "980101", "PRIJEM", 10.0, 10.0)])

    outcomes = aggregate_period_outcomes(
        transactions, [20, 10], period_start="1998-01-01", period_end="1998-12-31"
    ).set_index("account_id")

    inactive = outcomes.loc[20]
    assert outcomes.index.tolist() == [20, 10]
    assert inactive.transaction_count == 0
    assert inactive.active_months == 0
    assert inactive.credit_amount_total == 0.0
    assert inactive.debit_amount_total == 0.0
    assert inactive.net_cash_flow == 0.0
    assert inactive.avg_monthly_transaction_frequency == 0.0
    assert np.isnan(inactive.avg_transaction_frequency_per_active_month)
    assert np.isnan(inactive.median_transaction_amount)
    assert np.isnan(inactive.median_balance)
    assert np.isnan(inactive.negative_balance_frequency)
    assert np.isnan(inactive.interarrival_median_days)


def test_aggregation_is_deterministic_under_input_row_permutation():
    transactions = _transactions(
        [
            (1, 10, "980101", "PRIJEM", 10.0, 10.0),
            (2, 10, "980201", "VYDAJ", 2.0, 8.0),
            (3, 20, "980301", "VYBER", 5.0, -5.0),
        ]
    )
    expected = aggregate_period_outcomes(
        transactions, [10, 20, 30], period_start="1998-01-01", period_end="1998-12-31"
    )
    actual = aggregate_period_outcomes(
        transactions.sample(frac=1, random_state=2026),
        [10, 20, 30],
        period_start="1998-01-01",
        period_end="1998-12-31",
    )

    pd.testing.assert_frame_equal(expected, actual)


def test_frozen_assignments_preserve_original_labels_and_requested_id_order():
    assignments = pd.DataFrame(
        [
            (7, "kmeans", 2, 42, 0),
            (8, "kmeans", 2, 42, 1),
            (7, "gaussian_mixture", 2, 42, 1),
            (8, "gaussian_mixture", 2, 42, 0),
            (7, "kmeans", 2, 123, 1),
        ],
        columns=["account_id", "algorithm", "k", "seed", "cluster"],
    )

    selected = frozen_assignments_for_population(
        assignments, [8, 7], algorithm="kmeans", seed=42
    )

    assert selected.account_id.tolist() == [8, 7]
    assert selected.cluster.tolist() == [1, 0]
    with pytest.raises(ValueError, match="do not match held-out"):
        frozen_assignments_for_population(assignments, [7, 9], algorithm="kmeans", seed=42)


def test_segment_summary_and_bootstrap_tables_include_defined_paired_outcomes():
    outcomes = pd.DataFrame(
        {
            "account_id": [1, 2, 3, 4],
            "kmeans_cluster": [0, 0, 1, 1],
            "gaussian_mixture_cluster": [1, 0, 1, 0],
            "prior_365d_transaction_count": [10, 20, 30, 40],
            "outcome_1998_transaction_count": [12, 18, 33, 45],
        }
    )

    summary, contrasts, changes = _summary_tables(
        outcomes, bootstrap_repetitions=100, bootstrap_seed=13, confidence_level=0.95
    )

    assert not summary.empty
    assert not contrasts.empty
    assert not changes.empty
    assert set(summary.outcome) == {"transaction_count"}
    assert set(contrasts.period) == {"prior_365d", "outcome_1998"}
    assert changes.accounts_paired.eq(2).all()
    assert np.isfinite(changes.median_paired_change_ci95_low).all()
