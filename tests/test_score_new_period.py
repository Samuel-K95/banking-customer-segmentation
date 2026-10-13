import numpy as np
import pandas as pd
import pytest

from customer_segmentation.pipelines.score_new_period import psi, transition_summary


def test_psi_is_zero_for_identical_and_large_for_shifted_distributions():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    assert psi(base, base) == pytest.approx(0.0, abs=1e-9)
    assert psi(base, rng.normal(0, 1, 5000)) < 0.05  # same distribution, new sample
    assert psi(base, rng.normal(1.5, 1, 5000)) > 0.25  # clear shift


def test_psi_for_binary_columns_compares_category_shares():
    before = np.array([0] * 80 + [1] * 20)
    assert psi(before, before, categorical=True) == pytest.approx(0.0, abs=1e-9)
    assert psi(before, np.array([0] * 40 + [1] * 60), categorical=True) > 0.25


def test_transition_summary_counts_moves_and_stays():
    before = pd.Series([0, 0, 0, 1, 1, 2])
    after = pd.Series([0, 0, 1, 1, 1, 2])
    moves = transition_summary(before, after)
    assert moves["stay_rate"] == pytest.approx(5 / 6)
    assert moves["counts"].loc[0, 1] == 1 and moves["counts"].loc[0, 0] == 2
    assert moves["shares"].loc[0].sum() == pytest.approx(1.0)
