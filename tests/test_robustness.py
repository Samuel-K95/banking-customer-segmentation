import numpy as np
import pytest

from customer_segmentation.models.robustness import (
    apply_quantile_clip,
    cohort_segment_profiles,
    fit_quantile_clip_bounds,
    history_assignment_association,
    match_cluster_membership,
    transaction_history_cohort,
    validate_customer_order,
)


def test_clip_thresholds_are_fit_from_training_values_and_transform_is_nonmutating():
    training = np.arange(100, dtype=np.float64)
    evaluation = np.array([-500.0, 10.0, 500.0])
    bounds = fit_quantile_clip_bounds(
        training,
        feature_name="Avg_Spend_Ratio",
        lower_quantile=0.01,
        upper_quantile=0.99,
        method="linear",
    )
    original_evaluation = evaluation.copy()

    transformed, below, above = apply_quantile_clip(evaluation, bounds)

    assert bounds["fit_rows"] == 100
    assert bounds["lower_bound"] == np.quantile(training, 0.01, method="linear")
    assert bounds["upper_bound"] == np.quantile(training, 0.99, method="linear")
    assert transformed[0] == bounds["lower_bound"]
    assert transformed[-1] == bounds["upper_bound"]
    assert below.tolist() == [True, False, False]
    assert above.tolist() == [False, False, True]
    assert np.array_equal(evaluation, original_evaluation)
    assert bounds["fit_scope"] == "training customers only"


def test_clip_rejects_invalid_fit_values_and_quantiles():
    with pytest.raises(ValueError, match="finite vector"):
        fit_quantile_clip_bounds(
            np.array([0.0, np.inf]),
            feature_name="f",
            lower_quantile=0.01,
            upper_quantile=0.99,
        )
    with pytest.raises(ValueError, match="Quantiles"):
        fit_quantile_clip_bounds(
            np.arange(4),
            feature_name="f",
            lower_quantile=0.99,
            upper_quantile=0.01,
        )


def test_customer_id_alignment_rejects_reordered_or_duplicated_ids():
    ids = np.array(["c1", "c2", "c3"])
    validate_customer_order(ids, ids.copy())
    with pytest.raises(ValueError, match="ordering does not match"):
        validate_customer_order(ids, ids[::-1])
    with pytest.raises(ValueError, match="unique"):
        validate_customer_order(ids, np.array(["c1", "c1", "c3"]))


def test_cluster_membership_comparison_permutation_matches_labels_first():
    ids = np.array(["a", "b", "c", "d", "e"])
    reference = np.array([0, 0, 0, 1, 1])
    candidate = np.array([1, 1, 0, 0, 0])

    comparison = match_cluster_membership(ids, reference, ids.copy(), candidate)

    assert comparison["label_map_candidate_to_reference"] == {"0": 1, "1": 0}
    assert comparison["changed_customers_after_label_alignment"] == 1
    assert comparison["adjusted_rand_index"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="identical ordered CustomerIDs"):
        match_cluster_membership(ids, reference, ids[::-1], candidate)


def test_transaction_history_cohorts_and_segment_profiles():
    ids = np.array(["a", "b", "c", "d", "e", "f"])
    labels = np.array([0, 0, 0, 1, 1, 1])
    counts = np.array([1, 1, 2, 2, 3, 1])

    cohorts = transaction_history_cohort(counts)
    profiles = cohort_segment_profiles(ids, labels, counts)
    association = history_assignment_association(labels, counts)

    assert cohorts.tolist() == [
        "exactly_one",
        "exactly_one",
        "at_least_two",
        "at_least_two",
        "at_least_two",
        "exactly_one",
    ]
    one_tx = profiles.loc[profiles["cohort"] == "exactly_one"].set_index("cluster")
    repeat = profiles.loc[profiles["cohort"] == "at_least_two"].set_index("cluster")
    assert one_tx.loc[0, "customers_in_cluster"] == 2
    assert one_tx.loc[1, "customers_in_cluster"] == 1
    assert repeat.loc[0, "customers_in_cluster"] == 1
    assert repeat.loc[1, "customers_in_cluster"] == 2
    assert one_tx["percent_of_cohort_sample"].sum() == 100.0
    assert set(association) == {
        "ari_vs_singleton_status",
        "nmi_vs_singleton_status",
        "ari_vs_transaction_count",
        "nmi_vs_transaction_count",
    }
    assert all(np.isfinite(list(association.values())))


def test_cohort_profile_requires_aligned_unique_ids_and_positive_counts():
    with pytest.raises(ValueError, match="align row for row"):
        cohort_segment_profiles(
            np.array(["a", "b"]), np.array([0]), np.array([1, 2])
        )
    with pytest.raises(ValueError, match="finite positive counts"):
        transaction_history_cohort(np.array([1, 0]))
