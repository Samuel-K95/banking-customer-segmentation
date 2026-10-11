import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from customer_segmentation.models.p2_behavioral_pipeline import (
    BehavioralSegmentationPipeline,
    P2TrainFittedPreprocessor,
)

FEATURES = [
    "AccountBalance",
    "Transaction_Count",
    "Avg_Transaction_Amt",
    "Std_Transaction_Amt",
    "Total_Spend",
    "Avg_Spend_Ratio",
    "Recency_Days",
    "Weekend_Spend_Ratio",
    "Night_Spend_Ratio",
]
LOG_FEATURES = [
    "AccountBalance",
    "Avg_Transaction_Amt",
    "Std_Transaction_Amt",
    "Total_Spend",
]


def _frame(prefix: str, counts: list[int], ratios: list[float]) -> pd.DataFrame:
    rows = len(counts)
    return pd.DataFrame(
        {
            "CustomerID": [f"{prefix}-{index}" for index in range(rows)],
            "Last_Transaction": pd.date_range("2016-10-01", periods=rows, freq="D"),
            "AccountBalance": np.arange(rows, dtype=float) * 10 + 100,
            "Transaction_Count": counts,
            "Avg_Transaction_Amt": np.arange(rows, dtype=float) * 2 + 20,
            "Std_Transaction_Amt": np.arange(rows, dtype=float) * 1.5,
            "Total_Spend": np.arange(rows, dtype=float) * 4 + 30,
            "Avg_Spend_Ratio": ratios,
            "Recency_Days": np.zeros(rows),  # ignored; recomputed from Last_Transaction
            "Weekend_Spend_Ratio": np.linspace(0.0, 1.0, rows),
            "Night_Spend_Ratio": np.linspace(1.0, 0.0, rows),
        }
    )


def _preprocessor(*, minimum_transactions=2, clip=False):
    return P2TrainFittedPreprocessor(
        feature_order=FEATURES,
        log1p_features=LOG_FEATURES,
        as_of_date="2016-11-01",
        clip_feature="Avg_Spend_Ratio" if clip else None,
        clip_quantiles=(0.1, 0.9) if clip else None,
    )


def test_p2_preprocessing_parameters_and_ratio_bounds_are_train_only():
    train = _frame(
        "train",
        [2, 3, 2, 4, 3, 2, 4, 2, 3, 2],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    )
    evaluation = _frame("eval", [2, 3], [100_000.0, 200_000.0])
    processor = _preprocessor(clip=True).fit(train)
    original_bounds = dict(processor.clip_bounds)
    original_mean = processor.scaler.mean_.copy()
    original_scale = processor.scaler.scale_.copy()

    transformed = processor.transform(evaluation)

    assert transformed.shape == (2, 9)
    assert np.isfinite(transformed).all()
    assert processor.fit_rows == len(train)
    assert processor.clip_bounds == original_bounds
    assert np.array_equal(processor.scaler.mean_, original_mean)
    assert np.array_equal(processor.scaler.scale_, original_scale)
    assert processor.metadata()["scaler_fit_scope"] == "training customer IDs only"
    assert processor.metadata()["as_of_date"] == "2016-11-01"


def test_p2_pipeline_routes_insufficient_history_and_preserves_ids():
    train = _frame(
        "train",
        [2, 2, 3, 3, 4, 4, 2, 3],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    )
    processor = _preprocessor().fit(train)
    estimator = KMeans(n_clusters=2, random_state=42, n_init=5).fit(
        processor.transform(train)
    )
    pipeline = BehavioralSegmentationPipeline(
        preprocessor=processor,
        estimator=estimator,
        minimum_transactions=2,
        cohort="two_plus",
        metric_space="p2_two_plus_nine_feature_trainfit_scaled_9d",
    )
    incoming = _frame("incoming", [1, 2, 3], [1000.0, 0.03, 0.04])

    first = pipeline.predict_features(incoming)
    second = pipeline.predict_features(incoming)

    assert first["CustomerID"].tolist() == incoming["CustomerID"].tolist()
    assert first["segmentation_status"].tolist() == [
        "insufficient_observed_history",
        "modeled",
        "modeled",
    ]
    assert pd.isna(first.loc[0, "cluster"])
    assert first.loc[1:, "cluster"].notna().all()
    assert first["cluster"].equals(second["cluster"])
    assert "max_component_posterior" not in first.columns


def test_three_plus_pipeline_routes_two_transaction_customers_as_insufficient():
    train = _frame(
        "train",
        [3, 4, 3, 5, 4, 3, 5, 4],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    )
    processor = _preprocessor(minimum_transactions=3).fit(train)
    estimator = KMeans(n_clusters=2, random_state=2024, n_init=3).fit(
        processor.transform(train)
    )
    pipeline = BehavioralSegmentationPipeline(
        preprocessor=processor,
        estimator=estimator,
        minimum_transactions=3,
        cohort="three_plus",
        metric_space="p2_three_plus_nine_feature_trainfit_scaled_9d",
    )
    incoming = _frame("incoming", [2, 3], [0.03, 0.04])
    result = pipeline.predict_features(incoming)

    assert result["segmentation_status"].tolist() == [
        "insufficient_observed_history",
        "modeled",
    ]
    assert pd.isna(result.loc[0, "cluster"])
    assert pd.notna(result.loc[1, "cluster"])


def test_gmm_pipeline_emits_only_supported_posterior_uncertainty():
    train = _frame(
        "train",
        [2, 2, 3, 3, 4, 4, 2, 3, 4, 2],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    )
    processor = _preprocessor().fit(train)
    estimator = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        random_state=42,
        reg_covar=1e-5,
    ).fit(processor.transform(train))
    pipeline = BehavioralSegmentationPipeline(
        preprocessor=processor,
        estimator=estimator,
        minimum_transactions=2,
        cohort="two_plus",
        metric_space="p2_two_plus_nine_feature_trainfit_scaled_9d",
    )
    result = pipeline.predict_features(_frame("eval", [1, 2, 3], [0.02, 0.04, 0.06]))

    assert result.loc[0, "segmentation_status"] == "insufficient_observed_history"
    assert result["max_component_posterior"].iloc[1:].between(0.0, 1.0).all()
    assert result["uncertainty_method"].iloc[1:].eq(
        "maximum Gaussian-mixture posterior"
    ).all()
    assert pd.isna(result["max_component_posterior"].iloc[0])


def test_pipeline_serialization_preserves_transformation_and_inference(tmp_path):
    train = _frame(
        "train",
        [2, 2, 3, 3, 4, 4, 2, 3],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    )
    evaluation = _frame("eval", [2, 3, 4], [0.015, 0.04, 0.08])
    processor = _preprocessor().fit(train)
    estimator = KMeans(n_clusters=2, random_state=17, n_init=4).fit(
        processor.transform(train)
    )
    pipeline = BehavioralSegmentationPipeline(
        preprocessor=processor,
        estimator=estimator,
        minimum_transactions=2,
        cohort="two_plus",
        metric_space="p2_two_plus_nine_feature_trainfit_scaled_9d",
    )
    path = tmp_path / "candidate_pipeline.joblib"
    joblib.dump({"pipeline": pipeline, "metadata": processor.metadata()}, path)
    reloaded = joblib.load(path)["pipeline"]

    assert reloaded.preprocessor.metadata() == processor.metadata()
    assert reloaded.predict_features(evaluation)["cluster"].equals(
        pipeline.predict_features(evaluation)["cluster"]
    )
    assert np.array_equal(
        reloaded.preprocessor.transform(evaluation), processor.transform(evaluation)
    )


def test_p2_ledger_preserves_evaluation_clip_count(tmp_path):
    from customer_segmentation.models.p2_segmentation_experiments import (
        _make_p2_ledger_row,
    )

    result = {
        "metrics": {
            "silhouette_score": 0.1,
            "silhouette_sample_rows": 2,
            "davies_bouldin_index": 1.0,
            "calinski_harabasz_index": 1.0,
            "gmm_bic_evaluation": None,
            "gmm_aic_evaluation": None,
            "gmm_converged": None,
            "metric_status": "ok",
            "metric_error": None,
            "segment_counts": {"0": 1, "1": 1},
            "segment_percentages": {"0": 50.0, "1": 50.0},
            "smallest_segment_pct": 50.0,
        },
        "algorithm": "kmeans",
        "metric_space": "p2_two_plus_ratio_clip_trainfit_trainfit_scaled_9d",
        "training_ids": np.array(["train-1", "train-2"]),
        "evaluation_ids": np.array(["eval-1", "eval-2"]),
        "cohort": "two_plus",
        "preprocessor_path": tmp_path / "preprocessor.joblib",
        "preprocessor_metadata": {
            "fit_customer_id_sha256": "train-hash",
            "clip_training_changed_rows": 2,
            "clip_bounds": {"lower_bound": 0.1, "upper_bound": 10.0},
        },
        "metadata_payload": {"clip_evaluation_changed_rows": 1},
        "candidate_id": "candidate-1",
        "inertia": 1.0,
        "iterations": 1,
        "training_seconds": 0.5,
        "memory_before": 100,
        "memory_peak": 200,
        "memory_after": 100,
        "reload_predictions_identical": True,
        "model_path": tmp_path / "model.joblib",
        "assignment_path": tmp_path / "assignments.csv",
        "metadata_path": tmp_path / "metadata.json",
    }
    cohort_metadata = {
        "minimum_transactions": 2,
        "training": {"42": {"manifest_path": "train.csv"}},
        "evaluation": {"manifest_path": "evaluation.csv"},
    }
    config = {
        "data": {"feature_order": FEATURES, "as_of_date": "2016-11-01"},
        "models": {"kmeans": {"n_init": 1}},
    }

    row = _make_p2_ledger_row(
        root=tmp_path,
        config=config,
        result=result,
        cohort_metadata=cohort_metadata,
        variant_name="ratio_clip_trainfit",
        seed=42,
        k=2,
        data_hashes={
            "raw_transactions_sha256": "raw-hash",
            "interpretable_sha256": "interpretable-hash",
            "p1_scaler_sha256": "p1-hash",
        },
    )

    assert json.loads(row["parameters_json"])["clip_evaluation_changed_rows"] == 1
