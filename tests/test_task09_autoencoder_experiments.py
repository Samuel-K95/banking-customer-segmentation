import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from customer_segmentation.models.clustering_baselines import ordered_id_sha256
from customer_segmentation.models.p2_behavioral_pipeline import (
    P2TrainFittedPreprocessor,
)
from customer_segmentation.models.track_a_task09_autoencoder_experiments import (
    load_task08_baseline_assignments,
    validate_task08_preprocessor,
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


def _training_frame() -> pd.DataFrame:
    rows = 8
    return pd.DataFrame(
        {
            "CustomerID": [f"train-{index}" for index in range(rows)],
            "Transaction_Count": [2, 2, 3, 2, 4, 2, 3, 2],
            "Last_Transaction": pd.date_range("2016-10-01", periods=rows, freq="D"),
            "AccountBalance": np.arange(rows, dtype=float) + 100,
            "Avg_Transaction_Amt": np.arange(rows, dtype=float) + 20,
            "Std_Transaction_Amt": np.arange(rows, dtype=float) + 2,
            "Total_Spend": np.arange(rows, dtype=float) + 30,
            "Avg_Spend_Ratio": np.linspace(0.1, 0.8, rows),
            "Weekend_Spend_Ratio": np.linspace(0.0, 1.0, rows),
            "Night_Spend_Ratio": np.linspace(1.0, 0.0, rows),
        }
    )


def test_task08_preprocessor_must_match_exact_training_ids_and_feature_contract():
    frame = _training_frame()
    ids = frame["CustomerID"].astype(str).to_numpy()
    processor = P2TrainFittedPreprocessor(
        feature_order=FEATURES,
        log1p_features=LOG_FEATURES,
        as_of_date="2016-11-01",
    ).fit(frame)
    wrapper = {
        "fit_customer_id_sha256": ordered_id_sha256(ids),
        "variant": "nine_feature_trainfit",
        "cohort": "two_plus",
    }

    checked = validate_task08_preprocessor(
        processor,
        wrapper,
        training_ids=ids,
        feature_order=FEATURES,
        seed=42,
    )

    assert checked["fit_rows"] == 8
    assert checked["fit_customer_id_sha256"] == ordered_id_sha256(ids)
    assert checked["feature_order"] == FEATURES
    assert checked["clip_bounds"] is None
    with pytest.raises(ValueError, match="fit-ID hash"):
        validate_task08_preprocessor(
            processor,
            wrapper,
            training_ids=ids[::-1],
            feature_order=FEATURES,
            seed=42,
        )


def test_task08_baseline_loader_filters_geometry_and_requires_exact_order(tmp_path):
    ids = np.array(["eval-1", "eval-2", "eval-3", "eval-4"])
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    assignment_path = tmp_path / "task08_assignments.csv"
    pd.DataFrame(
        {
            "CustomerID": ids,
            "split": ["evaluation"] * len(ids),
            "cluster": labels,
        }
    ).to_csv(assignment_path, index=False)
    base = {
        "phase": "task08_p2_leakage_safe",
        "algorithm": "kmeans_p2",
        "seed": 42,
        "k": 2,
        "run_status": "completed",
        "assignment_path": str(assignment_path),
        "run_metadata_path": "metadata.json",
        "evaluation_customer_id_sha256": ordered_id_sha256(ids),
        "silhouette_score": 0.2,
        "davies_bouldin_index": 1.1,
        "calinski_harabasz_index": 50.0,
        "smallest_segment_pct": 50.0,
    }
    board = pd.DataFrame(
        [
            {
                **base,
                "run_id": "clipped-baseline",
                "parameters_json": json.dumps(
                    {"cohort": "two_plus", "preprocessing_variant": "ratio_clip_trainfit"}
                ),
            },
            {
                **base,
                "run_id": "matched-baseline",
                "parameters_json": json.dumps(
                    {"cohort": "two_plus", "preprocessing_variant": "nine_feature_trainfit"}
                ),
            },
        ]
    )

    loaded = load_task08_baseline_assignments(
        tmp_path,
        board,
        evaluation_ids=ids,
        seed=42,
        k=2,
        algorithm="kmeans_p2",
    )

    assert loaded["candidate_id"] == "matched-baseline"
    assert np.array_equal(loaded["labels"], labels)
    assert loaded["saved_metrics"]["silhouette_score"] == 0.2
    with pytest.raises(ValueError, match="ordered holdout"):
        load_task08_baseline_assignments(
            tmp_path,
            board,
            evaluation_ids=ids[::-1],
            seed=42,
            k=2,
            algorithm="kmeans_p2",
        )
