import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from customer_segmentation.models import clustering_baselines as baselines
from customer_segmentation.models import track_a_experiments as experiments


def _two_cluster_data(seed=17):
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=-2.0, scale=0.35, size=(80, 2))
    right = rng.normal(loc=2.0, scale=0.35, size=(80, 2))
    return np.vstack([left, right])


def test_feature_contract_selects_exact_order_and_excludes_ids_and_profiles():
    contract = {
        "model_features": ["f2", "f1"],
        "identifier_columns": ["CustomerID"],
        "profile_features": ["Age"],
        "excluded_features": ["CustomerID", "Age"],
        "excluded_column_patterns": [],
    }
    frame = pd.DataFrame(
        {
            "CustomerID": ["x", "y"],
            "Age": [40.0, 50.0],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        }
    )

    selected = baselines.select_baseline_features(frame, contract, ["f2", "f1"])

    assert selected.columns.tolist() == ["f2", "f1"]
    assert "CustomerID" not in selected
    assert "Age" not in selected
    with pytest.raises(ValueError, match="exactly match"):
        baselines.select_baseline_features(frame, contract, ["f1", "f2"])
    prohibited_contract = {
        **contract,
        "model_features": ["CustomerID", "f1"],
    }
    with pytest.raises(ValueError, match="cannot be clustering inputs"):
        baselines.validate_feature_contract(
            prohibited_contract, list(frame.columns), prohibited_contract["model_features"]
        )


def test_sampling_is_deterministic_disjoint_and_uses_shared_eval_positions():
    train_a, eval_a = baselines.make_deterministic_sample_indices(
        1_000, training_count=100, evaluation_count=50, training_seed=42, evaluation_seed=7
    )
    train_b, eval_b = baselines.make_deterministic_sample_indices(
        1_000, training_count=100, evaluation_count=50, training_seed=42, evaluation_seed=7
    )
    train_other_seed, eval_other = baselines.make_deterministic_sample_indices(
        1_000, training_count=120, evaluation_count=50, training_seed=2024, evaluation_seed=7
    )

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(eval_a, eval_b)
    assert np.array_equal(eval_a, eval_other)
    assert len(np.intersect1d(train_a, eval_a)) == 0
    assert len(np.intersect1d(train_other_seed, eval_other)) == 0


def test_chunk_sampler_preserves_ids_and_explicit_feature_order(tmp_path):
    path = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "CustomerID": [f"id-{i}" for i in range(12)],
            "f2": np.arange(12) + 20,
            "Age": np.arange(12) + 30,
            "f1": np.arange(12),
        }
    ).to_csv(path, index=False)

    samples = baselines.load_sampled_rows(
        path,
        identifier_column="CustomerID",
        feature_order=["f1", "f2"],
        requested_positions={"train": np.array([0, 4, 8]), "evaluation": np.array([1, 5])},
        expected_row_count=12,
        chunk_rows=3,
    )

    assert samples["train"].customer_ids.tolist() == ["id-0", "id-4", "id-8"]
    assert samples["train"].features.tolist() == [[0.0, 20.0], [4.0, 24.0], [8.0, 28.0]]
    assert samples["evaluation"].customer_ids.tolist() == ["id-1", "id-5"]


def test_runner_passes_identical_customer_samples_to_both_algorithms(monkeypatch, tmp_path):
    train_sample = baselines.SampledCustomers(
        name="smoke_train_seed42",
        row_positions=np.array([0, 2, 4]),
        customer_ids=np.array(["train-a", "train-b", "train-c"]),
        features=np.array([[0.0], [1.0], [2.0]]),
    )
    evaluation_sample = baselines.SampledCustomers(
        name="evaluation",
        row_positions=np.array([1, 3]),
        customer_ids=np.array(["eval-a", "eval-b"]),
        features=np.array([[0.5], [1.5]]),
    )
    observed = []

    def capture_run(**kwargs):
        observed.append(
            (
                kwargs["algorithm"],
                kwargs["train_sample"].customer_ids.copy(),
                kwargs["eval_sample"].customer_ids.copy(),
            )
        )
        return (
            {
                "algorithm": kwargs["algorithm"],
                "k": kwargs["k"],
                "seed": kwargs["seed"],
                "run_status": "completed",
            },
            np.array([0, 1]),
        )

    monkeypatch.setattr(experiments, "_run_one", capture_run)
    rows = experiments._run_phase(
        phase="smoke",
        root=tmp_path,
        config={"sampling": {"smoke": {"training_rows": 3, "seeds": [42], "k": 2}}},
        samples={
            "smoke_train_seed42": train_sample,
            "evaluation": evaluation_sample,
        },
        manifests={
            "smoke_train_seed42": (tmp_path / "train.csv", "train-hash"),
            "evaluation": (tmp_path / "evaluation.csv", "evaluation-hash"),
        },
        expected_row_count=5,
    )

    assert [row["algorithm"] for row in rows] == ["kmeans", "gaussian_mixture"]
    assert len(observed) == 2
    assert np.array_equal(observed[0][1], observed[1][1])
    assert np.array_equal(observed[0][2], observed[1][2])


def test_kmeans_and_diagonal_gmm_fit_and_predict():
    features = _two_cluster_data()
    kmeans = baselines.train_kmeans(
        features, n_clusters=2, random_state=42, n_init=3, max_iter=100
    )
    gmm = baselines.train_gaussian_mixture(
        features,
        n_components=2,
        covariance_type="diag",
        reg_covar=1e-6,
        random_state=42,
        n_init=1,
        max_iter=100,
    )

    assert len(kmeans.predict(features)) == len(features)
    assert len(gmm.predict(features)) == len(features)
    assert kmeans.n_clusters == 2
    assert gmm.covariance_type == "diag"
    assert np.array_equal(
        kmeans.predict(features),
        baselines.train_kmeans(
            features, n_clusters=2, random_state=42, n_init=3, max_iter=100
        ).predict(features),
    )
    assert np.array_equal(
        gmm.predict(features),
        baselines.train_gaussian_mixture(
            features,
            n_components=2,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=42,
            n_init=1,
            max_iter=100,
        ).predict(features),
    )


def test_evaluation_metrics_are_bounded_and_return_segment_metrics(monkeypatch):
    features = _two_cluster_data()
    labels = np.array([0] * 80 + [1] * 80)
    silhouette_shapes = []

    def bounded_silhouette(x, y):
        silhouette_shapes.append((len(x), len(y)))
        return 0.25

    monkeypatch.setattr(baselines, "silhouette_score", bounded_silhouette)
    metrics = baselines.evaluate_assignments(
        features, labels, max_silhouette_rows=20, random_state=11
    )

    assert silhouette_shapes == [(20, 20)]
    assert metrics["silhouette_sample_rows"] == 20
    assert metrics["silhouette_score"] == 0.25
    assert metrics["davies_bouldin_index"] is not None
    assert metrics["calinski_harabasz_index"] is not None
    assert metrics["segment_counts"] == {"0": 80, "1": 80}
    assert metrics["smallest_segment_pct"] == 50.0
    assert metrics["metric_status"] == "ok"


def test_degenerate_evaluation_and_invalid_training_data_are_reported():
    features = _two_cluster_data()
    metrics = baselines.evaluate_assignments(
        features, np.zeros(len(features), dtype=int), max_silhouette_rows=20
    )
    assert metrics["metric_status"] == "degenerate"
    assert metrics["silhouette_score"] is None

    invalid = features.copy()
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        baselines.train_kmeans(invalid, n_clusters=2, random_state=42)
    with pytest.raises(ValueError, match="smaller than training rows"):
        baselines.train_gaussian_mixture(
            features[:2], n_components=2, random_state=42
        )


def test_stability_ari_requires_identical_ordered_customer_ids():
    ids = np.array(["c1", "c2", "c3", "c4"])
    labels = np.array([0, 0, 1, 1])
    assert baselines.aligned_adjusted_rand_index(labels, ids, labels.copy(), ids.copy()) == 1.0
    with pytest.raises(ValueError, match="identical customer IDs"):
        baselines.aligned_adjusted_rand_index(labels, ids, labels[::-1], ids[::-1])


def test_assignments_preserve_customer_ids_and_serialized_model_predictions(tmp_path):
    features = _two_cluster_data()
    ids = np.array([f"cust-{i}" for i in range(len(features))])
    model = baselines.train_kmeans(features, n_clusters=2, random_state=42, n_init=2)
    labels = model.predict(features)
    assignments = baselines.make_assignment_frame(
        ids, labels, run_id="test-run", split="evaluation"
    )
    assert assignments["CustomerID"].tolist() == ids.tolist()

    model_path = tmp_path / "model.joblib"
    baselines.save_model_artifact(model_path, model, {"run_id": "test-run"})
    loaded = baselines.load_model_artifact(model_path)
    assert np.array_equal(loaded["estimator"].predict(features), labels)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        baselines.save_model_artifact(model_path, model, {"run_id": "replacement"})


def test_p1_preprocessor_metadata_is_verified_before_baseline_use(tmp_path):
    features = [
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
    scaled_features = ["Age", *features]
    scaler = StandardScaler().fit(
        pd.DataFrame(np.arange(20, dtype=float).reshape(2, 10), columns=scaled_features)
    )
    artifact = {
        "version": 1,
        "track": "track_a",
        "identifier_columns": ["CustomerID"],
        "feature_order": scaled_features,
        "transformations": {
            "input_representation": "unscaled_interpretable",
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaled_features": scaled_features,
            "log1p_columns": [
                "AccountBalance",
                "Avg_Transaction_Amt",
                "Std_Transaction_Amt",
                "Total_Spend",
            ],
            "clip_nonnegative_columns": [],
        },
        "scaler": scaler,
    }
    artifact_path = tmp_path / "preprocessor.joblib"
    joblib.dump(artifact, artifact_path)

    loaded = experiments.validate_p1_preprocessor(
        artifact_path, feature_order=features, identifier_column="CustomerID"
    )

    assert loaded["transformations"]["scaled_features"] == scaled_features

    artifact["transformations"]["scaled_features"] = list(reversed(scaled_features))
    joblib.dump(artifact, artifact_path)
    with pytest.raises(ValueError, match="feature order"):
        experiments.validate_p1_preprocessor(
            artifact_path, feature_order=features, identifier_column="CustomerID"
        )


def test_fit_failure_is_returned_as_a_failed_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(
        experiments,
        "train_kmeans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("deliberate fit failure")),
    )
    train_sample = baselines.SampledCustomers(
        name="train",
        row_positions=np.array([0, 1, 2]),
        customer_ids=np.array(["a", "b", "c"]),
        features=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    )
    eval_sample = baselines.SampledCustomers(
        name="evaluation",
        row_positions=np.array([3, 4]),
        customer_ids=np.array(["d", "e"]),
        features=np.array([[3.0, 3.0], [4.0, 4.0]]),
    )
    config = {
        "_runtime_evidence": {
            "model_matrix_sha256": "model-hash",
            "preprocessor_sha256": "preprocessor-hash",
        },
        "data": {
            "features": ["feature_a", "feature_b"],
            "identifier_column": "CustomerID",
        },
        "algorithms": {"kmeans": {"n_init": 1, "max_iter": 10, "algorithm": "lloyd"}},
        "execution": {
            "model_directory": "models",
            "assignment_directory": "assignments",
            "run_metadata_directory": "metadata",
            "thread_limit": 1,
        },
        "metrics": {"silhouette_max_rows": 10, "silhouette_seed": 1},
    }

    row, predictions = experiments._run_one(
        root=tmp_path,
        config=config,
        phase="test",
        algorithm="kmeans",
        seed=7,
        k=2,
        train_sample=train_sample,
        eval_sample=eval_sample,
        train_manifest=tmp_path / "train.csv",
        eval_manifest=tmp_path / "evaluation.csv",
        train_hash="train-hash",
        eval_hash="evaluation-hash",
    )

    assert row["run_status"] == "failed"
    assert "deliberate fit failure" in row["error"]
    assert predictions is None
