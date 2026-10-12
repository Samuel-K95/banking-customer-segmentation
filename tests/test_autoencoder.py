import numpy as np
import pytest

torch = pytest.importorskip("torch")

from customer_segmentation.models.autoencoder import (
    AutoencoderTrainingConfig,
    CustomerAutoencoder,
    encode_features,
    load_autoencoder_checkpoint,
    make_internal_validation_indices,
    save_autoencoder_checkpoint,
    select_autoencoder_feature_order,
    train_autoencoder,
    validate_train_evaluation_ids,
)
from customer_segmentation.models.clustering_baselines import (
    aligned_adjusted_rand_index,
    evaluate_assignments,
    make_assignment_frame,
)
from customer_segmentation.models.track_a_autoencoder_experiments import (
    fit_train_only_latent_scaler,
    metric_parameters,
)


def _config(*, seed=42, input_dim=9):
    return AutoencoderTrainingConfig(
        input_dim=input_dim,
        hidden_dims=(16, 8),
        latent_dim=4,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=12,
        patience=4,
        min_delta=1e-7,
        seed=seed,
        num_threads=1,
    )


def test_autoencoder_forward_backward_dimensions_and_finite_gradients():
    model = CustomerAutoencoder(input_dim=9, hidden_dims=(32, 16), latent_dim=4)
    features = torch.randn(24, 9, dtype=torch.float32)

    latent, reconstruction = model.forward_with_latent(features)
    loss = torch.nn.functional.smooth_l1_loss(reconstruction, features)
    loss.backward()

    assert latent.shape == (24, 4)
    assert reconstruction.shape == (24, 9)
    assert torch.isfinite(loss).item()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    assert model.architecture()["reconstruction_activation"] == "linear"


def test_training_is_finite_deterministic_and_restores_checkpoint(tmp_path):
    rng = np.random.default_rng(101)
    training = rng.normal(size=(256, 9)).astype(np.float32)
    validation = rng.normal(size=(64, 9)).astype(np.float32)
    result = train_autoencoder(training, validation, _config())

    assert 1 <= result.best_epoch <= result.epochs_completed <= 12
    assert np.isfinite(result.best_train_loss)
    assert np.isfinite(result.best_validation_loss)
    assert all(
        np.isfinite([entry["training_loss"], entry["validation_loss"]]).all()
        for entry in result.history
    )

    checkpoint = tmp_path / "candidate.pt"
    save_autoencoder_checkpoint(
        checkpoint,
        result.model,
        training_config=_config(),
        training_summary={"best_epoch": result.best_epoch},
    )
    reloaded = load_autoencoder_checkpoint(checkpoint)["model"]
    assert np.array_equal(
        encode_features(result.model, validation), encode_features(reloaded, validation)
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_autoencoder_checkpoint(
            checkpoint,
            result.model,
            training_config=_config(),
            training_summary={"best_epoch": result.best_epoch},
        )


def test_internal_validation_split_is_deterministic_and_training_only():
    customer_ids = np.array([f"cust-{index}" for index in range(100)])
    train_indices, validation_indices = make_internal_validation_indices(
        customer_ids, validation_fraction=0.1, seed=2024
    )
    train_again, validation_again = make_internal_validation_indices(
        customer_ids, validation_fraction=0.1, seed=2024
    )

    assert np.array_equal(train_indices, train_again)
    assert np.array_equal(validation_indices, validation_again)
    assert not np.intersect1d(train_indices, validation_indices).size
    validate_train_evaluation_ids(customer_ids, np.array(["heldout-1", "heldout-2"]))
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_train_evaluation_ids(customer_ids, np.array(["cust-0"]))


def test_latent_scaler_uses_training_embeddings_only():
    training_embeddings = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    evaluation_embeddings = np.array([[1000.0, -1000.0]])

    scaler = fit_train_only_latent_scaler(training_embeddings)
    transformed = scaler.transform(evaluation_embeddings)

    assert np.array_equal(scaler.mean_, training_embeddings.mean(axis=0))
    assert transformed.shape == (1, 2)
    assert np.all(np.abs(transformed) > 100)
    with pytest.raises(ValueError, match="non-finite"):
        fit_train_only_latent_scaler(np.array([[0.0], [np.inf]]))


def test_spending_ratio_ablation_preserves_the_remaining_feature_order():
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

    selected = select_autoencoder_feature_order(features, exclude_feature="Avg_Spend_Ratio")

    assert selected == [feature for feature in features if feature != "Avg_Spend_Ratio"]
    assert len(selected) == 8
    with pytest.raises(ValueError, match="not in the approved"):
        select_autoencoder_feature_order(features, exclude_feature="Age")


def test_customer_ids_assignments_and_metric_space_metadata_are_aligned():
    customer_ids = np.array(["cust-1", "cust-2", "cust-3", "cust-4"])
    labels = np.array([0, 0, 1, 1])
    assignments = make_assignment_frame(
        customer_ids, labels, run_id="test-ae", split="evaluation"
    )
    metrics = evaluate_assignments(
        np.array([[0.0, 0.1], [0.1, 0.0], [2.0, 2.1], [2.1, 2.0]]),
        labels,
        max_silhouette_rows=4,
        random_state=1,
    )
    parameters = metric_parameters(
        phase="task06_real_comparison",
        family="ae_nine_feature",
        input_feature_order=["f1", "f2"],
        metric_space="original_p1_9d",
        metric_feature_order=["f1", "f2"],
        seed=42,
        k=2,
        clustering={"n_init": 10},
    )

    assert assignments["CustomerID"].tolist() == customer_ids.tolist()
    assert metrics["metric_status"] == "ok"
    assert metrics["silhouette_sample_rows"] == 4
    assert parameters["metric_space"] == "original_p1_9d"
    assert parameters["input_feature_order"] == ["f1", "f2"]
    assert aligned_adjusted_rand_index(labels, customer_ids, labels.copy(), customer_ids.copy()) == 1.0
    with pytest.raises(ValueError, match="identical customer IDs"):
        aligned_adjusted_rand_index(labels, customer_ids, labels, customer_ids[::-1])
    with pytest.raises(ValueError, match="explicitly identified"):
        metric_parameters(
            phase="test",
            family="ae",
            input_feature_order=["f1"],
            metric_space="",
            metric_feature_order=["latent_0"],
            seed=1,
            k=2,
            clustering={},
        )


def test_nonfinite_autoencoder_inputs_fail_before_training():
    training = np.ones((8, 9), dtype=np.float32)
    training[0, 0] = np.nan
    validation = np.ones((4, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        train_autoencoder(training, validation, _config())
