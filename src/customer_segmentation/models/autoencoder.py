"""CPU-capable PyTorch autoencoder utilities for customer features."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class TrainingDivergenceError(RuntimeError):
    """Raised when losses, gradients, or model weights become non-finite."""


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    """Architecture and optimization settings for a deterministic CPU fit."""

    input_dim: int
    hidden_dims: tuple[int, int] = (32, 16)
    latent_dim: int = 4
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 10
    min_delta: float = 1e-5
    huber_beta: float = 1.0
    divergence_factor: float = 100.0
    max_loss: float = 1e6
    seed: int = 42
    num_threads: int = 1

    def validate(self) -> None:
        if self.input_dim < 1 or self.latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive.")
        if len(self.hidden_dims) != 2 or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain two positive layer widths.")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs, and patience must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.min_delta < 0 or self.huber_beta <= 0:
            raise ValueError("min_delta must be non-negative and huber_beta positive.")
        if self.divergence_factor <= 1 or self.max_loss <= 0 or self.num_threads < 1:
            raise ValueError("Invalid divergence, loss, or thread limits.")


class CustomerAutoencoder(nn.Module):
    """Symmetric dense autoencoder with a linear four-dimensional bottleneck."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (32, 16),
        latent_dim: int = 4,
    ) -> None:
        super().__init__()
        if input_dim < 1 or latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive.")
        if len(hidden_dims) != 2 or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain two positive layer widths.")
        width_1, width_2 = hidden_dims
        self.input_dim = int(input_dim)
        self.hidden_dims = (int(width_1), int(width_2))
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_2),
            nn.ReLU(),
            nn.Linear(width_2, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, width_2),
            nn.ReLU(),
            nn.Linear(width_2, width_1),
            nn.ReLU(),
            nn.Linear(width_1, self.input_dim),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward_with_latent(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(features)
        return latent, self.decoder(latent)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_latent(features)[1]

    def architecture(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "hidden_activation": "ReLU",
            "latent_activation": "linear",
            "reconstruction_activation": "linear",
        }


@dataclass
class AutoencoderTrainingResult:
    model: CustomerAutoencoder
    history: list[dict[str, float | int]]
    best_epoch: int
    epochs_completed: int
    best_train_loss: float
    best_validation_loss: float
    stopped_early: bool


def make_internal_validation_indices(
    customer_ids: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic internal split from training IDs only."""
    ids = np.asarray(customer_ids, dtype=str)
    if ids.ndim != 1 or len(ids) < 3 or np.any(ids == "") or len(np.unique(ids)) != len(ids):
        raise ValueError("Internal split requires at least three unique non-empty customer IDs.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one.")
    validation_count = max(1, round(len(ids) * validation_fraction))
    if validation_count >= len(ids) - 1:
        raise ValueError("Internal validation split must leave at least two training customers.")
    permutation = np.random.default_rng(seed).permutation(len(ids))
    validation_indices = np.sort(permutation[:validation_count]).astype(np.int64, copy=False)
    training_mask = np.ones(len(ids), dtype=bool)
    training_mask[validation_indices] = False
    training_indices = np.flatnonzero(training_mask).astype(np.int64, copy=False)
    return training_indices, validation_indices


def validate_train_evaluation_ids(training_ids: np.ndarray, evaluation_ids: np.ndarray) -> None:
    """Reject overlapping, null, or duplicated train/evaluation identifiers."""
    train = np.asarray(training_ids, dtype=str)
    evaluation = np.asarray(evaluation_ids, dtype=str)
    if train.ndim != 1 or evaluation.ndim != 1:
        raise ValueError("Training and evaluation IDs must be one-dimensional.")
    if np.any(train == "") or np.any(evaluation == ""):
        raise ValueError("Training and evaluation IDs must be non-empty.")
    if len(np.unique(train)) != len(train) or len(np.unique(evaluation)) != len(evaluation):
        raise ValueError("Training and evaluation IDs must be unique within each split.")
    if np.intersect1d(train, evaluation).size:
        raise ValueError("Training and held-out evaluation customer IDs must be disjoint.")


def select_autoencoder_feature_order(
    feature_order: list[str], *, exclude_feature: str | None = None
) -> list[str]:
    """Return a validated feature order, optionally omitting one sensitivity feature."""
    if not feature_order or len(feature_order) != len(set(feature_order)):
        raise ValueError("Autoencoder feature order must be non-empty and unique.")
    if exclude_feature is None:
        return list(feature_order)
    if exclude_feature not in feature_order:
        raise ValueError(f"Ablation feature is not in the approved feature order: {exclude_feature}")
    selected = [feature for feature in feature_order if feature != exclude_feature]
    if len(selected) != len(feature_order) - 1:
        raise RuntimeError("Feature ablation must remove exactly one input feature.")
    return selected


def _as_float32_matrix(features: np.ndarray, *, name: str, expected_dim: int) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != expected_dim:
        raise ValueError(f"{name} must have shape (rows, {expected_dim}).")
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values


def _state_dict_to_cpu(model: CustomerAutoencoder) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_autoencoder(
    training_features: np.ndarray,
    validation_features: np.ndarray,
    config: AutoencoderTrainingConfig,
) -> AutoencoderTrainingResult:
    """Fit with SmoothL1 loss, deterministic shuffling, and best-val restoration."""
    config.validate()
    training = _as_float32_matrix(
        training_features, name="training_features", expected_dim=config.input_dim
    )
    validation = _as_float32_matrix(
        validation_features, name="validation_features", expected_dim=config.input_dim
    )
    if len(training) < 2 or len(validation) < 1:
        raise ValueError("Autoencoder fitting requires at least two training and one validation row.")

    torch.set_num_threads(config.num_threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    train_tensor = torch.from_numpy(training).to(device)
    validation_tensor = torch.from_numpy(validation).to(device)
    model = CustomerAutoencoder(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_train_loss = float("inf")
    best_validation_loss = float("inf")
    previous_train_loss: float | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_tensor), generator=generator)
        total_train_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch = train_tensor.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = functional.smooth_l1_loss(
                reconstruction, batch, beta=config.huber_beta, reduction="mean"
            )
            if not torch.isfinite(loss).item() or loss.item() > config.max_loss:
                raise TrainingDivergenceError(
                    f"Non-finite or excessive training loss at epoch {epoch}: {loss.item()}."
                )
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            ):
                raise TrainingDivergenceError(f"Non-finite gradient at epoch {epoch}.")
            optimizer.step()
            if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise TrainingDivergenceError(f"Non-finite model parameter at epoch {epoch}.")
            total_train_loss += float(loss.detach().item()) * len(indices)

        train_loss = total_train_loss / len(train_tensor)
        if previous_train_loss is not None and train_loss > max(
            previous_train_loss, 1e-8
        ) * config.divergence_factor:
            raise TrainingDivergenceError(
                f"Training loss diverged at epoch {epoch}: {train_loss} after {previous_train_loss}."
            )
        previous_train_loss = train_loss

        model.eval()
        with torch.no_grad():
            validation_loss_tensor = functional.smooth_l1_loss(
                model(validation_tensor),
                validation_tensor,
                beta=config.huber_beta,
                reduction="mean",
            )
        validation_loss = float(validation_loss_tensor.item())
        if not np.isfinite(validation_loss) or validation_loss > config.max_loss:
            raise TrainingDivergenceError(
                f"Non-finite or excessive validation loss at epoch {epoch}: {validation_loss}."
            )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(train_loss),
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss - config.min_delta:
            best_epoch = epoch
            best_train_loss = float(train_loss)
            best_validation_loss = validation_loss
            best_state = _state_dict_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_early = epoch < config.max_epochs
                break

    if best_state is None:
        raise TrainingDivergenceError("Training ended without a finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return AutoencoderTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_train_loss=best_train_loss,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
    )


def encode_features(
    model: CustomerAutoencoder,
    features: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Encode finite inputs in bounded batches and return CPU float32 embeddings."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    values = _as_float32_matrix(
        features, name="features", expected_dim=model.input_dim
    )
    embeddings = np.empty((len(values), model.latent_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch = torch.from_numpy(values[start:stop])
            latent = model.encode(batch).cpu().numpy()
            if not np.isfinite(latent).all():
                raise TrainingDivergenceError("Autoencoder produced non-finite latent values.")
            embeddings[start:stop] = latent
    return embeddings


def reconstruction_errors(
    model: CustomerAutoencoder,
    features: np.ndarray,
    *,
    batch_size: int = 4096,
) -> dict[str, dict[str, float]]:
    """Return per-feature MAE and RMSE without retaining reconstructions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    values = _as_float32_matrix(
        features, name="features", expected_dim=model.input_dim
    )
    absolute_sums = np.zeros(model.input_dim, dtype=np.float64)
    squared_sums = np.zeros(model.input_dim, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch = torch.from_numpy(values[start:stop])
            reconstructed = model(batch)
            residual = reconstructed - batch
            if not torch.isfinite(residual).all().item():
                raise TrainingDivergenceError("Autoencoder reconstruction contains non-finite values.")
            absolute_sums += residual.abs().sum(dim=0).cpu().numpy()
            squared_sums += residual.square().sum(dim=0).cpu().numpy()
    return {
        str(index): {
            "mae": float(absolute_sums[index] / len(values)),
            "rmse": float(np.sqrt(squared_sums[index] / len(values))),
        }
        for index in range(model.input_dim)
    }


def save_autoencoder_checkpoint(
    path: str | Path,
    model: CustomerAutoencoder,
    *,
    training_config: AutoencoderTrainingConfig,
    training_summary: dict[str, Any],
) -> Path:
    """Atomically save CPU weights and architecture; never overwrite a checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing autoencoder checkpoint: {destination}")
    payload = {
        "state_dict": _state_dict_to_cpu(model),
        "architecture": model.architecture(),
        "training_config": asdict(training_config),
        "training_summary": training_summary,
    }
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{destination.stem}-", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = temporary.name
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return destination


def load_autoencoder_checkpoint(path: str | Path) -> dict[str, Any]:
    """Restore a saved architecture and state dictionary on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"state_dict", "architecture"}.issubset(payload):
        raise ValueError(f"Invalid autoencoder checkpoint: {path}")
    architecture = payload["architecture"]
    model = CustomerAutoencoder(
        input_dim=int(architecture["input_dim"]),
        hidden_dims=tuple(int(value) for value in architecture["hidden_dims"]),
        latent_dim=int(architecture["latent_dim"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {**payload, "model": model}
