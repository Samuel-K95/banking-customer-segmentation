"""Tabular autoencoder + clustering on latent space (K-Means or GMM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from customer_segmentation.models.metrics import evaluate_clustering, size_sanity

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_OK = True
except ImportError:  # pragma: no cover
    TORCH_OK = False


if TORCH_OK:

    class TabularAE(nn.Module):
        def __init__(self, n_feat: int, latent_dim: int, hidden: int = 32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_feat, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_feat),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            z = self.encoder(x)
            return self.decoder(z), z


@dataclass
class AETrainResult:
    latent_dim: int
    z: np.ndarray
    losses: list[float]
    model: Any


@dataclass
class AEResult:
    latent_dim: int
    labels: np.ndarray
    z: np.ndarray
    losses: list[float]
    metrics: dict[str, float]
    model: Any
    clusterer: Any
    cluster_method: str
    size: dict[str, Any]
    confidence: np.ndarray | None = None
    # backward-compat alias used by pipeline
    kmeans: Any = None


def train_autoencoder(
    X: np.ndarray,
    *,
    latent_dim: int,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    random_state: int = 42,
    device: str | None = None,
    hidden: int = 32,
    noise_std: float = 0.0,
) -> AETrainResult:
    """Train a tabular AE. If noise_std > 0, train as a denoising AE."""
    if not TORCH_OK:
        raise ImportError("PyTorch is required for the autoencoder track.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    Xt = torch.tensor(X, dtype=torch.float32)
    model = TabularAE(X.shape[1], latent_dim, hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=True)

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(dev)
            inp = batch
            if noise_std > 0:
                inp = batch + noise_std * torch.randn_like(batch)
            opt.zero_grad()
            recon, _ = model(inp)
            loss = crit(recon, batch)  # reconstruct clean target
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        _, z_t = model(Xt.to(dev))
        z = z_t.cpu().numpy()

    return AETrainResult(latent_dim=latent_dim, z=z, losses=losses, model=model.cpu())


def cluster_latent(
    z: np.ndarray,
    *,
    n_clusters: int,
    method: Literal["kmeans", "gmm"] = "kmeans",
    random_state: int = 42,
) -> tuple[np.ndarray, Any, np.ndarray]:
    """Cluster latent embeddings; return labels, fitted model, confidence."""
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        labels = model.fit_predict(z)
        conf = distance_confidence(z, labels, model.cluster_centers_)
        return labels, model, conf

    if method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
        )
        labels = model.fit_predict(z)
        proba = model.predict_proba(z)
        conf = proba.max(axis=1).astype(np.float64)
        return labels, model, conf

    raise ValueError(f"Unknown cluster method: {method}")


def train_ae_cluster(
    X: np.ndarray,
    *,
    latent_dim: int,
    n_clusters: int,
    method: Literal["kmeans", "gmm"] = "kmeans",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    random_state: int = 42,
    device: str | None = None,
    trained: AETrainResult | None = None,
) -> AEResult:
    """Train AE (or reuse trained) then cluster latent space with K-Means or GMM."""
    ae = trained or train_autoencoder(
        X,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        random_state=random_state,
        device=device,
    )
    labels, clusterer, conf = cluster_latent(
        ae.z, n_clusters=n_clusters, method=method, random_state=random_state
    )
    metrics = evaluate_clustering(ae.z, labels, random_state=random_state)
    return AEResult(
        latent_dim=ae.latent_dim,
        labels=labels,
        z=ae.z,
        losses=ae.losses,
        metrics=metrics,
        model=ae.model,
        clusterer=clusterer,
        cluster_method=method,
        size=size_sanity(labels),
        confidence=conf,
        kmeans=clusterer if method == "kmeans" else None,
    )


def train_ae_kmeans(
    X: np.ndarray,
    *,
    latent_dim: int,
    n_clusters: int,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    random_state: int = 42,
    device: str | None = None,
) -> AEResult:
    return train_ae_cluster(
        X,
        latent_dim=latent_dim,
        n_clusters=n_clusters,
        method="kmeans",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        random_state=random_state,
        device=device,
    )


def train_ae_gmm(
    X: np.ndarray,
    *,
    latent_dim: int,
    n_clusters: int,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    random_state: int = 42,
    device: str | None = None,
    trained: AETrainResult | None = None,
) -> AEResult:
    return train_ae_cluster(
        X,
        latent_dim=latent_dim,
        n_clusters=n_clusters,
        method="gmm",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        random_state=random_state,
        device=device,
        trained=trained,
    )


def distance_confidence(z: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Soft confidence from inverse distance to assigned centroid vs nearest other."""
    dists = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    own = dists[np.arange(len(labels)), labels]
    sorted_d = np.sort(dists, axis=1)
    second = sorted_d[:, 1] if sorted_d.shape[1] > 1 else sorted_d[:, 0] + 1e-6
    conf = second / (own + second + 1e-9)
    return conf.astype(np.float64)
