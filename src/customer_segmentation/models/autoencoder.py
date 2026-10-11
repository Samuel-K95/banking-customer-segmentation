"""Tabular autoencoder + K-Means on latent space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

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
class AEResult:
    latent_dim: int
    labels: np.ndarray
    z: np.ndarray
    losses: list[float]
    metrics: dict[str, float]
    model: Any
    kmeans: KMeans
    size: dict[str, Any]


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
    if not TORCH_OK:
        raise ImportError("PyTorch is required for the autoencoder track.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    Xt = torch.tensor(X, dtype=torch.float32)
    model = TabularAE(X.shape[1], latent_dim).to(dev)
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
            opt.zero_grad()
            recon, _ = model(batch)
            loss = crit(recon, batch)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        _, z_t = model(Xt.to(dev))
        z = z_t.cpu().numpy()

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = km.fit_predict(z)
    metrics = evaluate_clustering(z, labels, random_state=random_state)
    return AEResult(
        latent_dim=latent_dim,
        labels=labels,
        z=z,
        losses=losses,
        metrics=metrics,
        model=model.cpu(),
        kmeans=km,
        size=size_sanity(labels),
    )


def distance_confidence(z: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Soft confidence from inverse distance to assigned centroid vs nearest other."""
    dists = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    own = dists[np.arange(len(labels)), labels]
    # second-nearest
    sorted_d = np.sort(dists, axis=1)
    second = sorted_d[:, 1] if sorted_d.shape[1] > 1 else sorted_d[:, 0] + 1e-6
    # closer to own than second → high confidence
    conf = second / (own + second + 1e-9)
    return conf.astype(np.float64)
