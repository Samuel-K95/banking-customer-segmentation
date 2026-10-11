"""Persist segmentation figures to disk (not notebook-only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_kmeans_sweep(
    k_values: list[int],
    inertias: list[float],
    sils: list[float],
    dbs: list[float],
    out: Path,
    title: str = "K-Means sweep",
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(k_values, inertias, "b-o")
    axes[0].set_title("Elbow (inertia)")
    axes[0].set_xlabel("k")
    axes[1].plot(k_values, sils, "g-o")
    axes[1].set_title("Silhouette")
    axes[1].set_xlabel("k")
    axes[2].plot(k_values, dbs, "r-o")
    axes[2].set_title("Davies-Bouldin")
    axes[2].set_xlabel("k")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def plot_size_bars(labels: np.ndarray, out: Path, title: str = "Segment sizes") -> Path:
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#1a3a5c")
    for i, v in zip(counts.index, counts.values):
        ax.text(str(i), v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("segment_id")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def plot_pca_segments(X: np.ndarray, labels: np.ndarray, out: Path, title: str = "PCA (exploratory)") -> Path:
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg in np.unique(labels):
        mask = labels == seg
        ax.scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=0.55, label=f"Seg {seg}")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title + " — exploratory, not ground truth")
    fig.tight_layout()
    return _save(fig, out)


def plot_centroid_heatmap(profile: pd.DataFrame, out: Path, title: str = "Segment mean profiles") -> Path:
    # z-score columns for readable heatmap
    z = (profile - profile.mean()) / (profile.std(ddof=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(8, profile.shape[1] * 0.9), max(3, profile.shape[0] * 0.55)))
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(profile.columns)))
    ax.set_xticklabels(profile.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(profile.index)))
    ax.set_yticklabels([f"Seg {i}" for i in profile.index])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return _save(fig, out)


def plot_ae_loss(losses_by_lat: dict[int, list[float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for lat, losses in losses_by_lat.items():
        ax.plot(losses, label=f"latent={lat}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Autoencoder training loss")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def plot_stability_bars(ari_values: dict[str, float], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(ari_values.keys())
    vals = [ari_values[n] for n in names]
    ax.barh(names, vals, color="#2E7D32")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("ARI")
    ax.set_title("Stability (higher = more reproducible)")
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)
