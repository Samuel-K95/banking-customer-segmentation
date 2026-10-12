"""Traditional clustering sweeps: K-Means, GMM, Hierarchical."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from customer_segmentation.models.metrics import evaluate_clustering, size_sanity


def sweep_kmeans(
    X: np.ndarray,
    k_values: range | list[int],
    *,
    n_init: int = 20,
    random_state: int = 42,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        metrics = evaluate_clustering(X, labels, random_state=random_state)
        results[k] = {
            "model": km,
            "labels": labels,
            "inertia": float(km.inertia_),
            **metrics,
            "size": size_sanity(labels),
        }
    return results


def sweep_gmm(
    X: np.ndarray,
    k_values: range | list[int],
    *,
    random_state: int = 42,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for k in k_values:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state, n_init=3)
        labels = gmm.fit_predict(X)
        proba = gmm.predict_proba(X)
        metrics = evaluate_clustering(X, labels, random_state=random_state)
        confidence = proba.max(axis=1)
        results[k] = {
            "model": gmm,
            "labels": labels,
            "proba": proba,
            "confidence": confidence,
            "bic": float(gmm.bic(X)),
            "aic": float(gmm.aic(X)),
            **metrics,
            "size": size_sanity(labels),
            "low_confidence_frac": float((confidence < 0.9).mean()),
        }
    return results


def sweep_hierarchical(
    X: np.ndarray,
    k_values: range | list[int],
    *,
    linkage: str = "ward",
    max_n: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Ward linkage; optionally cap N for memory (pdist is O(N^2))."""
    X_fit = X if max_n is None or len(X) <= max_n else X[:max_n]
    results: dict[int, dict[str, Any]] = {}
    for k in k_values:
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = agg.fit_predict(X_fit)
        metrics = evaluate_clustering(X_fit, labels)
        results[k] = {
            "model": agg,
            "labels": labels,
            "n_used": int(len(X_fit)),
            **metrics,
            "size": size_sanity(labels),
        }
    return results


def best_by_silhouette(results: dict[int, dict[str, Any]]) -> int:
    return max(results.keys(), key=lambda k: results[k]["silhouette"])
