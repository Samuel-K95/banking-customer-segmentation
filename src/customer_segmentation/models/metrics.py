"""Internal cluster-quality metrics used across all tracks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int | None = 20_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Return silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

    Silhouette is optionally computed on a subsample for large N.
    """
    labels = np.asarray(labels)
    X = np.asarray(X)
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return {"silhouette": float("nan"), "db_index": float("nan"), "ch_index": float("nan")}

    if sample_size is not None and len(labels) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(labels), size=sample_size, replace=False)
        X_eval, y_eval = X[idx], labels[idx]
    else:
        X_eval, y_eval = X, labels

    return {
        "silhouette": float(silhouette_score(X_eval, y_eval)),
        "db_index": float(davies_bouldin_score(X_eval, y_eval)),
        "ch_index": float(calinski_harabasz_score(X_eval, y_eval)),
    }


def size_sanity(labels: np.ndarray, *, min_frac: float = 0.03, max_frac: float = 0.60) -> dict[str, Any]:
    """Check that no segment is tiny or swallows most of the book."""
    labels = np.asarray(labels)
    n = len(labels)
    counts = np.bincount(labels.astype(int))
    fracs = counts / max(n, 1)
    return {
        "counts": counts.tolist(),
        "fractions": fracs.tolist(),
        "ok": bool(fracs.min() >= min_frac and fracs.max() <= max_frac),
        "min_frac": float(fracs.min()),
        "max_frac": float(fracs.max()),
    }
