"""Stability protocols: multi-seed and subsample ARI."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import adjusted_rand_score


def multi_seed_ari(
    fit_predict: Callable[[np.ndarray, int], np.ndarray],
    X: np.ndarray,
    *,
    n_seeds: int = 10,
    base_seed: int = 0,
) -> dict[str, float]:
    """Re-fit with different seeds; report ARI vs seed-0 baseline."""
    base = fit_predict(X, base_seed)
    aris: list[float] = []
    for seed in range(1, n_seeds + 1):
        labels = fit_predict(X, seed)
        aris.append(float(adjusted_rand_score(base, labels)))
    arr = np.asarray(aris, dtype=float)
    return {
        "mean_ari": float(arr.mean()) if len(arr) else float("nan"),
        "std_ari": float(arr.std()) if len(arr) else float("nan"),
        "min_ari": float(arr.min()) if len(arr) else float("nan"),
        "n_seeds": float(n_seeds),
    }


def subsample_ari(
    fit_predict: Callable[[np.ndarray, int], np.ndarray],
    X: np.ndarray,
    *,
    frac: float = 0.75,
    n_rounds: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Fit on random 75% subsets; compare labels on the intersection via ARI."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    take = max(int(n * frac), 2)
    aris: list[float] = []
    full = fit_predict(X, random_state)
    for i in range(n_rounds):
        idx = rng.choice(n, size=take, replace=False)
        sub_labels = fit_predict(X[idx], random_state + i + 1)
        # Align: compare full labels on the same indices to the subset fit
        # by re-fitting is already on subset; ARI needs same length —
        # compare full[idx] vs sub_labels (same customers).
        aris.append(float(adjusted_rand_score(full[idx], sub_labels)))
    arr = np.asarray(aris, dtype=float)
    return {
        "mean_ari": float(arr.mean()) if len(arr) else float("nan"),
        "std_ari": float(arr.std()) if len(arr) else float("nan"),
        "min_ari": float(arr.min()) if len(arr) else float("nan"),
        "n_rounds": float(n_rounds),
        "frac": float(frac),
    }
