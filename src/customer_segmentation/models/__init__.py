"""Segmentation models: traditional, AE, and loadable champion."""

from customer_segmentation.models.metrics import evaluate_clustering, size_sanity
from customer_segmentation.models.traditional import (
    sweep_gmm,
    sweep_hierarchical,
    sweep_kmeans,
)

__all__ = [
    "ChampionModel",
    "evaluate_clustering",
    "size_sanity",
    "sweep_gmm",
    "sweep_hierarchical",
    "sweep_kmeans",
]


def __getattr__(name: str):
    if name == "ChampionModel":
        from customer_segmentation.models.champion import ChampionModel

        return ChampionModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
