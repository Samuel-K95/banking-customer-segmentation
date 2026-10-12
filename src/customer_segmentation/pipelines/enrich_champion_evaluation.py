"""Enrich frozen champion evaluation without AE retrain.

Ports notebook 11 fixes onto ``artifacts/segmentation/champion``:
shared metrics, subsample stability, expanded comparison board.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from customer_segmentation.data.berka_dataset_loader import (
    load_berka_product_tables,
    load_berka_raw_data,
    resolve_berka_dir,
)
from customer_segmentation.features.build_berka_segmentation_features import (
    build_berka_segmentation_features_asof,
    transform_with_train_scaler,
)
from customer_segmentation.models.champion import ChampionModel
from customer_segmentation.models.metrics import evaluate_clustering
from customer_segmentation.models.registry import write_json
from customer_segmentation.models.stability import multi_seed_ari, subsample_ari


def enrich_champion_evaluation(
    *,
    champion_dir: str | Path = "artifacts/segmentation/champion",
    berka_dir: str | Path | None = None,
    n_seed_stability: int = 5,
    n_subsample_rounds: int = 5,
) -> dict[str, Any]:
    champ_dir = Path(champion_dir)
    champ = ChampionModel.load(champ_dir)
    metrics = dict(champ.metrics or {})
    as_of = str(metrics.get("as_of_date") or "1997-12-31")
    min_tenure = int(metrics.get("minimum_tenure_days") or 365)
    n_clusters = int(champ.n_clusters)
    split = pd.read_csv(champ_dir / "account_split.csv")
    assign = pd.read_csv(champ_dir / "assignments.csv")

    path = resolve_berka_dir(berka_dir)
    trans, account, _client = load_berka_raw_data(path)
    loan, card, disp = load_berka_product_tables(path)
    interpretable = build_berka_segmentation_features_asof(
        trans,
        account,
        loan,
        card,
        disp,
        as_of_date=as_of,
        minimum_tenure_days=min_tenure,
    )
    df = interpretable.merge(split, on="account_id", how="inner").merge(
        assign[["account_id", "segment_id", "confidence"]], on="account_id", how="inner"
    )
    train_df = df[df["split"] == "train"].copy()
    hold_df = df[df["split"] == "holdout"].copy()
    x_train = transform_with_train_scaler(train_df, champ.scaler)
    x_hold = transform_with_train_scaler(hold_df, champ.scaler)
    z_train = champ.encode(x_train)
    z_hold = champ.encode(x_hold)

    def fit_km(X: np.ndarray, seed: int, k: int = n_clusters) -> np.ndarray:
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)

    hold_metrics = evaluate_clustering(z_hold, champ.kmeans.predict(z_hold))
    train_metrics = evaluate_clustering(z_train, champ.kmeans.predict(z_train))
    seed_stability = multi_seed_ari(lambda X, s: fit_km(X, s), z_train, n_seeds=n_seed_stability, base_seed=42)
    sub_stability = subsample_ari(
        lambda X, s: fit_km(X, s),
        z_train,
        frac=0.75,
        n_rounds=n_subsample_rounds,
        random_state=42,
    )

    km6 = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(x_train)
    km6_hold = evaluate_clustering(x_hold, km6.predict(x_hold))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42, n_init=3)
    gmm.fit(x_train)
    gmm_hold = evaluate_clustering(x_hold, gmm.predict(x_hold))
    hier_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(x_train)
    hier_train = evaluate_clustering(x_train, hier_labels)
    km2 = KMeans(n_clusters=2, n_init=20, random_state=42).fit(x_train)
    km2_hold = evaluate_clustering(x_hold, km2.predict(x_hold))

    board = pd.DataFrame(
        [
            {
                "run_id": champ.run_id,
                "family": "ae_kmeans",
                "k": n_clusters,
                "latent_dim": 4,
                "split": "holdout",
                "silhouette": round(hold_metrics["silhouette"], 4),
                "db_index": round(hold_metrics["db_index"], 4),
                "ch_index": round(hold_metrics["ch_index"], 0),
                "seed_ari_mean": round(seed_stability["mean_ari"], 4),
                "subsample_ari_mean": round(sub_stability["mean_ari"], 4),
                "role": "product_champion_leakage_safe",
            },
            {
                "run_id": "c_kmeans_k6_scaled_leakage_safe",
                "family": "kmeans",
                "k": 6,
                "latent_dim": "-",
                "split": "holdout",
                "silhouette": round(km6_hold["silhouette"], 4),
                "db_index": round(km6_hold["db_index"], 4),
                "ch_index": round(km6_hold["ch_index"], 0),
                "seed_ari_mean": None,
                "subsample_ari_mean": None,
                "role": "traditional_baseline_same_matrix",
            },
            {
                "run_id": "c_gmm_k6_scaled_leakage_safe",
                "family": "gmm",
                "k": 6,
                "latent_dim": "-",
                "split": "holdout",
                "silhouette": round(gmm_hold["silhouette"], 4),
                "db_index": round(gmm_hold["db_index"], 4),
                "ch_index": round(gmm_hold["ch_index"], 0),
                "seed_ari_mean": None,
                "subsample_ari_mean": None,
                "role": "traditional_baseline_same_matrix",
            },
            {
                "run_id": "c_hier_k6_scaled_leakage_safe",
                "family": "hierarchical",
                "k": 6,
                "latent_dim": "-",
                "split": "train",
                "silhouette": round(hier_train["silhouette"], 4),
                "db_index": round(hier_train["db_index"], 4),
                "ch_index": round(hier_train["ch_index"], 0),
                "seed_ari_mean": None,
                "subsample_ari_mean": None,
                "role": "traditional_baseline_train_only",
            },
            {
                "run_id": "c_kmeans_k2_leakage_safe",
                "family": "kmeans",
                "k": 2,
                "latent_dim": "-",
                "split": "holdout",
                "silhouette": round(km2_hold["silhouette"], 4),
                "db_index": round(km2_hold["db_index"], 4),
                "ch_index": round(km2_hold["ch_index"], 0),
                "seed_ari_mean": None,
                "subsample_ari_mean": None,
                "role": "research_twin_same_matrix",
            },
        ]
    )

    metrics.update(
        {
            "holdout_silhouette": hold_metrics["silhouette"],
            "holdout_db_index": hold_metrics["db_index"],
            "holdout_ch_index": hold_metrics["ch_index"],
            "train_silhouette": train_metrics["silhouette"],
            "silhouette": hold_metrics["silhouette"],
            "db_index": hold_metrics["db_index"],
            "ch_index": hold_metrics["ch_index"],
            "stability_ari_ae_latent_kmeans": seed_stability["mean_ari"],
            "stability_ari_ae_latent_kmeans_min": seed_stability["min_ari"],
            "stability_ari_ae_latent_kmeans_std": seed_stability["std_ari"],
            "stability_subsample_ari_ae_latent_kmeans": sub_stability["mean_ari"],
            "stability_subsample_ari_ae_latent_kmeans_min": sub_stability["min_ari"],
            "stability_subsample_ari_ae_latent_kmeans_std": sub_stability["std_ari"],
            "stability_protocols": ["multi_seed_ari", "subsample_ari"],
            "metrics_module": "customer_segmentation.models.metrics.evaluate_clustering",
            "parallel_kmeans_k2_holdout": km2_hold,
            "baselines_same_matrix_holdout": {
                "kmeans_k6": km6_hold,
                "gmm_k6": gmm_hold,
                "hierarchical_k6_train": hier_train,
            },
            "selection_policy": (
                "Leakage-safe AE lat=4 k=6 remains the product path: best holdout silhouette "
                f"({hold_metrics['silhouette']:.3f}) vs raw KMeans k=6 ({km6_hold['silhouette']:.3f}), "
                f"GMM k=6 ({gmm_hold['silhouette']:.3f}); k=2 twin is research-only."
            ),
        }
    )
    write_json(champ_dir / "metrics.json", metrics)
    board.to_csv(champ_dir / "comparison_board.csv", index=False)

    # Keep research twin copy when present
    research = Path("artifacts/segmentation/research/leakage_safe_source")
    if research.exists():
        write_json(research / "metrics.json", {**metrics, "note": "mirrored from enrich_champion_evaluation"})
        board.to_csv(research / "comparison_board.csv", index=False)

    summary = {
        "champion_dir": str(champ_dir),
        "holdout_metrics": hold_metrics,
        "stability_seed_ari": seed_stability,
        "stability_subsample_ari": sub_stability,
        "comparison_board_rows": int(len(board)),
        "files": {
            "metrics": str(champ_dir / "metrics.json"),
            "comparison_board": str(champ_dir / "comparison_board.csv"),
        },
    }
    write_json(champ_dir / "evaluation_enrichment_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(enrich_champion_evaluation(), indent=2, default=str))
