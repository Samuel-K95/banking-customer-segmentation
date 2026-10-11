"""Berka (Track C) bake-off → champion registration + artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from customer_segmentation.features.build_berka_features import build_and_save_berka_features
from customer_segmentation.models.autoencoder import (
    TORCH_OK,
    distance_confidence,
    train_ae_cluster,
    train_ae_kmeans,
    train_autoencoder,
)
from customer_segmentation.models.registry import register_champion
from customer_segmentation.models.segment_intel import build_segment_catalog
from customer_segmentation.models.stability import multi_seed_ari, subsample_ari
from customer_segmentation.models.traditional import (
    best_by_silhouette,
    sweep_gmm,
    sweep_hierarchical,
    sweep_kmeans,
)
from customer_segmentation.models import viz

RANDOM_STATE = 42
K_RANGE = list(range(3, 9))
FEATURE_VERSION = "berka_rfm_product_v1"
MODEL_VERSION = "v1"
CHAMPION_RUN_ID = "c_ae_lat4_k6"


def _row(
    run_id: str,
    method: str,
    k: int,
    latent_dim: str | int,
    metrics: dict[str, float],
    stability_ari: float | None = None,
    recommended: bool = False,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "track": "berka",
        "method": method,
        "k": k,
        "latent_dim": latent_dim,
        "silhouette": round(metrics["silhouette"], 4),
        "db_index": round(metrics["db_index"], 4),
        "ch_index": round(metrics["ch_index"], 0),
        "stability_ari": None if stability_ari is None else round(stability_ari, 3),
        "recommended": recommended,
    }


def run_berka_segmentation(
    *,
    berka_dir: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
    processed_dir: str | Path = "data/processed",
    epochs: int = 50,
) -> dict[str, Any]:
    """Full Track C pipeline: features → sweeps → AE → register champion."""
    artifact_root = Path(artifact_root)
    exp_dir = artifact_root / "experiments"
    champ_dir = artifact_root / "segmentation" / "champion_v1"
    fig_dir = champ_dir / "figures"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Berka features ===")
    bundle = build_and_save_berka_features(berka_dir=berka_dir, processed_dir=processed_dir)
    id_col = bundle.id_col
    feat_cols = bundle.feature_columns
    X = bundle.model[feat_cols].fillna(0).values.astype(np.float64)
    print(f"  matrix={X.shape} cols={feat_cols}")

    rows: list[dict[str, Any]] = []

    # ── Traditional ──────────────────────────────────────────────
    print("=== K-Means sweep ===")
    km_res = sweep_kmeans(X, K_RANGE, random_state=RANDOM_STATE)
    best_k_km = best_by_silhouette(km_res)
    viz.plot_kmeans_sweep(
        K_RANGE,
        [km_res[k]["inertia"] for k in K_RANGE],
        [km_res[k]["silhouette"] for k in K_RANGE],
        [km_res[k]["db_index"] for k in K_RANGE],
        fig_dir / "kmeans_sweep.png",
        title="Track C — K-Means sweep",
    )
    for k, r in km_res.items():
        rows.append(
            _row(
                f"c_kmeans_k{k}",
                "kmeans",
                k,
                "-",
                r,
                recommended=(k == 6),  # notebook prior / dendrogram narrative
            )
        )

    print("=== GMM sweep ===")
    gmm_res = sweep_gmm(X, K_RANGE, random_state=RANDOM_STATE)
    for k, r in gmm_res.items():
        rows.append(_row(f"c_gmm_k{k}", "gmm", k, "-", r))

    print("=== Hierarchical (Ward) ===")
    hier_res = sweep_hierarchical(X, [5, 6, 7], linkage="ward")
    for k, r in hier_res.items():
        rows.append(_row(f"c_hier_k{k}", "hierarchical", k, "-", r, recommended=(k == 6)))

    # Stability on K-Means k=6 (and later on AE)
    k_stab = 6 if 6 in km_res else best_k_km

    def _km_fp(X_in: np.ndarray, seed: int) -> np.ndarray:
        return KMeans(n_clusters=k_stab, n_init=10, random_state=seed).fit_predict(X_in)

    print("=== Stability (K-Means) ===")
    stab_seed = multi_seed_ari(_km_fp, X, n_seeds=10, base_seed=0)
    stab_sub = subsample_ari(_km_fp, X, frac=0.75, n_rounds=5, random_state=RANDOM_STATE)
    for row in rows:
        if row["run_id"] == f"c_kmeans_k{k_stab}":
            row["stability_ari"] = round(stab_seed["mean_ari"], 3)

    viz.plot_stability_bars(
        {
            "kmeans_multiseed": stab_seed["mean_ari"],
            "kmeans_subsample": stab_sub["mean_ari"],
        },
        fig_dir / "stability_ari.png",
    )

    # ── Autoencoder ──────────────────────────────────────────────
    ae_results: dict[int, Any] = {}
    champion = None
    if TORCH_OK:
        print("=== Autoencoder + K-Means / GMM ===")
        losses_by_lat: dict[int, list[float]] = {}
        ae_gmm_lat4 = None
        for lat in (4, 8):
            trained = train_autoencoder(
                X, latent_dim=lat, epochs=epochs, random_state=RANDOM_STATE
            )
            ae = train_ae_cluster(
                X,
                latent_dim=lat,
                n_clusters=k_stab,
                method="kmeans",
                trained=trained,
                random_state=RANDOM_STATE,
            )
            ae_results[lat] = ae
            losses_by_lat[lat] = ae.losses
            rows.append(
                _row(
                    f"c_ae_lat{lat}_k{k_stab}",
                    "ae_kmeans",
                    k_stab,
                    lat,
                    ae.metrics,
                    recommended=False,
                )
            )
            print(
                f"  AE+KMeans lat={lat} sil={ae.metrics['silhouette']:.3f} "
                f"DB={ae.metrics['db_index']:.3f} CH={ae.metrics['ch_index']:.0f}"
            )
            # GMM head on the same latent (winner latent=4 gets the full compare)
            if lat == 4:
                ae_gmm_lat4 = train_ae_cluster(
                    X,
                    latent_dim=lat,
                    n_clusters=k_stab,
                    method="gmm",
                    trained=trained,
                    random_state=RANDOM_STATE,
                )
                rows.append(
                    _row(
                        f"c_ae_lat{lat}_gmm_k{k_stab}",
                        "ae_gmm",
                        k_stab,
                        lat,
                        ae_gmm_lat4.metrics,
                        recommended=False,
                    )
                )
                print(
                    f"  AE+GMM    lat={lat} sil={ae_gmm_lat4.metrics['silhouette']:.3f} "
                    f"DB={ae_gmm_lat4.metrics['db_index']:.3f} "
                    f"CH={ae_gmm_lat4.metrics['ch_index']:.0f}"
                )
        viz.plot_ae_loss(losses_by_lat, fig_dir / "ae_recon_loss.png")
        # Champion by project policy: AE lat=4 + K-Means (AE+GMM is on the board for comparison)
        champion = ae_results[4]
    else:
        print("PyTorch unavailable — champion falls back to K-Means k=6")
        champion_labels = km_res[k_stab]["labels"]
        champion = type("C", (), {})()
        champion.labels = champion_labels
        champion.z = X
        champion.metrics = {
            "silhouette": km_res[k_stab]["silhouette"],
            "db_index": km_res[k_stab]["db_index"],
            "ch_index": km_res[k_stab]["ch_index"],
        }
        champion.kmeans = km_res[k_stab]["model"]
        champion.model = None
        champion.latent_dim = None
        champion.size = km_res[k_stab]["size"]

    # AE stability (multi-seed on latent K-Means only — encoder frozen from champion)
    ae_stab = {"mean_ari": float("nan")}
    if TORCH_OK and champion is not None and getattr(champion, "z", None) is not None:
        z = champion.z

        def _km_z(X_in: np.ndarray, seed: int) -> np.ndarray:
            return KMeans(n_clusters=k_stab, n_init=10, random_state=seed).fit_predict(X_in)

        ae_stab = multi_seed_ari(_km_z, z, n_seeds=10, base_seed=0)
        for row in rows:
            if row["run_id"] == CHAMPION_RUN_ID:
                row["stability_ari"] = round(ae_stab["mean_ari"], 3)
                row["recommended"] = False  # selection is policy, not notebook prior

    board = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    board_path = exp_dir / "comparison_board_track_c.csv"
    board.to_csv(board_path, index=False)
    print(f"Saved comparison board -> {board_path}")

    labels = champion.labels
    if TORCH_OK and hasattr(champion, "kmeans") and hasattr(champion, "z"):
        conf = distance_confidence(champion.z, labels, champion.kmeans.cluster_centers_)
        space = champion.z
    else:
        centers = km_res[k_stab]["model"].cluster_centers_
        conf = distance_confidence(X, labels, centers)
        space = X

    assignments = pd.DataFrame(
        {
            id_col: bundle.interpretable[id_col].values,
            "segment_id": labels,
            "confidence": np.round(conf, 4),
        }
    )

    profiles = (
        bundle.interpretable.assign(segment_id=labels)
        .groupby("segment_id")[feat_cols]
        .mean()
        .round(2)
    )
    pop = bundle.interpretable[feat_cols].mean()
    counts = {int(s): int(n) for s, n in zip(*np.unique(labels, return_counts=True))}
    segment_catalog = build_segment_catalog(
        profiles,
        population=pop,
        sizes=counts,
        total_n=len(labels),
    )

    viz.plot_size_bars(labels, fig_dir / "size_bars.png", title="Champion segment sizes")
    viz.plot_pca_segments(space, labels, fig_dir / "pca_segments.png", title="Champion latent/feature PCA")
    viz.plot_centroid_heatmap(profiles, fig_dir / "centroid_heatmap.png")

    encoder_state = None
    if TORCH_OK and getattr(champion, "model", None) is not None:
        encoder_state = {
            "state_dict": champion.model.state_dict(),
            "latent_dim": champion.latent_dim,
            "n_feat": X.shape[1],
            "hidden": 32,
        }

    metrics_payload = {
        "run_id": CHAMPION_RUN_ID if TORCH_OK else f"c_kmeans_k{k_stab}",
        "method": "ae_kmeans" if TORCH_OK else "kmeans",
        "k": k_stab,
        "latent_dim": getattr(champion, "latent_dim", None),
        "silhouette": champion.metrics["silhouette"],
        "db_index": champion.metrics["db_index"],
        "ch_index": champion.metrics["ch_index"],
        "stability_ari_kmeans_multiseed": stab_seed["mean_ari"],
        "stability_ari_kmeans_subsample": stab_sub["mean_ari"],
        "stability_ari_ae_latent_kmeans": ae_stab.get("mean_ari"),
        "size": getattr(champion, "size", {}),
        "low_confidence_frac": float((conf < 0.55).mean()),
        "selection_policy": (
            "Berka AE lat=4 selected for behavioral+product coverage; "
            "not max silhouette across tracks (see champion_recommendation.md)"
        ),
    }

    paths = register_champion(
        champ_dir,
        run_id=metrics_payload["run_id"],
        track="berka",
        method=metrics_payload["method"],
        feature_version=FEATURE_VERSION,
        model_version=MODEL_VERSION,
        feature_columns=feat_cols,
        metrics=metrics_payload,
        assignments=assignments,
        profiles=profiles,
        comparison_board=board,
        scaler=bundle.scaler,
        kmeans=getattr(champion, "kmeans", None),
        encoder_state=encoder_state,
        segment_catalog=segment_catalog,
        notes="Champion = Track C AE lat=4 k=6 per assignment behavioral criteria.",
    )

    # Also copy board to experiments (already written)
    summary = {
        "champion_dir": str(champ_dir),
        "comparison_board": str(board_path),
        "metrics": metrics_payload,
        "files": {k: str(v) for k, v in paths.items()},
    }
    (champ_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Champion registered -> {champ_dir}")
    return summary


if __name__ == "__main__":
    run_berka_segmentation()
