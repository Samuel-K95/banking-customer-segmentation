"""Broad Berka exploration: AE variants + simpler/experimental clustering.

Saves a full comparison board under artifacts/experiments/berka_explore/.
Does NOT change the registered champion — this is an experiment factory run.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    HDBSCAN,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from customer_segmentation.features.build_berka_features import build_and_save_berka_features
from customer_segmentation.models.tabular_ae import TORCH_OK, train_autoencoder
from customer_segmentation.models.metrics import evaluate_clustering, size_sanity
from customer_segmentation.models import viz

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
K_FIXED = 6


def _row(
    run_id: str,
    family: str,
    method: str,
    k: Any,
    extra: dict[str, Any],
    metrics: dict[str, float],
    size: dict[str, Any],
    elapsed_s: float,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "family": family,
        "method": method,
        "k": k,
        "silhouette": round(metrics.get("silhouette", float("nan")), 4),
        "db_index": round(metrics.get("db_index", float("nan")), 4),
        "ch_index": round(metrics.get("ch_index", float("nan")), 1),
        "n_clusters_found": size.get("n_clusters"),
        "noise_frac": size.get("noise_frac"),
        "min_frac": round(size.get("min_frac", float("nan")), 4) if size.get("min_frac") is not None else None,
        "max_frac": round(size.get("max_frac", float("nan")), 4) if size.get("max_frac") is not None else None,
        "size_ok": size.get("ok"),
        "elapsed_s": round(elapsed_s, 2),
        "notes": notes,
        **extra,
    }


def _label_stats(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    noise_frac = float((labels < 0).mean()) if np.any(labels < 0) else 0.0
    valid = labels[labels >= 0]
    if len(valid) == 0 or len(np.unique(valid)) < 2:
        return {
            "n_clusters": int(len(np.unique(valid))),
            "noise_frac": noise_frac,
            "min_frac": None,
            "max_frac": None,
            "ok": False,
            "counts": [],
        }
    base = size_sanity(valid)
    base["n_clusters"] = int(len(np.unique(valid)))
    base["noise_frac"] = noise_frac
    return base


def _safe_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        return {"silhouette": float("nan"), "db_index": float("nan"), "ch_index": float("nan")}
    return evaluate_clustering(X[mask], labels[mask], random_state=RANDOM_STATE)


def run_explore(
    *,
    berka_dir: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
    ae_epochs: int = 40,
) -> pd.DataFrame:
    out_dir = Path(artifact_root) / "experiments" / "berka_explore"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Berka features ===")
    bundle = build_and_save_berka_features(berka_dir=berka_dir)
    feat_cols = bundle.feature_columns
    X = bundle.model[feat_cols].fillna(0).values.astype(np.float64)
    print(f"  matrix={X.shape}")

    rows: list[dict[str, Any]] = []
    ae_loss_curves: dict[str, list[float]] = {}

    # ── 1. Simple / classical on raw features ─────────────────────
    print("\n=== 1. Classical / experimental on raw features ===")
    classical = [
        ("kmeans_k6", "kmeans", KMeans(n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE), K_FIXED, ""),
        ("minibatch_kmeans_k6", "minibatch_kmeans", MiniBatchKMeans(n_clusters=K_FIXED, random_state=RANDOM_STATE, batch_size=256, n_init=10), K_FIXED, "scalable K-Means"),
        ("gmm_k6", "gmm", GaussianMixture(n_components=K_FIXED, covariance_type="full", random_state=RANDOM_STATE, n_init=5), K_FIXED, "soft membership"),
        ("birch_k6", "birch", Birch(n_clusters=K_FIXED), K_FIXED, "hierarchical CF-tree"),
        ("hier_ward_k6", "hierarchical", AgglomerativeClustering(n_clusters=K_FIXED, linkage="ward"), K_FIXED, "ward"),
        ("hier_average_k6", "hierarchical", AgglomerativeClustering(n_clusters=K_FIXED, linkage="average"), K_FIXED, "average linkage"),
        ("spectral_k6", "spectral", SpectralClustering(n_clusters=K_FIXED, affinity="nearest_neighbors", random_state=RANDOM_STATE, n_neighbors=15), K_FIXED, "graph / manifold"),
    ]
    for run_id, method, model, k, notes in classical:
        t0 = time.perf_counter()
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - t0
        metrics = _safe_metrics(X, labels)
        size = _label_stats(labels)
        rows.append(_row(run_id, "classical", method, k, {}, metrics, size, elapsed, notes))
        print(f"  {run_id:22s} sil={metrics['silhouette']:.3f} DB={metrics['db_index']:.3f}")

    # GMM BIC sweep for k
    print("  GMM BIC sweep k=3..8 ...")
    bic_rows = []
    for k in range(3, 9):
        t0 = time.perf_counter()
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE, n_init=3)
        labels = gmm.fit_predict(X)
        elapsed = time.perf_counter() - t0
        metrics = _safe_metrics(X, labels)
        size = _label_stats(labels)
        bic = float(gmm.bic(X))
        bic_rows.append((k, bic, metrics["silhouette"]))
        rows.append(
            _row(
                f"gmm_bic_k{k}",
                "classical",
                "gmm_bic",
                k,
                {"bic": round(bic, 1)},
                metrics,
                size,
                elapsed,
                "BIC-scored GMM",
            )
        )
    best_bic_k = min(bic_rows, key=lambda t: t[1])[0]
    print(f"  best BIC k={best_bic_k}")

    # Density-based
    print("\n=== 2. Density-based ===")
    # Heuristic eps from k-distance
    nn = NearestNeighbors(n_neighbors=10).fit(X)
    dists, _ = nn.kneighbors(X)
    eps_guess = float(np.percentile(dists[:, -1], 70))
    for run_id, model, notes in [
        ("hdbscan_default", HDBSCAN(min_cluster_size=40, min_samples=10), "noise allowed"),
        ("hdbscan_small", HDBSCAN(min_cluster_size=20, min_samples=5), "finer density modes"),
        ("dbscan_eps_auto", DBSCAN(eps=eps_guess, min_samples=10), f"eps≈{eps_guess:.2f} from knn"),
    ]:
        t0 = time.perf_counter()
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - t0
        metrics = _safe_metrics(X, labels)
        size = _label_stats(labels)
        rows.append(
            _row(run_id, "density", run_id.split("_")[0], size["n_clusters"], {}, metrics, size, elapsed, notes)
        )
        print(
            f"  {run_id:22s} k={size['n_clusters']} noise={size['noise_frac']:.2%} "
            f"sil={metrics['silhouette']:.3f}"
        )

    # PCA linear baseline
    print("\n=== 3. PCA + K-Means (linear representation baseline) ===")
    for n_comp in (2, 4, 8):
        t0 = time.perf_counter()
        z = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(X)
        labels = KMeans(n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE).fit_predict(z)
        elapsed = time.perf_counter() - t0
        metrics = _safe_metrics(z, labels)
        size = _label_stats(labels)
        rows.append(
            _row(
                f"pca{n_comp}_kmeans_k6",
                "linear_rep",
                "pca_kmeans",
                K_FIXED,
                {"latent_dim": n_comp},
                metrics,
                size,
                elapsed,
                "linear baseline vs AE",
            )
        )
        print(f"  pca{n_comp}+kmeans sil={metrics['silhouette']:.3f}")

    # ── 4. Autoencoder exploration ────────────────────────────────
    if not TORCH_OK:
        print("\n=== 4. AE skipped (no torch) ===")
    else:
        print("\n=== 4. Autoencoder variants ===")
        ae_configs = [
            # name, latent, hidden, noise, epochs, notes
            ("ae_lat2_h32", 2, 32, 0.0, ae_epochs, "very compressed"),
            ("ae_lat4_h32", 4, 32, 0.0, ae_epochs, "current champion latent"),
            ("ae_lat8_h32", 8, 32, 0.0, ae_epochs, "wider latent"),
            ("ae_lat16_h32", 16, 32, 0.0, ae_epochs, "near-identity risk"),
            ("ae_lat4_h64", 4, 64, 0.0, ae_epochs, "deeper capacity"),
            ("dae_lat4_n01", 4, 32, 0.10, ae_epochs, "denoising noise=0.10"),
            ("dae_lat4_n02", 4, 32, 0.20, ae_epochs, "denoising noise=0.20"),
            ("ae_lat4_long", 4, 32, 0.0, max(ae_epochs, 80), "longer train"),
        ]

        for name, lat, hidden, noise, epochs, notes in ae_configs:
            print(f"  training {name} ...")
            t0 = time.perf_counter()
            ae = train_autoencoder(
                X,
                latent_dim=lat,
                hidden=hidden,
                noise_std=noise,
                epochs=epochs,
                random_state=RANDOM_STATE,
            )
            train_s = time.perf_counter() - t0
            ae_loss_curves[name] = ae.losses
            final_loss = ae.losses[-1]

            # Heads on this latent
            heads = [
                ("kmeans", lambda Z: KMeans(n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE).fit_predict(Z)),
                ("gmm", lambda Z: GaussianMixture(n_components=K_FIXED, covariance_type="full", random_state=RANDOM_STATE, n_init=3).fit_predict(Z)),
                ("hdbscan", lambda Z: HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(Z)),
                ("minibatch", lambda Z: MiniBatchKMeans(n_clusters=K_FIXED, random_state=RANDOM_STATE, batch_size=256, n_init=10).fit_predict(Z)),
            ]
            for head_name, head_fn in heads:
                t1 = time.perf_counter()
                labels = head_fn(ae.z)
                elapsed = train_s + (time.perf_counter() - t1)
                metrics = _safe_metrics(ae.z, labels)
                size = _label_stats(labels)
                run_id = f"{name}_{head_name}_k{K_FIXED}" if head_name != "hdbscan" else f"{name}_{head_name}"
                rows.append(
                    _row(
                        run_id,
                        "autoencoder",
                        f"ae_{head_name}",
                        size["n_clusters"] if head_name == "hdbscan" else K_FIXED,
                        {
                            "latent_dim": lat,
                            "hidden": hidden,
                            "noise_std": noise,
                            "epochs": epochs,
                            "recon_loss": round(final_loss, 5),
                        },
                        metrics,
                        size,
                        elapsed,
                        notes,
                    )
                )
            best_head = max(
                [r for r in rows if r["run_id"].startswith(name + "_") and r["method"].startswith("ae_")],
                key=lambda r: (-999 if np.isnan(r["silhouette"]) else r["silhouette"]),
            )
            print(
                f"    recon={final_loss:.4f} best_head={best_head['method']} "
                f"sil={best_head['silhouette']:.3f}"
            )

        viz.plot_ae_loss(
            {k: v for k, v in ae_loss_curves.items() if k.startswith("ae_lat") or k.startswith("dae")},
            fig_dir / "ae_recon_curves.png",
        )

    board = pd.DataFrame(rows)
    # Rank: prefer higher sil, then lower db, ignore nan sil at bottom
    board["_rank_sil"] = board["silhouette"].fillna(-1)
    board = board.sort_values(["_rank_sil", "db_index"], ascending=[False, True]).drop(columns=["_rank_sil"])
    board = board.reset_index(drop=True)
    board_path = out_dir / "comparison_board_explore.csv"
    board.to_csv(board_path, index=False)

    # Top charts
    top = board.dropna(subset=["silhouette"]).head(12)
    if len(top):
        fig_vals = {
            str(r.run_id)[:28]: float(r.silhouette) for r in top.itertuples()
        }
        viz.plot_stability_bars(fig_vals, fig_dir / "top12_silhouette.png")

    # Family summary
    fam = (
        board.dropna(subset=["silhouette"])
        .groupby("family")["silhouette"]
        .agg(["count", "max", "mean"])
        .round(4)
        .sort_values("max", ascending=False)
    )
    fam.to_csv(out_dir / "family_summary.csv")

    summary = {
        "n_runs": int(len(board)),
        "best_overall": board.iloc[0].to_dict() if len(board) else {},
        "best_by_family": {
            fam_name: board[board["family"] == fam_name].iloc[0].to_dict()
            for fam_name in board["family"].unique()
            if len(board[board["family"] == fam_name].dropna(subset=["silhouette"]))
        },
        "champion_reference": {
            "run_id": "c_ae_lat4_k6",
            "note": "Registered champion remains AE lat4 + KMeans unless you promote a new winner",
        },
        "gmm_best_bic_k": best_bic_k,
        "outputs": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== TOP 10 by silhouette ===")
    cols = ["run_id", "family", "method", "k", "silhouette", "db_index", "ch_index", "notes"]
    print(board[cols].head(10).to_string(index=False))
    print(f"\nSaved board -> {board_path}")
    return board


if __name__ == "__main__":
    run_explore()
