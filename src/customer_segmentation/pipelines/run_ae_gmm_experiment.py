"""Compare AE+K-Means vs AE+GMM on Berka (winner dataset), same latent space."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from customer_segmentation.features.build_berka_features import build_and_save_berka_features
from customer_segmentation.models.autoencoder import TORCH_OK, train_ae_cluster, train_autoencoder
from customer_segmentation.models import viz
from customer_segmentation.models.stability import multi_seed_ari
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

RANDOM_STATE = 42
LATENT_DIM = 4
N_CLUSTERS = 6
EPOCHS = 50


def main(
    *,
    berka_dir: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
    epochs: int = EPOCHS,
) -> dict:
    if not TORCH_OK:
        raise SystemExit("PyTorch required for AE+GMM experiment.")

    out_dir = Path(artifact_root) / "experiments" / "ae_gmm_berka"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Features (Berka) ===")
    bundle = build_and_save_berka_features(berka_dir=berka_dir)
    feat_cols = bundle.feature_columns
    X = bundle.model[feat_cols].fillna(0).values.astype(np.float64)
    print(f"  matrix={X.shape}")

    print(f"=== Train AE once (latent={LATENT_DIM}, epochs={epochs}) ===")
    ae = train_autoencoder(X, latent_dim=LATENT_DIM, epochs=epochs, random_state=RANDOM_STATE)
    print(f"  final recon loss={ae.losses[-1]:.5f}")

    print("=== Cluster latent: K-Means vs GMM ===")
    km = train_ae_cluster(
        X, latent_dim=LATENT_DIM, n_clusters=N_CLUSTERS, method="kmeans",
        trained=ae, random_state=RANDOM_STATE,
    )
    gmm = train_ae_cluster(
        X, latent_dim=LATENT_DIM, n_clusters=N_CLUSTERS, method="gmm",
        trained=ae, random_state=RANDOM_STATE,
    )

    ari_agree = float(adjusted_rand_score(km.labels, gmm.labels))

    def _km_fp(Z: np.ndarray, seed: int) -> np.ndarray:
        return KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed).fit_predict(Z)

    def _gmm_fp(Z: np.ndarray, seed: int) -> np.ndarray:
        return GaussianMixture(
            n_components=N_CLUSTERS, covariance_type="full",
            random_state=seed, n_init=3,
        ).fit_predict(Z)

    stab_km = multi_seed_ari(_km_fp, ae.z, n_seeds=10, base_seed=0)
    stab_gmm = multi_seed_ari(_gmm_fp, ae.z, n_seeds=10, base_seed=0)

    rows = [
        {
            "run_id": f"c_ae_lat{LATENT_DIM}_kmeans_k{N_CLUSTERS}",
            "method": "ae_kmeans",
            "latent_dim": LATENT_DIM,
            "k": N_CLUSTERS,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in km.metrics.items()},
            "stability_ari": round(stab_km["mean_ari"], 4),
            "low_confidence_frac": round(float((km.confidence < 0.55).mean()), 4),
            "size_min_frac": round(km.size["min_frac"], 4),
            "size_max_frac": round(km.size["max_frac"], 4),
            "size_ok": km.size["ok"],
        },
        {
            "run_id": f"c_ae_lat{LATENT_DIM}_gmm_k{N_CLUSTERS}",
            "method": "ae_gmm",
            "latent_dim": LATENT_DIM,
            "k": N_CLUSTERS,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in gmm.metrics.items()},
            "stability_ari": round(stab_gmm["mean_ari"], 4),
            "low_confidence_frac": round(float((gmm.confidence < 0.55).mean()), 4),
            "size_min_frac": round(gmm.size["min_frac"], 4),
            "size_max_frac": round(gmm.size["max_frac"], 4),
            "size_ok": gmm.size["ok"],
            "bic": round(float(gmm.clusterer.bic(ae.z)), 1),
            "aic": round(float(gmm.clusterer.aic(ae.z)), 1),
        },
    ]
    board = pd.DataFrame(rows)
    board_path = out_dir / "ae_kmeans_vs_gmm.csv"
    board.to_csv(board_path, index=False)

    # Profiles for GMM result
    profiles_gmm = (
        bundle.interpretable.assign(segment_id=gmm.labels)
        .groupby("segment_id")[feat_cols]
        .mean()
        .round(2)
    )
    profiles_km = (
        bundle.interpretable.assign(segment_id=km.labels)
        .groupby("segment_id")[feat_cols]
        .mean()
        .round(2)
    )
    profiles_gmm.to_csv(out_dir / "profiles_ae_gmm.csv")
    profiles_km.to_csv(out_dir / "profiles_ae_kmeans.csv")

    assignments = pd.DataFrame({
        "account_id": bundle.interpretable["account_id"].values,
        "segment_ae_kmeans": km.labels,
        "conf_ae_kmeans": np.round(km.confidence, 4),
        "segment_ae_gmm": gmm.labels,
        "conf_ae_gmm": np.round(gmm.confidence, 4),
    })
    assignments.to_csv(out_dir / "assignments_both.csv", index=False)

    viz.plot_ae_loss({LATENT_DIM: ae.losses}, fig_dir / "ae_recon_loss.png")
    viz.plot_size_bars(km.labels, fig_dir / "sizes_ae_kmeans.png", title="AE + K-Means sizes")
    viz.plot_size_bars(gmm.labels, fig_dir / "sizes_ae_gmm.png", title="AE + GMM sizes")
    viz.plot_pca_segments(ae.z, km.labels, fig_dir / "pca_ae_kmeans.png", title="AE+KMeans latent PCA")
    viz.plot_pca_segments(ae.z, gmm.labels, fig_dir / "pca_ae_gmm.png", title="AE+GMM latent PCA")
    viz.plot_centroid_heatmap(profiles_gmm, fig_dir / "heatmap_ae_gmm.png", title="AE+GMM profiles")
    viz.plot_stability_bars(
        {"ae_kmeans": stab_km["mean_ari"], "ae_gmm": stab_gmm["mean_ari"]},
        fig_dir / "stability.png",
    )

    winner_metric = "ae_gmm" if gmm.metrics["silhouette"] > km.metrics["silhouette"] else "ae_kmeans"
    summary = {
        "latent_dim": LATENT_DIM,
        "k": N_CLUSTERS,
        "same_encoder": True,
        "ari_kmeans_vs_gmm_labels": round(ari_agree, 4),
        "ae_kmeans": {**km.metrics, "stability_ari": stab_km["mean_ari"], "sizes": km.size["counts"]},
        "ae_gmm": {
            **gmm.metrics,
            "stability_ari": stab_gmm["mean_ari"],
            "sizes": gmm.size["counts"],
            "bic": float(gmm.clusterer.bic(ae.z)),
            "aic": float(gmm.clusterer.aic(ae.z)),
            "low_confidence_frac": float((gmm.confidence < 0.55).mean()),
        },
        "higher_silhouette": winner_metric,
        "note": (
            "Same AE latent space; only the clustering head differs. "
            "GMM adds soft membership (confidence = max component probability)."
        ),
        "outputs": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== RESULTS (same AE latent) ===")
    print(board.to_string(index=False))
    print(f"\nLabel agreement ARI (KMeans vs GMM): {ari_agree:.3f}")
    print(f"Higher silhouette head: {winner_metric}")
    print(f"Saved -> {out_dir}")
    return summary


if __name__ == "__main__":
    main()
