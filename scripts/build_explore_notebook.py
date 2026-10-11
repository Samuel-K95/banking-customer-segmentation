"""Generate notebooks/09_Berka_Exploration_AE_and_Clustering.ipynb"""

from __future__ import annotations

import json
from pathlib import Path


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = [
    md(
        """# 09 — Berka Exploration: AE Variants + Extra Clustering Models

**Dataset:** Track C (Berka) — project champion dataset  
**Goal:** Explore more autoencoder designs and alternative clustering heads without changing the registered champion until we decide to promote one.

**Registered champion (reference):** `c_ae_lat4_k6` = AE(latent=4) + K-Means k=6

This notebook mirrors `customer_segmentation.pipelines.run_berka_explore` with richer plots so you can inspect every family.

Related: `reports/berka_explore_findings.md`, `artifacts/experiments/berka_explore/`
"""
    ),
    md("## 0. Setup"),
    code(
        """import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import (
    AgglomerativeClustering, Birch, DBSCAN, HDBSCAN,
    KMeans, MiniBatchKMeans, SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.features.build_berka_features import build_and_save_berka_features
from customer_segmentation.models.autoencoder import TORCH_OK, train_autoencoder
from customer_segmentation.models.metrics import evaluate_clustering, size_sanity

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
RANDOM_STATE = 42
K_FIXED = 6
np.random.seed(RANDOM_STATE)

OUT = ROOT / "artifacts" / "experiments" / "berka_explore"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

print("TORCH_OK =", TORCH_OK)
print("Output  =", OUT)
"""
    ),
    md(
        """## 1. Load / build Berka features

RFM + balance + `has_loan` / `has_card` (scaled model matrix).
"""
    ),
    code(
        """bundle = build_and_save_berka_features(
    berka_dir=ROOT / "data" / "raw" / "berka-dataset",
    processed_dir=ROOT / "data" / "processed",
)
feat_cols = bundle.feature_columns
X = bundle.model[feat_cols].fillna(0).values.astype(np.float64)
interp = bundle.interpretable

print("Shape:", X.shape)
print("Features:", feat_cols)
display(interp[feat_cols].describe().T.round(2))
"""
    ),
    md("## 2. Helper: score a labeling"),
    code(
        """rows = []

def label_stats(labels):
    labels = np.asarray(labels)
    noise = float((labels < 0).mean()) if np.any(labels < 0) else 0.0
    valid = labels[labels >= 0]
    if len(valid) < 10 or len(np.unique(valid)) < 2:
        return {
            "n_clusters": int(len(np.unique(valid))),
            "noise_frac": noise,
            "min_frac": np.nan,
            "max_frac": np.nan,
            "ok": False,
        }
    s = size_sanity(valid)
    return {
        "n_clusters": int(len(np.unique(valid))),
        "noise_frac": noise,
        "min_frac": s["min_frac"],
        "max_frac": s["max_frac"],
        "ok": s["ok"],
        "counts": s["counts"],
    }

def score(run_id, family, method, labels, space=None, k=None, **extra):
    space = X if space is None else space
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        m = {"silhouette": np.nan, "db_index": np.nan, "ch_index": np.nan}
    else:
        m = evaluate_clustering(space[mask], labels[mask], random_state=RANDOM_STATE)
    st = label_stats(labels)
    row = dict(
        run_id=run_id,
        family=family,
        method=method,
        k=k if k is not None else st["n_clusters"],
        silhouette=round(m["silhouette"], 4),
        db_index=round(m["db_index"], 4),
        ch_index=round(m["ch_index"], 1),
        n_clusters=st["n_clusters"],
        noise_frac=round(st["noise_frac"], 4),
        min_frac=None if st["min_frac"] != st["min_frac"] else round(st["min_frac"], 4),
        max_frac=None if st["max_frac"] != st["max_frac"] else round(st["max_frac"], 4),
        size_ok=st["ok"],
        **extra,
    )
    rows.append(row)
    print(
        f"{run_id:28s} sil={row['silhouette']!s:>7}  "
        f"DB={row['db_index']!s:>7}  k={row['n_clusters']}  "
        f"noise={row['noise_frac']:.2%}"
    )
    return labels, m, st

def show_sizes(labels, title):
    vc = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(vc.index.astype(str), vc.values, color="#1a3a5c")
    ax.set_title(title)
    ax.set_xlabel("segment")
    ax.set_ylabel("count")
    for i, v in zip(vc.index, vc.values):
        ax.text(str(i), v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.show()

def pca_scatter(space, labels, title):
    xy = PCA(2, random_state=RANDOM_STATE).fit_transform(space)
    fig, ax = plt.subplots(figsize=(6, 5))
    for seg in np.unique(labels):
        m = labels == seg
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.55, label=f"{seg}")
    ax.legend(markerscale=2, fontsize=8, title="seg")
    ax.set_title(title + " (PCA exploratory)")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 3. Classical / simple clustering on raw features"),
    code(
        """classical = {
    "kmeans_k6": KMeans(n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE),
    "minibatch_kmeans_k6": MiniBatchKMeans(
        n_clusters=K_FIXED, random_state=RANDOM_STATE, batch_size=256, n_init=10
    ),
    "gmm_k6": GaussianMixture(
        n_components=K_FIXED, covariance_type="full", random_state=RANDOM_STATE, n_init=5
    ),
    "birch_k6": Birch(n_clusters=K_FIXED),
    "hier_ward_k6": AgglomerativeClustering(n_clusters=K_FIXED, linkage="ward"),
    "hier_average_k6": AgglomerativeClustering(n_clusters=K_FIXED, linkage="average"),
    "spectral_k6": SpectralClustering(
        n_clusters=K_FIXED,
        affinity="nearest_neighbors",
        random_state=RANDOM_STATE,
        n_neighbors=15,
    ),
}

classic_labels = {}
for run_id, model in classical.items():
    lbl = model.fit_predict(X)
    classic_labels[run_id] = lbl
    score(run_id, "classical", run_id.split("_")[0], lbl, k=K_FIXED)
"""
    ),
    md(
        """### Watch out: high silhouette can still be a bad segmentation

`hier_average` often spikes silhouette while stuffing almost everyone into one cluster.
"""
    ),
    code(
        """show_sizes(classic_labels["hier_average_k6"], "Hierarchical average — check imbalance")
show_sizes(classic_labels["kmeans_k6"], "K-Means k=6 — reference baseline")
pca_scatter(X, classic_labels["kmeans_k6"], "K-Means on raw features")
"""
    ),
    md("### GMM BIC sweep (let the data suggest k)"),
    code(
        """bic_scores, sil_scores, ks = [], [], list(range(3, 9))
for k in ks:
    gmm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=RANDOM_STATE, n_init=3
    )
    lbl = gmm.fit_predict(X)
    bic_scores.append(gmm.bic(X))
    m = evaluate_clustering(X, lbl, random_state=RANDOM_STATE)
    sil_scores.append(m["silhouette"])
    score(
        f"gmm_bic_k{k}",
        "classical",
        "gmm_bic",
        lbl,
        k=k,
        bic=round(float(gmm.bic(X)), 1),
    )

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(ks, bic_scores, "o-")
ax[0].set_title("GMM BIC (lower better)")
ax[0].set_xlabel("k")
ax[1].plot(ks, sil_scores, "g-o")
ax[1].set_title("GMM silhouette")
ax[1].set_xlabel("k")
plt.tight_layout()
plt.show()
print("Best BIC k =", ks[int(np.argmin(bic_scores))])
"""
    ),
    md(
        """## 4. Density-based (HDBSCAN / DBSCAN)

Noise points get label `-1` — useful as a "human review" bucket.
"""
    ),
    code(
        """nn = NearestNeighbors(n_neighbors=10).fit(X)
dists, _ = nn.kneighbors(X)
eps_guess = float(np.percentile(dists[:, -1], 70))
print(f"DBSCAN eps heuristic (70th pct of 10-NN distance): {eps_guess:.3f}")

density = {
    "hdbscan_default": HDBSCAN(min_cluster_size=40, min_samples=10),
    "hdbscan_small": HDBSCAN(min_cluster_size=20, min_samples=5),
    "dbscan_eps_auto": DBSCAN(eps=eps_guess, min_samples=10),
}
for run_id, model in density.items():
    lbl = model.fit_predict(X)
    score(run_id, "density", run_id.split("_")[0], lbl)
    show_sizes(lbl, run_id)
"""
    ),
    md(
        """## 5. PCA + K-Means (linear representation baseline)

If PCA matches AE quality, the neural net is not buying much.
"""
    ),
    code(
        """for n in (2, 4, 8):
    z = PCA(n_components=n, random_state=RANDOM_STATE).fit_transform(X)
    lbl = KMeans(n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE).fit_predict(z)
    score(
        f"pca{n}_kmeans_k6",
        "linear_rep",
        "pca_kmeans",
        lbl,
        space=z,
        k=K_FIXED,
        latent_dim=n,
    )
    if n == 4:
        pca_scatter(z, lbl, "PCA(4) + K-Means")
"""
    ),
    md(
        """## 6. Autoencoder exploration

Train several AEs, then cluster each latent with **K-Means, GMM, MiniBatch, HDBSCAN**.
"""
    ),
    code(
        """if not TORCH_OK:
    raise ImportError("PyTorch required for AE section")

ae_configs = [
    # name, latent, hidden, noise_std, epochs, note
    ("ae_lat2_h32", 2, 32, 0.0, 40, "very compressed"),
    ("ae_lat4_h32", 4, 32, 0.0, 40, "champion-like latent"),
    ("ae_lat8_h32", 8, 32, 0.0, 40, "wider latent"),
    ("ae_lat16_h32", 16, 32, 0.0, 40, "near-identity risk"),
    ("ae_lat4_h64", 4, 64, 0.0, 40, "more capacity"),
    ("dae_lat4_n01", 4, 32, 0.10, 40, "denoising 0.10"),
    ("dae_lat4_n02", 4, 32, 0.20, 40, "denoising 0.20"),
    ("ae_lat4_long", 4, 32, 0.0, 80, "longer train"),
]

ae_cache = {}
loss_curves = {}

for name, lat, hidden, noise, epochs, note in ae_configs:
    print(f"\\n=== Training {name} ({note}) ===")
    ae = train_autoencoder(
        X,
        latent_dim=lat,
        hidden=hidden,
        noise_std=noise,
        epochs=epochs,
        random_state=RANDOM_STATE,
    )
    ae_cache[name] = ae
    loss_curves[name] = ae.losses
    print(f"  final recon loss = {ae.losses[-1]:.5f}")

    heads = {
        "kmeans": KMeans(
            n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE
        ).fit_predict(ae.z),
        "gmm": GaussianMixture(
            n_components=K_FIXED,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=3,
        ).fit_predict(ae.z),
        "minibatch": MiniBatchKMeans(
            n_clusters=K_FIXED,
            random_state=RANDOM_STATE,
            batch_size=256,
            n_init=10,
        ).fit_predict(ae.z),
        "hdbscan": HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(ae.z),
    }
    for head, lbl in heads.items():
        rid = f"{name}_{head}_k{K_FIXED}" if head != "hdbscan" else f"{name}_{head}"
        score(
            rid,
            "autoencoder",
            f"ae_{head}",
            lbl,
            space=ae.z,
            latent_dim=lat,
            hidden=hidden,
            noise_std=noise,
            epochs=epochs,
            recon_loss=round(ae.losses[-1], 5),
            notes=note,
        )
"""
    ),
    md("### AE reconstruction loss curves"),
    code(
        """fig, ax = plt.subplots(figsize=(10, 4.5))
for name, losses in loss_curves.items():
    ax.plot(losses, label=name, alpha=0.85)
ax.set_xlabel("epoch")
ax.set_ylabel("MSE recon loss")
ax.set_title("Autoencoder training curves")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
fig.savefig(OUT / "figures" / "ae_recon_curves_nb.png", dpi=140, bbox_inches="tight")
"""
    ),
    md("### Inspect champion-like AE (lat=4) vs stronger encoder (hidden=64)"),
    code(
        """for name in ["ae_lat4_h32", "ae_lat4_h64", "ae_lat4_long", "dae_lat4_n01"]:
    z = ae_cache[name].z
    lbl = KMeans(n_clusters=K_FIXED, n_init=20, random_state=RANDOM_STATE).fit_predict(z)
    print("\\n", name)
    show_sizes(lbl, f"{name} + K-Means")
    pca_scatter(z, lbl, name + " + K-Means")
    prof = interp.assign(segment_id=lbl).groupby("segment_id")[feat_cols].mean().round(2)
    display(prof)
"""
    ),
    md("### AE latent + HDBSCAN (often high silhouette — check noise + k)"),
    code(
        """name = "ae_lat4_h64"
z = ae_cache[name].z
lbl = HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(z)
score("view_ae_h64_hdbscan", "autoencoder", "ae_hdbscan", lbl, space=z)
show_sizes(lbl, "AE(h64) + HDBSCAN")
pca_scatter(z, lbl, "AE(h64) + HDBSCAN")
print("Noise fraction:", (lbl < 0).mean())
prof = interp.assign(segment_id=lbl).groupby("segment_id")[feat_cols].mean().round(2)
display(prof)
"""
    ),
    md("## 7. Comparison board"),
    code(
        """board = pd.DataFrame(rows)
board = board.sort_values("silhouette", ascending=False, na_position="last").reset_index(drop=True)
board_path = OUT / "comparison_board_explore_from_notebook.csv"
board.to_csv(board_path, index=False)

print("TOP 15 by silhouette")
display(board.head(15))

print("\\nBest AE + K-Means only")
display(board[board["method"] == "ae_kmeans"].head(10))

print("\\nFamily max silhouette")
display(
    board.dropna(subset=["silhouette"])
    .groupby("family")["silhouette"]
    .agg(["count", "max", "mean"])
    .round(4)
    .sort_values("max", ascending=False)
)
"""
    ),
    md("### Top silhouette bar chart (read sizes before trusting the leader)"),
    code(
        """top = board.dropna(subset=["silhouette"]).head(12)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top["run_id"].str[:32][::-1], top["silhouette"][::-1], color="#2E7D32")
ax.set_xlabel("silhouette")
ax.set_title("Top 12 runs (higher ≠ automatically better)")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """## 8. Takeaways

1. **`hier_average`** can top the board and still be useless (one giant cluster).
2. **AE latent ≈ 4** remains the sweet spot; 8/16 overfit/identity.
3. **Wider / longer AE** (`h64`, `long`) helps especially with HDBSCAN/GMM heads.
4. **Denoising AE** is competitive, not clearly better here.
5. **Registered champion AE(lat=4)+K-Means** stays the safe CRM choice (fixed k, stable, interpretable).
6. **v2 candidates:** AE(h64 or long)+K-Means, or AE+HDBSCAN if you accept noise + variable k.

Full write-up: `reports/berka_explore_findings.md`
"""
    ),
    code(
        """print("Notebook complete.")
print("Board saved to", board_path)
print("Reference champion: c_ae_lat4_k6 (AE lat4 + KMeans k=6)")
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out = Path("notebooks/09_Berka_Exploration_AE_and_Clustering.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    pass
