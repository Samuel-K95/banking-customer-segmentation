"""Generate notebooks/10_Champion_Berka_Decision_Layer.ipynb"""

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
        """# 10 — Champion Segmentation → Decision Layer (Berka)

**Champion:** `c_ae_lat4_k6` = Autoencoder (latent=4) + K-Means **k=6** on Track C (Berka)

This notebook is the single place to see the full story:

1. **Why Berka + AE** (selection vs max silhouette)  
2. **EDA** on the behavioral feature set  
3. **Champion model** (load registered artifacts, re-score)  
4. **Cluster profiles** (names, personas, lift)  
5. **Decision layer** — explainability, fairness notes, banking plays, decision map  

Aligned to `03-segmentation-layer.md` and `04-analysis-and-decision-layer.md`.

**Artifacts:** `artifacts/segmentation/champion_v1/` · **Reports:** `champion_recommendation.md`, `cluster_profiles.md`
"""
    ),
    md(
        """## 0. How this maps to the decision layer

| Decision-layer block (`04`) | Where in this notebook |
|-----------------------------|------------------------|
| Segment intelligence / profiles | §6–7 |
| Explainability | §8 |
| Fairness diagnostics | §9 |
| Decision maps (≥3 applications) | §10 |
| Score → profile → play workflow | §11 |

Segmentation finds groups. This notebook makes them **bank-usable**.
"""
    ),
    md("## 1. Setup"),
    code(
        """import json
import warnings
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import silhouette_score, davies_bouldin_score

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.features.build_berka_features import (
    MODEL_COLUMNS,
    build_and_save_berka_features,
)
from customer_segmentation.data.berka_dataset_loader import resolve_berka_dir
from customer_segmentation.models.tabular_ae import TORCH_OK, distance_confidence
from customer_segmentation.models.champion import ChampionModel
from customer_segmentation.models.metrics import evaluate_clustering, size_sanity
from customer_segmentation.models.segment_intel import PERSONAS, build_segment_catalog

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CHAMP = ROOT / "artifacts" / "segmentation" / "champion_v1"
FIG = CHAMP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT_NB = ROOT / "artifacts" / "experiments" / "champion_notebook"
OUT_NB.mkdir(parents=True, exist_ok=True)

RUN_ID = "c_ae_lat4_k6"
print("ROOT   =", ROOT)
print("CHAMP  =", CHAMP)
print("TORCH  =", TORCH_OK)
print("exists =", CHAMP.exists())
"""
    ),
    md(
        """## 2. Why this champion? (selection decision)

> Highest silhouette is **recorded**, not used as the selection rule.

| Track | Data reality | Metric note | Role |
|-------|--------------|-------------|------|
| A Indian Bank | Timing-heavy FE; weak recency | Lower sil | Required starting track |
| B Santander | Products + demos, **no transactions** | Highest sil | Metric / representation comparison |
| **C Berka** | **Transactions + balance + loan/card** | Strong DB, usable sil | **Selected segmentation** |

Berka is the only track that answers: *what behaviors distinguish groups, and what should the bank do?*
"""
    ),
    code(
        """# Cross-track snapshot (from reports — for context in one place)
cross = pd.DataFrame(
    [
        {
            "track": "C Berka (selected)",
            "model": "AE+KMeans lat4 k6",
            "silhouette": 0.417,
            "db_index": 0.731,
            "stability_ari": 0.864,
            "has_transactions": True,
            "has_products": True,
        },
        {
            "track": "B Santander",
            "model": "AE+KMeans lat8 k3",
            "silhouette": 0.512,
            "db_index": 0.891,
            "stability_ari": 1.000,
            "has_transactions": False,
            "has_products": True,
        },
        {
            "track": "A Indian Bank",
            "model": "AE+KMeans lat8 k6",
            "silhouette": 0.332,
            "db_index": 1.099,
            "stability_ari": np.nan,
            "has_transactions": True,
            "has_products": False,
        },
    ]
)
display(cross)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
colors = ["#1b7a6e", "#888888", "#bbbbbb"]
ax[0].barh(cross["track"], cross["silhouette"], color=colors)
ax[0].set_xlabel("Silhouette (higher better)")
ax[0].set_title("Metric leader ≠ selected")
ax[0].invert_yaxis()
ax[1].barh(cross["track"], cross["db_index"], color=colors)
ax[1].set_xlabel("Davies–Bouldin (lower better)")
ax[1].set_title("Berka wins compactness/separation")
ax[1].invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_NB / "cross_track_metrics.png", dpi=120, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        """## 3. EDA — Berka behavioral feature set

**Model features (8):** RFM-style activity + balance + product flags.

| Feature | Meaning |
|---------|---------|
| `Transaction_Count` | Lifetime activity volume |
| `Account_Balance` | Latest balance (CZK) |
| `Total_Credit` / `Total_Withdrawal` | Flow intensity |
| `Recency_Days` | Days since last txn |
| `Account_Age_Days` | Tenure |
| `has_loan` / `has_card` | Product holding |

These are the axes a bank can **name** and **act on**.
"""
    ),
    code(
        """bundle = build_and_save_berka_features(
    berka_dir=ROOT / "data" / "raw" / "berka-dataset",
    processed_dir=ROOT / "data" / "processed",
)
feat_cols = list(bundle.feature_columns)
interp = bundle.interpretable.copy()
X = bundle.model[feat_cols].fillna(0).values.astype(np.float64)
id_col = bundle.id_col

print("Accounts:", len(interp))
print("Features:", feat_cols)
display(interp[feat_cols].describe().T.round(2))
"""
    ),
    code(
        """# Univariate distributions (original scale)
plot_cols = [
    "Transaction_Count",
    "Account_Balance",
    "Total_Credit",
    "Total_Withdrawal",
    "Recency_Days",
    "Account_Age_Days",
]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes.ravel(), plot_cols):
    s = interp[col].clip(upper=interp[col].quantile(0.99))
    ax.hist(s, bins=40, color="#1b7a6e", alpha=0.85, edgecolor="white")
    ax.set_title(col, fontsize=10)
plt.suptitle("Feature distributions (99th-pct clipped for readability)", y=1.02)
plt.tight_layout()
plt.savefig(OUT_NB / "eda_distributions.png", dpi=120, bbox_inches="tight")
plt.show()

# Product penetration
prod = pd.Series(
    {
        "has_loan": interp["has_loan"].mean(),
        "has_card": interp["has_card"].mean(),
        "loan_and_card": ((interp["has_loan"] == 1) & (interp["has_card"] == 1)).mean(),
        "neither": ((interp["has_loan"] == 0) & (interp["has_card"] == 0)).mean(),
    }
)
fig, ax = plt.subplots(figsize=(6, 3.5))
prod.plot(kind="barh", ax=ax, color="#c45c26")
ax.set_xlabel("Share of accounts")
ax.set_title("Product penetration (population)")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(OUT_NB / "eda_product_penetration.png", dpi=120, bbox_inches="tight")
plt.show()
display(prod.round(3).to_frame("rate"))
"""
    ),
    code(
        """# Correlations — why AE compression helps
corr = interp[feat_cols].corr()
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
ax.set_title("Feature correlation (scaled space ≈ same ranks)")
plt.tight_layout()
plt.savefig(OUT_NB / "eda_correlation.png", dpi=120, bbox_inches="tight")
plt.show()

print(
    "Note: Credit/Withdrawal/Balance co-move; AE latent=4 targets ~4 independent axes "
    "(activity, financial depth, spend/tenure, product)."
)
"""
    ),
    md(
        """## 4. Load champion via modular `ChampionModel`

Production path: `ChampionModel.load(champion_v1)` — encoder + K-Means + scaler + segment catalog.
Notebooks should not re-implement scoring; they call this module.
"""
    ),
    code(
        """champ = ChampionModel.load(CHAMP)
display(pd.Series(champ.summary()).to_frame("value"))

board = pd.read_csv(CHAMP / "comparison_board.csv")
assign = pd.read_csv(CHAMP / "assignments.csv")
metrics_art = champ.metrics
manifest = champ.manifest

# Score through the modular API (same scaler + encoder + kmeans)
scored = champ.score_interpretable(interp)
df = interp.merge(scored, on=id_col, how="inner")
labels = df["segment_id"].values.astype(int)
print("Joined rows:", len(df), "| segments:", sorted(df["segment_id"].unique()))
display(scored.head())
"""
    ),
    code(
        """# Latent embeddings from ChampionModel.encode
z = champ.encode(X)
recon_err = None
if champ.encoder is not None:
    import torch

    champ.encoder.eval()
    with torch.no_grad():
        xhat, _ = champ.encoder(torch.tensor(X, dtype=torch.float32))
        recon_err = ((xhat.numpy() - X) ** 2).mean(axis=1)
    print(f"Encoder latent_dim={champ.latent_dim}, z={z.shape}")
else:
    print("No encoder — using feature space")

m_live = evaluate_clustering(z, labels) if z.shape[1] >= 2 else evaluate_clustering(X, labels)
sz = size_sanity(labels)
print("Live metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_live.items()})
print("Size sanity:", sz)
"""
    ),
    md("## 5. Bake-off board (Track C) — why AE + k=6"),
    code(
        """display(board.sort_values("silhouette", ascending=False).head(12))

fig, ax = plt.subplots(figsize=(8, 4))
top = board.nlargest(min(10, len(board)), "silhouette")
colors = ["#1b7a6e" if rid == RUN_ID else "#8a8a8a" for rid in top["run_id"]]
ax.barh(top["run_id"][::-1], top["silhouette"][::-1], color=list(reversed(colors)))
ax.set_xlabel("Silhouette")
ax.set_title("Track C comparison board (champion highlighted if present)")
plt.tight_layout()
plt.savefig(OUT_NB / "bakeoff_top.png", dpi=120, bbox_inches="tight")
plt.show()

print(
    "Selection policy:",
    metrics_art.get("selection_policy", "behavioral coverage + interpretability + banking use"),
)
"""
    ),
    md(
        """## 6. Segment sizes & latent view

Decision layer requires **size sanity** and a composition chart before naming personas.
"""
    ),
    code(
        """size_tbl = (
    df.groupby("segment_id")
    .size()
    .rename("n")
    .to_frame()
    .assign(pct=lambda t: (100 * t["n"] / t["n"].sum()).round(2))
)
display(size_tbl)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(size_tbl.index.astype(str), size_tbl["pct"], color="#1b7a6e")
ax.set_xlabel("segment_id")
ax.set_ylabel("% of book")
ax.set_title("Segment composition")
plt.tight_layout()
plt.savefig(OUT_NB / "segment_sizes.png", dpi=120, bbox_inches="tight")
plt.show()

# 2D view of latent (or PCA of latent)
z2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(z)
fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(z2[:, 0], z2[:, 1], c=labels, cmap="tab10", s=8, alpha=0.55)
ax.set_title("Champion latent space (PCA-2 of z) colored by segment")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.colorbar(sc, ax=ax, label="segment_id")
plt.tight_layout()
plt.savefig(OUT_NB / "latent_pca_segments.png", dpi=120, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        """## 7. Cluster profiles — intelligence pack

For each segment: **centroid (original scale)**, **lift vs population**, auto-**name** from defining drivers, and a short banking persona.
"""
    ),
    code(
        """# Centroids + lift
centroids = df.groupby("segment_id")[feat_cols].mean()
pop = df[feat_cols].mean()
lift = centroids.div(pop.replace(0, np.nan))

display(centroids.round(2).style.background_gradient(cmap="YlGnBu", axis=None))
print("Lift vs population (1.0 = average):")
display(lift.round(2))

# Heatmap of standardized centroids
cent_z = (centroids - centroids.mean()) / centroids.std(ddof=0)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.heatmap(cent_z, cmap="RdBu_r", center=0, annot=False, ax=ax)
ax.set_title("Standardized segment × feature profile")
plt.tight_layout()
plt.savefig(OUT_NB / "profile_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """# Names / personas come from modular segment catalog (frozen at registration)
if not champ.catalog.get("segments"):
    counts = {int(s): int(n) for s, n in zip(*np.unique(labels, return_counts=True))}
    champ.ensure_catalog(centroids, population=pop, sizes=counts, total_n=len(df), persist=True)

name_map = {int(k): v["name"] for k, v in champ.catalog["segments"].items()}
if "segment_name" not in df.columns:
    df["segment_name"] = df["segment_id"].map(name_map)

profile_rows = []
for sid_str, meta in champ.catalog["segments"].items():
    sid = int(sid_str)
    profile_rows.append(
        {
            "segment_id": sid,
            "name": meta["name"],
            "n": meta.get("n") or int((df["segment_id"] == sid).sum()),
            "pct": meta.get("pct") or round(100 * (df["segment_id"] == sid).mean(), 2),
            "one_liner": meta.get("one_liner") or PERSONAS.get(meta["name"], {}).get("one_liner"),
            "default_play": meta.get("default_play") or PERSONAS.get(meta["name"], {}).get("default_play"),
            "recency": round(centroids.loc[sid, "Recency_Days"], 1),
            "balance": round(centroids.loc[sid, "Account_Balance"], 0),
            "txn": round(centroids.loc[sid, "Transaction_Count"], 1),
            "loan": round(centroids.loc[sid, "has_loan"], 2),
            "card": round(centroids.loc[sid, "has_card"], 2),
        }
    )
profiles_named = pd.DataFrame(profile_rows).sort_values("segment_id")
display(profiles_named)
profiles_named.to_csv(OUT_NB / "segment_profiles_named.csv", index=False)
print("Name map:", name_map)
"""
    ),
    code(
        """# Per-segment narrative cards
for _, r in profiles_named.iterrows():
    print("=" * 72)
    print(f"Segment {r['segment_id']} — {r['name']}  ({r['n']} accts, {r['pct']}%)")
    print(
        f"Signature: recency={r['recency']}d | balance={r['balance']:,.0f} CZK | "
        f"txn={r['txn']} | loan={r['loan']} | card={r['card']}"
    )
    print(f"Who: {r['one_liner']}")
    print(f"Bank: {r['default_play']}")
"""
    ),
    code(
        """# Lift bars for top drivers per segment
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, sid in zip(axes.ravel(), sorted(centroids.index)):
    L = lift.loc[sid].sort_values(ascending=False)
    colors = ["#1b7a6e" if v >= 1 else "#c45c26" for v in L.values]
    ax.barh(L.index, L.values, color=colors)
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_title(f"{sid}: {name_map[sid][:28]}", fontsize=9)
plt.suptitle("Feature lift vs population (>1 over-indexed)", y=1.01)
plt.tight_layout()
plt.savefig(OUT_NB / "lift_by_segment.png", dpi=120, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        """## 8. Explainability

Profile in **original feature space**. Optional: decision-tree **surrogate** that predicts `segment_id` — a communication tool, not the clustering objective.
"""
    ),
    code(
        """# Global drivers via tree surrogate
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, random_state=RANDOM_STATE)
clf.fit(df[feat_cols], df["segment_id"])
print(export_text(clf, feature_names=feat_cols, max_depth=4))

imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 3.5))
imp.plot(kind="barh", ax=ax, color="#1b7a6e")
ax.set_title("Surrogate tree — feature importance for segment_id")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_NB / "surrogate_importance.png", dpi=120, bbox_inches="tight")
plt.show()

# Latent ↔ raw correlations (AE-specific)
if z is not None and z.shape[1] >= 2:
    zdf = pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])])
    lat_corr = pd.concat([df[feat_cols].reset_index(drop=True), zdf], axis=1).corr().loc[feat_cols, zdf.columns]
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(lat_corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Raw features ↔ latent dims")
    plt.tight_layout()
    plt.savefig(OUT_NB / "latent_feature_corr.png", dpi=120, bbox_inches="tight")
    plt.show()

if recon_err is not None:
    tmp = df.assign(recon_mse=recon_err[: len(df)])
    display(tmp.groupby("segment_id")["recon_mse"].mean().round(4).to_frame("mean_recon_mse"))
"""
    ),
    code(
        """# Exemplars: medoid-like (closest to segment centroid in scaled space)
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(df[feat_cols])
exemplars = []
for sid in sorted(df["segment_id"].unique()):
    idx = np.where(df["segment_id"].values == sid)[0]
    center = Xs[idx].mean(axis=0)
    d = ((Xs[idx] - center) ** 2).sum(axis=1)
    take = idx[np.argsort(d)[:3]]
    for rank, i in enumerate(take, 1):
        row = df.iloc[i]
        exemplars.append(
            {
                "segment_id": sid,
                "segment_name": name_map[sid],
                "rank": rank,
                "account_id": int(row[id_col]),
                "Transaction_Count": row["Transaction_Count"],
                "Account_Balance": round(row["Account_Balance"], 0),
                "Recency_Days": row["Recency_Days"],
                "has_loan": int(row["has_loan"]),
                "has_card": int(row["has_card"]),
            }
        )
ex_df = pd.DataFrame(exemplars)
display(ex_df)
ex_df.to_csv(OUT_NB / "exemplars.csv", index=False)
"""
    ),
    md(
        """## 9. Fairness diagnostics (minimum pack)

Segmentation features are **behavioral + product** (no age/gender in the model matrix).  
We still check whether segments concentrate by **district** (proxy) when joinable — concentrations are reported as risks, not “proof of no bias.”
"""
    ),
    code(
        """# Try district concentration via account → district_id
fair_notes = []
try:
    berka_dir = resolve_berka_dir(ROOT / "data" / "raw" / "berka-dataset")
    account = pd.read_csv(berka_dir / "account.csv", sep=";")
    if "district_id" not in account.columns:
        account = pd.read_csv(berka_dir / "account.csv")
    merged_f = df[[id_col, "segment_id", "segment_name"]].merge(
        account[["account_id", "district_id"]],
        left_on=id_col,
        right_on="account_id",
        how="left",
    )
    # Top districts share per segment
    ct = pd.crosstab(merged_f["segment_name"], merged_f["district_id"], normalize="index")
    # concentration: max district share within segment
    conc = ct.max(axis=1).sort_values(ascending=False).rename("max_district_share")
    display(conc.to_frame().round(3))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    conc.plot(kind="barh", ax=ax, color="#6b5b95")
    ax.set_xlabel("Largest single-district share within segment")
    ax.set_title("Geographic concentration (risk flag if extreme)")
    plt.tight_layout()
    plt.savefig(OUT_NB / "fairness_district_concentration.png", dpi=120, bbox_inches="tight")
    plt.show()
    fair_notes.append(
        "District is not a model feature; concentration here is a usage risk if CRM maps offers by region."
    )
except Exception as e:
    print("Fairness join skipped:", e)
    fair_notes.append("Could not join district demographics — document as limitation.")

# Behavior vs product drivers (already behavior-first matrix)
print("Model features exclude explicit demographics (age/gender).")
print("Guardrail: do not use these segments for credit eligibility / pricing without a supervised risk model.")
for n in fair_notes:
    print("-", n)
"""
    ),
    md(
        """## 10. Decision map — banking applications

Rules (`04` §5): name segment(s) → cite evidence → concrete action → expected value → guardrail.  
**Not** a churn/default predictor unless a separate supervised model exists.
"""
    ),
    code(
        """# Decision map from modular ChampionModel.catalog
decision_map = champ.decision_map()
display(decision_map)
decision_map.to_csv(OUT_NB / "decision_map.csv", index=False)
print(f"Applications mapped: {len(decision_map)} (need >=3) —", "OK" if len(decision_map) >= 3 else "SHORT")
"""
    ),
    code(
        """# Visual: segment → play
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis("off")
ax.set_title("Decision map (champion segments → bank plays)", loc="left", fontsize=12, pad=12)
y = 0.9
for _, r in decision_map.iterrows():
    ax.text(
        0.02,
        y,
        f"{r['segment_id']}  {r['segment']}",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        y - 0.06,
        f"-> {r['play']}  |  owner: {r['owner']}",
        fontsize=9,
        color="#333",
        transform=ax.transAxes,
    )
    y -= 0.14
plt.tight_layout()
plt.savefig(OUT_NB / "decision_map_diagram.png", dpi=120, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        """## 11. Product workflow (score → decide)

```text
new / batch accounts
  → Berka RFM + loan/card features (scaled)
  → champion encoder → z (4-d)
  → nearest K-Means centroid → segment_id (+ confidence)
  → attach named profile
  → lookup decision map → recommended play + guardrail
  → low confidence / stressed segment → human review queue
```

Registered bundle: `artifacts/segmentation/champion_v1/` (`encoder.pt`, `kmeans.joblib`, `assignments.csv`, profiles).
"""
    ),
    code(
        """# Confidence summary if present
if "confidence" in df.columns:
    conf = df.groupby(["segment_id", "segment_name"])["confidence"].agg(["mean", "median", lambda s: (s < 0.55).mean()])
    conf.columns = ["mean_conf", "median_conf", "low_conf_frac"]
    display(conf.round(3))
else:
    # recompute distance confidence if kmeans artifact exists
    if (CHAMP / "kmeans.joblib").exists() and z is not None and z.shape[1] == 4:
        km = joblib.load(CHAMP / "kmeans.joblib")
        conf_arr = distance_confidence(z, labels, km.cluster_centers_)
        df["confidence"] = conf_arr
        print("low_confidence_frac (<0.55):", round(float((conf_arr < 0.55).mean()), 4))
        display(
            df.groupby("segment_name")["confidence"]
            .agg(mean="mean", low_frac=lambda s: (s < 0.55).mean())
            .round(3)
        )
    else:
        print("No confidence column — distance confidence skipped")

print("\\nChampion acceptance snapshot")
print("- Profiles named:", profiles_named["name"].tolist())
print("- Decision map rows:", len(decision_map))
print("- Manifest:", manifest["run_id"], manifest.get("tag"))
"""
    ),
    md(
        """## 12. Takeaways

1. **Champion stays** `c_ae_lat4_k6` — Berka behavior + products beat Santander’s higher silhouette for the assignment’s banking questions.  
2. **EDA** shows correlated flows and sparse products — AE latent=4 is the right inductive bias.  
3. **Six named segments** are explainable from recency, balance, activity, loan, and card.  
4. **Decision map** gives each segment a different play, owner, KPI, and guardrail.  
5. **Fairness:** demographics are out of the model matrix; still monitor district concentration and ban eligibility misuse.

Next: serve via API (`POST /v1/customers/segment`) or CRM batch export from `assignments.csv` + `decision_map.csv`.
"""
    ),
]


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "10_Champion_Berka_Decision_Layer.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
