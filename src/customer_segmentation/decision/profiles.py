"""Full segment profile packs (04 §2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLS_DEFAULT = [
    "Transaction_Count",
    "Account_Balance",
    "Total_Credit",
    "Total_Withdrawal",
    "Recency_Days",
    "Account_Age_Days",
    "has_loan",
    "has_card",
]


def compute_centroids_and_lift(
    df: pd.DataFrame,
    *,
    segment_col: str = "segment_id",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cols = feature_cols or [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    centroids = df.groupby(segment_col)[cols].mean()
    population = df[cols].mean()
    lift = centroids.div(population.replace(0, np.nan))
    return centroids, lift, population


def nearest_segment_contrast(centroids: pd.DataFrame) -> pd.DataFrame:
    """For each segment, name the nearest other segment in standardized space."""
    if len(centroids) < 2:
        return pd.DataFrame(columns=["segment_id", "nearest_segment_id", "distance", "top_diff_features"])
    z = (centroids - centroids.mean()) / centroids.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    rows = []
    ids = list(centroids.index)
    for sid in ids:
        others = [o for o in ids if o != sid]
        dists = {o: float(np.linalg.norm(z.loc[sid] - z.loc[o])) for o in others}
        nearest = min(dists, key=dists.get)
        delta = (centroids.loc[sid] - centroids.loc[nearest]).abs().sort_values(ascending=False)
        rows.append(
            {
                "segment_id": int(sid),
                "nearest_segment_id": int(nearest),
                "distance": round(dists[nearest], 4),
                "top_diff_features": ", ".join(delta.head(3).index.tolist()),
            }
        )
    return pd.DataFrame(rows)


def select_exemplars(
    df: pd.DataFrame,
    *,
    segment_col: str = "segment_id",
    id_col: str = "account_id",
    feature_cols: list[str] | None = None,
    n: int = 3,
) -> pd.DataFrame:
    """Medoid-like exemplars closest to segment centroid in scaled feature space."""
    cols = feature_cols or [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    Xs = StandardScaler().fit_transform(df[cols].astype(float))
    rows = []
    for sid in sorted(df[segment_col].unique()):
        idx = np.where(df[segment_col].values == sid)[0]
        center = Xs[idx].mean(axis=0)
        d = ((Xs[idx] - center) ** 2).sum(axis=1)
        take = idx[np.argsort(d)[:n]]
        for rank, i in enumerate(take, 1):
            r = df.iloc[i]
            rows.append(
                {
                    "segment_id": int(sid),
                    "rank": rank,
                    id_col: r[id_col],
                    **{c: r[c] for c in cols},
                }
            )
    return pd.DataFrame(rows)


def confidence_by_segment(df: pd.DataFrame, *, segment_col: str = "segment_id", conf_col: str = "confidence") -> pd.DataFrame:
    if conf_col not in df.columns:
        return pd.DataFrame(columns=[segment_col, "mean_conf", "median_conf", "low_conf_frac"])
    g = df.groupby(segment_col)[conf_col]
    out = g.agg(mean_conf="mean", median_conf="median").copy()
    out["low_conf_frac"] = g.apply(lambda s: float((s < 0.55).mean()))
    return out.reset_index()


def build_profile_packs(
    df: pd.DataFrame,
    catalog: dict[str, Any],
    *,
    segment_col: str = "segment_id",
    id_col: str = "account_id",
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble per-segment intelligence packs for artifacts + report."""
    cols = feature_cols or [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    centroids, lift, population = compute_centroids_and_lift(df, segment_col=segment_col, feature_cols=cols)
    contrast = nearest_segment_contrast(centroids)
    exemplars = select_exemplars(df, segment_col=segment_col, id_col=id_col, feature_cols=cols)
    conf = confidence_by_segment(df, segment_col=segment_col)

    packs = []
    for sid_str, meta in catalog.get("segments", {}).items():
        sid = int(sid_str)
        lift_row = lift.loc[sid].sort_values(ascending=False) if sid in lift.index else pd.Series(dtype=float)
        top_lift = [{"feature": f, "lift": round(float(v), 3)} for f, v in lift_row.head(4).items()]
        bottom_lift = [{"feature": f, "lift": round(float(v), 3)} for f, v in lift_row.tail(2).items()]
        c_row = contrast.loc[contrast["segment_id"] == sid]
        conf_row = conf.loc[conf[segment_col] == sid]
        packs.append(
            {
                "segment_id": sid,
                "name": meta.get("name"),
                "model_version": "champion_v1",
                "n": meta.get("n") or int((df[segment_col] == sid).sum()),
                "pct": meta.get("pct"),
                "one_liner": meta.get("one_liner"),
                "centroid": {c: float(centroids.loc[sid, c]) for c in cols} if sid in centroids.index else {},
                "behavioral_signature": {"over_indexed": top_lift, "under_indexed": bottom_lift},
                "contrast": c_row.to_dict(orient="records")[0] if len(c_row) else {},
                "confidence": conf_row.to_dict(orient="records")[0] if len(conf_row) else {},
                "default_play": meta.get("default_play"),
                "decision": meta.get("decision") or {},
            }
        )

    return {
        "population_means": {k: float(v) for k, v in population.items()},
        "centroids": centroids.reset_index().to_dict(orient="records"),
        "lift": lift.reset_index().to_dict(orient="records"),
        "contrast": contrast.to_dict(orient="records"),
        "exemplars": exemplars.to_dict(orient="records"),
        "confidence_by_segment": conf.to_dict(orient="records"),
        "packs": packs,
    }
