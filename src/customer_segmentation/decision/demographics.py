"""Berka owner demographics helpers for fairness ablation (not clustering inputs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from customer_segmentation.data.berka_dataset_loader import (
    load_berka_product_tables,
    load_berka_raw_data,
    resolve_berka_dir,
)
from customer_segmentation.features.build_berka_segmentation_features import (
    MODEL_COLUMNS,
)


def berka_age_gender(birth_number: int, *, as_of_year: int = 1997) -> tuple[int, int]:
    """Czech YYMMDD (+50 female month) → (age, is_female)."""
    bn = int(birth_number)
    yy, mm = bn // 10000, (bn // 100) % 100
    is_female = 0
    if mm > 50:
        mm -= 50
        is_female = 1
    year = 1900 + yy
    age = max(as_of_year - year, 0)
    return age, is_female


def owner_demographics_frame(
    *,
    berka_dir: str | Path | None = None,
    as_of_year: int = 1997,
) -> pd.DataFrame:
    """One row per account from OWNER disposition: age, is_female, district_id."""
    path = resolve_berka_dir(berka_dir)
    _trans, account, client = load_berka_raw_data(path)
    _loan, _card, disp = load_berka_product_tables(path)
    owners = disp[disp["type"].astype(str).str.upper().eq("OWNER")].copy()
    owners = owners.merge(client[["client_id", "birth_number"]], on="client_id", how="left")
    ages, females = zip(
        *[berka_age_gender(b, as_of_year=as_of_year) for b in owners["birth_number"].fillna(0).astype(int)]
    )
    owners["age"] = ages
    owners["is_female"] = females
    owners = owners.merge(account[["account_id", "district_id"]], on="account_id", how="left")
    return owners.groupby("account_id", as_index=False).agg(
        age=("age", "first"),
        is_female=("is_female", "first"),
        district_id=("district_id", "first"),
    )


def run_behavior_vs_demographic_ablation(
    feature_df: pd.DataFrame,
    *,
    segment_col: str = "segment_id",
    id_col: str = "account_id",
    feature_cols: list[str] | None = None,
    n_clusters: int = 6,
    berka_dir: str | Path | None = None,
    as_of_year: int = 1997,
    random_state: int = 42,
) -> dict[str, Any]:
    """Recluster train-like rows: behavior-only vs behavior+demographics; ARI vs champion.

    Returns quantitative ablation (not a narrative-only claim).
    """
    cols = feature_cols or [c for c in MODEL_COLUMNS if c in feature_df.columns]
    demo = owner_demographics_frame(berka_dir=berka_dir, as_of_year=as_of_year)
    work = feature_df.merge(demo, on=id_col, how="left")
    work["age"] = work["age"].fillna(work["age"].median())
    work["is_female"] = work["is_female"].fillna(0).astype(int)
    work["district_id"] = work["district_id"].fillna(-1).astype(int)

    top_d = work["district_id"].value_counts().head(15).index
    work["district_top"] = work["district_id"].where(work["district_id"].isin(top_d), other=-1)
    dist_oh = pd.get_dummies(work["district_top"], prefix="dist")

    x_beh = StandardScaler().fit_transform(work[cols].astype(float))
    x_full = StandardScaler().fit_transform(
        np.hstack(
            [
                work[cols].astype(float).values,
                work[["age", "is_female"]].astype(float).values,
                dist_oh.astype(float).values,
            ]
        )
    )
    lab_beh = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(x_beh)
    lab_full = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(x_full)

    ari_beh_full = float(adjusted_rand_score(lab_beh, lab_full))
    out: dict[str, Any] = {
        "n": len(work),
        "k": int(n_clusters),
        "behavior_features": list(cols),
        "added_demographics": ["age", "is_female", "district_top15_onehot"],
        "explicit_demographics_in_champion_model": False,
        "ari_behavior_vs_behavior_plus_demo": ari_beh_full,
        "ablation_status": "executed_recluster",
        "interpretation": (
            "Low ARI(behavior, behavior+demo) means age/gender/district would rewrite "
            "segments if included — supporting the champion's behavior-only feature set. "
            "High ARI would mean demographics are mostly redundant with behavior."
            if ari_beh_full < 0.5
            else "High ARI(behavior, behavior+demo) means demographics do not dominate "
            "the behavior partition on this book; champion still excludes them by policy."
        ),
        "driver_audit": (
            "Champion clustering inputs remain the 8 CRM behavioral/product features. "
            "Demographics are post-hoc ablation / fairness monitors only."
        ),
    }
    if segment_col in work.columns:
        champ = work[segment_col].to_numpy()
        out["ari_champion_vs_behavior_kmeans"] = float(adjusted_rand_score(champ, lab_beh))
        out["ari_champion_vs_behavior_plus_demo_kmeans"] = float(adjusted_rand_score(champ, lab_full))
    return out
