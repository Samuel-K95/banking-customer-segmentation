"""Membership explainability (04 §3) — surrogate tree + rules + AE hooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from customer_segmentation.decision.profiles import FEATURE_COLS_DEFAULT


def fit_segment_surrogate(
    df: pd.DataFrame,
    *,
    segment_col: str = "segment_id",
    feature_cols: list[str] | None = None,
    max_depth: int = 4,
    min_samples_leaf: int = 40,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, list[str], str]:
    """Train a communication-only tree that predicts segment_id from features."""
    cols = feature_cols or [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(df[cols], df[segment_col])
    rules = export_text(clf, feature_names=cols, max_depth=max_depth)
    return clf, cols, rules


def surrogate_importance(clf: DecisionTreeClassifier, feature_cols: list[str]) -> pd.DataFrame:
    return (
        pd.Series(clf.feature_importances_, index=feature_cols, name="importance")
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
    )


def latent_feature_correlations(
    features: pd.DataFrame,
    z: np.ndarray,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Correlate original features with AE latent dims (AE-specific explainability)."""
    cols = feature_cols or [c for c in FEATURE_COLS_DEFAULT if c in features.columns]
    zdf = pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])])
    joined = pd.concat([features[cols].reset_index(drop=True), zdf], axis=1)
    return joined.corr().loc[cols, zdf.columns]


def reconstruction_error_by_segment(
    recon_mse: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"segment_id": labels, "recon_mse": recon_mse})
    return (
        frame.groupby("segment_id")["recon_mse"]
        .agg(mean="mean", median="median", p90=lambda s: float(np.quantile(s, 0.9)))
        .reset_index()
    )


def build_explainability_bundle(
    df: pd.DataFrame,
    *,
    segment_col: str = "segment_id",
    feature_cols: list[str] | None = None,
    z: np.ndarray | None = None,
    recon_mse: np.ndarray | None = None,
) -> dict[str, Any]:
    clf, cols, rules = fit_segment_surrogate(df, segment_col=segment_col, feature_cols=feature_cols)
    imp = surrogate_importance(clf, cols)
    payload: dict[str, Any] = {
        "disclaimer": (
            "Surrogate tree / importances explain segment labels for stakeholders; "
            "they are not the clustering objective."
        ),
        "feature_columns": cols,
        "surrogate_rules": rules,
        "surrogate_importance": imp.to_dict(orient="records"),
        "train_accuracy": float(clf.score(df[cols], df[segment_col])),
    }
    if z is not None and len(z) == len(df):
        corr = latent_feature_correlations(df, z, cols)
        payload["latent_feature_corr"] = corr.round(3).reset_index().rename(columns={"index": "feature"}).to_dict(
            orient="records"
        )
    if recon_mse is not None and len(recon_mse) == len(df):
        payload["reconstruction_by_segment"] = reconstruction_error_by_segment(
            recon_mse, df[segment_col].to_numpy()
        ).to_dict(orient="records")
    return payload
