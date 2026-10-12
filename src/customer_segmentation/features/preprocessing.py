"""Shared fitted preprocessing persistence for the feature builders."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PREPROCESSOR_VERSION = 1


def prepare_model_matrix(
    interpretable_df: pd.DataFrame,
    *,
    track: str,
    identifier_columns: list[str],
    log1p_columns: list[str],
    scaled_features: list[str],
    clip_nonnegative_columns: list[str] | None = None,
    artifact_path: str | Path | None = None,
    fit: bool = True,
) -> pd.DataFrame:
    """Transform an unscaled interpretable frame using a saved or newly fit scaler.

    A fitted artifact stores its schema/order and transform recipe along with the
    scaler. ``fit=False`` never fits; it loads the exact artifact and rejects a
    schema or transform mismatch. Pass raw/interpretable values, never a model
    matrix, to this function.
    """
    if interpretable_df.attrs.get("feature_representation") == "model":
        raise ValueError("Preprocessing expects unscaled interpretable values, not a model matrix.")

    clip_nonnegative_columns = clip_nonnegative_columns or []
    feature_names = [column for column in interpretable_df.columns if column not in identifier_columns]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Input feature columns must be unique.")
    if not set(identifier_columns).issubset(interpretable_df.columns):
        missing_ids = sorted(set(identifier_columns) - set(interpretable_df.columns))
        raise ValueError(f"Missing identifier columns: {missing_ids}")

    declared = [*log1p_columns, *scaled_features, *clip_nonnegative_columns]
    missing_features = sorted(set(declared) - set(feature_names))
    if missing_features:
        raise ValueError(f"Preprocessing features are missing from input: {missing_features}")

    transformations = {
        "log1p_columns": list(log1p_columns),
        "clip_nonnegative_columns": list(clip_nonnegative_columns),
        "scaler": "sklearn.preprocessing.StandardScaler" if scaled_features else None,
        "scaled_features": list(scaled_features),
        "input_representation": "unscaled_interpretable",
    }
    expected_metadata = {
        "artifact_type": "customer-segmentation-preprocessor",
        "version": PREPROCESSOR_VERSION,
        "track": track,
        "feature_names": feature_names,
        "feature_order": feature_names.copy(),
        "identifier_columns": list(identifier_columns),
        "transformations": transformations,
    }

    if not fit:
        if artifact_path is None:
            raise ValueError("artifact_path is required when fit=False.")
        artifact = joblib.load(artifact_path)
        for key, expected in expected_metadata.items():
            if artifact.get(key) != expected:
                raise ValueError(
                    f"Preprocessor metadata mismatch for '{key}': "
                    f"expected {expected!r}, got {artifact.get(key)!r}."
                )
        scaler = artifact.get("scaler")
        if scaled_features and scaler is None:
            raise ValueError("Preprocessor artifact has no scaler for its declared scaled features.")
    else:
        scaler = StandardScaler() if scaled_features else None

    model_df = interpretable_df.copy()
    for column in log1p_columns:
        values = pd.to_numeric(model_df[column], errors="raise").to_numpy(dtype=np.float64)
        if column in clip_nonnegative_columns:
            values = np.maximum(values, 0.0)
        if not np.isfinite(values).all() or np.any(values <= -1.0):
            raise ValueError(f"Cannot apply log1p to invalid values in '{column}'.")
        model_df[column] = np.log1p(values)

    for column in scaled_features:
        values = pd.to_numeric(model_df[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Cannot scale non-finite values in '{column}'.")

    if fit and scaler is not None:
        scaler.fit(model_df[scaled_features])

    if scaler is not None:
        model_df[scaled_features] = scaler.transform(model_df[scaled_features])

    if fit and artifact_path is not None:
        artifact = {**expected_metadata, "scaler": scaler}
        destination = Path(artifact_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{destination.stem}-", suffix=".tmp", dir=destination.parent, delete=False
            ) as temporary_file:
                temporary_path = temporary_file.name
            joblib.dump(artifact, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    model_df.attrs["feature_representation"] = "model"
    return model_df
