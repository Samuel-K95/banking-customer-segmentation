"""Train-only Track A preprocessing and history-aware inference interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from customer_segmentation.models.clustering_baselines import ordered_id_sha256
from customer_segmentation.models.robustness import (
    apply_quantile_clip,
    fit_quantile_clip_bounds,
)


@dataclass
class P2TrainFittedPreprocessor:
    """P1 transform semantics with training-only clipping/scaler parameters."""

    feature_order: list[str]
    log1p_features: list[str]
    as_of_date: str
    clip_feature: str | None = None
    clip_quantiles: tuple[float, float] | None = None
    scaler: StandardScaler | None = None
    clip_bounds: dict[str, Any] | None = None
    fit_rows: int = 0
    fit_customer_id_sha256: str | None = None
    transformations_: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_order or len(self.feature_order) != len(set(self.feature_order)):
            raise ValueError("feature_order must be non-empty and unique.")
        if "Recency_Days" not in self.feature_order:
            raise ValueError("feature_order must include as-of-derived Recency_Days.")
        if not set(self.log1p_features).issubset(self.feature_order):
            raise ValueError("log1p_features must be in the configured feature order.")
        if self.clip_feature is not None and self.clip_feature not in self.feature_order:
            raise ValueError("clip_feature must be in the configured feature order.")
        if self.clip_quantiles is not None:
            if self.clip_feature is None:
                raise ValueError("clip_feature is required when clipping quantiles are configured.")
            lower, upper = self.clip_quantiles
            if not 0 <= lower < upper <= 1:
                raise ValueError("clip_quantiles must satisfy 0 <= lower < upper <= 1.")
        pd.Timestamp(self.as_of_date)

    @property
    def learned(self) -> bool:
        return self.scaler is not None and self.fit_rows > 0

    def _raw_values(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        required = {
            "CustomerID",
            "Transaction_Count",
            "Last_Transaction",
            *(feature for feature in self.feature_order if feature != "Recency_Days"),
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"P2 input is missing required columns: {missing}")
        if frame["CustomerID"].isna().any() or frame["CustomerID"].duplicated().any():
            raise ValueError("CustomerID values must be non-null and unique within a P2 batch.")
        columns = list(
            dict.fromkeys(
                [
                    "CustomerID",
                    "Transaction_Count",
                    "Last_Transaction",
                    *[feature for feature in self.feature_order if feature != "Recency_Days"],
                ]
            )
        )
        working = frame.loc[:, columns].copy()
        as_of = pd.Timestamp(self.as_of_date)
        last_transaction = pd.to_datetime(working["Last_Transaction"], errors="raise")
        if last_transaction.isna().any() or (last_transaction > as_of).any():
            raise ValueError("Last_Transaction must be valid and no later than as_of_date.")
        working["Recency_Days"] = (as_of - last_transaction).dt.days.astype(np.int64)
        values = working.loc[:, self.feature_order].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("P2 interpretable input features contain non-finite values.")
        if np.any(values[:, self.feature_order.index("Transaction_Count")] < 1):
            raise ValueError("Transaction_Count must be positive.")
        return working, values

    def _deterministic_transforms(self, values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=np.float64).copy()
        if self.clip_bounds is not None and self.clip_feature is not None:
            index = self.feature_order.index(self.clip_feature)
            transformed[:, index], _, _ = apply_quantile_clip(
                transformed[:, index], self.clip_bounds
            )
        for feature in self.log1p_features:
            index = self.feature_order.index(feature)
            if np.any(transformed[:, index] <= -1):
                raise ValueError(f"Cannot apply log1p to values <= -1 for {feature}.")
            transformed[:, index] = np.log1p(transformed[:, index])
        if not np.isfinite(transformed).all():
            raise ValueError("P2 deterministic transforms produced non-finite values.")
        return transformed

    def fit(self, training_frame: pd.DataFrame) -> P2TrainFittedPreprocessor:
        """Learn clipping and StandardScaler parameters from training IDs only."""
        prepared, raw_values = self._raw_values(training_frame)
        if len(prepared) < 2:
            raise ValueError("At least two training customers are required to fit preprocessing.")
        clip_training_below = 0
        clip_training_above = 0
        if self.clip_feature is not None and self.clip_quantiles is not None:
            ratio_values = raw_values[:, self.feature_order.index(self.clip_feature)]
            self.clip_bounds = fit_quantile_clip_bounds(
                ratio_values,
                feature_name=self.clip_feature,
                lower_quantile=float(self.clip_quantiles[0]),
                upper_quantile=float(self.clip_quantiles[1]),
                method="linear",
            )
            _, below, above = apply_quantile_clip(ratio_values, self.clip_bounds)
            clip_training_below = int(below.sum())
            clip_training_above = int(above.sum())
        transformed = self._deterministic_transforms(raw_values)
        scaler = StandardScaler().fit(transformed)
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
            raise ValueError("Training-fitted scaler contains non-finite parameters.")
        self.scaler = scaler
        self.fit_rows = len(prepared)
        self.fit_customer_id_sha256 = ordered_id_sha256(
            prepared["CustomerID"].astype("string").to_numpy(dtype=str)
        )
        self.transformations_ = {
            "input_representation": "unscaled_interpretable_customer_aggregates",
            "feature_order": list(self.feature_order),
            "log1p_features": list(self.log1p_features),
            "clip_feature": self.clip_feature,
            "clip_quantiles": list(self.clip_quantiles) if self.clip_quantiles else None,
            "clip_bounds": self.clip_bounds,
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_fit_scope": "training customer IDs only",
            "scaler_fit_rows": self.fit_rows,
            "fit_customer_id_sha256": self.fit_customer_id_sha256,
            "as_of_date": self.as_of_date,
            "clip_training_below_lower": clip_training_below,
            "clip_training_above_upper": clip_training_above,
            "clip_training_changed_rows": clip_training_below + clip_training_above,
        }
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply saved training parameters only; held-out/inference data never refits."""
        if not self.learned or self.scaler is None:
            raise RuntimeError("P2 preprocessing must be fit on training customers before transform.")
        _, raw_values = self._raw_values(frame)
        transformed = self._deterministic_transforms(raw_values)
        scaled = self.scaler.transform(transformed)
        if not np.isfinite(scaled).all():
            raise ValueError("P2 scaler produced non-finite transformed values.")
        return scaled

    def metadata(self) -> dict[str, Any]:
        if not self.learned or self.scaler is None:
            raise RuntimeError("Unfitted preprocessing has no learned metadata.")
        return {
            **self.transformations_,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }


@dataclass
class BehavioralSegmentationPipeline:
    """P4-ready feature-to-segment interface with insufficient-history routing."""

    preprocessor: P2TrainFittedPreprocessor
    estimator: Any
    minimum_transactions: int
    cohort: str
    metric_space: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.preprocessor.learned:
            raise ValueError("BehavioralSegmentationPipeline requires fitted preprocessing.")
        if self.minimum_transactions < 1:
            raise ValueError("minimum_transactions must be positive.")
        if not self.metric_space:
            raise ValueError("metric_space must be explicit.")

    def predict_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return CustomerID-aligned assignments and route insufficient histories."""
        if "CustomerID" not in frame or "Transaction_Count" not in frame:
            raise ValueError("Inference inputs must include CustomerID and Transaction_Count.")
        customer_ids = frame["CustomerID"].astype("string")
        if customer_ids.isna().any() or customer_ids.duplicated().any():
            raise ValueError("Inference CustomerID values must be non-null and unique.")
        transaction_counts = pd.to_numeric(frame["Transaction_Count"], errors="raise").to_numpy(dtype=np.int64)
        if np.any(transaction_counts < 1):
            raise ValueError("Inference Transaction_Count values must be positive.")
        eligible = transaction_counts >= self.minimum_transactions
        output = pd.DataFrame(
            {
                "CustomerID": customer_ids.to_numpy(dtype=str),
                "transaction_count": transaction_counts,
                "segmentation_status": np.where(
                    eligible, "modeled", "insufficient_observed_history"
                ),
                "cluster": pd.array([pd.NA] * len(frame), dtype="Int64"),
                "metric_space": self.metric_space,
            }
        )
        if not eligible.any():
            return output
        selected = frame.loc[eligible].copy()
        transformed = self.preprocessor.transform(selected)
        labels = self.estimator.predict(transformed).astype(np.int32)
        output.loc[eligible, "cluster"] = labels
        if hasattr(self.estimator, "predict_proba"):
            max_posterior = self.estimator.predict_proba(transformed).max(axis=1)
            output["max_component_posterior"] = np.nan
            output.loc[eligible, "max_component_posterior"] = max_posterior
            output["uncertainty_method"] = np.where(
                eligible, "maximum Gaussian-mixture posterior", None
            )
        return output
