"""Loadable champion segmentation model (AE encoder + K-Means + catalog).

Usage:
    champ = ChampionModel.load("artifacts/segmentation/champion")
    out = champ.score(X_scaled)           # segment_id, confidence, name, play
    out = champ.score_interpretable(df)   # applies log1p + frozen scaler
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from customer_segmentation.features.build_berka_segmentation_features import (
    MODEL_COLUMNS,
    transform_berka_model_matrix,
)
from customer_segmentation.models.tabular_ae import TORCH_OK, distance_confidence
from customer_segmentation.models.segment_intel import (
    build_segment_catalog,
    decision_map_frame,
)


@dataclass
class ChampionModel:
    """Production-facing wrapper around registered champion_v1 artifacts."""

    artifact_dir: Path
    run_id: str
    method: str
    feature_columns: list[str]
    feature_version: str
    model_version: str
    latent_dim: int | None
    n_clusters: int
    scaler: Any
    kmeans: Any
    encoder: Any | None
    encoder_hidden: int | None
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    catalog: dict[str, Any] = field(default_factory=dict)

    # ── load / save helpers ──────────────────────────────────────

    @classmethod
    def default_dir(cls, root: str | Path = ".") -> Path:
        return Path(root) / "artifacts" / "segmentation" / "champion"

    @classmethod
    def load(cls, artifact_dir: str | Path | None = None) -> ChampionModel:
        """Load frozen champion bundle from disk."""
        path = Path(artifact_dir) if artifact_dir else cls.default_dir()
        if not path.exists():
            raise FileNotFoundError(f"Champion artifacts not found at {path}")

        manifest = _read_json(path / "manifest.json")
        metrics = _read_json(path / "metrics.json") if (path / "metrics.json").exists() else {}
        feat_meta = _read_json(path / "feature_columns.json")
        feature_columns = list(feat_meta.get("columns") or MODEL_COLUMNS)

        scaler = joblib.load(path / "scaler.joblib") if (path / "scaler.joblib").exists() else None
        kmeans = joblib.load(path / "kmeans.joblib") if (path / "kmeans.joblib").exists() else None
        if kmeans is None:
            raise FileNotFoundError(f"Missing kmeans.joblib under {path}")

        encoder = None
        latent_dim = metrics.get("latent_dim") or manifest.get("latent_dim")
        hidden = None
        if TORCH_OK and (path / "encoder.pt").exists():
            import torch

            from customer_segmentation.models.tabular_ae import TabularAE

            payload = torch.load(path / "encoder.pt", map_location="cpu", weights_only=False)
            state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            latent_dim = int(payload.get("latent_dim", latent_dim or 4)) if isinstance(payload, dict) else int(latent_dim or 4)
            hidden = int(payload.get("hidden", 32)) if isinstance(payload, dict) else 32
            n_feat = int(payload.get("n_feat", len(feature_columns))) if isinstance(payload, dict) else len(feature_columns)
            encoder = TabularAE(n_feat=n_feat, latent_dim=latent_dim, hidden=hidden)
            encoder.load_state_dict(state)
            encoder.eval()

        catalog_path = path / "segment_catalog.json"
        if catalog_path.exists():
            catalog = _read_json(catalog_path)
        else:
            # Rebuild from profiles if older bundle
            catalog = {}
            profiles_path = path / "segment_profiles.csv"
            if profiles_path.exists():
                profiles = pd.read_csv(profiles_path)
                if "segment_id" in profiles.columns:
                    profiles = profiles.set_index("segment_id")
                # drop non-feature cols if present
                cols = [c for c in feature_columns if c in profiles.columns]
                catalog = build_segment_catalog(profiles[cols])

        n_clusters = int(getattr(kmeans, "n_clusters", metrics.get("k") or 6))

        return cls(
            artifact_dir=path,
            run_id=str(manifest.get("run_id") or metrics.get("run_id") or "unknown"),
            method=str(manifest.get("method") or metrics.get("method") or "ae_kmeans"),
            feature_columns=feature_columns,
            feature_version=str(manifest.get("feature_version") or feat_meta.get("feature_version") or ""),
            model_version=str(manifest.get("model_version") or "v1"),
            latent_dim=int(latent_dim) if latent_dim is not None else None,
            n_clusters=n_clusters,
            scaler=scaler,
            kmeans=kmeans,
            encoder=encoder,
            encoder_hidden=hidden,
            manifest=manifest,
            metrics=metrics,
            catalog=catalog,
        )

    # ── inference ────────────────────────────────────────────────

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Map scaled model matrix → latent z (or identity if no encoder)."""
        X = np.asarray(X, dtype=np.float64)
        if self.encoder is None:
            return X
        import torch

        self.encoder.eval()
        with torch.no_grad():
            _, z = self.encoder(torch.tensor(X, dtype=torch.float32))
        return z.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard segment labels for scaled feature matrix."""
        z = self.encode(X)
        return self.kmeans.predict(z).astype(int)

    def predict_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Labels + distance-to-centroid confidence in latent (or feature) space."""
        z = self.encode(X)
        labels = self.kmeans.predict(z).astype(int)
        conf = distance_confidence(z, labels, self.kmeans.cluster_centers_)
        return labels, conf

    def score(self, X: np.ndarray, *, ids: np.ndarray | pd.Series | None = None) -> pd.DataFrame:
        """Score scaled matrix → id, segment_id, confidence, name, play."""
        labels, conf = self.predict_confidence(X)
        return self._attach_catalog(labels, conf, ids=ids)

    def score_interpretable(
        self,
        interpretable: pd.DataFrame,
        *,
        id_col: str = "account_id",
    ) -> pd.DataFrame:
        """Score original-scale Berka features (applies frozen log1p + scaler)."""
        if self.scaler is None:
            raise RuntimeError("Champion scaler missing; cannot score interpretable features")
        missing = [c for c in self.feature_columns if c not in interpretable.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X = transform_berka_model_matrix(interpretable, self.scaler, self.feature_columns)
        ids = interpretable[id_col].values if id_col in interpretable.columns else None
        return self.score(X, ids=ids)

    def segment_name(self, segment_id: int) -> str:
        meta = self.catalog.get("segments", {}).get(str(int(segment_id)), {})
        return str(meta.get("name") or f"Segment {segment_id}")

    def decision_map(self) -> pd.DataFrame:
        return decision_map_frame(self.catalog)

    def ensure_catalog(
        self,
        centroids: pd.DataFrame,
        *,
        population: pd.Series | None = None,
        sizes: dict[int, int] | None = None,
        total_n: int | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Build / refresh catalog from centroids; optionally write to artifact dir."""
        self.catalog = build_segment_catalog(
            centroids, population=population, sizes=sizes, total_n=total_n
        )
        if persist:
            out = self.artifact_dir / "segment_catalog.json"
            out.write_text(json.dumps(self.catalog, indent=2, default=str), encoding="utf-8")
        return self.catalog

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "method": self.method,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "latent_dim": self.latent_dim,
            "n_clusters": self.n_clusters,
            "has_encoder": self.encoder is not None,
            "has_scaler": self.scaler is not None,
            "n_catalog_segments": len(self.catalog.get("segments", {})),
            "artifact_dir": str(self.artifact_dir),
            "metrics": {
                k: self.metrics.get(k)
                for k in ("silhouette", "db_index", "stability_ari_ae_latent_kmeans")
                if k in self.metrics
            },
        }

    def _attach_catalog(
        self,
        labels: np.ndarray,
        conf: np.ndarray,
        *,
        ids: np.ndarray | pd.Series | None,
    ) -> pd.DataFrame:
        names = [self.segment_name(int(s)) for s in labels]
        plays = []
        for s in labels:
            meta = self.catalog.get("segments", {}).get(str(int(s)), {})
            plays.append((meta.get("decision") or {}).get("play"))
        out = pd.DataFrame(
            {
                "segment_id": labels,
                "confidence": np.round(conf, 4),
                "segment_name": names,
                "play": plays,
            }
        )
        if ids is not None:
            out.insert(0, "account_id", np.asarray(ids))
        return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
