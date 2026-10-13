"""Champion registration: freeze config + export assignments/artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def register_champion(
    out_dir: Path,
    *,
    run_id: str,
    track: str,
    method: str,
    feature_version: str,
    model_version: str,
    feature_columns: list[str],
    metrics: dict[str, Any],
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    comparison_board: pd.DataFrame,
    scaler: Any | None = None,
    kmeans: Any | None = None,
    encoder_state: dict[str, Any] | None = None,
    segment_catalog: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Path]:
    """Write a versioned champion bundle under artifacts/segmentation/."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["assignments"] = out_dir / "assignments.csv"
    assignments.to_csv(paths["assignments"], index=False)

    paths["profiles"] = out_dir / "segment_profiles.csv"
    profiles.to_csv(paths["profiles"])

    paths["comparison_board"] = out_dir / "comparison_board.csv"
    comparison_board.to_csv(paths["comparison_board"], index=False)

    paths["metrics"] = write_json(out_dir / "metrics.json", metrics)
    paths["feature_columns"] = write_json(
        out_dir / "feature_columns.json",
        {"feature_version": feature_version, "columns": feature_columns},
    )

    if segment_catalog is not None:
        paths["segment_catalog"] = write_json(out_dir / "segment_catalog.json", segment_catalog)
        paths["decision_map"] = out_dir / "decision_map.csv"
        from customer_segmentation.models.segment_intel import decision_map_frame

        decision_map_frame(segment_catalog).to_csv(paths["decision_map"], index=False)

    if scaler is not None:
        paths["scaler"] = out_dir / "scaler.joblib"
        joblib.dump(scaler, paths["scaler"])
    if kmeans is not None:
        paths["kmeans"] = out_dir / "kmeans.joblib"
        joblib.dump(kmeans, paths["kmeans"])
    if encoder_state is not None:
        try:
            import torch

            paths["encoder"] = out_dir / "encoder.pt"
            torch.save(encoder_state, paths["encoder"])
        except ImportError:
            paths["encoder_state_json"] = write_json(out_dir / "encoder_state_meta.json", {"note": "torch missing"})

    manifest = {
        "run_id": run_id,
        "track": track,
        "method": method,
        "model_version": model_version,
        "feature_version": feature_version,
        "registered_at": datetime.now(UTC).isoformat(),
        "tag": f"prod_segmentation_{model_version}",
        "n_customers": len(assignments),
        "n_segments": int(assignments["segment_id"].nunique()) if "segment_id" in assignments else None,
        "notes": notes,
        "files": {k: str(v.relative_to(out_dir)) for k, v in paths.items()},
        "load": "customer_segmentation.models.champion.ChampionModel.load",
    }
    paths["manifest"] = write_json(out_dir / "manifest.json", manifest)
    return paths
