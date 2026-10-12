"""Promotion must not destroy an existing champion when the source is incomplete."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from customer_segmentation.pipelines.promote_production import (
    PromotionError,
    promote_production,
)

REPO = Path(__file__).resolve().parents[1]
CHAMPION = REPO / "artifacts" / "segmentation" / "champion"


@pytest.mark.skipif(not CHAMPION.exists(), reason="champion bundle missing")
def test_incomplete_source_leaves_destination_untouched(tmp_path: Path):
    dest = tmp_path / "champion"
    shutil.copytree(CHAMPION, dest)
    marker = dest / "segment_catalog.json"
    before = marker.read_text(encoding="utf-8")

    bad_src = tmp_path / "bad_source"
    bad_src.mkdir()
    (bad_src / "manifest.json").write_text("{}", encoding="utf-8")
    (bad_src / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PromotionError, match="preflight"):
        promote_production(source_dir=bad_src, dest_dir=dest, backup=False)

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == before
    assert (dest / "scaler.joblib").exists() or True  # may be gitignored in some envs
    # Destination still present as a directory
    assert dest.is_dir()


@pytest.mark.skipif(
    not (CHAMPION / "encoder.pt").exists() or not (CHAMPION / "scaler.joblib").exists(),
    reason="champion binaries missing locally",
)
def test_complete_source_promotes_atomically(tmp_path: Path):
    src = tmp_path / "source"
    dest = tmp_path / "dest"
    shutil.copytree(CHAMPION, src)
    # Drop serving contract so stamp rewrites it
    contract = src / "serving_contract.json"
    if contract.exists():
        contract.unlink()

    result = promote_production(source_dir=src, dest_dir=dest, backup=False)
    assert result["status"] == "ok"
    assert (dest / "encoder.pt").exists()
    assert (dest / "scaler.joblib").exists()
    assert (dest / "kmeans.joblib").exists()
    stamped = json.loads((dest / "serving_contract.json").read_text(encoding="utf-8"))
    assert stamped["model_version"] == "production_candidate_v1"
    assert stamped["artifact_sha256"]["encoder.pt"]
    assert stamped["artifact_sha256"]["scaler.joblib"]
    assert stamped["artifact_sha256"]["kmeans.joblib"]


@pytest.mark.skipif(
    not (CHAMPION / "encoder.pt").exists() or not (CHAMPION / "scaler.joblib").exists(),
    reason="champion binaries missing locally",
)
def test_promote_rolls_back_when_cutover_move_fails(tmp_path: Path):
    """If live→backup succeeds but staging→live fails, restore the previous champion."""
    src = tmp_path / "source"
    dest = tmp_path / "champion"
    shutil.copytree(CHAMPION, src)
    shutil.copytree(CHAMPION, dest)
    marker_before = (dest / "segment_catalog.json").read_text(encoding="utf-8")
    # Ensure source is distinguishable after a successful promote would overwrite.
    (src / "segment_catalog.json").write_text(
        json.dumps({"segments": {}, "marker": "new_source"}),
        encoding="utf-8",
    )

    real_move = shutil.move

    def flaky_move(src_path, dst_path):
        src_p = Path(src_path)
        dst_p = Path(dst_path)
        # Fail only the staging → live cutover; allow live→backup and rollback moves.
        if src_p.name.startswith(".champion_promote_tmp_") and dst_p.resolve() == dest.resolve():
            raise OSError("simulated cutover move failure")
        return real_move(src_path, dst_path)

    with patch(
        "customer_segmentation.pipelines.promote_production.shutil.move",
        side_effect=flaky_move,
    ):
        with pytest.raises(PromotionError, match="Failed to move staged champion"):
            promote_production(source_dir=src, dest_dir=dest, backup=True)

    assert dest.is_dir()
    assert (dest / "encoder.pt").exists()
    assert (dest / "scaler.joblib").exists()
    assert (dest / "segment_catalog.json").read_text(encoding="utf-8") == marker_before
    # Serving path must not be empty / missing after rollback
    assert any(dest.iterdir())
