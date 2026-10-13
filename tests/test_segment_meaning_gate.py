"""segment_meaning freeze gate is computed from catalog + exception policy."""

from __future__ import annotations

import json
from pathlib import Path

from customer_segmentation.pipelines.production_freeze import _segment_meaning_gate


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_segment_meaning_pass_with_full_catalog(tmp_path: Path) -> None:
    segments = {
        str(i): {
            "segment_id": i,
            "name": f"Seg {i}",
            "one_liner": "persona",
            "default_play": "play",
            "decision": {"play": "play", "guardrail": "guard"},
        }
        for i in range(3)
    }
    _write(tmp_path / "segment_catalog.json", {"segments": segments})
    gate = _segment_meaning_gate(
        tmp_path,
        contract={"exception_segment_ids": [2]},
        metrics={"n_clusters": 3},
    )
    assert gate["verdict"] == "PASS"
    assert "exception-queue segments=[2]" in gate["detail"]


def test_segment_meaning_fails_without_exceptions(tmp_path: Path) -> None:
    _write(
        tmp_path / "segment_catalog.json",
        {
            "segments": {
                "0": {
                    "name": "A",
                    "one_liner": "x",
                    "decision": {"play": "p", "guardrail": "g"},
                }
            }
        },
    )
    gate = _segment_meaning_gate(tmp_path, contract={}, metrics={"n_clusters": 1})
    assert gate["verdict"] == "FAIL"
    assert "exception_segment_ids" in gate["detail"]


def test_segment_meaning_fails_incomplete_persona(tmp_path: Path) -> None:
    _write(
        tmp_path / "segment_catalog.json",
        {"segments": {"0": {"name": "A", "decision": {"play": "p"}}}},
    )
    gate = _segment_meaning_gate(
        tmp_path,
        contract={"exception_segment_ids": [0]},
        metrics={"n_clusters": 1},
    )
    assert gate["verdict"] == "FAIL"
    assert "Incomplete" in gate["detail"]
