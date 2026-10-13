"""The DQ report: gates every feature track's output and records pass/fail per run."""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from customer_segmentation.config import data_path, load_data_config
from customer_segmentation.features.io import read_track_summary
from customer_segmentation.validate.raw_checks import DataQualityError


def check_track(summary: dict, gates: dict) -> list[str]:
    """Returns the failed gates for one track's save summary (empty list = pass)."""
    failed = []
    if summary["rows"] < gates["min_rows"]:
        failed.append(f"rows {summary['rows']} < min_rows {gates['min_rows']}")
    if not summary["has_id_column"]:
        failed.append(f"id column {summary['id_column']!r} missing from an output matrix")
    elif summary["duplicate_ids"] > gates["max_duplicate_ids"]:
        failed.append(f"{summary['duplicate_ids']} duplicate {summary['id_column']} values")
    if summary["null_cells"] > gates["max_null_cells"]:
        failed.append(f"{summary['null_cells']} null cells")
    return failed


def build_report(run_id: str, track_summaries: list[dict], raw_sources: list[dict] | None = None) -> dict:
    gates = load_data_config()["quality"]
    tracks = []
    for summary in track_summaries:
        failed = check_track(summary, gates)
        tracks.append({**summary, "status": "fail" if failed else "pass", "failed_gates": failed})
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "fail" if any(t["status"] == "fail" for t in tracks) else "pass",
        "gates": gates,
        "raw_sources": raw_sources or [],
        "tracks": tracks,
    }


def publish_report(report: dict, out_dir: Path | None = None) -> Path:
    """Writes the report (also as dq_report_latest.json), then raises if any gate failed.

    Writing before raising is deliberate: a failed run still leaves its report behind.
    """
    out_dir = out_dir or data_path("quality_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", report["run_id"])
    path = out_dir / f"dq_report_{safe_run_id}.json"
    text = json.dumps(report, indent=2)
    path.write_text(text)
    (out_dir / "dq_report_latest.json").write_text(text)
    print(f"DQ report: {report['status'].upper()} -> {path}")

    if report["status"] != "pass":
        failed = [f"{t['track']}: {'; '.join(t['failed_gates'])}" for t in report["tracks"] if t["failed_gates"]]
        raise DataQualityError("DQ report failed:\n  " + "\n  ".join(failed))
    return path


def main() -> int:
    """Standalone run (shell script): re-reads the saved matrices to build the report."""
    run_id = "local__" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    summaries = [read_track_summary(track) for track in load_data_config()["tracks"]]
    try:
        publish_report(build_report(run_id, summaries))
    except DataQualityError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
