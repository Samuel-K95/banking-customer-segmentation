"""Day-1 data layer: per source download -> raw check -> features, then one DQ report.

Every task calls the project's own functions (mounted at /opt/project, see
docker-compose.yaml), so no pipeline logic lives in this file. Project imports sit
inside the tasks to keep DAG parsing fast.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task

# track -> (source in configs/data.yaml, feature builder module)
TRACKS = {
    "track_a": ("indian_bank", "build_indian_bank_features"),
    "track_b": ("santander", "build_santander_features"),
    "track_c": ("berka", "build_berka_features"),
}


@task(retries=2, retry_delay=timedelta(minutes=1))
def download(source: str) -> None:
    """Skips files already in data/raw; retried because Kaggle downloads can flake."""
    from customer_segmentation.config import data_path
    from customer_segmentation.ingest.download_datasets import DOWNLOADERS

    DOWNLOADERS[source](data_path("raw_dir"))


@task
def check_raw(source: str) -> dict:
    from customer_segmentation.validate.raw_checks import check_raw_source

    return check_raw_source(source)


@task
def build_features(track: str) -> dict:
    builder = importlib.import_module(f"customer_segmentation.features.{TRACKS[track][1]}")
    return builder.run()


@task
def dq_report(raw_sources: list[dict], track_summaries: list[dict]) -> str:
    """Writes artifacts/quality/dq_report_<run_id>.json; fails the run if any gate fails."""
    from customer_segmentation.validate.report import build_report, publish_report

    run_id = get_current_context()["run_id"]
    return str(publish_report(build_report(run_id, list(track_summaries), list(raw_sources))))


@dag(
    dag_id="bcip_data_layer",
    description="Ingest, check and featurize the three banking datasets, then gate on the DQ report",
    schedule=None,  # the Kaggle sources are static snapshots; trigger by hand
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,  # runs overwrite the same files in data/processed
    default_args={"retries": 0},
    tags=["bcip", "day-1", "data-layer"],
)
def bcip_data_layer():
    raw_sources, summaries = [], []
    for track, (source, _) in TRACKS.items():
        landed = download.override(task_id=f"download_{source}")(source)
        note = check_raw.override(task_id=f"check_raw_{source}")(source)
        summary = build_features.override(task_id=f"build_{track}")(track)
        landed >> note >> summary
        raw_sources.append(note)
        summaries.append(summary)
    dq_report(raw_sources, summaries)


bcip_data_layer()
