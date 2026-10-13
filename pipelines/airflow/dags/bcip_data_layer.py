"""Data layer + segmentation, step for step with scripts/run_data_layer.sh.

    download_<source> -> check_raw_<source> -> build_<track>      (per selected track)
        -> dq_report -> feature_contract
        -> berka_snapshot -> berka_baselines -> berka_feature_sensitivity
                                             -> berka_temporal_validation
        -> track_d_segmentation

Run parameters mirror the shell flags: `tracks` (--tracks), `skip_downloads`
(--skip-downloads) and `reuse_preprocessors` (--reuse-preprocessors). Unselected tracks
are skipped, not failed. A segmentation stage runs only when its track is selected, and is
skipped when its versioned output already exists.

Every task calls the project's own code from /opt/project (see docker-compose.yaml), the
same working directory the shell script uses, so repo-relative paths such as
artifacts/preprocessors/ and configs/models/ resolve identically.
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta

from airflow.exceptions import AirflowSkipException
from airflow.sdk import Param, dag, get_current_context, task

PROJECT_DIR = os.environ.get("BCIP_PROJECT_DIR", "/opt/project")
ALL_TRACKS = ["track_a", "track_b", "track_c", "track_d"]

# track -> (source in configs/data.yaml, feature builder module)
TRACKS = {
    "track_a": ("indian_bank", "build_indian_bank_features"),
    "track_b": ("santander", "build_santander_features"),
    "track_c": ("berka", "build_berka_features"),
    "track_d": ("banksim", "build_banksim_features"),
}


def selected_tracks() -> list[str]:
    tracks = list(get_current_context()["params"]["tracks"])
    unknown = [t for t in tracks if t not in TRACKS]
    if unknown or not tracks:
        raise ValueError(f"tracks must be a non-empty subset of {ALL_TRACKS}, got {tracks}")
    return tracks


def skip_unless_selected(track: str) -> None:
    if track not in selected_tracks():
        raise AirflowSkipException(f"{track} not selected for this run")


def module_command(module: str, *args: str) -> str:
    return " ".join(["python", "-m", f"customer_segmentation.{module}", *args])


@task(retries=2, retry_delay=timedelta(minutes=1))
def download(track: str) -> None:
    """Skips files already in data/raw; retried because Kaggle downloads can flake."""
    skip_unless_selected(track)
    if get_current_context()["params"]["skip_downloads"]:
        raise AirflowSkipException("skip_downloads is set; raw files are still checked next")
    from customer_segmentation.config import data_path
    from customer_segmentation.ingest.download_datasets import DOWNLOADERS

    DOWNLOADERS[TRACKS[track][0]](data_path("raw_dir"))


@task(trigger_rule="none_failed")  # still runs when the download was skipped on purpose
def check_raw(track: str) -> dict:
    skip_unless_selected(track)
    from customer_segmentation.validate.raw_checks import check_raw_source

    return check_raw_source(TRACKS[track][0])


@task
def build_features(track: str) -> dict:
    skip_unless_selected(track)
    os.chdir(PROJECT_DIR)  # builders save preprocessors to repo-relative artifacts/preprocessors/
    builder = importlib.import_module(f"customer_segmentation.features.{TRACKS[track][1]}")
    return builder.run(fit_preprocessor=not get_current_context()["params"]["reuse_preprocessors"])


@task(trigger_rule="none_failed_min_one_success")
def dq_report() -> str:
    """Writes artifacts/quality/dq_report_<run_id>.json; fails the run if any gate fails."""
    from customer_segmentation.validate.report import build_report, publish_report

    context = get_current_context()
    ti, tracks = context["ti"], selected_tracks()
    summaries = [ti.xcom_pull(task_ids=f"build_{t}") for t in tracks]
    raw_notes = [ti.xcom_pull(task_ids=f"check_raw_{TRACKS[t][0]}") for t in tracks]
    missing = [t for t, s in zip(tracks, summaries) if s is None]
    if missing:
        raise RuntimeError(f"No build summary for selected track(s) {missing}")
    report = build_report(context["run_id"], summaries, [n for n in raw_notes if n])
    return str(publish_report(report))


@task.bash(cwd=PROJECT_DIR)
def feature_contract() -> str:
    """Shell step 4: selected outputs against configs/models/feature_contracts.yaml."""
    return module_command(
        "models.preflight",
        "--config configs/models/feature_contracts.yaml",
        "--tracks {{ params.tracks | join(' ') }}",
    )


# Segmentation stages write versioned outputs once and refuse to overwrite them.
# stage -> (track it needs, module to run, module whose DEFAULT_CONFIG names the output,
#           keys to that output)
SEGMENTATION_STAGES = {
    "berka_snapshot": (
        "track_c",
        "features.build_berka_snapshot_features",
        "models.berka_behavioral_baselines",
        ("data", "snapshot_directory"),
    ),
    "berka_baselines": (
        "track_c",
        "models.berka_behavioral_baselines",
        "models.berka_behavioral_baselines",
        ("data", "output_directory"),
    ),
    "berka_feature_sensitivity": (
        "track_c",
        "models.berka_feature_sensitivity",
        "models.berka_feature_sensitivity",
        ("data", "output_directory"),
    ),
    "berka_temporal_validation": (
        "track_c",
        "models.berka_temporal_validation",
        "models.berka_temporal_validation",
        ("artifacts", "output_directory"),
    ),
    "track_d_segmentation": (
        "track_d",
        "pipelines.run_track_segmentation",
        "pipelines.run_track_segmentation",
        ("artifacts", "output_directory"),
    ),
}


def write_once_command(stage: str) -> str:
    """Runs a stage unless this version's output already exists (exit 99 = skipped).

    Rebuilding means bumping the version/output path in the stage's config, which is how
    the stages are designed to be versioned. The config path comes from the module itself,
    so moving configs only needs a change there.
    """
    track, module, config_owner, keys = SEGMENTATION_STAGES[stage]
    if track not in selected_tracks():
        raise AirflowSkipException(f"{stage} needs {track}, which is not selected")
    import yaml

    config = importlib.import_module(f"customer_segmentation.{config_owner}").DEFAULT_CONFIG
    with open(os.path.join(PROJECT_DIR, config)) as f:
        output = yaml.safe_load(f)
    for key in keys:
        output = output[key]
    if stage == "berka_snapshot":
        output = os.path.join(output, "berka_account_snapshot_features.csv")
    message = f"{output} already exists for this version; reusing it (bump {config} to rebuild)"
    return f'if [ -e "{output}" ]; then echo "{message}"; exit 99; fi; {module_command(module)}'


@task.bash(cwd=PROJECT_DIR)
def berka_snapshot() -> str:
    """Leakage-safe account snapshot as of the configured observation date."""
    return write_once_command("berka_snapshot")


@task.bash(cwd=PROJECT_DIR, trigger_rule="none_failed")
def berka_baselines() -> str:
    """Train-fitted K-Means / GMM baselines with a shared holdout."""
    return write_once_command("berka_baselines")


@task.bash(cwd=PROJECT_DIR, trigger_rule="none_failed")
def berka_feature_sensitivity() -> str:
    return write_once_command("berka_feature_sensitivity")


@task.bash(cwd=PROJECT_DIR, trigger_rule="none_failed")
def berka_temporal_validation() -> str:
    return write_once_command("berka_temporal_validation")


@task.bash(cwd=PROJECT_DIR)
def track_d_segmentation() -> str:
    """BankSim: candidate comparison, selection, profiles, explainability, fairness, outcomes."""
    return write_once_command("track_d_segmentation")


@dag(
    dag_id="bcip_data_layer",
    description="Ingest, check and featurize the banking datasets, gate on DQ + feature contract, then segment",
    schedule=None,  # the Kaggle sources are static snapshots; trigger by hand
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,  # runs overwrite the same files in data/processed and artifacts/
    default_args={"retries": 0},
    params={
        "tracks": Param(ALL_TRACKS, type="array", items={"type": "string", "enum": ALL_TRACKS},
                        description="Feature tracks to process (shell: --tracks)"),
        "skip_downloads": Param(False, type="boolean",
                                description="Use existing raw files; never contact Kaggle (shell: --skip-downloads)"),
        "reuse_preprocessors": Param(False, type="boolean",
                                     description="Load saved preprocessors instead of fitting (shell: --reuse-preprocessors)"),
    },
    tags=["bcip", "data-layer", "segmentation"],
)
def bcip_data_layer():
    builds = []
    for track, (source, _) in TRACKS.items():
        landed = download.override(task_id=f"download_{source}")(track)
        checked = check_raw.override(task_id=f"check_raw_{source}")(track)
        built = build_features.override(task_id=f"build_{track}")(track)
        landed >> checked >> built
        builds.append(built)

    contract = feature_contract()
    builds >> dq_report() >> contract

    baselines = berka_baselines()
    contract >> berka_snapshot() >> baselines
    baselines >> [berka_feature_sensitivity(), berka_temporal_validation()]
    contract >> track_d_segmentation()


bcip_data_layer()
