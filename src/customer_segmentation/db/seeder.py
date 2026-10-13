"""Seeder utilities to populate BCIP PostgreSQL database from artifacts and raw data."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from customer_segmentation.db.models import (
    Account,
    AssignmentRecord,
    District,
    EvaluationRecord,
    ModelRecord,
    SegmentCatalogRecord,
)


def seed_districts_and_accounts(
    session: Session, berka_dir: Path | str = "data/raw/berka"
) -> dict[str, int]:
    """Seeds districts and accounts from raw Berka CSV files if they exist."""
    berka_path = Path(berka_dir)
    counts = {"districts": 0, "accounts": 0}
    if not berka_path.exists():
        return counts

    # Seed districts
    district_file = berka_path / "district.csv"
    if district_file.exists():
        with open(district_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                d_id = int(row[0].strip('"\n'))
                exists = session.query(District).filter_by(district_id=d_id).first()
                if not exists:
                    district = District(
                        district_id=d_id,
                        district_name=row[1].strip('"\n'),
                        region=row[2].strip('"\n'),
                        avg_salary=float(row[10].strip('"\n'))
                        if len(row) > 10 and row[10] != "?"
                        else None,
                    )
                    session.add(district)
                    counts["districts"] += 1
        session.flush()

    # Seed accounts
    account_file = berka_path / "account.csv"
    if account_file.exists():
        with open(account_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                acc_id = int(row[0].strip('"\n'))
                exists = session.query(Account).filter_by(account_id=acc_id).first()
                if not exists:
                    raw_date = row[3].strip('"\n')
                    # format YYMMDD
                    parsed_date = datetime.strptime(raw_date, "%y%m%d").date()
                    account = Account(
                        account_id=acc_id,
                        district_id=int(row[1].strip('"\n')),
                        frequency=row[2].strip('"\n'),
                        created_date=parsed_date,
                    )
                    session.add(account)
                    counts["accounts"] += 1
        session.flush()

    return counts


def seed_champion_model(
    session: Session,
    champion_dir: Path | str = "artifacts/segmentation/champion",
) -> bool:
    """Seeds the champion model entry in ml_registry.models and evaluations."""
    champ_path = Path(champion_dir)
    manifest_file = champ_path / "manifest.json"
    summary_file = champ_path / "run_summary.json"
    contract_file = champ_path / "serving_contract.json"
    metrics_file = champ_path / "metrics.json"

    if not summary_file.exists():
        return False

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    serving_contract = {}
    if contract_file.exists():
        with open(contract_file, "r", encoding="utf-8") as f:
            serving_contract = json.load(f)

    manifest = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    model_metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            model_metrics = json.load(f)

    model_id = summary.get("metrics", {}).get("run_id", "c_ae_lat4_k6_leakage_safe")
    model_version = serving_contract.get("model_version", "production_candidate_v1")

    existing = session.query(ModelRecord).filter_by(model_id=model_id).first()
    if not existing:
        as_of_str = summary.get("as_of_date", "1997-12-31")
        as_of_date = datetime.strptime(as_of_str, "%Y-%m-%d").date()

        model_rec = ModelRecord(
            model_id=model_id,
            model_version=model_version,
            architecture=summary.get("metrics", {}).get("method", "ae_kmeans"),
            dataset_track="TRACK_C_BERKA",
            latent_dim=summary.get("metrics", {}).get("latent_dim", 4),
            k_clusters=summary.get("metrics", {}).get("k", 6),
            as_of_date=as_of_date,
            minimum_tenure_days=summary.get("minimum_tenure_days", 365),
            artifact_uri=str(champ_path),
            artifact_sha256_hashes=manifest.get("sha256", {}),
            serving_contract_json=serving_contract,
            is_champion=True,
            freeze_status=summary.get("freeze_status", "PROVISIONAL"),
        )
        session.add(model_rec)
        session.flush()

        # Add evaluation record
        eval_rec = EvaluationRecord(
            model_id=model_id,
            holdout_silhouette=model_metrics.get("holdout_silhouette"),
            holdout_db_index=model_metrics.get("holdout_db_index"),
            holdout_ch_index=model_metrics.get("holdout_ch_index"),
            stability_ari=model_metrics.get("stability_ari_ae_latent_kmeans"),
            low_confidence_frac=model_metrics.get("low_confidence_frac"),
            metrics_json=model_metrics,
        )
        session.add(eval_rec)
        session.flush()
        return True

    return False


def seed_segment_catalog(
    session: Session,
    catalog_path: Path | str = "artifacts/segmentation/champion/segment_catalog.json",
    model_version: str = "production_candidate_v1",
) -> int:
    """Seeds the segment catalog in decision.segment_catalog."""
    path = Path(catalog_path)
    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", {})
    count = 0
    for seg_id_str, seg_info in segments.items():
        seg_id = int(seg_id_str)
        existing = (
            session.query(SegmentCatalogRecord)
            .filter_by(model_version=model_version, segment_id=seg_id)
            .first()
        )
        decision_info = seg_info.get("decision", {})
        if not existing:
            catalog_rec = SegmentCatalogRecord(
                model_version=model_version,
                segment_id=seg_id,
                segment_name=seg_info.get("name", f"Segment {seg_id}"),
                percentage_share=seg_info.get("pct"),
                persona_description=seg_info.get("one_liner", ""),
                default_play=seg_info.get("default_play", ""),
                play_owner=decision_info.get("owner", "CRM"),
                target_kpi=decision_info.get("kpi", "Engagement"),
                guardrail_policy=decision_info.get("guardrail", ""),
                centroid_json=seg_info.get("centroid", {}),
            )
            session.add(catalog_rec)
            count += 1

    session.flush()
    return count


def seed_champion_assignments(
    session: Session,
    assignments_csv: Path | str = "artifacts/segmentation/champion/assignments.csv",
    model_id: str = "c_ae_lat4_k6_leakage_safe",
    as_of_date_str: str = "1997-12-31",
) -> int:
    """Seeds the customer assignments into segmentation.assignments."""
    path = Path(assignments_csv)
    if not path.exists():
        return 0

    as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc_id = int(row.get("account_id") or row.get("customer_id") or 0)
            if not acc_id:
                continue

            # Check if account exists in ods.accounts
            acc_exists = session.query(Account).filter_by(account_id=acc_id).first()
            if not acc_exists:
                # Create a placeholder account if needed to satisfy FK
                placeholder = Account(
                    account_id=acc_id,
                    district_id=None,
                    frequency="POPLATEK MESICNE",
                    created_date=as_of_date,
                )
                session.add(placeholder)
                session.flush()

            existing = (
                session.query(AssignmentRecord)
                .filter_by(account_id=acc_id, model_id=model_id, as_of_date=as_of_date)
                .first()
            )
            if not existing:
                seg_id = int(row.get("segment_id", 0))
                rec = AssignmentRecord(
                    account_id=acc_id,
                    model_id=model_id,
                    as_of_date=as_of_date,
                    segment_id=seg_id,
                    segment_name=row.get("segment_name", f"Segment {seg_id}"),
                    confidence_score=float(row.get("confidence", 1.0))
                    if row.get("confidence")
                    else None,
                    is_guardrail_exception=(seg_id == 4),
                    guardrail_reason="exception_queue_human_review"
                    if seg_id == 4
                    else None,
                )
                session.add(rec)
                count += 1

    session.flush()
    return count
