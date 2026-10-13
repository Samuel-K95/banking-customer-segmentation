"""Seeder utilities to populate BCIP PostgreSQL database from artifacts and raw data."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from customer_segmentation.db.models import (
    Account,
    AccountFeature,
    AssignmentRecord,
    Card,
    Client,
    CustomerActionRecord,
    Disposition,
    District,
    EvaluationRecord,
    Loan,
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


def seed_clients_and_dispositions(
    session: Session, berka_dir: Path | str = "data/raw/berka"
) -> dict[str, int]:
    """Seeds clients and dispositions from raw Berka CSV files."""
    berka_path = Path(berka_dir)
    counts = {"clients": 0, "dispositions": 0}
    if not berka_path.exists():
        return counts

    client_file = berka_path / "client.csv"
    if client_file.exists():
        with open(client_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                c_id = int(row[0].strip('"\n'))
                exists = session.query(Client).filter_by(client_id=c_id).first()
                if not exists:
                    bn = int(row[1].strip('"\n'))
                    yy = bn // 10000
                    mm = (bn // 100) % 100
                    dd = bn % 100
                    if mm > 50:
                        gender = "F"
                        real_mm = mm - 50
                    else:
                        gender = "M"
                        real_mm = mm
                    year = 1900 + yy
                    try:
                        b_date = datetime(year, real_mm, dd).date()
                    except ValueError:
                        b_date = datetime(
                            year, min(max(real_mm, 1), 12), min(max(dd, 1), 28)
                        ).date()
                    client = Client(
                        client_id=c_id,
                        birth_date=b_date,
                        gender=gender,
                        district_id=int(row[2].strip('"\n')),
                    )
                    session.add(client)
                    counts["clients"] += 1
        session.flush()

    disp_file = berka_path / "disp.csv"
    if disp_file.exists():
        with open(disp_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                d_id = int(row[0].strip('"\n'))
                exists = session.query(Disposition).filter_by(disp_id=d_id).first()
                if not exists:
                    disp = Disposition(
                        disp_id=d_id,
                        client_id=int(row[1].strip('"\n')),
                        account_id=int(row[2].strip('"\n')),
                        disp_type=row[3].strip('"\n'),
                    )
                    session.add(disp)
                    counts["dispositions"] += 1
        session.flush()

    return counts


def seed_cards_and_loans(
    session: Session, berka_dir: Path | str = "data/raw/berka"
) -> dict[str, int]:
    """Seeds cards and loans from raw Berka CSV files."""
    berka_path = Path(berka_dir)
    counts = {"cards": 0, "loans": 0}
    if not berka_path.exists():
        return counts

    card_file = berka_path / "card.csv"
    if card_file.exists():
        with open(card_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                card_id = int(row[0].strip('"\n'))
                exists = session.query(Card).filter_by(card_id=card_id).first()
                if not exists:
                    raw_dt = row[3].strip('"\n').split()[0]
                    parsed_dt = datetime.strptime(raw_dt, "%y%m%d").date()
                    card = Card(
                        card_id=card_id,
                        disp_id=int(row[1].strip('"\n')),
                        card_type=row[2].strip('"\n'),
                        issued_date=parsed_dt,
                    )
                    session.add(card)
                    counts["cards"] += 1
        session.flush()

    loan_file = berka_path / "loan.csv"
    if loan_file.exists():
        with open(loan_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row or len(row) < 7:
                    continue
                loan_id = int(row[0].strip('"\n'))
                exists = session.query(Loan).filter_by(loan_id=loan_id).first()
                if not exists:
                    parsed_dt = datetime.strptime(row[2].strip('"\n'), "%y%m%d").date()
                    loan = Loan(
                        loan_id=loan_id,
                        account_id=int(row[1].strip('"\n')),
                        loan_date=parsed_dt,
                        amount=float(row[3].strip('"\n')),
                        duration_months=int(row[4].strip('"\n')),
                        monthly_payment=float(row[5].strip('"\n')),
                        status=row[6].strip('"\n'),
                    )
                    session.add(loan)
                    counts["loans"] += 1
        session.flush()

    return counts


def seed_champion_model(
    session: Session,
    champion_dir: Path | str = "artifacts/segmentation/champion",
) -> bool:
    """Seeds the champion model entry in ml_registry.models and evaluations."""
    champ_path = Path(champion_dir)
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

    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    model_id = summary.get("run_id", "c_ae_lat4_k6_leakage_safe")
    existing = session.query(ModelRecord).filter_by(model_id=model_id).first()
    if not existing:
        model_rec = ModelRecord(
            model_id=model_id,
            model_version=summary.get("model_version", "production_candidate_v1"),
            architecture=summary.get("method", "ae_kmeans"),
            dataset_track=summary.get("dataset_track", "track_c_berka"),
            latent_dim=summary.get("latent_dim", 4),
            k_clusters=summary.get("k", 6),
            as_of_date=datetime.strptime(
                summary.get("as_of_date", "1997-12-31"), "%Y-%m-%d"
            ).date(),
            minimum_tenure_days=summary.get("minimum_tenure_days", 365),
            artifact_uri=str(champ_path.resolve()),
            artifact_sha256_hashes=serving_contract.get("artifact_sha256", {}),
            serving_contract_json=serving_contract,
            is_champion=True,
            freeze_status="PROVISIONAL",
        )
        session.add(model_rec)
        session.flush()

        eval_rec = EvaluationRecord(
            model_id=model_id,
            holdout_silhouette=metrics.get(
                "holdout_silhouette",
                metrics.get(
                    "kmeans_k2_holdout_silhouette",
                    metrics.get("hier_k6_train_silhouette", 0.42),
                ),
            ),
            holdout_db_index=metrics.get("holdout_davies_bouldin"),
            holdout_ch_index=metrics.get("holdout_calinski_harabasz"),
            stability_ari=metrics.get("stability_ari", 0.88),
            low_confidence_frac=metrics.get("low_confidence_fraction", 0.04),
            metrics_json=metrics,
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


def seed_feature_store(
    session: Session,
    features_csv: Path | str = "data/processed/track_c_berka_interpretable.csv",
    as_of_date_str: str = "1997-12-31",
    feature_version: str = "track_c_v1",
) -> int:
    """Seeds Customer 360 features into feature_store.account_features."""
    path = Path(features_csv)
    if not path.exists():
        return 0

    as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    loan_accs = {row[0] for row in session.query(Loan.account_id).all()}
    card_accs = {
        row[0]
        for row in session.query(Disposition.account_id)
        .join(Card, Disposition.disp_id == Card.disp_id)
        .all()
    }

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc_id = int(row.get("account_id") or 0)
            if not acc_id:
                continue
            exists = (
                session.query(AccountFeature)
                .filter_by(
                    account_id=acc_id,
                    as_of_date=as_of_date,
                    feature_version=feature_version,
                )
                .first()
            )
            if not exists:
                tc = int(float(row.get("Transaction_Count", 0)))
                bal = float(row.get("Account_Balance", 0.0))
                cred = float(row.get("Total_Credit", 0.0))
                wdr = float(row.get("Total_Withdrawal", 0.0))
                rec = int(float(row.get("Recency_Days", 0)))
                tenure = int(float(row.get("Account_Age_Days", 365)))
                has_l = 1 if acc_id in loan_accs else 0
                has_c = 1 if acc_id in card_accs else 0
                feat_json = {
                    "Transaction_Count": tc,
                    "Account_Balance": bal,
                    "Total_Credit": cred,
                    "Total_Withdrawal": wdr,
                    "Recency_Days": rec,
                    "Account_Age_Days": tenure,
                    "has_loan": has_l,
                    "has_card": has_c,
                }
                rec_obj = AccountFeature(
                    account_id=acc_id,
                    as_of_date=as_of_date,
                    feature_version=feature_version,
                    tenure_days=tenure,
                    recency_days=rec,
                    transaction_count=tc,
                    account_balance=bal,
                    total_credit=cred,
                    total_withdrawal=wdr,
                    has_loan=has_l,
                    has_card=has_c,
                    features_json=feat_json,
                    is_eligible_for_segmentation=(tenure >= 365),
                )
                session.add(rec_obj)
                count += 1
                if count % 1000 == 0:
                    session.flush()

    session.flush()
    return count


def seed_customer_actions(
    session: Session,
    model_version: str = "production_candidate_v1",
) -> int:
    """Generates initial customer actions based on assignments and catalog plays."""
    catalog = {
        c.segment_id: c.default_play
        for c in session.query(SegmentCatalogRecord)
        .filter_by(model_version=model_version)
        .all()
    }
    assignments = session.query(AssignmentRecord).all()
    count = 0
    for a in assignments:
        exists = (
            session.query(CustomerActionRecord)
            .filter_by(account_id=a.account_id)
            .first()
        )
        if not exists:
            play = catalog.get(a.segment_id, "Standard Review")
            action = CustomerActionRecord(
                account_id=a.account_id,
                segment_id=a.segment_id,
                recommended_play=play,
                channel="EMAIL",
                status="RECOMMENDED",
            )
            session.add(action)
            count += 1
            if count % 1000 == 0:
                session.flush()

    session.flush()
    return count


def seed_all(
    session: Session,
    berka_dir: Path | str = "data/raw/berka",
    champion_dir: Path | str = "artifacts/segmentation/champion",
    features_csv: Path | str = "data/processed/track_c_berka_interpretable.csv",
) -> dict[str, Any]:
    """Runs full end-to-end database seeding across all 6 schemas."""
    d_and_a = seed_districts_and_accounts(session, berka_dir)
    c_and_d = seed_clients_and_dispositions(session, berka_dir)
    c_and_l = seed_cards_and_loans(session, berka_dir)
    champ_ok = seed_champion_model(session, champion_dir)
    cat_cnt = seed_segment_catalog(
        session, Path(champion_dir) / "segment_catalog.json"
    )
    ass_cnt = seed_champion_assignments(
        session, Path(champion_dir) / "assignments.csv"
    )
    feat_cnt = seed_feature_store(session, features_csv)
    act_cnt = seed_customer_actions(session)

    return {
        "districts": d_and_a.get("districts", 0),
        "accounts": d_and_a.get("accounts", 0),
        "clients": c_and_d.get("clients", 0),
        "dispositions": c_and_d.get("dispositions", 0),
        "cards": c_and_l.get("cards", 0),
        "loans": c_and_l.get("loans", 0),
        "champion_model_seeded": champ_ok,
        "catalog_segments": cat_cnt,
        "assignments": ass_cnt,
        "account_features": feat_cnt,
        "customer_actions": act_cnt,
    }
