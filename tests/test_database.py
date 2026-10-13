"""Unit and integration tests for PostgreSQL database models, connection, and seeder."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from customer_segmentation.db.audit import log_inference_requests
from customer_segmentation.db.connection import (
    Base,
    check_db_connection,
    get_database_url,
    get_session_factory,
)
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
    InferenceLogRecord,
    Loan,
    ModelRecord,
    SegmentCatalogRecord,
    Transaction,
)


def is_live_postgres_available() -> bool:
    """Check if the real PostgreSQL container is reachable."""
    try:
        status = check_db_connection()
        return bool(status.get("connected"))
    except Exception:  # noqa: BLE001
        return False


requires_postgres = pytest.mark.skipif(
    not is_live_postgres_available(),
    reason="Live PostgreSQL service is not reachable (start with docker compose up -d postgres)",
)


def test_database_url_normalization(monkeypatch):
    """Verifies that postgresql:// URLs get rewritten with psycopg2 driver."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    url = get_database_url()
    assert url == "postgresql+psycopg2://user:pass@localhost:5432/testdb"

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/testdb"
    )
    assert (
        get_database_url()
        == "postgresql+psycopg2://user:pass@localhost:5432/testdb"
    )

    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    assert get_database_url() == "sqlite:///test.db"


def test_check_db_connection_handles_failure(monkeypatch):
    """Verifies check_db_connection returns graceful dict when connection fails."""
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = RuntimeError("Connection refused")
    monkeypatch.setattr(
        "customer_segmentation.db.connection.get_engine",
        lambda: mock_engine,
    )

    result = check_db_connection()
    assert result["connected"] is False
    assert "Connection refused" in str(result["error"])


def test_models_metadata_contains_all_schemas_and_tables():
    """Verifies that declarative models register all expected schema tables."""
    registered = set(Base.metadata.tables.keys())
    expected_tables = {
        "ods.districts",
        "ods.accounts",
        "feature_store.account_features",
        "ml_registry.models",
        "ml_registry.evaluations",
        "segmentation.assignments",
        "decision.segment_catalog",
        "decision.customer_actions",
        "audit.inference_log",
        "audit.drift_snapshots",
    }
    for t in expected_tables:
        assert t in registered, f"Missing table {t} in Base.metadata.tables"


def test_audit_logging_resilient_to_db_failure(monkeypatch):
    """Verifies that log_inference_requests never raises even if DB fails."""
    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.side_effect = RuntimeError("DB connection dropped")
    monkeypatch.setattr(
        "customer_segmentation.db.audit.get_db_session",
        lambda: mock_session_ctx,
    )

    # Should not raise exception
    log_inference_requests(
        scored_records=[{"account_id": 101, "segment_id": 2, "segment_name": "Core"}],
        raw_payloads=[{"account_id": 101}],
        model_version="c_ae_lat4_k6_leakage_safe",
        as_of_date="1998-12-31",
    )


@pytest.mark.integration
@requires_postgres
def test_live_postgres_status_and_tables():
    """Verifies connectivity to live PostgreSQL container and queried tables."""
    status = check_db_connection()
    assert status["connected"] is True
    assert "PostgreSQL 16" in str(status.get("version", ""))

    factory = get_session_factory()
    with factory() as session:
        # Check ods tables populated
        n_districts = session.query(District).count()
        assert n_districts >= 77

        n_accounts = session.query(Account).count()
        assert n_accounts >= 4500

        n_clients = session.query(Client).count()
        assert n_clients >= 5300

        n_disps = session.query(Disposition).count()
        assert n_disps >= 5300

        n_cards = session.query(Card).count()
        assert n_cards >= 800

        n_loans = session.query(Loan).count()
        assert n_loans >= 600

        n_trans = session.query(Transaction).count()
        assert n_trans >= 1000000

        # Check feature store populated
        n_feats = session.query(AccountFeature).count()
        assert n_feats >= 4500

        # Check champion model registered
        champ = (
            session.query(ModelRecord)
            .filter_by(model_id="c_ae_lat4_k6_leakage_safe")
            .first()
        )
        assert champ is not None
        assert champ.architecture == "ae_kmeans"
        assert champ.is_champion is True

        # Check evaluation record exists
        eval_rec = (
            session.query(EvaluationRecord)
            .filter_by(model_id=champ.model_id)
            .first()
        )
        assert eval_rec is not None
        assert eval_rec.holdout_silhouette is not None

        # Check catalog
        catalog = session.query(SegmentCatalogRecord).all()
        assert len(catalog) == 6

        # Check assignments
        n_assignments = session.query(AssignmentRecord).count()
        assert n_assignments >= 3600

        # Check customer actions
        n_actions = session.query(CustomerActionRecord).count()
        assert n_actions >= 3600


@pytest.mark.integration
@requires_postgres
def test_audit_log_live_insertion():
    """Verifies that log_inference_requests writes records into audit.inference_log."""
    test_acc_id = 999999
    log_inference_requests(
        scored_records=[
            {
                "account_id": test_acc_id,
                "segment_id": 3,
                "segment_name": "High Volume Transactors",
                "confidence": 0.95,
                "guardrail": False,
            }
        ],
        raw_payloads=[{"account_id": test_acc_id, "amount_mean": 5400.0}],
        model_version="c_ae_lat4_k6_leakage_safe",
        as_of_date="1998-12-31",
        latency_ms=12.5,
        client_ip="127.0.0.1",
    )

    factory = get_session_factory()
    with factory() as session:
        rec = (
            session.query(InferenceLogRecord)
            .filter_by(account_id=test_acc_id)
            .order_by(InferenceLogRecord.created_at.desc())
            .first()
        )
        assert rec is not None
        assert rec.predicted_segment_id == 3
        assert rec.predicted_segment_name == "High Volume Transactors"
        assert rec.latency_ms == 12.5

        # Cleanup test record
        session.delete(rec)
        session.commit()


@pytest.mark.integration
@requires_postgres
def test_cli_db_status_invocation(capsys):
    """Verifies that running 'bcs db-status' produces structured JSON output."""
    from customer_segmentation.cli import main

    with patch("sys.argv", ["bcs", "db-status"]):
        main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert "connection" in payload
    assert "table_counts" in payload
    assert "ods.districts" in payload["table_counts"]
