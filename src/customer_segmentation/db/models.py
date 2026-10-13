"""SQLAlchemy ORM models for the 6 BCIP database schemas."""

from __future__ import annotations

from typing import Any, ClassVar

from sqlalchemy import (
    BIGINT,
    BOOLEAN,
    DATE,
    FLOAT,
    INT,
    NUMERIC,
    SMALLINT,
    TEXT,
    TIMESTAMP,
    VARCHAR,
    Column,
    ForeignKey,
    Index,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from customer_segmentation.db.connection import Base

TIMESTAMPTZ = TIMESTAMP(timezone=True)


# ============================================================================
# Domain 1: Operational Data Store (ods)
# ============================================================================


class District(Base):
    __tablename__ = "districts"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ods"}

    district_id = Column(INT, primary_key=True)
    district_name = Column(VARCHAR(100), nullable=False)
    region = Column(VARCHAR(100), nullable=False)
    n_inhabitants = Column(INT)
    n_municipalities_gt_499 = Column(INT)
    n_municipalities_500_1999 = Column(INT)
    n_municipalities_2000_9999 = Column(INT)
    n_municipalities_gt_10000 = Column(INT)
    n_cities = Column(INT)
    ratio_urban_inhabitants = Column(NUMERIC(6, 2))
    avg_salary = Column(NUMERIC(12, 2))
    unemployment_rate_95 = Column(NUMERIC(6, 2))
    unemployment_rate_96 = Column(NUMERIC(6, 2))
    n_entrepreneurs_per_1000 = Column(INT)
    n_crimes_95 = Column(INT)
    n_crimes_96 = Column(INT)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class Client(Base):
    __tablename__ = "clients"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ods"}

    client_id = Column(INT, primary_key=True)
    birth_date = Column(DATE, nullable=False)
    gender = Column(VARCHAR(1), nullable=False)
    district_id = Column(INT, ForeignKey("ods.districts.district_id"))
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class Account(Base):
    __tablename__ = "accounts"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ods"}

    account_id = Column(INT, primary_key=True)
    district_id = Column(INT, ForeignKey("ods.districts.district_id"))
    frequency = Column(VARCHAR(50), nullable=False)
    created_date = Column(DATE, nullable=False)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class Disposition(Base):
    __tablename__ = "dispositions"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ods"}

    disp_id = Column(INT, primary_key=True)
    client_id = Column(INT, ForeignKey("ods.clients.client_id"), nullable=False)
    account_id = Column(INT, ForeignKey("ods.accounts.account_id"), nullable=False)
    disp_type = Column(VARCHAR(20), nullable=False)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        Index("idx_trans_account_date", "account_id", "trans_date"),
        {"schema": "ods"},
    )

    trans_id = Column(INT, primary_key=True)
    account_id = Column(INT, ForeignKey("ods.accounts.account_id"), nullable=False)
    trans_date = Column(DATE, nullable=False)
    trans_type = Column(VARCHAR(20), nullable=False)
    operation = Column(VARCHAR(50))
    amount = Column(NUMERIC(12, 2), nullable=False)
    balance_after = Column(NUMERIC(12, 2), nullable=False)
    k_symbol = Column(VARCHAR(50))
    bank_partner = Column(VARCHAR(20))
    account_partner = Column(INT)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class Card(Base):
    __tablename__ = "cards"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ods"}

    card_id = Column(INT, primary_key=True)
    disp_id = Column(INT, ForeignKey("ods.dispositions.disp_id"), nullable=False)
    card_type = Column(VARCHAR(20), nullable=False)
    issued_date = Column(DATE, nullable=False)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class Loan(Base):
    __tablename__ = "loans"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ods"}

    loan_id = Column(INT, primary_key=True)
    account_id = Column(INT, ForeignKey("ods.accounts.account_id"), nullable=False)
    loan_date = Column(DATE, nullable=False)
    amount = Column(NUMERIC(12, 2), nullable=False)
    duration_months = Column(INT, nullable=False)
    monthly_payment = Column(NUMERIC(12, 2), nullable=False)
    status = Column(VARCHAR(5), nullable=False)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


# ============================================================================
# Domain 2: Customer 360 Feature Store (feature_store)
# ============================================================================


class AccountFeature(Base):
    __tablename__ = "account_features"
    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "as_of_date",
            "feature_version",
            name="uq_account_as_of_version",
        ),
        Index("idx_features_account_asof", "account_id", "as_of_date"),
        {"schema": "feature_store"},
    )

    feature_id = Column(BIGINT, primary_key=True, autoincrement=True)
    account_id = Column(INT, ForeignKey("ods.accounts.account_id"), nullable=False)
    as_of_date = Column(DATE, nullable=False)
    feature_version = Column(VARCHAR(50), nullable=False)
    tenure_days = Column(INT, nullable=False)
    recency_days = Column(INT, nullable=False)
    transaction_count = Column(INT, nullable=False)
    account_balance = Column(NUMERIC(14, 2), nullable=False)
    total_credit = Column(NUMERIC(14, 2), nullable=False)
    total_withdrawal = Column(NUMERIC(14, 2), nullable=False)
    has_loan = Column(SMALLINT, nullable=False, default=0)
    has_card = Column(SMALLINT, nullable=False, default=0)
    features_json = Column(JSONB)
    is_eligible_for_segmentation = Column(BOOLEAN, default=True)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


# ============================================================================
# Domain 3: ML Model Registry & Governance (ml_registry)
# ============================================================================


class ModelRecord(Base):
    __tablename__ = "models"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ml_registry"}

    model_id = Column(VARCHAR(100), primary_key=True)
    model_version = Column(VARCHAR(50), nullable=False)
    architecture = Column(VARCHAR(100), nullable=False)
    dataset_track = Column(VARCHAR(50), nullable=False)
    latent_dim = Column(INT)
    k_clusters = Column(INT, nullable=False)
    as_of_date = Column(DATE, nullable=False)
    minimum_tenure_days = Column(INT, default=365)
    artifact_uri = Column(TEXT, nullable=False)
    artifact_sha256_hashes = Column(JSONB, nullable=False)
    serving_contract_json = Column(JSONB, nullable=False)
    is_champion = Column(BOOLEAN, default=False)
    freeze_status = Column(VARCHAR(30), default="PROVISIONAL")
    promoted_at = Column(TIMESTAMPTZ)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class EvaluationRecord(Base):
    __tablename__ = "evaluations"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "ml_registry"}

    eval_id = Column(INT, primary_key=True, autoincrement=True)
    model_id = Column(
        VARCHAR(100), ForeignKey("ml_registry.models.model_id"), nullable=False
    )
    holdout_silhouette = Column(FLOAT)
    holdout_db_index = Column(FLOAT)
    holdout_ch_index = Column(FLOAT)
    stability_ari = Column(FLOAT)
    low_confidence_frac = Column(FLOAT)
    metrics_json = Column(JSONB)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


# ============================================================================
# Domain 4: Segmentation & Latent Embeddings (segmentation)
# ============================================================================


class AssignmentRecord(Base):
    __tablename__ = "assignments"
    __table_args__ = (
        UniqueConstraint(
            "account_id", "model_id", "as_of_date", name="uq_account_model_asof"
        ),
        Index("idx_assignments_segment", "segment_id"),
        {"schema": "segmentation"},
    )

    assignment_id = Column(BIGINT, primary_key=True, autoincrement=True)
    account_id = Column(INT, ForeignKey("ods.accounts.account_id"), nullable=False)
    model_id = Column(
        VARCHAR(100), ForeignKey("ml_registry.models.model_id"), nullable=False
    )
    as_of_date = Column(DATE, nullable=False)
    segment_id = Column(SMALLINT, nullable=False)
    segment_name = Column(VARCHAR(100), nullable=False)
    confidence_score = Column(FLOAT)
    is_guardrail_exception = Column(BOOLEAN, default=False)
    guardrail_reason = Column(VARCHAR(255))
    assigned_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


# ============================================================================
# Domain 5: Decision Catalog & Closed-Loop Tracking (decision)
# ============================================================================


class SegmentCatalogRecord(Base):
    __tablename__ = "segment_catalog"
    __table_args__ = (
        UniqueConstraint("model_version", "segment_id", name="uq_model_segment"),
        {"schema": "decision"},
    )

    catalog_id = Column(INT, primary_key=True, autoincrement=True)
    model_version = Column(VARCHAR(50), nullable=False)
    segment_id = Column(SMALLINT, nullable=False)
    segment_name = Column(VARCHAR(100), nullable=False)
    percentage_share = Column(NUMERIC(5, 2))
    persona_description = Column(TEXT, nullable=False)
    default_play = Column(TEXT, nullable=False)
    play_owner = Column(VARCHAR(100), nullable=False)
    target_kpi = Column(VARCHAR(200), nullable=False)
    guardrail_policy = Column(TEXT)
    centroid_json = Column(JSONB)
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class CustomerActionRecord(Base):
    __tablename__ = "customer_actions"
    __table_args__ = (
        Index("idx_actions_account", "account_id"),
        Index("idx_actions_status", "status"),
        {"schema": "decision"},
    )

    action_id = Column(BIGINT, primary_key=True, autoincrement=True)
    account_id = Column(INT, ForeignKey("ods.accounts.account_id"), nullable=False)
    segment_id = Column(SMALLINT, nullable=False)
    recommended_play = Column(VARCHAR(200), nullable=False)
    channel = Column(VARCHAR(50), nullable=False, default="EMAIL")
    status = Column(VARCHAR(30), nullable=False, default="RECOMMENDED")
    dispatched_at = Column(TIMESTAMPTZ)
    responded_at = Column(TIMESTAMPTZ)
    financial_impact_amt = Column(NUMERIC(12, 2))
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


# ============================================================================
# Domain 6: Inference Observability & Auditing (audit)
# ============================================================================


class InferenceLogRecord(Base):
    __tablename__ = "inference_log"
    __table_args__ = (
        Index("idx_audit_created", "created_at"),
        {"schema": "audit"},
    )

    request_id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()")
    )
    model_version = Column(VARCHAR(50), nullable=False)
    as_of_date = Column(VARCHAR(30), nullable=False)
    account_id = Column(INT, nullable=True)
    input_payload_json = Column(JSONB, nullable=False)
    predicted_segment_id = Column(SMALLINT, nullable=False)
    predicted_segment_name = Column(VARCHAR(100), nullable=False)
    confidence_score = Column(FLOAT)
    guardrail_triggered = Column(BOOLEAN, default=False)
    latency_ms = Column(FLOAT)
    client_ip = Column(VARCHAR(50))
    created_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))


class DriftSnapshotRecord(Base):
    __tablename__ = "drift_snapshots"
    __table_args__: ClassVar[dict[str, Any]] = {"schema": "audit"}

    snapshot_id = Column(INT, primary_key=True, autoincrement=True)
    model_version = Column(VARCHAR(50), nullable=False)
    window_start = Column(TIMESTAMPTZ, nullable=False)
    window_end = Column(TIMESTAMPTZ, nullable=False)
    batch_size = Column(INT, nullable=False)
    observed_segment_mix = Column(JSONB, nullable=False)
    baseline_segment_mix = Column(JSONB, nullable=False)
    max_segment_drift_delta = Column(FLOAT, nullable=False)
    drift_status = Column(VARCHAR(20), nullable=False, default="NORMAL")
    evaluated_at = Column(TIMESTAMPTZ, server_default=text("NOW()"))
