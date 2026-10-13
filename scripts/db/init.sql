-- ----------------------------------------------------------------------------
-- Banking Customer Intelligence Platform (BCIP) - Database Initialization
-- Engine: PostgreSQL 16 with pgvector extension
-- ----------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 1. Create Logical Schemas
CREATE SCHEMA IF NOT EXISTS ods;
CREATE SCHEMA IF NOT EXISTS feature_store;
CREATE SCHEMA IF NOT EXISTS ml_registry;
CREATE SCHEMA IF NOT EXISTS segmentation;
CREATE SCHEMA IF NOT EXISTS decision;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO public, ods, feature_store, ml_registry, segmentation, decision, audit;

-- ============================================================================
-- Domain 1: Operational Data Store (ods)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ods.districts (
    district_id INT PRIMARY KEY,
    district_name VARCHAR(100) NOT NULL,
    region VARCHAR(100) NOT NULL,
    n_inhabitants INT,
    n_municipalities_gt_499 INT,
    n_municipalities_500_1999 INT,
    n_municipalities_2000_9999 INT,
    n_municipalities_gt_10000 INT,
    n_cities INT,
    ratio_urban_inhabitants NUMERIC(6,2),
    avg_salary NUMERIC(12,2),
    unemployment_rate_95 NUMERIC(6,2),
    unemployment_rate_96 NUMERIC(6,2),
    n_entrepreneurs_per_1000 INT,
    n_crimes_95 INT,
    n_crimes_96 INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ods.clients (
    client_id INT PRIMARY KEY,
    birth_date DATE NOT NULL,
    gender VARCHAR(1) NOT NULL CHECK (gender IN ('M', 'F')),
    district_id INT REFERENCES ods.districts(district_id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ods.accounts (
    account_id INT PRIMARY KEY,
    district_id INT REFERENCES ods.districts(district_id),
    frequency VARCHAR(50) NOT NULL,
    created_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ods.dispositions (
    disp_id INT PRIMARY KEY,
    client_id INT NOT NULL REFERENCES ods.clients(client_id),
    account_id INT NOT NULL REFERENCES ods.accounts(account_id),
    disp_type VARCHAR(20) NOT NULL CHECK (disp_type IN ('OWNER', 'DISPONENT')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ods.transactions (
    trans_id INT PRIMARY KEY,
    account_id INT NOT NULL REFERENCES ods.accounts(account_id),
    trans_date DATE NOT NULL,
    trans_type VARCHAR(20) NOT NULL,
    operation VARCHAR(50),
    amount NUMERIC(12,2) NOT NULL,
    balance_after NUMERIC(12,2) NOT NULL,
    k_symbol VARCHAR(50),
    bank_partner VARCHAR(20),
    account_partner INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trans_account_date ON ods.transactions(account_id, trans_date);

CREATE TABLE IF NOT EXISTS ods.cards (
    card_id INT PRIMARY KEY,
    disp_id INT NOT NULL REFERENCES ods.dispositions(disp_id),
    card_type VARCHAR(20) NOT NULL CHECK (card_type IN ('classic', 'junior', 'gold')),
    issued_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ods.loans (
    loan_id INT PRIMARY KEY,
    account_id INT NOT NULL REFERENCES ods.accounts(account_id),
    loan_date DATE NOT NULL,
    amount NUMERIC(12,2) NOT NULL,
    duration_months INT NOT NULL,
    monthly_payment NUMERIC(12,2) NOT NULL,
    status VARCHAR(5) NOT NULL CHECK (status IN ('A', 'B', 'C', 'D')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Domain 2: Customer 360 Feature Store (feature_store)
-- ============================================================================

CREATE TABLE IF NOT EXISTS feature_store.account_features (
    feature_id BIGSERIAL PRIMARY KEY,
    account_id INT NOT NULL REFERENCES ods.accounts(account_id),
    as_of_date DATE NOT NULL,
    feature_version VARCHAR(50) NOT NULL,
    tenure_days INT NOT NULL,
    recency_days INT NOT NULL,
    transaction_count INT NOT NULL,
    account_balance NUMERIC(14,2) NOT NULL,
    total_credit NUMERIC(14,2) NOT NULL,
    total_withdrawal NUMERIC(14,2) NOT NULL,
    has_loan SMALLINT NOT NULL DEFAULT 0,
    has_card SMALLINT NOT NULL DEFAULT 0,
    features_json JSONB,
    is_eligible_for_segmentation BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_account_as_of_version UNIQUE (account_id, as_of_date, feature_version)
);
CREATE INDEX IF NOT EXISTS idx_features_account_asof ON feature_store.account_features(account_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_features_asof ON feature_store.account_features USING brin(as_of_date);

-- ============================================================================
-- Domain 3: ML Model Registry & Governance (ml_registry)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_registry.models (
    model_id VARCHAR(100) PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    architecture VARCHAR(100) NOT NULL,
    dataset_track VARCHAR(50) NOT NULL,
    latent_dim INT,
    k_clusters INT NOT NULL,
    as_of_date DATE NOT NULL,
    minimum_tenure_days INT DEFAULT 365,
    artifact_uri TEXT NOT NULL,
    artifact_sha256_hashes JSONB NOT NULL,
    serving_contract_json JSONB NOT NULL,
    is_champion BOOLEAN DEFAULT FALSE,
    freeze_status VARCHAR(30) DEFAULT 'PROVISIONAL',
    promoted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_registry.evaluations (
    eval_id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL REFERENCES ml_registry.models(model_id),
    holdout_silhouette FLOAT,
    holdout_db_index FLOAT,
    holdout_ch_index FLOAT,
    stability_ari FLOAT,
    low_confidence_frac FLOAT,
    metrics_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Domain 4: Segmentation & Latent Embeddings (segmentation)
-- ============================================================================

CREATE TABLE IF NOT EXISTS segmentation.assignments (
    assignment_id BIGSERIAL PRIMARY KEY,
    account_id INT NOT NULL REFERENCES ods.accounts(account_id),
    model_id VARCHAR(100) NOT NULL REFERENCES ml_registry.models(model_id),
    as_of_date DATE NOT NULL,
    segment_id SMALLINT NOT NULL,
    segment_name VARCHAR(100) NOT NULL,
    confidence_score FLOAT,
    is_guardrail_exception BOOLEAN DEFAULT FALSE,
    guardrail_reason VARCHAR(255),
    latent_vector vector(4),
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_account_model_asof UNIQUE (account_id, model_id, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_assignments_segment ON segmentation.assignments(segment_id);

-- ============================================================================
-- Domain 5: Decision Catalog & Closed-Loop Tracking (decision)
-- ============================================================================

CREATE TABLE IF NOT EXISTS decision.segment_catalog (
    catalog_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    segment_id SMALLINT NOT NULL,
    segment_name VARCHAR(100) NOT NULL,
    percentage_share NUMERIC(5,2),
    persona_description TEXT NOT NULL,
    default_play TEXT NOT NULL,
    play_owner VARCHAR(100) NOT NULL,
    target_kpi VARCHAR(200) NOT NULL,
    guardrail_policy TEXT,
    centroid_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_model_segment UNIQUE (model_version, segment_id)
);

CREATE TABLE IF NOT EXISTS decision.customer_actions (
    action_id BIGSERIAL PRIMARY KEY,
    account_id INT NOT NULL REFERENCES ods.accounts(account_id),
    segment_id SMALLINT NOT NULL,
    recommended_play VARCHAR(200) NOT NULL,
    channel VARCHAR(50) NOT NULL DEFAULT 'EMAIL',
    status VARCHAR(30) NOT NULL DEFAULT 'RECOMMENDED' CHECK (status IN ('RECOMMENDED', 'SENT', 'VIEWED', 'ACCEPTED', 'REJECTED', 'CURED', 'EXPIRED')),
    dispatched_at TIMESTAMPTZ,
    responded_at TIMESTAMPTZ,
    financial_impact_amt NUMERIC(12,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_actions_account ON decision.customer_actions(account_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON decision.customer_actions(status);

-- ============================================================================
-- Domain 6: Inference Observability & Auditing (audit)
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit.inference_log (
    request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    as_of_date VARCHAR(30) NOT NULL,
    account_id INT,
    input_payload_json JSONB NOT NULL,
    predicted_segment_id SMALLINT NOT NULL,
    predicted_segment_name VARCHAR(100) NOT NULL,
    confidence_score FLOAT,
    guardrail_triggered BOOLEAN DEFAULT FALSE,
    latency_ms FLOAT,
    client_ip VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit.inference_log(created_at);

CREATE TABLE IF NOT EXISTS audit.drift_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    batch_size INT NOT NULL,
    observed_segment_mix JSONB NOT NULL,
    baseline_segment_mix JSONB NOT NULL,
    max_segment_drift_delta FLOAT NOT NULL,
    drift_status VARCHAR(20) NOT NULL DEFAULT 'NORMAL' CHECK (drift_status IN ('NORMAL', 'WARNING', 'CRITICAL')),
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Permissions
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'bcip_admin') THEN
        GRANT ALL ON ALL TABLES IN SCHEMA ods TO bcip_admin;
        GRANT ALL ON ALL TABLES IN SCHEMA feature_store TO bcip_admin;
        GRANT ALL ON ALL TABLES IN SCHEMA ml_registry TO bcip_admin;
        GRANT ALL ON ALL TABLES IN SCHEMA segmentation TO bcip_admin;
        GRANT ALL ON ALL TABLES IN SCHEMA decision TO bcip_admin;
        GRANT ALL ON ALL TABLES IN SCHEMA audit TO bcip_admin;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA ods TO bcip_admin;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA feature_store TO bcip_admin;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA ml_registry TO bcip_admin;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA segmentation TO bcip_admin;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA decision TO bcip_admin;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA audit TO bcip_admin;
    END IF;
END $$;
