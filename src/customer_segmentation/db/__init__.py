"""Database package for Banking Customer Intelligence Platform (BCIP)."""

from customer_segmentation.db.connection import (
    Base,
    check_db_connection,
    get_database_url,
    get_db_session,
    get_engine,
)
from customer_segmentation.db.models import (
    Account,
    AccountFeature,
    AssignmentRecord,
    Client,
    CustomerActionRecord,
    District,
    DriftSnapshotRecord,
    EvaluationRecord,
    InferenceLogRecord,
    ModelRecord,
    SegmentCatalogRecord,
    Transaction,
)

__all__ = [
    "Account",
    "AccountFeature",
    "AssignmentRecord",
    "Base",
    "Client",
    "CustomerActionRecord",
    "District",
    "DriftSnapshotRecord",
    "EvaluationRecord",
    "InferenceLogRecord",
    "ModelRecord",
    "SegmentCatalogRecord",
    "Transaction",
    "check_db_connection",
    "get_database_url",
    "get_db_session",
    "get_engine",
]
