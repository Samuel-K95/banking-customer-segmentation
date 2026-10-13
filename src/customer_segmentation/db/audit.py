"""Audit logging utilities for API inference and monitoring."""

from __future__ import annotations

import logging
from typing import Any

from customer_segmentation.db.connection import get_db_session
from customer_segmentation.db.models import InferenceLogRecord

logger = logging.getLogger(__name__)


def log_inference_requests(
    scored_records: list[dict[str, Any]],
    raw_payloads: list[dict[str, Any]],
    *,
    model_version: str,
    as_of_date: str,
    latency_ms: float = 0.0,
    client_ip: str | None = None,
) -> None:
    """Best-effort persistence of inference requests to audit.inference_log.

    Fails silently so that database outages do not block real-time scoring.
    """
    try:
        with get_db_session() as session:
            for idx, scored in enumerate(scored_records):
                raw = raw_payloads[idx] if idx < len(raw_payloads) else {}
                acc_id = raw.get("account_id") or scored.get("account_id")
                try:
                    acc_id = int(acc_id) if acc_id is not None else None
                except (ValueError, TypeError):
                    acc_id = None

                rec = InferenceLogRecord(
                    model_version=model_version,
                    as_of_date=as_of_date,
                    account_id=acc_id,
                    input_payload_json=raw,
                    predicted_segment_id=int(scored.get("segment_id", 0)),
                    predicted_segment_name=str(scored.get("segment_name", "")),
                    confidence_score=float(scored.get("confidence", 1.0))
                    if scored.get("confidence") is not None
                    else None,
                    guardrail_triggered=bool(scored.get("guardrail", False)),
                    latency_ms=latency_ms,
                    client_ip=client_ip,
                )
                session.add(rec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to persist inference audit log: %s", exc)
