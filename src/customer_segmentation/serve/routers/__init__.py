"""API routers for the champion service."""

from customer_segmentation.serve.routers import (
    comparison,
    decisions,
    inference,
    meta,
    model,
    reports,
)

__all__ = ["comparison", "decisions", "inference", "meta", "model", "reports"]
