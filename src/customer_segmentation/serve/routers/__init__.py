"""API routers for the champion service."""

from customer_segmentation.serve.routers import (
    decisions,
    inference,
    meta,
    model,
    reports,
)

__all__ = ["decisions", "inference", "meta", "model", "reports"]
