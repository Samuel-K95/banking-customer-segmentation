"""FastAPI app — champion model, decisions, reports, inference."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from customer_segmentation import __version__
from customer_segmentation.serve.routers import (
    comparison,
    decisions,
    inference,
    meta,
    model,
    reports,
)

app = FastAPI(
    title="Banking Customer Segmentation API",
    description=(
        "Production-candidate champion API: model introspection, decision layer, "
        "reports, single/batch inference with serving-contract enforcement."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/v1"

app.include_router(meta.router, prefix=API_PREFIX)
app.include_router(model.router, prefix=API_PREFIX)
app.include_router(decisions.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(inference.router, prefix=API_PREFIX)
app.include_router(comparison.router, prefix=API_PREFIX)


@app.get("/")
def root() -> dict:
    return {
        "service": "banking-customer-segmentation",
        "version": __version__,
        "health": "/v1/health",
        "catalog": "/v1/catalog",
        "docs": "/docs",
    }
