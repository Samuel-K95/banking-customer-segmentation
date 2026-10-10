"""FastAPI skeleton — frontend must call these endpoints only."""

from __future__ import annotations

from fastapi import FastAPI

from customer_segmentation import __version__

app = FastAPI(
    title="Banking Customer Segmentation and Behavioral Intelligence",
    version=__version__,
)


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/v1/segments")
def list_segments() -> dict[str, list]:
    return {"segments": []}
