"""FastAPI — segment scoring via modular ChampionModel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from customer_segmentation import __version__

app = FastAPI(
    title="Banking Customer Segmentation and Behavioral Intelligence",
    version=__version__,
)


class FeatureRow(BaseModel):
    account_id: int | str | None = None
    Transaction_Count: float
    Account_Balance: float
    Total_Credit: float
    Total_Withdrawal: float
    Recency_Days: float
    Account_Age_Days: float
    has_loan: float = Field(ge=0, le=1)
    has_card: float = Field(ge=0, le=1)


class ScoreRequest(BaseModel):
    customers: list[FeatureRow]


@lru_cache(maxsize=1)
def _champion():
    from customer_segmentation.models.champion import ChampionModel

    # Prefer cwd artifact; fall back to package-relative if needed
    for cand in (
        ChampionModel.default_dir("."),
        Path(__file__).resolve().parents[3] / "artifacts" / "segmentation" / "champion_v1",
    ):
        if cand.exists():
            return ChampionModel.load(cand)
    raise FileNotFoundError("champion_v1 artifacts not found")


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/v1/champion")
def champion_info() -> dict[str, Any]:
    try:
        return _champion().summary()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/segments")
def list_segments() -> dict[str, Any]:
    try:
        champ = _champion()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"segments": list(champ.catalog.get("segments", {}).values())}


@app.get("/v1/segments/{segment_id}/applications")
def segment_applications(segment_id: int) -> dict[str, Any]:
    try:
        champ = _champion()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    meta = champ.catalog.get("segments", {}).get(str(segment_id))
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown segment_id={segment_id}")
    return {
        "segment_id": segment_id,
        "name": meta.get("name"),
        "application": meta.get("decision"),
        "one_liner": meta.get("one_liner"),
    }


@app.post("/v1/customers/segment")
def score_customers(body: ScoreRequest) -> dict[str, Any]:
    try:
        champ = _champion()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    import pandas as pd

    rows = [c.model_dump() for c in body.customers]
    df = pd.DataFrame(rows)
    scored = champ.score_interpretable(df, id_col="account_id")
    return {"model": champ.run_id, "n": len(scored), "assignments": scored.to_dict(orient="records")}
