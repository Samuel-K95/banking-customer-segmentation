"""Expose champion / explore comparison boards as JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from customer_segmentation.serve.deps import resolve_path

router = APIRouter(tags=["comparison"])

_BOARDS: dict[str, Path] = {
    "champion": Path("artifacts") / "segmentation" / "champion" / "comparison_board.csv",
    "explore": Path("artifacts")
    / "experiments"
    / "berka_explore"
    / "comparison_board_explore.csv",
}


@router.get("/comparison/board")
def comparison_board(
    board: str = Query("champion", description="champion | explore"),
) -> dict[str, Any]:
    key = board.strip().lower()
    if key not in _BOARDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown board={board!r}; expected one of {sorted(_BOARDS)}",
        )
    path = resolve_path(_BOARDS[key])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Comparison board missing: {path}")
    frame = pd.read_csv(path)
    rows = frame.where(pd.notnull(frame), None).to_dict(orient="records")
    return {
        "board": key,
        "path": str(path).replace("\\", "/"),
        "n": len(rows),
        "columns": list(frame.columns),
        "rows": rows,
    }
