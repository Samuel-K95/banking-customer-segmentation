"""Champion consistency verifier — catches reviewer-class integrity gaps."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from customer_segmentation.models.artifact_integrity import integrity_report
from customer_segmentation.pipelines.promote_production import PRODUCTION_DIR

DECISION_DIR = Path("artifacts/decision/champion")

# Paths scanned for stale references (active product surface only).
SCAN_GLOBS = (
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "scripts/**/*.sh",
    "src/customer_segmentation/**/*.py",
    "docs/api.md",
    "reports/champion/*.md",
    "CURRENT_CHAMPION.md",
    "README.md",
)

FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("stale_path_champion_v1_dir", re.compile(r"artifacts/segmentation/champion_v1")),
    ("stale_path_decision_production", re.compile(r"artifacts/decision/production")),
    ("stale_path_segmentation_product", re.compile(r"artifacts/segmentation/product/")),
    (
        "docker_old_bakeoff_entrypoint",
        re.compile(r"run_berka_segmentation[^_]|pipelines\.run_berka_segmentation\b"),
    ),
]

# Allow known historical / research mentions in specific files.
ALLOWLIST_FILES = frozenset(
    {
        "src/customer_segmentation/pipelines/run_berka_segmentation.py",
        "src/customer_segmentation/pipelines/run_berka_segmentation_leakage_safe.py",
        "src/customer_segmentation/pipelines/verify_champion.py",  # defines forbidden patterns
        "src/customer_segmentation/cli.py",  # segment-berka still documents archive bake-off
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _check_bundle(champion_dir: Path) -> dict[str, Any]:
    report = integrity_report(champion_dir)
    return {
        "name": "bundle_integrity",
        "ok": report["ok"],
        "errors": report["errors"],
        "missing_required": report.get("missing_required") or [],
    }


def _check_load(champion_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        from customer_segmentation.models.champion import ChampionModel

        champ = ChampionModel.load(champion_dir, verify_hashes=True)
        if champ.encoder is None:
            errors.append("ChampionModel loaded without encoder")
        if champ.scaler is None:
            errors.append("ChampionModel loaded without scaler")
        if champ.kmeans is None:
            errors.append("ChampionModel loaded without kmeans")
        summary = {
            "run_id": champ.run_id,
            "model_version": champ.model_version,
            "has_encoder": champ.encoder is not None,
            "n_clusters": champ.n_clusters,
        }
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        summary = {}
    return {"name": "loadability", "ok": len(errors) == 0, "errors": errors, "summary": summary}


def _check_decision(decision_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    summary_path = decision_dir / "run_summary.json"
    map_path = decision_dir / "decision_map_full.csv"
    stale_markers = (
        "decision/production",
        "segmentation/champion_v1",
        "segmentation/product/",
    )
    if not summary_path.exists():
        errors.append("decision run_summary.json missing")
    else:
        raw = summary_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        acceptance = payload.get("acceptance") or {}
        for key, val in acceptance.items():
            if val is False:
                errors.append(f"decision acceptance flag false: {key}")
        if int(payload.get("n_plays") or 0) < 3:
            errors.append("decision n_plays < 3")
        artifact_dir = str(payload.get("artifact_dir") or "").replace("\\", "/")
        if "artifacts/decision/champion" not in artifact_dir and decision_dir.name == "champion":
            errors.append(f"decision artifact_dir stale/unexpected: {artifact_dir}")
        for marker in stale_markers:
            if marker in raw.replace("\\", "/"):
                errors.append(f"decision run_summary contains stale path marker: {marker}")
        for key, rel in (payload.get("files") or {}).items():
            path = Path(str(rel).replace("\\", "/"))
            if not path.is_absolute():
                cand = Path(path)
            else:
                cand = path
            if not cand.exists() and not (Path.cwd() / path).exists():
                # also try under decision_dir by filename
                if not (decision_dir / Path(str(rel)).name).exists():
                    errors.append(f"decision files[{key}] missing: {rel}")
    if not map_path.exists():
        errors.append("decision_map_full.csv missing")
    else:
        lines = map_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            errors.append("decision_map_full.csv empty")
    return {"name": "decision_layer", "ok": len(errors) == 0, "errors": errors}


def _check_stale_paths(root: Path) -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    for pattern in SCAN_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in ALLOWLIST_FILES:
                continue
            # skip archive write-ups and leakage-safe report generators that mention history
            if "/archive/" in rel:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, regex in FORBIDDEN_PATTERNS:
                if label == "docker_old_bakeoff_entrypoint" and "run_segmentation_layer.sh" not in rel:
                    # only enforce old bake-off ban on the docker entrypoint script
                    if not rel.endswith("run_segmentation_layer.sh"):
                        continue
                for i, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        # allow comments describing migration away from old paths in docs
                        if "archive" in line.lower() and "champion_v1" in line:
                            continue
                        hits.append({"file": rel, "line": str(i), "rule": label, "text": line.strip()[:200]})
    return {
        "name": "stale_path_scan",
        "ok": len(hits) == 0,
        "errors": [f"{h['file']}:{h['line']} [{h['rule']}] {h['text']}" for h in hits],
        "hits": hits,
    }


def _check_api_smoke(champion_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        from fastapi.testclient import TestClient

        from customer_segmentation.serve.app import app
        from customer_segmentation.serve.deps import get_champion

        get_champion.cache_clear()
        # Ensure resolve finds the champion under cwd
        os.chdir(champion_dir.parents[2] if (champion_dir.parents[2] / "pyproject.toml").exists() else Path.cwd())
        with TestClient(app) as client:
            health = client.get("/v1/health")
            if health.status_code != 200:
                errors.append(f"/v1/health -> {health.status_code}: {health.text[:300]}")
            else:
                body = health.json()
                if not body.get("has_encoder"):
                    errors.append("/v1/health missing has_encoder=true")

            contract = client.get("/v1/inference/contract").json()
            mv = contract["model_version"]
            asof = contract["as_of_date"]
            bad = client.post(
                "/v1/inference",
                json={
                    "model_version": mv,
                    "as_of_date": asof,
                    "Transaction_Count": 10,
                    "Account_Balance": 1000,
                    "Total_Credit": 2000,
                    "Total_Withdrawal": 500,
                    "Recency_Days": 1,
                    "Account_Age_Days": 30,
                    "has_loan": 0,
                    "has_card": 0,
                },
            )
            if bad.status_code != 400:
                errors.append(f"under-tenure expected 400, got {bad.status_code}")

            good = client.post(
                "/v1/inference",
                json={
                    "model_version": mv,
                    "as_of_date": asof,
                    "Transaction_Count": 120,
                    "Account_Balance": 45000,
                    "Total_Credit": 80000,
                    "Total_Withdrawal": 35000,
                    "Recency_Days": 5,
                    "Account_Age_Days": 900,
                    "has_loan": 0,
                    "has_card": 1,
                },
            )
            if good.status_code != 200:
                errors.append(f"valid inference expected 200, got {good.status_code}: {good.text[:300]}")
        get_champion.cache_clear()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"API smoke failed: {exc}")
    return {"name": "api_smoke", "ok": len(errors) == 0, "errors": errors}


def _check_replay(champion_dir: Path) -> dict[str, Any]:
    """Holdout replay when Berka raw is available; otherwise skip (not fail)."""
    errors: list[str] = []
    skipped = False
    try:
        from customer_segmentation.data.berka_dataset_loader import resolve_berka_dir

        try:
            resolve_berka_dir(None)
        except Exception:
            return {
                "name": "holdout_replay",
                "ok": True,
                "skipped": True,
                "errors": [],
                "detail": "Berka raw not available; replay skipped",
            }

        import numpy as np
        import pandas as pd

        from customer_segmentation.data.berka_dataset_loader import (
            load_berka_product_tables,
            load_berka_raw_data,
        )
        from customer_segmentation.features.build_berka_segmentation_features import (
            build_berka_segmentation_features_asof,
            transform_with_train_scaler,
        )
        from customer_segmentation.models.champion import ChampionModel

        champ = ChampionModel.load(champion_dir)
        split = pd.read_csv(champion_dir / "account_split.csv")
        assign = pd.read_csv(champion_dir / "assignments.csv")
        split["account_id"] = split["account_id"].astype(np.int64)
        assign["account_id"] = assign["account_id"].astype(np.int64)
        hold_ids = set(split.loc[split["split"] == "holdout", "account_id"].tolist())
        hold = assign[assign["account_id"].isin(hold_ids)].sort_values("account_id")
        as_of = str((champ.metrics or {}).get("as_of_date") or "1997-12-31")
        min_tenure = int((champ.metrics or {}).get("minimum_tenure_days") or 365)
        path = resolve_berka_dir(None)
        trans, account, _ = load_berka_raw_data(path)
        loan, card, disp = load_berka_product_tables(path)
        feats = build_berka_segmentation_features_asof(
            trans, account, loan, card, disp, as_of_date=as_of, minimum_tenure_days=min_tenure
        )
        feats["account_id"] = feats["account_id"].astype(np.int64)
        feats = feats[feats["account_id"].isin(hold_ids)].sort_values("account_id")
        hold = hold[hold["account_id"].isin(set(feats["account_id"]))].sort_values("account_id")
        feats = feats.set_index("account_id").loc[hold["account_id"].to_numpy()].reset_index()
        X = transform_with_train_scaler(feats, champ.scaler)
        pred = champ.predict(X)
        stored = hold["segment_id"].astype(int).to_numpy()
        if not np.array_equal(pred, stored):
            errors.append("holdout replay labels do not match stored assignments")
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    return {
        "name": "holdout_replay",
        "ok": len(errors) == 0,
        "skipped": skipped,
        "errors": errors,
    }


def verify_champion(
    *,
    champion_dir: str | Path | None = None,
    decision_dir: str | Path | None = None,
    include_api: bool = True,
    include_replay: bool = True,
    include_path_scan: bool = True,
) -> dict[str, Any]:
    root = _repo_root()
    # Prefer cwd champion if present (developer checkout)
    cwd_champ = Path("artifacts/segmentation/champion")
    champ = Path(champion_dir) if champion_dir else (cwd_champ if cwd_champ.exists() else root / PRODUCTION_DIR)
    dec = Path(decision_dir) if decision_dir else (
        Path("artifacts/decision/champion")
        if Path("artifacts/decision/champion").exists()
        else root / DECISION_DIR
    )

    checks = [
        _check_bundle(champ),
        _check_load(champ),
        _check_decision(dec),
    ]
    if include_path_scan:
        checks.append(_check_stale_paths(root if (root / "pyproject.toml").exists() else Path.cwd()))
    if include_api:
        checks.append(_check_api_smoke(champ))
    if include_replay:
        checks.append(_check_replay(champ))

    ok = all(c.get("ok") for c in checks)
    return {
        "ok": ok,
        "champion_dir": str(champ),
        "decision_dir": str(dec),
        "checks": checks,
    }


def main() -> int:
    report = verify_champion()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
