"""Champion bundle integrity: required files + SHA-256 vs serving contract."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "scaler.joblib",
    "kmeans.joblib",
    "encoder.pt",
    "manifest.json",
    "metrics.json",
    "feature_columns.json",
    "segment_catalog.json",
    "assignments.csv",
    "account_split.csv",
)

REQUIRED_BINARY_HASH_KEYS: tuple[str, ...] = (
    "scaler.joblib",
    "kmeans.joblib",
    "encoder.pt",
)


def compute_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text_file_to_lf(path: Path) -> bool:
    """Rewrite path with LF newlines. Return True if bytes changed."""
    if not path.exists() or not path.is_file():
        return False
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized == raw:
        return False
    path.write_bytes(normalized)
    return True


TEXT_HASHED_ARTIFACTS: tuple[str, ...] = (
    "account_split.csv",
    "assignments.csv",
)


def sync_serving_contract_hashes(
    bundle_dir: str | Path,
    *,
    normalize_text: bool = True,
) -> dict[str, Any]:
    """Recompute serving_contract.artifact_sha256 from on-disk bytes (LF-normalized CSVs)."""
    root = Path(bundle_dir)
    contract_path = root / "serving_contract.json"
    contract = read_serving_contract(root) if contract_path.exists() else {}
    if normalize_text:
        for name in TEXT_HASHED_ARTIFACTS:
            normalize_text_file_to_lf(root / name)

    hashes: dict[str, str] = {}
    missing: list[str] = []
    for name in (*REQUIRED_BINARY_HASH_KEYS, *TEXT_HASHED_ARTIFACTS):
        digest = compute_sha256(root / name)
        if digest is None:
            missing.append(name)
        else:
            hashes[name] = digest
    if missing:
        raise FileNotFoundError(
            f"Cannot sync serving_contract hashes; missing: {', '.join(missing)}"
        )

    contract = dict(contract)
    contract["artifact_sha256"] = hashes
    contract.setdefault("model_version", "production_candidate_v1")
    contract.setdefault("as_of_date", "1997-12-31")
    contract.setdefault("minimum_tenure_days", 365)
    from datetime import datetime

    contract["updated_at"] = datetime.now(UTC).isoformat()
    contract_path.write_text(json.dumps(contract, indent=2, default=str) + "\n", encoding="utf-8")
    return contract


def missing_required_artifacts(bundle_dir: str | Path) -> list[str]:
    root = Path(bundle_dir)
    missing: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        path = root / name
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            missing.append(name)
    return missing


def preflight_bundle(bundle_dir: str | Path, *, load_binaries: bool = True) -> list[str]:
    """Return human-readable errors if the bundle is incomplete or unloadable."""
    root = Path(bundle_dir)
    errors: list[str] = []
    if not root.exists():
        return [f"Bundle directory missing: {root}"]

    for name in missing_required_artifacts(root):
        errors.append(f"Missing or empty required artifact: {name}")

    if errors or not load_binaries:
        return errors

    try:
        import joblib

        joblib.load(root / "scaler.joblib")
        joblib.load(root / "kmeans.joblib")
    except Exception as exc:  # noqa: BLE001 — surface any load failure
        errors.append(f"Failed to load scaler/kmeans: {exc}")

    try:
        import torch

        from customer_segmentation.models.tabular_ae import TORCH_OK

        if not TORCH_OK:
            errors.append("PyTorch not available; cannot load encoder.pt")
        else:
            torch.load(root / "encoder.pt", map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Failed to load encoder.pt: {exc}")

    return errors


def read_serving_contract(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir) / "serving_contract.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_serving_contract(
    bundle_dir: str | Path,
    contract: dict[str, Any] | None = None,
    *,
    require_binary_hashes: bool = True,
) -> list[str]:
    """Recompute SHA-256 and compare to serving_contract.artifact_sha256."""
    root = Path(bundle_dir)
    errors: list[str] = []
    contract = contract if contract is not None else read_serving_contract(root)
    if not contract:
        errors.append("serving_contract.json missing")
        return errors

    declared = contract.get("artifact_sha256") or {}
    if not isinstance(declared, dict) or not declared:
        errors.append("serving_contract.artifact_sha256 missing or empty")
        return errors

    if require_binary_hashes:
        for key in REQUIRED_BINARY_HASH_KEYS:
            digest = declared.get(key)
            if digest is None or digest == "":
                errors.append(f"serving_contract hash missing/null for required binary: {key}")

    for name, expected in declared.items():
        if expected is None or expected == "":
            # already flagged for required binaries; skip optional nulls
            continue
        path = root / str(name)
        actual = compute_sha256(path)
        if actual is None:
            errors.append(f"Declared artifact missing on disk: {name}")
        elif actual != expected:
            errors.append(
                f"SHA-256 mismatch for {name}: declared={expected} actual={actual}"
            )
    return errors


def integrity_report(bundle_dir: str | Path) -> dict[str, Any]:
    """Existence → SHA-256 vs contract → only then deserialize binaries."""
    root = Path(bundle_dir)
    existence = preflight_bundle(root, load_binaries=False)
    sha_errors = verify_serving_contract(root) if not existence else []
    load_errors: list[str] = []
    if not existence and not sha_errors:
        load_errors = preflight_bundle(root, load_binaries=True)
    errors = existence + sha_errors + load_errors
    return {
        "bundle_dir": str(root),
        "ok": len(errors) == 0,
        "errors": errors,
        "missing_required": missing_required_artifacts(root),
    }
