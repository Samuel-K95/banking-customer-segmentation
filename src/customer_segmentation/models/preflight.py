"""Memory-conscious validation of processed feature matrices before modeling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype


def _append_unique(items: list[str], message: str) -> None:
    if message not in items:
        items.append(message)


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return next(csv.reader(csv_file), [])


def _identifier_key(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _scan_csv(
    path: Path,
    *,
    numeric_columns: list[str],
    identifier_columns: list[str],
    binary_columns: list[str],
    inspected_features: list[str],
    chunk_size: int,
) -> dict[str, Any]:
    """Scan selected columns only; use SQLite for disk-backed ID uniqueness."""
    result: dict[str, Any] = {
        "rows": 0,
        "identity_digest": None,
        "errors": [],
        "constant_features": [],
        "duplicate_features": [],
        "binary_values": {},
    }
    columns = list(dict.fromkeys([*numeric_columns, *identifier_columns]))
    if not columns:
        result["errors"].append("No contract columns are available to scan.")
        return result

    dtype = {column: "string" for column in identifier_columns}
    numeric_type_errors: set[str] = set()
    nonfinite_errors: set[str] = set()
    binary_invalid: dict[str, set[float | str]] = {column: set() for column in binary_columns}
    feature_hashes = {column: hashlib.sha256() for column in inspected_features}
    feature_values: dict[str, set[float]] = {column: set() for column in inspected_features}
    invalid_inspected: set[str] = set()

    database_path: str | None = None
    connection: sqlite3.Connection | None = None
    identity_hash = hashlib.sha256() if identifier_columns else None
    if identifier_columns:
        with tempfile.NamedTemporaryFile(prefix="ml-preflight-", suffix=".sqlite", delete=False) as temporary:
            database_path = temporary.name
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("CREATE TABLE ids (value TEXT PRIMARY KEY)")

    try:
        chunks = pd.read_csv(
            path,
            usecols=columns,
            dtype=dtype or None,
            chunksize=chunk_size,
        )
        for frame in chunks:
            count = len(frame)
            result["rows"] += count

            for column in numeric_columns:
                series = frame[column]
                if not is_numeric_dtype(series.dtype):
                    numeric_type_errors.add(column)
                converted = pd.to_numeric(series, errors="coerce")
                values = converted.to_numpy(dtype=np.float64, na_value=np.nan)
                finite = np.isfinite(values)
                if not finite.all():
                    nonfinite_errors.add(column)
                    if column in inspected_features:
                        invalid_inspected.add(column)

                if column in binary_invalid:
                    invalid_values = values[finite & (values != 0) & (values != 1)]
                    if invalid_values.size:
                        bucket = binary_invalid[column]
                        bucket.update(float(value) for value in np.unique(invalid_values)[:5])

                if column in feature_hashes and finite.all():
                    feature_hashes[column].update(np.asarray(values, dtype="<f8").tobytes())
                    known = feature_values[column]
                    if len(known) < 2:
                        for value in np.unique(values):
                            known.add(float(value))
                            if len(known) >= 2:
                                break

            if identifier_columns:
                id_arrays = [frame[column].astype("string").tolist() for column in identifier_columns]
                batch_keys: list[tuple[str]] = []
                for row in zip(*id_arrays):
                    if any(pd.isna(value) for value in row):
                        _append_unique(
                            result["errors"],
                            f"Identifier column(s) {identifier_columns} contain null values.",
                        )
                    values = tuple("<NULL>" if pd.isna(value) else str(value) for value in row)
                    key = _identifier_key(values)
                    batch_keys.append((key,))
                    identity_hash.update(key.encode("utf-8"))
                    identity_hash.update(b"\n")
                connection.executemany("INSERT OR IGNORE INTO ids(value) VALUES (?)", batch_keys)

        if result["rows"] == 0:
            result["errors"].append("CSV contains no data rows.")

        for column in sorted(numeric_type_errors):
            result["errors"].append(f"Column '{column}' is not numeric in the CSV.")
        for column in sorted(nonfinite_errors):
            result["errors"].append(f"Column '{column}' contains null, NaN, or infinite values.")

        if connection is not None:
            unique_count = connection.execute("SELECT COUNT(*) FROM ids").fetchone()[0]
            if unique_count != result["rows"]:
                result["errors"].append(
                    f"Identifier combination {identifier_columns} is not unique "
                    f"({unique_count} unique of {result['rows']} rows)."
                )
            result["identity_digest"] = identity_hash.hexdigest()

        result["constant_features"] = sorted(
            column
            for column, values in feature_values.items()
            if column not in invalid_inspected and result["rows"] > 0 and len(values) <= 1
        )
        digest_columns: dict[str, list[str]] = {}
        for column, digest in feature_hashes.items():
            if column not in invalid_inspected:
                digest_columns.setdefault(digest.hexdigest(), []).append(column)
        result["duplicate_features"] = sorted(
            group
            for group in digest_columns.values()
            if len(group) > 1
        )
        result["binary_values"] = {
            column: sorted(values, key=str) for column, values in binary_invalid.items() if values
        }
        for column, values in result["binary_values"].items():
            result["errors"].append(
                f"Binary feature '{column}' contains values outside {{0, 1}}: {values}."
            )
    except (OSError, pd.errors.ParserError, ValueError, sqlite3.Error) as exc:
        result["errors"].append(f"Could not scan CSV: {exc}")
    finally:
        if connection is not None:
            connection.close()
        if database_path and os.path.exists(database_path):
            os.unlink(database_path)

    return result


def _validate_track(
    track_key: str,
    spec: dict[str, Any],
    data_dir: Path,
    chunk_size: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    model_features = list(spec.get("model_features", []))
    profile_features = list(spec.get("profile_features", []))
    profile_categorical_features = list(spec.get("profile_categorical_features", []))
    identifiers = list(spec.get("identifier_columns", []))
    binary_features = list(spec.get("binary_features", []))
    excluded = set(spec.get("excluded_features", []))
    patterns = [re.compile(pattern) for pattern in spec.get("excluded_column_patterns", [])]

    if not model_features:
        errors.append("Contract must define at least one model feature.")
    if len(model_features) != len(set(model_features)):
        errors.append("Contract contains duplicate model feature names.")
    if set(model_features) & set(identifiers):
        errors.append("Identifier columns must not appear in model_features.")
    for feature in model_features:
        if feature in excluded or any(pattern.search(feature) for pattern in patterns):
            errors.append(f"Excluded/profile-only feature '{feature}' appears in model_features.")
    if not set(binary_features).issubset(model_features):
        errors.append("binary_features must be a subset of model_features.")

    model_path = data_dir / spec["model_file"]
    interpretable_path = data_dir / spec["interpretable_file"]
    output: dict[str, Any] = {
        "name": spec.get("name", track_key),
        "grain": spec.get("grain", "unspecified"),
        "model_file": str(model_path),
        "interpretable_file": str(interpretable_path),
        "model_features": model_features,
        "binary_features_checked": [],
        "rows": {},
        "errors": errors,
        "warnings": warnings,
        "status": "FAIL",
    }

    file_specs = (
        ("model", model_path, [*identifiers, *model_features]),
        (
            "interpretable",
            interpretable_path,
            [*identifiers, *model_features, *profile_features, *profile_categorical_features],
        ),
    )
    scan_results: dict[str, dict[str, Any]] = {}
    for kind, path, required_columns in file_specs:
        if not path.is_file():
            errors.append(f"Missing expected {kind} file: {path}")
            continue
        try:
            header = _read_header(path)
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"Could not read {kind} CSV header at {path}: {exc}")
            continue

        duplicates = sorted({column for column in header if header.count(column) > 1})
        if duplicates:
            errors.append(f"{kind} CSV has duplicate header names: {duplicates}.")
        missing = [column for column in dict.fromkeys(required_columns) if column not in header]
        if missing:
            errors.append(f"{kind} CSV is missing required columns: {missing}.")

        numeric_required = [*model_features]
        if kind == "interpretable":
            numeric_required.extend(profile_features)
            for pattern in spec.get("profile_column_patterns", []):
                regex = re.compile(pattern)
                matches = [column for column in header if regex.search(column)]
                numeric_required.extend(matches)
                if not matches:
                    warnings.append(
                        f"No interpretable columns matched profile pattern '{pattern}'."
                    )
        numeric_columns = list(dict.fromkeys(column for column in numeric_required if column in header))
        available_identifiers = [column for column in identifiers if column in header]
        available_binary = [column for column in binary_features if column in header]
        if kind == "model":
            output["binary_features_checked"] = available_binary
        scan = _scan_csv(
            path,
            numeric_columns=numeric_columns,
            identifier_columns=available_identifiers,
            binary_columns=available_binary,
            inspected_features=[column for column in model_features if column in numeric_columns]
            if kind == "model"
            else [],
            chunk_size=chunk_size,
        )
        scan_results[kind] = scan
        output["rows"][kind] = scan["rows"]
        errors.extend(f"{kind} file: {message}" for message in scan["errors"])
        min_rows = int(spec.get("min_rows", 1))
        if scan["rows"] < min_rows:
            errors.append(
                f"{kind} file has {scan['rows']} rows; contract minimum is {min_rows}."
            )
        if kind == "model":
            if scan["constant_features"]:
                errors.append(f"Constant model features: {scan['constant_features']}.")
            if scan["duplicate_features"]:
                errors.append(f"Duplicate model feature columns: {scan['duplicate_features']}.")

    if "model" in scan_results and "interpretable" in scan_results:
        model_scan = scan_results["model"]
        interpretable_scan = scan_results["interpretable"]
        if model_scan["rows"] != interpretable_scan["rows"]:
            errors.append(
                "Model/interpretable row counts differ "
                f"({model_scan['rows']} vs {interpretable_scan['rows']})."
            )
        if identifiers:
            model_digest = model_scan["identity_digest"]
            interpretable_digest = interpretable_scan["identity_digest"]
            if model_digest is not None and interpretable_digest is not None:
                if model_digest != interpretable_digest:
                    errors.append("Identifier order does not align between model and interpretable files.")
            else:
                errors.append("Cannot verify model/interpretable identity alignment: identifier is missing.")

    if not errors:
        output["status"] = "PASS"
    return output


def run_preflight(
    config_path: str | Path,
    selected_tracks: list[str] | None = None,
) -> dict[str, Any]:
    """Run all contracts, or only selected track keys, in a YAML file."""
    config_path = Path(config_path).expanduser().resolve()
    report: dict[str, Any] = {
        "config": str(config_path),
        "data_dir": None,
        "tracks": {},
        "ready": False,
        "errors": [],
    }
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as exc:
        report["errors"].append(f"Could not load config: {exc}")
        return report

    data_dir_value = config.get("data_dir")
    if not data_dir_value:
        report["errors"].append("Config must define data_dir.")
        return report
    data_dir = Path(data_dir_value)
    if not data_dir.is_absolute():
        data_dir = (config_path.parent / data_dir).resolve()
    report["data_dir"] = str(data_dir)

    tracks = config.get("tracks")
    if not isinstance(tracks, dict) or not tracks:
        report["errors"].append("Config must define a non-empty tracks mapping.")
        return report

    if selected_tracks is not None:
        selected_tracks = list(dict.fromkeys(selected_tracks))
        if not selected_tracks:
            report["errors"].append("At least one track must be selected.")
            return report
        unknown_tracks = [track for track in selected_tracks if track not in tracks]
        if unknown_tracks:
            report["errors"].append(f"Unknown track key(s): {unknown_tracks}.")
            return report
        tracks = {track: tracks[track] for track in selected_tracks}

    try:
        chunk_size = int(config.get("chunk_size", 50000))
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    except (TypeError, ValueError) as exc:
        report["errors"].append(f"Invalid chunk_size: {exc}")
        return report

    for track_key, spec in tracks.items():
        try:
            report["tracks"][track_key] = _validate_track(
                track_key, spec, data_dir, chunk_size
            )
        except (KeyError, TypeError, ValueError, re.error) as exc:
            report["tracks"][track_key] = {
                "name": spec.get("name", track_key) if isinstance(spec, dict) else track_key,
                "status": "FAIL",
                "errors": [f"Invalid track contract: {exc}"],
                "warnings": [],
                "rows": {},
                "model_features": [],
            }

    report["ready"] = all(track["status"] == "PASS" for track in report["tracks"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/models/feature_contracts.yaml",
        help="Path to the feature contract YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        metavar="TRACK",
        help="Validate selected track keys only (default: validate all configured tracks).",
    )
    args = parser.parse_args(argv)
    report = run_preflight(args.config, selected_tracks=args.tracks)

    print("ML preflight: " + ("READY" if report["ready"] else "BLOCKED"))
    if report["data_dir"]:
        print(f"Processed data: {report['data_dir']}")
    for message in report["errors"]:
        print(f"ERROR: {message}")
    for key, track in report["tracks"].items():
        row_counts = ", ".join(f"{kind}={count}" for kind, count in track.get("rows", {}).items())
        print(f"\n[{track['status']}] {key}: {track.get('name', key)} ({track.get('grain', 'unspecified')})")
        if row_counts:
            print(f"Rows: {row_counts}")
        print(f"Baseline inputs ({len(track.get('model_features', []))}): {', '.join(track.get('model_features', []))}")
        for warning in track.get("warnings", []):
            print(f"WARNING: {warning}")
        for message in track.get("errors", []):
            print(f"ERROR: {message}")

    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
