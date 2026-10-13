from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from customer_segmentation.models.preflight import main, run_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "configs" / "models" / "feature_contracts.yaml"


def _write_synthetic_data(tmp_path: Path) -> tuple[Path, dict]:
    with DEFAULT_CONTRACT.open(encoding="utf-8") as contract_file:
        contract = yaml.safe_load(contract_file)

    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    contract["data_dir"] = "processed"
    contract["chunk_size"] = 2
    config_path = tmp_path / "feature_contracts.yaml"
    config_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    row_count = 6
    for track_key, spec in contract["tracks"].items():
        identifier = spec["identifier_columns"][0]
        frame = pd.DataFrame({identifier: [f"{track_key}-{index}" for index in range(row_count)]})

        for feature_index, feature in enumerate(spec["model_features"]):
            if feature in spec.get("binary_features", []):
                # Each flag gets a distinct, non-constant six-row binary pattern.
                pattern = feature_index + 1
                frame[feature] = [(pattern >> row) & 1 for row in range(row_count)]
            else:
                frame[feature] = np.arange(1, row_count + 1, dtype=float) * (feature_index + 1) + feature_index

        for profile_index, feature in enumerate(spec.get("profile_features", [])):
            if feature not in frame:
                frame[feature] = np.arange(row_count, dtype=float) + profile_index

        for categorical_feature in spec.get("profile_categorical_features", []):
            frame[categorical_feature] = ["H", "V", None, "H", "V", None]

        if track_key == "track_b":
            frame["total_products_owned"] = frame[spec["binary_features"]].sum(axis=1)
            frame["customer_segment_02 - PARTICULARES"] = [0, 1, 0, 1, 0, 1]

        model_columns = [identifier, *spec["model_features"]]
        interpretable_columns = list(frame.columns)
        (data_dir / spec["model_file"]).parent.mkdir(parents=True, exist_ok=True)  # per-track subdirectory
        frame[model_columns].to_csv(data_dir / spec["model_file"], index=False)
        frame[interpretable_columns].to_csv(data_dir / spec["interpretable_file"], index=False)

    return config_path, contract


def _messages(report: dict, track: str) -> str:
    return "\n".join(report["tracks"][track]["errors"])


def test_preflight_accepts_aligned_numeric_synthetic_tracks(tmp_path):
    config_path, contract = _write_synthetic_data(tmp_path)

    report = run_preflight(config_path)

    assert report["ready"] is True
    for track_key, spec in contract["tracks"].items():
        track_report = report["tracks"][track_key]
        assert track_report["status"] == "PASS"
        assert track_report["rows"] == {"model": 6, "interpretable": 6}
        assert track_report["binary_features_checked"] == spec["binary_features"]


def test_preflight_rejects_duplicate_identifiers_and_misaligned_rows(tmp_path):
    config_path, contract = _write_synthetic_data(tmp_path)
    data_dir = tmp_path / "processed"
    spec = contract["tracks"]["track_a"]
    model_path = data_dir / spec["model_file"]
    model = pd.read_csv(model_path)
    model.loc[1, "CustomerID"] = model.loc[0, "CustomerID"]
    model.to_csv(model_path, index=False)

    interpretable_path = data_dir / spec["interpretable_file"]
    interpretable = pd.read_csv(interpretable_path)
    interpretable = interpretable.iloc[:-1]
    interpretable.to_csv(interpretable_path, index=False)

    report = run_preflight(config_path)

    messages = _messages(report, "track_a")
    assert report["ready"] is False
    assert "is not unique" in messages
    assert "row counts differ" in messages


def test_preflight_rejects_nonfinite_and_nonbinary_values(tmp_path):
    config_path, contract = _write_synthetic_data(tmp_path)
    data_dir = tmp_path / "processed"

    track_a = contract["tracks"]["track_a"]
    path_a = data_dir / track_a["model_file"]
    frame_a = pd.read_csv(path_a)
    frame_a.loc[0, "Avg_Transaction_Amt"] = np.inf
    frame_a.to_csv(path_a, index=False)

    track_b = contract["tracks"]["track_b"]
    path_b = data_dir / track_b["interpretable_file"]
    frame_b = pd.read_csv(path_b)
    frame_b.loc[0, "has_current_account"] = 2
    frame_b.to_csv(path_b, index=False)

    report = run_preflight(config_path)

    assert "contains null, NaN, or infinite" in _messages(report, "track_a")
    assert "outside {0, 1}" in _messages(report, "track_b")
    assert report["ready"] is False


def test_preflight_rejects_missing_columns_and_non_numeric_features(tmp_path):
    config_path, contract = _write_synthetic_data(tmp_path)
    data_dir = tmp_path / "processed"

    track_a = contract["tracks"]["track_a"]
    path_a = data_dir / track_a["model_file"]
    frame_a = pd.read_csv(path_a)
    frame_a["Avg_Transaction_Amt"] = frame_a["Avg_Transaction_Amt"].astype(object)
    frame_a.loc[0, "Avg_Transaction_Amt"] = "not-a-number"
    frame_a.to_csv(path_a, index=False)

    track_c = contract["tracks"]["track_c"]
    path_c = data_dir / track_c["interpretable_file"]
    pd.read_csv(path_c).drop(columns="Recency_Days").to_csv(path_c, index=False)

    report = run_preflight(config_path)

    assert "is not numeric" in _messages(report, "track_a")
    assert "missing required columns" in _messages(report, "track_c")
    assert report["ready"] is False


def test_preflight_rejects_constant_and_duplicate_model_features(tmp_path):
    config_path, contract = _write_synthetic_data(tmp_path)
    data_dir = tmp_path / "processed"
    spec = contract["tracks"]["track_c"]
    model_path = data_dir / spec["model_file"]
    frame = pd.read_csv(model_path)
    frame["Recency_Days"] = frame["Account_Age_Days"]
    frame["Total_Withdrawal"] = 1.0
    frame.to_csv(model_path, index=False)

    report = run_preflight(config_path)

    messages = _messages(report, "track_c")
    assert "Constant model features" in messages
    assert "Duplicate model feature columns" in messages


def test_preflight_detects_identifier_order_mismatch(tmp_path):
    config_path, contract = _write_synthetic_data(tmp_path)
    data_dir = tmp_path / "processed"
    spec = contract["tracks"]["track_c"]
    path = data_dir / spec["interpretable_file"]
    frame = pd.read_csv(path).iloc[::-1]
    frame.to_csv(path, index=False)

    report = run_preflight(config_path)

    assert "Identifier order does not align" in _messages(report, "track_c")
    assert report["ready"] is False


def test_preflight_selected_track_passes_without_other_track_files(tmp_path, capsys):
    config_path, contract = _write_synthetic_data(tmp_path)
    data_dir = tmp_path / "processed"
    for track_key in ("track_b", "track_c", "track_d"):
        spec = contract["tracks"][track_key]
        (data_dir / spec["model_file"]).unlink()
        (data_dir / spec["interpretable_file"]).unlink()

    selected_report = run_preflight(config_path, selected_tracks=["track_a"])
    assert selected_report["ready"] is True
    assert list(selected_report["tracks"]) == ["track_a"]

    all_tracks_report = run_preflight(config_path)
    assert all_tracks_report["ready"] is False
    assert set(all_tracks_report["tracks"]) == {"track_a", "track_b", "track_c", "track_d"}

    assert main(["--config", str(config_path), "--tracks", "track_a"]) == 0
    assert "ML preflight: READY" in capsys.readouterr().out
