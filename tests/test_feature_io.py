import pandas as pd

from customer_segmentation.config import source_files
from customer_segmentation.features.io import save_matrices, track_outputs


def test_track_outputs_follow_config_prefix(tmp_path):
    interpretable, model = track_outputs("track_c", tmp_path)
    assert interpretable == tmp_path / "track_c_berka_interpretable.csv"
    assert model == tmp_path / "track_c_berka_model.csv"


def test_save_matrices_writes_both_files_and_counts_dq_issues(tmp_path):
    interpretable = pd.DataFrame({"CustomerID": ["C1", "C1", "C2"], "Age": [30.0, None, 40.0]})
    model = interpretable.fillna(0.0)

    summary = save_matrices("track_a", interpretable, model, tmp_path)

    assert summary["rows"] == 3
    assert summary["has_id_column"]
    assert summary["duplicate_ids"] == 1
    assert summary["null_cells"] == 1
    assert pd.read_csv(summary["interpretable_path"]).shape == (3, 2)
    assert pd.read_csv(summary["model_path"]).shape == (3, 2)


def test_source_files_resolve_under_raw_dir():
    (santander_csv,) = source_files("santander")
    assert santander_csv.parts[-3:] == ("raw", "santander", "train_ver2.csv")
    assert santander_csv.is_absolute()


def test_nullable_columns_are_reported_but_not_counted(tmp_path):
    interpretable = pd.DataFrame({"customer_id": [1, 2], "gender": ["H", None], "age": [30.0, None]})
    model = interpretable.copy()

    summary = save_matrices("track_b", interpretable, model, tmp_path)

    assert summary["null_cells"] == 2  # age only: one per matrix
    assert summary["allowed_null_cells"] == {"gender": 2}  # is_female absent from this frame
