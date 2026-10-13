from pathlib import Path

import pandas as pd
import pytest
import yaml

from customer_segmentation.data.berka_dataset_loader import load_berka_raw_data
from customer_segmentation.data.indian_bank_dataset_loader import (
    load_indian_bank_raw_data,
)
from customer_segmentation.data.santander_dataset_loader import (
    SANTANDER_PRODUCT_COLUMN_MAP,
    SANTANDER_PRODUCT_FEATURES,
    SANTANDER_SOURCE_DTYPES,
    load_santander_raw_data,
)


def _santander_source_rows(row_count=2):
    rows = []
    for row_index in range(row_count):
        row = {}
        for column in SANTANDER_SOURCE_DTYPES:
            if column == "ncodpers":
                value = 100 + row_index
            elif column == "age":
                value = 30 + row_index
            elif column == "antiguedad":
                value = 12 + row_index
            elif column == "renta":
                value = 50_000 + row_index
            elif column == "sexo":
                value = "H" if row_index == 0 else "V"
            elif column == "fecha_dato":
                value = "2016-05-28"
            elif column == "nomprov":
                value = "MADRID"
            elif column in SANTANDER_PRODUCT_COLUMN_MAP:
                value = row_index % 2
            elif column == "segmento":
                value = "02 - PARTICULARES"
            elif column in {"ind_nuevo", "indrel", "tipodom", "ind_actividad_cliente"}:
                value = 1
            elif column == "indext" or column in {"conyuemp", "indfall"}:
                value = "N"
            elif column == "indrel_1mes":
                value = "1"
            elif column == "tiprel_1mes":
                value = "I"
            else:
                value = 0
            row[column] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_indian_bank_loader_handles_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_indian_bank_raw_data(tmp_path / "non_existent_file.csv")


def test_indian_bank_loader_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_indian_bank_raw_data(tmp_path / "missing.csv")


def test_indian_bank_loader_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="empty"):
        load_indian_bank_raw_data(path)


def test_santander_loader_maps_all_24_product_columns(tmp_path):
    path = tmp_path / "train_ver2.csv"
    _santander_source_rows().to_csv(path, index=False)

    loaded = load_santander_raw_data(path, chunksize=1)

    assert len(SANTANDER_PRODUCT_FEATURES) == 24
    assert set(SANTANDER_PRODUCT_FEATURES).issubset(loaded.columns)
    assert loaded["customer_id"].tolist() == [100, 101]
    assert "has_savings_account" in loaded.columns
    assert "has_bank_guarantees" in loaded.columns


def test_santander_feature_contract_matches_all_loader_product_features():
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "configs" / "models" / "feature_contracts.yaml").open(encoding="utf-8") as stream:
        contracts = yaml.safe_load(stream)
    track_b = contracts["tracks"]["track_b"]

    assert track_b["model_features"] == list(SANTANDER_PRODUCT_FEATURES)
    assert track_b["binary_features"] == list(SANTANDER_PRODUCT_FEATURES)


def test_santander_loader_fails_when_a_source_product_column_is_missing(tmp_path):
    path = tmp_path / "train_ver2.csv"
    _santander_source_rows().drop(columns="ind_ahor_fin_ult1").to_csv(path, index=False)

    with pytest.raises(ValueError, match="ind_ahor_fin_ult1"):
        load_santander_raw_data(path, chunksize=1)


def test_berka_loader_reads_csv_and_asc_tables_and_fails_on_partial_input(tmp_path):
    berka_dir = tmp_path / "berka"
    berka_dir.mkdir()
    pd.DataFrame({"trans_id": [1]}).to_csv(berka_dir / "trans.asc", sep=";", index=False)
    pd.DataFrame({"account_id": [1]}).to_csv(berka_dir / "account.csv", sep=";", index=False)
    pd.DataFrame({"client_id": [2]}).to_csv(berka_dir / "client.asc", sep=";", index=False)

    trans, account, client = load_berka_raw_data(berka_dir)

    assert trans.loc[0, "trans_id"] == 1
    assert account.loc[0, "account_id"] == 1
    assert client.loc[0, "client_id"] == 2

    (berka_dir / "client.asc").unlink()
    with pytest.raises(FileNotFoundError, match="client"):
        load_berka_raw_data(berka_dir)
