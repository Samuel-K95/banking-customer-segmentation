import json

import pytest

from customer_segmentation.validate.raw_checks import DataQualityError, check_raw_source
from customer_segmentation.validate.report import (
    build_report,
    check_track,
    publish_report,
)

GATES = {"min_rows": 2, "max_null_cells": 0, "max_duplicate_ids": 0}


def write_berka(raw_dir, trans_header='"trans_id";"account_id";"date";"type";"amount";"balance"'):
    berka = raw_dir / "berka"
    berka.mkdir()
    (berka / "trans.csv").write_text(trans_header + "\n1;1;930101;PRIJEM;10;10\n")
    (berka / "account.csv").write_text('"account_id";"district_id";"frequency";"date"\n1;1;x;930101\n')
    (berka / "client.csv").write_text('"client_id";"birth_number";"district_id"\n1;1;1\n')


def test_raw_check_passes_and_records_checksums(tmp_path):
    write_berka(tmp_path)
    note = check_raw_source("berka", tmp_path)
    assert [f["file"] for f in note["files"]] == ["berka/trans.csv", "berka/account.csv", "berka/client.csv"]
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in note["files"])


def test_raw_check_fails_on_missing_column(tmp_path):
    write_berka(tmp_path, trans_header='"trans_id";"account_id";"date";"type";"amount"')
    with pytest.raises(DataQualityError, match=r"missing required columns \['balance'\]"):
        check_raw_source("berka", tmp_path)


def test_raw_check_fails_on_missing_file(tmp_path):
    with pytest.raises(DataQualityError, match="bank_transactions.csv: missing or empty"):
        check_raw_source("indian_bank", tmp_path)


def summary(**overrides):
    base = {"track": "track_a", "rows": 5000, "columns": 2, "id_column": "CustomerID",
            "has_id_column": True, "duplicate_ids": 0, "null_cells": 0}
    return {**base, **overrides}


def test_check_track_gates():
    assert check_track(summary(), GATES) == []
    assert check_track(summary(rows=1, null_cells=4, duplicate_ids=2), GATES) == [
        "rows 1 < min_rows 2", "2 duplicate CustomerID values", "4 null cells",
    ]
    assert check_track(summary(has_id_column=False, duplicate_ids=None), GATES) == [
        "id column 'CustomerID' missing from an output matrix",
    ]


def test_failed_report_is_written_before_raising(tmp_path):
    report = build_report("manual__2026-10-11T18:00:00+00:00", [summary(), summary(track="track_b", null_cells=1)])
    assert report["status"] == "fail"

    with pytest.raises(DataQualityError, match="track_b: 1 null cells"):
        publish_report(report, tmp_path)

    written = json.loads((tmp_path / "dq_report_manual__2026-10-11T18_00_00_00_00.json").read_text())
    assert written["status"] == "fail"
    assert json.loads((tmp_path / "dq_report_latest.json").read_text()) == written
