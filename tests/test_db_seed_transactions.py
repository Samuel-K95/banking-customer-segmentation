"""Regression tests for bulk Berka database transaction seeding.

Copy into tests/test_db_seed_transactions.py on the integration-fix branch.
Tests deliberately do not require a live PostgreSQL instance.
"""
import csv
from datetime import date
from decimal import Decimal

import pytest

from customer_segmentation.db.seeder import seed_transactions

HEADERS = [
    "trans_id", "account_id", "date", "type", "operation", "amount",
    "balance", "k_symbol", "bank", "account",
]


class FakeCursor:
    def __init__(self):
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeRawConnection:
    def cursor(self):
        return FakeCursor()


class FakeSQLAlchemyConnection:
    connection = FakeRawConnection()


class FakeSession:
    def flush(self):
        pass

    def connection(self):
        return FakeSQLAlchemyConnection()


def make_csv(tmp_path, rows):
    root = tmp_path / "berka"
    root.mkdir()
    with (root / "trans.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADERS, delimiter=";", quotechar='"')
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_transactions_bulk_and_idempotent(monkeypatch, tmp_path):
    from psycopg2 import extras

    seen = set()
    inserted_rows = []

    def fake_execute_values(cursor, sql, rows, page_size):
        assert "ON CONFLICT (trans_id) DO NOTHING" in sql
        assert page_size == len(rows)
        cursor.rowcount = 0
        for item in rows:
            if item[0] not in seen:
                seen.add(item[0])
                inserted_rows.append(item)
                cursor.rowcount += 1

    monkeypatch.setattr(extras, "execute_values", fake_execute_values)
    root = make_csv(tmp_path, [
        dict(zip(HEADERS, ("695247", "2378", "930101", "PRIJEM", "VKLAD", "700.00", "700.00", "", "", ""))),
        dict(zip(HEADERS, ("695248", "2378", "980123", "VYDAJ", "VYBER", "500.25", "199.75", "", "AB", "12345"))),
    ])

    assert seed_transactions(FakeSession(), root, batch_size=1) == 2
    assert seed_transactions(FakeSession(), root, batch_size=2) == 0
    assert inserted_rows[0][2] == date(1993, 1, 1)
    assert inserted_rows[1][2] == date(1998, 1, 23)
    assert inserted_rows[1][5] == Decimal("500.25")
    assert inserted_rows[0][9] is None
    assert inserted_rows[1][9] == 12345


def test_missing_transaction_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="trans.csv"):
        seed_transactions(FakeSession(), tmp_path)


def test_invalid_transaction_date_fails(monkeypatch, tmp_path):
    from psycopg2 import extras

    monkeypatch.setattr(extras, "execute_values", lambda *_args, **_kwargs: None)
    root = make_csv(tmp_path, [
        dict(zip(HEADERS, ("1", "1", "notadate", "PRIJEM", "VKLAD", "1.0", "1.0", "", "", "")))
    ])
    with pytest.raises(ValueError, match="row 2"):
        seed_transactions(FakeSession(), root)
