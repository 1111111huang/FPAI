"""US#157: tests for scripts/backup_databases.py."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import duckdb
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.backup_databases import backup_duckdb, backup_sqlite, restore


@pytest.fixture()
def duckdb_config(tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "core.db"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE raw_matches (id INTEGER, home TEXT)")
    conn.execute("INSERT INTO raw_matches VALUES (1, 'Arsenal'), (2, 'Chelsea')")
    conn.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return config_path, db_path


def test_backup_duckdb_produces_a_readable_copy_with_all_rows(duckdb_config: tuple[Path, Path], tmp_path: Path) -> None:
    config_path, db_path = duckdb_config

    dest = backup_duckdb(config_path=str(config_path), dest_dir=tmp_path / "backups")

    assert dest.exists()
    assert dest.name.startswith("core.db.") and dest.name.endswith("_backup")
    conn = duckdb.connect(str(dest), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone() == (2,)
    finally:
        conn.close()
    # original untouched
    assert db_path.exists()


def test_backup_sqlite_produces_a_readable_copy_with_all_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "recs.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE recs (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO recs VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    conn.commit()
    conn.close()

    dest = backup_sqlite(db_path, dest_dir=tmp_path / "backups")

    assert dest.exists()
    check = sqlite3.connect(dest)
    try:
        assert check.execute("SELECT COUNT(*) FROM recs").fetchone() == (3,)
    finally:
        check.close()


def test_restore_refuses_to_overwrite_an_existing_target_without_force(tmp_path: Path) -> None:
    backup_file = tmp_path / "recs.db.20260101T000000Z_backup"
    backup_file.write_bytes(b"backup-content")
    target = tmp_path / "recs.db"
    target.write_bytes(b"current-content")

    with pytest.raises(FileExistsError):
        restore(backup_file, target)

    assert target.read_bytes() == b"current-content"  # untouched


def test_restore_with_force_overwrites_the_target(tmp_path: Path) -> None:
    backup_file = tmp_path / "recs.db.20260101T000000Z_backup"
    backup_file.write_bytes(b"backup-content")
    target = tmp_path / "recs.db"
    target.write_bytes(b"current-content")

    restore(backup_file, target, force=True)

    assert target.read_bytes() == b"backup-content"


def test_restore_to_a_fresh_path_needs_no_force(tmp_path: Path) -> None:
    backup_file = tmp_path / "recs.db.20260101T000000Z_backup"
    backup_file.write_bytes(b"backup-content")
    target = tmp_path / "restored" / "recs.db"

    restore(backup_file, target)

    assert target.read_bytes() == b"backup-content"
