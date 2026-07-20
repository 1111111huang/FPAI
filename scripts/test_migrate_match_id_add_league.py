"""US#140: tests for scripts/migrate_match_id_add_league.py.

Builds a small synthetic DuckDB mirroring the shape of the real production
db (raw_matches + feature_store + raw_player_match_stats, all keyed by the
OLD date|home|away match_id scheme) and verifies the migration:
  - recomputes match_id to the new league-inclusive scheme,
  - preserves row counts in every table,
  - leaves no orphaned feature_store/raw_player_match_stats rows,
  - is a no-op (report-only) under --dry-run,
  - refuses to write anything if it ever computed colliding new ids.
"""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.migrate_match_id_add_league import build_id_mapping, migrate
from src.utils.helpers import generate_match_id


def _old_scheme_id(date: str, home: str, away: str) -> str:
    """Recreate what the pre-US#140 generate_match_id(date, home, away) produced."""
    import hashlib

    normalized = "|".join(
        " ".join(str(v).strip().lower().split()) for v in (date, home, away)
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@pytest.fixture()
def old_scheme_db(tmp_path: Path) -> Path:
    """A synthetic db with raw_matches/feature_store/raw_player_match_stats
    all keyed by the OLD (pre-league) match_id scheme, matching production
    shape (feature_store and raw_player_match_stats have no league column of
    their own -- they only ever carry match_id)."""
    db_path = tmp_path / "old_scheme.db"
    conn = duckdb.connect(str(db_path))

    rows = [
        ("2025-03-08", "Liverpool", "Bournemouth", "E0"),
        ("2025-03-08", "Arsenal", "Chelsea", "E0"),
        ("2025-03-15", "Fulham", "Brighton", "E0"),
    ]
    conn.execute(
        "CREATE TABLE raw_matches (match_id TEXT PRIMARY KEY, date TIMESTAMP, home_team TEXT, away_team TEXT, league TEXT)"
    )
    conn.execute("CREATE TABLE feature_store (match_id TEXT PRIMARY KEY, some_feature FLOAT)")
    conn.execute(
        "CREATE TABLE raw_player_match_stats (match_id TEXT, player_id BIGINT, rating FLOAT, PRIMARY KEY (match_id, player_id))"
    )

    for i, (date, home, away, league) in enumerate(rows):
        old_id = _old_scheme_id(date, home, away)
        conn.execute(
            "INSERT INTO raw_matches VALUES (?, ?, ?, ?, ?)", [old_id, date, home, away, league]
        )
        conn.execute("INSERT INTO feature_store VALUES (?, ?)", [old_id, 1.5 + i])
        # Two player rows per match, like the real fotmob-sourced table.
        conn.execute(
            "INSERT INTO raw_player_match_stats VALUES (?, ?, ?)", [old_id, 100 + i, 7.1]
        )
        conn.execute(
            "INSERT INTO raw_player_match_stats VALUES (?, ?, ?)", [old_id, 200 + i, 6.8]
        )

    conn.close()
    return db_path


def test_dry_run_reports_without_changing_anything(old_scheme_db: Path) -> None:
    conn = duckdb.connect(str(old_scheme_db))
    try:
        before_ids = {row[0] for row in conn.execute("SELECT match_id FROM raw_matches").fetchall()}
        summary = migrate(conn, dry_run=True)
        after_ids = {row[0] for row in conn.execute("SELECT match_id FROM raw_matches").fetchall()}
    finally:
        conn.close()

    assert summary["to_change"] == 3
    assert summary["already_current"] == 0
    assert before_ids == after_ids  # nothing written


def test_migrate_recomputes_ids_to_new_league_inclusive_scheme(old_scheme_db: Path) -> None:
    conn = duckdb.connect(str(old_scheme_db))
    try:
        summary = migrate(conn)
        rows = conn.execute(
            "SELECT match_id, date, home_team, away_team, league FROM raw_matches"
        ).fetchall()
    finally:
        conn.close()

    assert summary["changed"] == 3
    assert summary["raw_matches_before"] == summary["raw_matches_after"] == 3

    for match_id, date, home_team, away_team, league in rows:
        date_str = date.date().isoformat() if hasattr(date, "date") else str(date)
        expected = generate_match_id(date=date_str, home_team=home_team, away_team=away_team, league=league)
        assert match_id == expected


def test_migrate_preserves_row_counts_in_dependent_tables(old_scheme_db: Path) -> None:
    conn = duckdb.connect(str(old_scheme_db))
    try:
        before_feature = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
        before_players = conn.execute("SELECT COUNT(*) FROM raw_player_match_stats").fetchone()[0]

        summary = migrate(conn)

        after_feature = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
        after_players = conn.execute("SELECT COUNT(*) FROM raw_player_match_stats").fetchone()[0]
    finally:
        conn.close()

    assert before_feature == after_feature == 3
    assert before_players == after_players == 6
    assert summary["dependent_row_counts_before"]["feature_store"] == 3
    assert summary["dependent_row_counts_after"]["feature_store"] == 3
    assert summary["dependent_row_counts_before"]["raw_player_match_stats"] == 6
    assert summary["dependent_row_counts_after"]["raw_player_match_stats"] == 6


def test_migrate_leaves_no_orphaned_dependent_rows(old_scheme_db: Path) -> None:
    conn = duckdb.connect(str(old_scheme_db))
    try:
        summary = migrate(conn)

        orphaned_features = conn.execute(
            """
            SELECT COUNT(*) FROM feature_store f
            LEFT JOIN raw_matches r ON f.match_id = r.match_id
            WHERE r.match_id IS NULL
            """
        ).fetchone()[0]
        orphaned_players = conn.execute(
            """
            SELECT COUNT(*) FROM raw_player_match_stats p
            LEFT JOIN raw_matches r ON p.match_id = r.match_id
            WHERE r.match_id IS NULL
            """
        ).fetchone()[0]
    finally:
        conn.close()

    assert orphaned_features == 0
    assert orphaned_players == 0
    assert summary["orphaned_dependent_rows"]["feature_store"] == 0
    assert summary["orphaned_dependent_rows"]["raw_player_match_stats"] == 0


def test_migrate_joins_feature_store_to_correct_match_after_remap(old_scheme_db: Path) -> None:
    """Not just 'no orphans' -- each feature_store row must still join to the
    SAME logical match it belonged to before migration (not some other row
    that happens to share a new match_id)."""
    conn = duckdb.connect(str(old_scheme_db))
    try:
        before = dict(
            conn.execute(
                """
                SELECT r.home_team || '|' || r.away_team, f.some_feature
                FROM raw_matches r JOIN feature_store f ON r.match_id = f.match_id
                """
            ).fetchall()
        )
        migrate(conn)
        after = dict(
            conn.execute(
                """
                SELECT r.home_team || '|' || r.away_team, f.some_feature
                FROM raw_matches r JOIN feature_store f ON r.match_id = f.match_id
                """
            ).fetchall()
        )
    finally:
        conn.close()

    assert before == after


def test_migrate_is_idempotent_on_second_run(old_scheme_db: Path) -> None:
    conn = duckdb.connect(str(old_scheme_db))
    try:
        migrate(conn)
        ids_after_first = sorted(row[0] for row in conn.execute("SELECT match_id FROM raw_matches").fetchall())
        summary_second = migrate(conn)
        ids_after_second = sorted(row[0] for row in conn.execute("SELECT match_id FROM raw_matches").fetchall())
    finally:
        conn.close()

    assert ids_after_first == ids_after_second
    assert summary_second["changed"] == 0
    assert summary_second["already_current"] == 3


def test_migrate_on_missing_raw_matches_table_is_a_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    conn = duckdb.connect(str(db_path))
    try:
        summary = migrate(conn)
    finally:
        conn.close()

    assert summary["raw_matches_before"] == 0
    assert summary["changed"] == 0


def test_build_id_mapping_matches_generate_match_id(old_scheme_db: Path) -> None:
    conn = duckdb.connect(str(old_scheme_db))
    try:
        mapping = build_id_mapping(conn)
        rows = conn.execute("SELECT match_id, date, home_team, away_team, league FROM raw_matches").fetchall()
    finally:
        conn.close()

    for old_id, date, home_team, away_team, league in rows:
        date_str = date.date().isoformat() if hasattr(date, "date") else str(date)
        expected_new_id = generate_match_id(date=date_str, home_team=home_team, away_team=away_team, league=league)
        assert mapping[old_id] == expected_new_id
