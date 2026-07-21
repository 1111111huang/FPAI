"""Tests for resolving FotMob player rows to raw_matches and persisting them."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.fotmob.merge import upsert_player_match_stats
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _seed_raw_matches(db_manager: DuckDBManager) -> None:
    with db_manager.connection() as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, date TIMESTAMP, home_team TEXT, away_team TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO raw_matches VALUES ('match-abc', '2024-05-19', 'Arsenal', 'Everton')"
        )


def _fotmob_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fotmob_match_id": 4193901, "match_date": pd.Timestamp("2024-05-19"),
                "home_team": "Arsenal", "away_team": "Everton",
                "player_id": 23354, "player_name": "Ashley Young", "opta_id": "18892", "team_name": "Everton",
                "rating": 5.58, "minutes_played": 90, "goals": 0, "assists": 0,
                "xg": None, "xa": 0.01, "xgot": None, "shots": 0,
            },
            {
                "fotmob_match_id": 4193901, "match_date": pd.Timestamp("2024-05-19"),
                "home_team": "Arsenal", "away_team": "Everton",
                "player_id": 99001, "player_name": "Idrissa Gana Gueye", "opta_id": "55001", "team_name": "Everton",
                "rating": 8.16, "minutes_played": 90, "goals": 1, "assists": 0,
                "xg": 0.06, "xa": 0.02, "xgot": 0.31, "shots": 1,
            },
        ]
    )


def test_upsert_player_match_stats_matches_by_date_and_team(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    result = upsert_player_match_stats(_fotmob_rows(), db_manager)

    assert result == {"matched": 2, "unmatched": 0, "players_upserted": 2, "rows_upserted": 2}

    with db_manager.connection() as conn:
        stats_rows = conn.execute(
            "SELECT match_id, player_id, rating, xg, xa FROM raw_player_match_stats ORDER BY player_id"
        ).fetchall()
        player_rows = conn.execute("SELECT player_id, player_name, opta_id FROM player_dim ORDER BY player_id").fetchall()

    assert stats_rows == [
        ("match-abc", 23354, pytest.approx(5.58), None, pytest.approx(0.01)),
        ("match-abc", 99001, pytest.approx(8.16), pytest.approx(0.06), pytest.approx(0.02)),
    ]
    assert player_rows == [
        (23354, "Ashley Young", "18892"),
        (99001, "Idrissa Gana Gueye", "55001"),
    ]


def test_upsert_player_match_stats_counts_unmatched_rows(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    fotmob_df = _fotmob_rows()
    fotmob_df["home_team"] = "Some Other Team"  # won't match the seeded raw_matches row

    result = upsert_player_match_stats(fotmob_df, db_manager)

    assert result["matched"] == 0
    assert result["unmatched"] == 2
    assert result["rows_upserted"] == 0

    with db_manager.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_player_match_stats").fetchone()[0]
    assert count == 0


def _seed_raw_matches_multi_league(db_manager: DuckDBManager) -> None:
    """Two competitions' rows in one table, one holding a real exact-name trap.

    "Everton" (E0) is the correct target. "Everston" (a fictional SWE club)
    exists as an *exact* string in the SWE row -- if the candidate pool isn't
    scoped by league, a misspelled FotMob away_team of "Everston" would find
    an exact (score 1.0) match against the wrong league's team before fuzzy
    matching ever gets a chance to prefer the correct "Everton".
    """
    with db_manager.connection() as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, date TIMESTAMP, home_team TEXT, away_team TEXT, league TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO raw_matches VALUES "
            "('match-e0', '2024-05-19', 'Arsenal', 'Everton', 'E0'), "
            "('match-swe', '2024-05-19', 'AIK', 'Everston', 'SWE')"
        )


def test_resolve_match_ids_unscoped_can_lose_a_real_match_to_the_wrong_league(tmp_path: Path) -> None:
    """Pre-US#141 behavior (league=None): proves the bug is real, not just theoretical."""
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches_multi_league(db_manager)

    fotmob_df = _fotmob_rows()
    fotmob_df["away_team"] = "Everston"  # misspelling of E0's "Everton" that happens to exactly match SWE's own club name

    result = upsert_player_match_stats(fotmob_df, db_manager, league=None)
    # The away name resolves to SWE's "Everston" (exact match beats fuzzy),
    # so the join against (Arsenal, Everston) finds no row at all -- the
    # real Arsenal-Everton match is silently lost, not wrongly merged.
    assert result["matched"] == 0
    assert result["unmatched"] == 2


def test_resolve_match_ids_scoped_by_league_resolves_the_same_row_correctly(tmp_path: Path) -> None:
    """US#141 fix: scoping the candidate pool by league finds the correct match."""
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches_multi_league(db_manager)

    fotmob_df = _fotmob_rows()
    fotmob_df["away_team"] = "Everston"  # same misspelling as the unscoped test above

    result = upsert_player_match_stats(fotmob_df, db_manager, league="E0")
    assert result["matched"] == 2

    with db_manager.connection() as conn:
        matched_ids = conn.execute("SELECT DISTINCT match_id FROM raw_player_match_stats").fetchall()
    assert matched_ids == [("match-e0",)]


def test_upsert_player_match_stats_is_idempotent_on_rerun(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    upsert_player_match_stats(_fotmob_rows(), db_manager)

    updated_rows = _fotmob_rows()
    updated_rows.loc[updated_rows["player_id"] == 99001, "rating"] = 9.0
    result = upsert_player_match_stats(updated_rows, db_manager)

    assert result["matched"] == 2
    with db_manager.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_player_match_stats").fetchone()[0]
        rating = conn.execute(
            "SELECT rating FROM raw_player_match_stats WHERE player_id = 99001"
        ).fetchone()[0]
    assert count == 2  # no duplicate rows from the second run
    assert rating == pytest.approx(9.0)  # second run's value won


def test_upsert_player_match_stats_handles_empty_input(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    result = upsert_player_match_stats(pd.DataFrame(columns=[
        "fotmob_match_id", "match_date", "home_team", "away_team",
        "player_id", "player_name", "opta_id", "team_name",
        "rating", "minutes_played", "goals", "assists", "xg", "xa", "xgot", "shots",
    ]), db_manager)

    assert result == {"matched": 0, "unmatched": 0, "players_upserted": 0, "rows_upserted": 0}
