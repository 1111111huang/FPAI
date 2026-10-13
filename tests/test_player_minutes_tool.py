"""Tests for get_player_minutes (A131): the get_player_rating-sibling tool
that returns a flagged player's real recent involvement, so the LLM can
judge whether a web_search-cited absence is actually meaningful rather than
taking the search text's own framing at face value. Mirrors
test_player_rating_tool.py's exact conventions -- same per-match-roster-
scoped tool-building pattern, same real (not mocked) DuckDB."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

duckdb = pytest.importorskip("duckdb")

from src.agent.player_minutes_tool import build_player_minutes_tool
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _seed(db_manager: DuckDBManager) -> None:
    with db_manager.connection() as conn:
        conn.execute("CREATE TABLE raw_matches (match_id TEXT, date TIMESTAMP, home_team TEXT, away_team TEXT)")
        conn.execute(
            "CREATE TABLE player_dim (player_id BIGINT PRIMARY KEY, player_name TEXT, opta_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE raw_player_match_stats (match_id TEXT, player_id BIGINT, team_name TEXT, "
            "minutes_played INTEGER, rating FLOAT, goals INTEGER, assists INTEGER, xg FLOAT, xa FLOAT, "
            "xgot FLOAT, shots INTEGER, interceptions FLOAT, recoveries FLOAT, PRIMARY KEY (match_id, player_id))"
        )
        conn.execute(
            "INSERT INTO player_dim VALUES (1, 'Erling Haaland', NULL), (2, 'Jordan Pickford', NULL), "
            "(3, 'Bench Warmer', NULL)"
        )
        # Man City play 3 matches in the trailing window; Everton play only 1
        # (a quiet stretch for Everton -- an international break, a
        # postponement -- independent of any one player's own involvement).
        matches = [
            ("m1", "2026-07-01", "Man City", "Other FC"),
            ("m2", "2026-08-01", "Man City", "Other FC"),
            ("m3", "2026-09-01", "Man City", "Everton"),
        ]
        for mid, date, home, away in matches:
            conn.execute("INSERT INTO raw_matches VALUES (?, ?, ?, ?)", [mid, date, home, away])

        # Haaland starts and plays full minutes in all 3 Man City matches.
        for mid in ("m1", "m2", "m3"):
            conn.execute(
                "INSERT INTO raw_player_match_stats VALUES (?, 1, 'Man City', 90, 8.1, 1, 0, 0.8, 0.1, 0.5, 3, 0, 1)",
                [mid],
            )
        # Pickford only features in Everton's one match, full 90.
        conn.execute(
            "INSERT INTO raw_player_match_stats VALUES ('m3', 2, 'Everton', 90, 6.5, 0, 0, 0, 0, 0, 0, 1, 5)"
        )
        # Bench Warmer is a real Man City squad player (so roster-matchable)
        # but has zero recorded minutes anywhere in the window -- a real,
        # meaningful "not a regular" answer, not a data gap.
        conn.execute(
            "INSERT INTO raw_player_match_stats VALUES ('m1', 3, 'Man City', 0, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )


def test_tool_returns_minutes_appearances_and_team_matches_for_a_regular(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_minutes_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Erling Haaland"})

    assert result == {
        "matched": True,
        "minutes_last_90_days": 270,
        "appearances_last_90_days": 3,
        "team_matches_available": 3,
    }


def test_team_matches_available_reflects_the_players_own_team_not_the_opponent(tmp_path: Path):
    """Direct user catch (2026-10-13): the denominator must be the player's
    OWN team's match count, not a fixed window -- Everton only played 1
    match in this window (their own quiet stretch) even though Man City (the
    other side of this fixture) played 3, and Pickford's involvement must be
    read against Everton's own 1, not City's 3."""
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_minutes_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Jordan Pickford"})

    assert result == {
        "matched": True,
        "minutes_last_90_days": 90,
        "appearances_last_90_days": 1,
        "team_matches_available": 1,
    }


def test_tool_reports_zero_minutes_for_a_rostered_but_unused_player(tmp_path: Path):
    """A real, meaningful answer (this player is not currently a rotation
    regular) -- not a 'matched: false' data gap. The player is still
    correctly resolved against the roster; they just have zero involvement."""
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_minutes_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Bench Warmer"})

    assert result == {
        "matched": True,
        "minutes_last_90_days": 0,
        "appearances_last_90_days": 1,  # one row exists (0 minutes played) -- a squad inclusion, not a start
        "team_matches_available": 3,
    }


def test_tool_degrades_to_matched_false_for_an_unknown_name(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_minutes_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Nobody On This Roster"})

    assert result == {"matched": False}
