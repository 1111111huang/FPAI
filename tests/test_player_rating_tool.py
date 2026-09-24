"""Tests for get_player_rating: roster lookup + the LLM-callable tool
itself. The tool is built fresh per match (build_player_rating_tool), closing
over that match's own roster -- the first per-match-context-bound tool in
this codebase, since get_default_tools()/build_graph() already run fresh per
run_agent() call (no new global state needed)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

duckdb = pytest.importorskip("duckdb")

from src.agent.player_rating_tool import build_player_rating_tool, get_team_roster
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _seed(db_manager: DuckDBManager) -> None:
    with db_manager.connection() as conn:
        conn.execute("CREATE TABLE raw_matches (match_id TEXT, date TIMESTAMP, home_team TEXT, away_team TEXT)")
        conn.execute("INSERT INTO raw_matches VALUES ('m1', '2026-09-01', 'Man City', 'Everton')")
        conn.execute("CREATE TABLE player_dim (player_id BIGINT PRIMARY KEY, player_name TEXT, opta_id TEXT)")
        conn.execute("INSERT INTO player_dim VALUES (1, 'Erling Haaland', NULL), (2, 'Jordan Pickford', NULL)")
        conn.execute(
            "CREATE TABLE raw_player_match_stats (match_id TEXT, player_id BIGINT, team_name TEXT, minutes_played INTEGER, rating FLOAT, goals INTEGER, assists INTEGER, xg FLOAT, xa FLOAT, xgot FLOAT, shots INTEGER, interceptions FLOAT, recoveries FLOAT, PRIMARY KEY (match_id, player_id))"
        )
        conn.execute("INSERT INTO raw_player_match_stats VALUES ('m1', 1, 'Man City', 90, 8.1, 1, 0, 0.8, 0.1, 0.5, 3, 0, 1)")
        conn.execute("INSERT INTO raw_player_match_stats VALUES ('m1', 2, 'Everton', 90, 6.5, 0, 0, 0, 0, 0, 0, 1, 5)")
        conn.execute("CREATE TABLE player_market_values (fotmob_player_id BIGINT, transfermarkt_player_id BIGINT, snapshot_date TEXT, market_value_eur BIGINT)")
        conn.execute("INSERT INTO player_market_values VALUES (1, 100, '2026-09-01', 220000000)")


def test_get_team_roster_returns_player_names_for_a_team(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    roster = get_team_roster(db_manager, team_name="Man City", as_of_date="2026-09-15")
    assert roster == ["Erling Haaland"]


def test_tool_returns_value_for_a_known_player(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Erling Haaland"})

    assert result == {"matched": True, "market_value_eur": 220000000}


def test_tool_degrades_to_matched_false_for_an_unknown_name(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Nobody On This Roster"})

    assert result == {"matched": False}


def test_tool_degrades_to_matched_false_when_player_has_no_market_value(tmp_path: Path):
    """Jordan Pickford is on the roster but has no player_market_values row
    seeded -- a real, non-blocking gap (unmapped crosswalk entry, no listed
    value, or never fetched), not an error."""
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Jordan Pickford"})

    assert result == {"matched": False}


def test_tool_never_reads_a_snapshot_dated_after_as_of_date(tmp_path: Path):
    """Point-in-time correctness -- the same lookahead-bias class of bug
    W179 already had to fix once for closing-line odds features. Two
    snapshots for the same player (an older, lower value and a newer,
    higher one); a lookup dated between them must return the OLDER one,
    never peek at the not-yet-existing newer snapshot."""
    db_manager = _make_db_manager(tmp_path)
    _seed(db_manager)
    with db_manager.connection() as conn:
        conn.execute("INSERT INTO player_market_values VALUES (1, 100, '2026-10-01', 300000000)")  # future, higher

    tool = build_player_rating_tool(
        db_manager, home_team="Man City", away_team="Everton", as_of_date="2026-09-15",
    )
    result = tool.invoke({"player_name": "Erling Haaland"})

    assert result == {"matched": True, "market_value_eur": 220000000}
