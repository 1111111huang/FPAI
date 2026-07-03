"""Tests for FotMob lineup ingestion: position group mapping, fetch parsing, and DB upsert."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import duckdb
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.fotmob.lineup import (
    _position_group,
    _create_lineup_table,
    fetch_match_lineup,
    upsert_match_lineups,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _mock_resp(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = payload
    return mock


def _lineup_payload(
    match_id: int = 4506586,
    lineup_type: str = "standard",
    home_starters: list[dict] | None = None,
    away_starters: list[dict] | None = None,
) -> dict:
    if home_starters is None:
        home_starters = [
            {"id": 1131987, "name": "Bart Verbruggen", "positionId": 11, "shirtNumber": "1"},
            {"id": 1111111, "name": "Home Defender", "positionId": 32, "shirtNumber": "4"},
        ]
    if away_starters is None:
        away_starters = [
            {"id": 2222222, "name": "Away Midfielder", "positionId": 64, "shirtNumber": "8"},
            {"id": 3333333, "name": "Away Forward", "positionId": 83, "shirtNumber": "9"},
        ]
    return {
        "content": {
            "lineup": {
                "matchId": match_id,
                "lineupType": lineup_type,
                "homeTeam": {
                    "id": 10204,
                    "name": "Brighton",
                    "formation": "4-2-3-1",
                    "starters": home_starters,
                    "subs": [],
                },
                "awayTeam": {
                    "id": 10205,
                    "name": "Arsenal",
                    "formation": "4-3-3",
                    "starters": away_starters,
                    "subs": [],
                },
            }
        }
    }


class _InMemoryDBManager:
    """Minimal DuckDBManager substitute backed by an in-memory DuckDB connection."""

    def __init__(self) -> None:
        self._conn = duckdb.connect(":memory:")

    def connection(self, read_only: bool = False):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            yield self._conn

        return _ctx()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Test 1: position group mapping
# ---------------------------------------------------------------------------

def test_position_group_mapping():
    assert _position_group(11) == "GK"
    assert _position_group(32) == "DEF"
    assert _position_group(64) == "MID"
    assert _position_group(83) == "FWD"
    assert _position_group(115) == "FWD"
    assert _position_group(None) == "UNK"
    # Boundary values
    assert _position_group(30) == "DEF"
    assert _position_group(39) == "DEF"
    assert _position_group(60) == "MID"
    assert _position_group(69) == "MID"
    assert _position_group(80) == "FWD"
    assert _position_group(89) == "FWD"
    assert _position_group(110) == "FWD"
    # Unknown range
    assert _position_group(50) == "UNK"


# ---------------------------------------------------------------------------
# Test 2: fetch_match_lineup parses starters correctly
# ---------------------------------------------------------------------------

def test_fetch_match_lineup_parses_starters():
    payload = _lineup_payload()
    with patch("src.ingestion.fotmob.lineup.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.lineup.time.sleep"):
        rows = fetch_match_lineup(4506586, delay=0)

    assert len(rows) == 4  # 2 home + 2 away

    home_rows = [r for r in rows if r["side"] == "home"]
    away_rows = [r for r in rows if r["side"] == "away"]

    assert len(home_rows) == 2
    assert len(away_rows) == 2

    # GK
    gk = next(r for r in home_rows if r["player_id"] == 1131987)
    assert gk["position_group"] == "GK"
    assert gk["team_name"] == "Brighton"
    assert gk["fotmob_match_id"] == 4506586
    assert gk["shirt_number"] == "1"
    assert gk["player_name"] == "Bart Verbruggen"

    # DEF
    defender = next(r for r in home_rows if r["player_id"] == 1111111)
    assert defender["position_group"] == "DEF"

    # MID
    mid = next(r for r in away_rows if r["player_id"] == 2222222)
    assert mid["position_group"] == "MID"
    assert mid["team_name"] == "Arsenal"

    # FWD
    fwd = next(r for r in away_rows if r["player_id"] == 3333333)
    assert fwd["position_group"] == "FWD"


# ---------------------------------------------------------------------------
# Test 3: absent lineup returns empty list
# ---------------------------------------------------------------------------

def test_fetch_match_lineup_absent_returns_empty():
    # Payload with no 'lineup' key inside content
    payload = {"content": {"playerStats": {}}}
    with patch("src.ingestion.fotmob.lineup.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.lineup.time.sleep"):
        rows = fetch_match_lineup(9999999, delay=0)

    assert rows == []


def test_fetch_match_lineup_null_lineuptype_returns_empty():
    """lineupType=None signals unknown/unavailable lineup — must return empty."""
    payload = _lineup_payload(lineup_type=None)
    # Manually set lineupType to None
    payload["content"]["lineup"]["lineupType"] = None
    with patch("src.ingestion.fotmob.lineup.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.lineup.time.sleep"):
        rows = fetch_match_lineup(4506586, delay=0)

    assert rows == []


# ---------------------------------------------------------------------------
# Test 4: upsert creates table and inserts rows
# ---------------------------------------------------------------------------

def test_upsert_creates_table_and_inserts():
    db = _InMemoryDBManager()

    payload = _lineup_payload()
    with patch("src.ingestion.fotmob.lineup.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.lineup.time.sleep"):
        total = upsert_match_lineups([4506586], db, delay=0)

    assert total == 4

    with db.connection() as conn:
        # Table must exist
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='match_lineups'"
        ).fetchall()
        assert len(tables) == 1

        rows = conn.execute("SELECT * FROM match_lineups ORDER BY player_id").fetchdf()

    assert len(rows) == 4
    assert set(rows["side"].unique()) == {"home", "away"}
    assert set(rows["position_group"].unique()).issubset({"GK", "DEF", "MID", "FWD", "UNK"})
    assert (rows["fotmob_match_id"] == 4506586).all()

    db.close()


def test_upsert_idempotent_on_conflict():
    """Upserting the same match twice must not duplicate rows."""
    db = _InMemoryDBManager()

    payload = _lineup_payload()
    with patch("src.ingestion.fotmob.lineup.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.lineup.time.sleep"):
        upsert_match_lineups([4506586], db, delay=0)

    with patch("src.ingestion.fotmob.lineup.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.lineup.time.sleep"):
        upsert_match_lineups([4506586], db, delay=0)

    with db.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM match_lineups").fetchone()[0]

    assert count == 4

    db.close()
