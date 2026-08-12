"""Tests for FotMob player-stats fetching: match discovery and player-stat extraction."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.fotmob.fetcher import (
    LEAGUE_IDS,
    fetch_finished_match_ids,
    fetch_match_player_stats,
    fetch_player_match_stats,
)


def _matches_payload(matches: list[dict], league_id: int = 47) -> dict:
    return {"leagues": [{"id": league_id, "matches": matches}]}


def _match_entry(
    match_id: int = 4193901,
    home: str = "Arsenal",
    away: str = "Everton",
    finished: bool = True,
    utc_time: str = "2024-05-19T15:00:00.000Z",
) -> dict:
    return {
        "id": match_id,
        "leagueId": 47,
        "home": {"id": 9825, "name": home},
        "away": {"id": 8668, "name": away},
        "status": {"finished": finished, "utcTime": utc_time},
    }


def _match_details_payload(player_stats: dict) -> dict:
    return {"content": {"playerStats": player_stats}}


def _player_entry(
    player_id: int = 23354,
    name: str = "Ashley Young",
    opta_id: str = "18892",
    team_name: str = "Everton",
    top_stats: dict | None = None,
) -> dict:
    stats = top_stats if top_stats is not None else {
        "FotMob rating": {"stat": {"value": 5.58}},
        "Minutes played": {"stat": {"value": 90}},
        "Goals": {"stat": {"value": 0}},
        "Assists": {"stat": {"value": 0}},
        "Expected assists (xA)": {"stat": {"value": 0.01}},
        # "Expected goals (xG)" deliberately absent, matching real FotMob payloads
        # for players with no attacking involvement.
    }
    return {
        str(player_id): {
            "id": player_id,
            "name": name,
            "optaId": opta_id,
            "teamName": team_name,
            "stats": [{"key": "top_stats", "title": "Top stats", "stats": stats}],
        }
    }


def _mock_resp(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = payload
    return mock


# ---------------------------------------------------------------------------
# fetch_finished_match_ids
# ---------------------------------------------------------------------------

def test_fetch_finished_match_ids_includes_finished_matches():
    payload = _matches_payload([_match_entry()])
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2024, 5, 19), league_id=47, delay=0)

    assert len(matches) == 1
    assert matches[0]["fotmob_match_id"] == 4193901
    assert matches[0]["home_team"] == "Arsenal"
    assert matches[0]["away_team"] == "Everton"
    assert matches[0]["match_date"] == pd.Timestamp("2024-05-19")


def test_fetch_finished_match_ids_excludes_unfinished_matches():
    payload = _matches_payload([_match_entry(finished=False)])
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2024, 5, 19), league_id=47, delay=0)

    assert matches == []


def test_fetch_finished_match_ids_treats_a_null_payload_as_no_matches():
    """BUG-041: found live -- a real 200 OK response with a literal `null`
    body (observed for a date FotMob's matches endpoint has no data for)
    must degrade to an empty list, not crash with an uncaught
    AttributeError on the .get() calls that assume a dict."""
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(None)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2011, 8, 1), league_id=47, delay=0)

    assert matches == []


def test_fetch_finished_match_ids_treats_a_list_payload_as_no_matches():
    """Same defensive contract for any other unexpected non-dict shape."""
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp([])), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2011, 8, 1), league_id=47, delay=0)

    assert matches == []


def test_fetch_finished_match_ids_filters_to_requested_league():
    payload = {
        "leagues": [
            {"id": 47, "matches": [_match_entry(match_id=1)]},
            {"id": 87, "matches": [_match_entry(match_id=2)]},
        ]
    }
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2024, 5, 19), league_id=47, delay=0)

    assert len(matches) == 1
    assert matches[0]["fotmob_match_id"] == 1


# ---------------------------------------------------------------------------
# fetch_match_player_stats
# ---------------------------------------------------------------------------

def test_fetch_match_player_stats_extracts_present_fields():
    payload = _match_details_payload(_player_entry())
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        rows = fetch_match_player_stats(4193901, delay=0)

    assert len(rows) == 1
    row = rows[0]
    assert row["player_id"] == 23354
    assert row["player_name"] == "Ashley Young"
    assert row["opta_id"] == "18892"
    assert row["team_name"] == "Everton"
    assert row["rating"] == pytest.approx(5.58)
    assert row["minutes_played"] == 90
    assert row["xa"] == pytest.approx(0.01)


def test_fetch_match_player_stats_returns_none_for_missing_fields():
    """A field absent from FotMob's payload (e.g. xG for a non-attacking player)
    must become None, not 0.0 — the two mean different things downstream."""
    payload = _match_details_payload(_player_entry())
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        rows = fetch_match_player_stats(4193901, delay=0)

    assert rows[0]["xg"] is None


def test_fetch_match_player_stats_handles_multiple_players():
    player_stats = {**_player_entry(player_id=1, name="Player One"), **_player_entry(player_id=2, name="Player Two")}
    payload = _match_details_payload(player_stats)
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        rows = fetch_match_player_stats(4193901, delay=0)

    assert {row["player_id"] for row in rows} == {1, 2}


# ---------------------------------------------------------------------------
# fetch_player_match_stats (end-to-end across a date range)
# ---------------------------------------------------------------------------

def test_fetch_player_match_stats_combines_match_and_player_rows():
    matches_payload = _matches_payload([_match_entry()])
    details_payload = _match_details_payload(_player_entry())

    def fake_get(url, **kwargs):
        if "matchDetails" in url:
            return _mock_resp(details_payload)
        return _mock_resp(matches_payload)

    with patch("src.ingestion.fotmob.fetcher.requests.get", side_effect=fake_get), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        df = fetch_player_match_stats("E0", date(2024, 5, 19), date(2024, 5, 19), delay=0)

    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Ashley Young"
    assert df.iloc[0]["home_team"] == "Arsenal"
    assert df.iloc[0]["fotmob_match_id"] == 4193901


def test_fetch_player_match_stats_rejects_unsupported_league():
    with pytest.raises(ValueError, match="Unsupported league"):
        fetch_player_match_stats("XX", date(2024, 5, 19), date(2024, 5, 19), delay=0)


def test_league_ids_maps_sp1_to_the_real_fotmob_league_id():
    """US#146: La Liga's real FotMob league id is 87 (live-verified 2026-08-06
    against /api/data/matches, entry name 'LaLiga', ccode 'ESP')."""
    assert LEAGUE_IDS["SP1"] == 87
