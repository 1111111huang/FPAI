"""Tests for Understat xG integration: fetcher, team mapper, and DB updater."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.understat import TeamNameMapper, update_raw_matches_xg
from src.ingestion.understat_fetcher import fetch_league_season


# ---------------------------------------------------------------------------
# Helpers — build mock API responses matching the real getLeagueData JSON
# ---------------------------------------------------------------------------

def _match_record(
    home: str = "Arsenal",
    away: str = "Chelsea",
    xg_h: str = "1.23",
    xg_a: str = "0.87",
    is_result: bool = True,
) -> dict:
    return {
        "id": "12345",
        "isResult": is_result,
        "datetime": "2024-08-17 15:00:00",
        "h": {"id": "1", "title": home, "short_title": "ARS"},
        "a": {"id": "2", "title": away, "short_title": "CHE"},
        "goals": {"h": "2", "a": "1"},
        "xG": {"h": xg_h, "a": xg_a},
    }


def _api_response(matches: list[dict]) -> dict:
    return {"dates": matches, "teams": {}, "players": {}}


def _mock_resp(matches: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = _api_response(matches)
    return mock


# ---------------------------------------------------------------------------
# fetch_league_season (mocked HTTP)
# ---------------------------------------------------------------------------

def test_fetch_league_season_returns_dataframe():
    records = [_match_record(), _match_record(is_result=False)]  # one result, one fixture
    with patch("src.ingestion.understat_fetcher.requests.get", return_value=_mock_resp(records)), \
         patch("src.ingestion.understat_fetcher.time.sleep"):
        df = fetch_league_season("E0", 2023, delay=0)

    assert len(df) == 1  # upcoming fixture (isResult=False) excluded
    assert df.iloc[0]["home_team"] == "Arsenal"
    assert df.iloc[0]["xg_h"] == pytest.approx(1.23)
    assert df.iloc[0]["xg_a"] == pytest.approx(0.87)
    assert "season" in df.columns


def test_fetch_league_season_maps_e0_to_epl():
    captured_urls = []

    def fake_get(url, **kwargs):
        captured_urls.append(url)
        return _mock_resp([])

    with patch("src.ingestion.understat_fetcher.requests.get", side_effect=fake_get), \
         patch("src.ingestion.understat_fetcher.time.sleep"):
        fetch_league_season("E0", 2023, delay=0)

    assert "EPL" in captured_urls[0]
    assert "2023" in captured_urls[0]


def test_fetch_league_season_sends_xhr_header():
    """Server requires X-Requested-With: XMLHttpRequest to return JSON."""
    captured_headers = {}

    def fake_get(url, headers=None, **kwargs):
        captured_headers.update(headers or {})
        return _mock_resp([])

    with patch("src.ingestion.understat_fetcher.requests.get", side_effect=fake_get), \
         patch("src.ingestion.understat_fetcher.time.sleep"):
        fetch_league_season("E0", 2023, delay=0)

    assert captured_headers.get("X-Requested-With") == "XMLHttpRequest"


def test_fetch_league_season_skips_malformed_records():
    records = [
        _match_record(),
        {"id": "bad", "isResult": True, "datetime": "2024-01-01",
         "h": {}, "a": {}, "xG": {}},
    ]
    with patch("src.ingestion.understat_fetcher.requests.get", return_value=_mock_resp(records)), \
         patch("src.ingestion.understat_fetcher.time.sleep"):
        df = fetch_league_season("E0", 2023, delay=0)

    assert len(df) == 1  # malformed record skipped without raising


# ---------------------------------------------------------------------------
# TeamNameMapper
# ---------------------------------------------------------------------------

def test_team_name_mapper_explicit_mapping(tmp_path):
    mapping = {"Wolverhampton Wanderers": "Wolves"}
    mapping_file = tmp_path / "team_mapping.json"
    mapping_file.write_text(json.dumps(mapping))

    mapper = TeamNameMapper(mapping_path=str(mapping_file))
    assert mapper.map_team("Wolverhampton Wanderers") == "Wolves"


def test_team_name_mapper_fuzzy_fallback(tmp_path):
    mapping_file = tmp_path / "team_mapping.json"
    mapping_file.write_text("{}")
    mapper = TeamNameMapper(mapping_path=str(mapping_file), min_similarity=0.70)
    candidates = ["Arsenal", "Chelsea", "Liverpool"]
    result = mapper.map_team("Arsenal FC", candidates=candidates)
    assert result == "Arsenal"


def test_team_name_mapper_missing_file_returns_name(tmp_path):
    mapper = TeamNameMapper(mapping_path=str(tmp_path / "nonexistent.json"))
    assert mapper.map_team("Arsenal") == "Arsenal"


def test_team_name_mapper_production_mapping():
    """Smoke-test that config/team_mapping.json covers key EPL mismatches."""
    mapper = TeamNameMapper(mapping_path="config/team_mapping.json")
    candidates = ["Wolves", "Nott'm Forest", "Newcastle", "West Ham", "Brighton", "West Brom"]
    assert mapper.map_team("Wolverhampton Wanderers", candidates) == "Wolves"
    assert mapper.map_team("Nottingham Forest", candidates) == "Nott'm Forest"
    assert mapper.map_team("Newcastle United", candidates) == "Newcastle"
    assert mapper.map_team("West Ham United", candidates) == "West Ham"
    assert mapper.map_team("Brighton & Hove Albion", candidates) == "Brighton"
    assert mapper.map_team("West Bromwich Albion", candidates) == "West Brom"


# ---------------------------------------------------------------------------
# update_raw_matches_xg
# ---------------------------------------------------------------------------

def _make_db_manager(raw_rows: list[tuple]) -> MagicMock:
    raw_df = pd.DataFrame(raw_rows, columns=["match_id", "date", "home_team", "away_team"])
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = raw_df
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    mock_db = MagicMock()
    mock_db.connection.return_value = mock_conn
    return mock_db


def test_update_raw_matches_xg_matches_and_updates(tmp_path):
    raw_rows = [("match_001", "2024-08-17", "Arsenal", "Chelsea")]
    mock_db = _make_db_manager(raw_rows)

    understat_df = pd.DataFrame([{
        "date": pd.Timestamp("2024-08-17"),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "xg_h": 1.23,
        "xg_a": 0.87,
    }])

    mapping = tmp_path / "team_mapping.json"
    mapping.write_text("{}")
    result = update_raw_matches_xg(understat_df, mock_db, mapping_path=str(mapping))

    assert result["matched"] == 1
    assert result["updated"] == 1
    assert result["unmatched"] == 0


def test_update_raw_matches_xg_team_name_remapped(tmp_path):
    raw_rows = [("match_002", "2024-09-01", "Wolves", "West Brom")]
    mock_db = _make_db_manager(raw_rows)

    understat_df = pd.DataFrame([{
        "date": pd.Timestamp("2024-09-01"),
        "home_team": "Wolverhampton Wanderers",
        "away_team": "West Bromwich Albion",
        "xg_h": 1.5,
        "xg_a": 0.4,
    }])

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "Wolverhampton Wanderers": "Wolves",
        "West Bromwich Albion": "West Brom",
    }))
    result = update_raw_matches_xg(understat_df, mock_db, mapping_path=str(mapping))

    assert result["matched"] == 1
    assert result["unmatched"] == 0


def test_update_raw_matches_xg_unmatched_date(tmp_path):
    raw_rows = [("match_003", "2024-08-17", "Arsenal", "Chelsea")]
    mock_db = _make_db_manager(raw_rows)

    understat_df = pd.DataFrame([{
        "date": pd.Timestamp("2024-08-18"),  # wrong date
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "xg_h": 1.0,
        "xg_a": 1.0,
    }])

    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}")
    result = update_raw_matches_xg(understat_df, mock_db, mapping_path=str(mapping))

    assert result["matched"] == 0
    assert result["unmatched"] == 1


def test_update_raw_matches_xg_empty_understat(tmp_path):
    raw_rows = [("match_004", "2024-08-17", "Arsenal", "Chelsea")]
    mock_db = _make_db_manager(raw_rows)

    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}")
    result = update_raw_matches_xg(pd.DataFrame(), mock_db, mapping_path=str(mapping))

    assert result["updated"] == 0
