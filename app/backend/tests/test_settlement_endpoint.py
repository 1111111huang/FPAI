"""W13: POST /api/bets/settle-open -- on-demand settlement trigger (no
scheduler; W08/W09 are deliberately deferred). Reuses get_fixtures_client
(W05's FootballDataClient) rather than a second client instance -- results
and fixtures are the same underlying API/rate-limit budget."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
from fastapi.testclient import TestClient

from app.backend import bets
from app.backend.bet_tracker import BetTracker
from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app


def _override_tracker(tmp_path: Path) -> BetTracker:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    app.dependency_overrides[bets.get_bet_tracker] = lambda: tracker
    return tracker


@pytest.fixture(autouse=True)
def sweden_client_mock():
    """W57: settle_open now also consults the Sweden results client
    alongside football-data.org -- defaulted to empty so every pre-existing
    EPL-only test here keeps working unchanged (mirrors the same pattern in
    test_fixtures_endpoint.py)."""
    with patch("app.backend.main.get_sweden_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_results.return_value = []
        yield mock_get_client.return_value


def test_settle_open_endpoint_settles_a_finished_match_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    try:
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = [
                NormalizedMatch(
                    match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED",
                    home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
                ),
            ]
            with TestClient(app) as client:
                response = client.post("/api/bets/settle-open")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["outcome"] == "won"
        assert body[0]["profit_loss"] == 10.0
        assert tracker.get_bet(bet.id).outcome == "won"
    finally:
        app.dependency_overrides.clear()


def test_settle_open_endpoint_settles_a_swedish_bet(tmp_path: Path, sweden_client_mock):
    """W57: a Swedish bet's match_id is only known to The Odds API's results,
    never football-data.org's -- confirms the endpoint actually wires
    get_sweden_fixtures_client() through, not just the settlement module's
    own optional-param support (already covered in test_settlement.py)."""
    tracker = _override_tracker(tmp_path)
    bet = tracker.create_bet(
        match_id="sw1", date="2026-08-22", home_team="Malmo FF", away_team="AIK",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    sweden_client_mock.get_results.return_value = [
        NormalizedMatch(
            match_id="sw1", utc_date="2026-08-22T15:00:00Z", status="FINISHED",
            home_team="Malmo FF", away_team="AIK", home_goals=2, away_goals=1,
        ),
    ]
    try:
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.post("/api/bets/settle-open")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["outcome"] == "won"
        assert tracker.get_bet(bet.id).outcome == "won"
    finally:
        app.dependency_overrides.clear()


def test_settle_open_endpoint_returns_empty_list_when_nothing_settles(tmp_path: Path):
    _override_tracker(tmp_path)
    try:
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.post("/api/bets/settle-open")
        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.clear()
