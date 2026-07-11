"""W02: POST /api/recommendations -- end-to-end endpoint behavior. run_agent
is mocked at the app boundary (no real LLM call in this test suite, matching
W20's own future zero-network-calls requirement, applied here from the start)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.main import app

_VALID_MARKET = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 2.1,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
}

_VALID_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [_VALID_MARKET],
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def test_valid_request_returns_schema_valid_recommendation():
    with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION) as mock_run:
        with TestClient(app) as client:
            response = client.post(
                "/api/recommendations",
                json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "league": "E0"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["overall"] == "direct_bet"
    assert len(body["markets"]) == 1
    assert body["invalid_market_count"] == 0
    mock_run.assert_called_once()
    match_info = mock_run.call_args.args[0]
    assert match_info == {"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "league": "E0"}


def test_malformed_market_degrades_gracefully_not_a_500():
    bad_market = {**_VALID_MARKET, "value_edge": "high"}
    raw = {**_VALID_RECOMMENDATION, "markets": [bad_market]}

    with patch("app.backend.recommendations.run_agent", return_value=raw):
        with TestClient(app) as client:
            response = client.post(
                "/api/recommendations",
                json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["markets"] == []
    assert body["invalid_market_count"] == 1


def test_odds_are_passed_through_to_match_info():
    with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION) as mock_run:
        with TestClient(app) as client:
            client.post(
                "/api/recommendations",
                json={
                    "home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22",
                    "odds": {"home": 1.5, "draw": 4.0, "away": 6.0},
                },
            )

    match_info = mock_run.call_args.args[0]
    assert match_info["odds"] == {"home": 1.5, "draw": 4.0, "away": 6.0}


def test_missing_required_field_returns_422():
    with TestClient(app) as client:
        response = client.post("/api/recommendations", json={"home_team": "Arsenal"})
    assert response.status_code == 422
