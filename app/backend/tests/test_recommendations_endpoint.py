"""W02: POST /api/recommendations -- end-to-end endpoint behavior. run_agent
is mocked at the app boundary (no real LLM call in this test suite, matching
W20's own future zero-network-calls requirement, applied here from the start)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.main import app
from app.backend.odds_api_client import NormalizedOdds

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


# W49: create_recommendation() now fetches odds itself (via
# build_odds_client()/eod_batch's match_odds team-matching, reused not
# reimplemented) when the caller didn't already supply them -- matching
# what the scheduled EOD/T-30 pipeline already does, instead of leaving the
# agent to rely on its own web_search (which can't succeed for a
# historical/sandboxed match and frequently fails for real matches too).


def _odds_client_returning(*events: NormalizedOdds) -> MagicMock:
    client = MagicMock()
    client.get_odds.return_value = list(events)
    return client


def test_odds_are_fetched_and_attached_when_not_supplied_and_a_match_is_found():
    fetched_event = NormalizedOdds(
        home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
        home_odds=1.5, draw_odds=4.0, away_odds=6.0,
    )
    with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION) as mock_run:
        with patch("app.backend.main.build_odds_client", return_value=_odds_client_returning(fetched_event)):
            with TestClient(app) as client:
                response = client.post(
                    "/api/recommendations",
                    json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "league": "E0"},
                )

    assert response.status_code == 200
    mock_run.assert_called_once()
    match_info = mock_run.call_args.args[0]
    assert match_info["odds"] == {"home": 1.5, "draw": 4.0, "away": 6.0}


def test_explicit_odds_take_precedence_over_fetched_odds():
    fetched_event = NormalizedOdds(
        home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
        home_odds=9.9, draw_odds=9.9, away_odds=9.9,
    )
    explicit_odds = {"home": 1.5, "draw": 4.0, "away": 6.0}
    with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION) as mock_run:
        with patch("app.backend.main.build_odds_client", return_value=_odds_client_returning(fetched_event)) as mock_build:
            with TestClient(app) as client:
                response = client.post(
                    "/api/recommendations",
                    json={
                        "home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22",
                        "odds": explicit_odds,
                    },
                )

    assert response.status_code == 200
    match_info = mock_run.call_args.args[0]
    assert match_info["odds"] == explicit_odds
    # Explicit odds skip the fetch entirely -- no reason to pay for/attempt
    # a lookup whose result would be discarded anyway.
    mock_build.assert_not_called()


def test_no_odds_client_configured_degrades_to_no_odds_not_an_error():
    with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION) as mock_run:
        with patch("app.backend.main.build_odds_client", return_value=None):
            with TestClient(app) as client:
                response = client.post(
                    "/api/recommendations",
                    json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "league": "E0"},
                )

    assert response.status_code == 200
    match_info = mock_run.call_args.args[0]
    assert "odds" not in match_info


def test_no_matching_odds_event_degrades_to_no_odds_not_an_error():
    unrelated_event = NormalizedOdds(
        home_team="Chelsea", away_team="Fulham", commence_time="2026-08-22T15:00:00Z",
        home_odds=1.5, draw_odds=4.0, away_odds=6.0,
    )
    with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION) as mock_run:
        with patch("app.backend.main.build_odds_client", return_value=_odds_client_returning(unrelated_event)):
            with TestClient(app) as client:
                response = client.post(
                    "/api/recommendations",
                    json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "league": "E0"},
                )

    assert response.status_code == 200
    match_info = mock_run.call_args.args[0]
    assert "odds" not in match_info


def test_odds_client_raising_degrades_to_no_odds_not_a_500():
    broken_client = MagicMock()
    broken_client.get_odds.side_effect = RuntimeError("simulated odds API failure")
    with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION) as mock_run:
        with patch("app.backend.main.build_odds_client", return_value=broken_client):
            with TestClient(app) as client:
                response = client.post(
                    "/api/recommendations",
                    json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "league": "E0"},
                )

    assert response.status_code == 200
    match_info = mock_run.call_args.args[0]
    assert "odds" not in match_info


def test_fetched_odds_are_recorded_in_the_cache_not_just_used_for_generation():
    """W49 follow-up (code review finding): the cache write must persist
    what was actually used for generation (match_info["odds"], including a
    server-side fetch), not just request.odds -- otherwise a fetched-odds
    generation is cached with odds={}, and t30_refresh.py's "odds unchanged,
    skip regeneration" dedup check sees a spurious change on every
    subsequent comparison."""
    fetched_event = NormalizedOdds(
        home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
        home_odds=1.5, draw_odds=4.0, away_odds=6.0,
    )
    mock_cache = MagicMock()
    app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
    try:
        with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION):
            with patch("app.backend.main.build_odds_client", return_value=_odds_client_returning(fetched_event)):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/recommendations",
                        json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "league": "E0"},
                    )
    finally:
        app.dependency_overrides.pop(recommendations.get_cache, None)

    assert response.status_code == 200
    mock_cache.record_generation.assert_called_once()
    recorded_odds = mock_cache.record_generation.call_args.kwargs["odds"]
    assert recorded_odds == {"home": 1.5, "draw": 4.0, "away": 6.0}
