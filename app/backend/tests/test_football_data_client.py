"""W05: football-data.org client -- typed wrapper, rate-limit respect, and a
normalized internal shape independent of the provider's own field names.
Fixture payloads below are real, captured responses (2026-07-11) from
GET /v4/competitions/PL/matches, not guessed shapes."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import FootballDataClient, NormalizedMatch, _RateLimiter

_SCHEDULED_MATCH = {
    "area": {"id": 2072, "name": "England", "code": "ENG"},
    "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
    "season": {"id": 2502, "startDate": "2026-08-21", "endDate": "2027-05-30"},
    "id": 560542,
    "utcDate": "2026-08-21T19:00:00Z",
    "status": "SCHEDULED",
    "matchday": 1,
    "homeTeam": {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS"},
    "awayTeam": {"id": 1076, "name": "Coventry City FC", "shortName": "Coventry City", "tla": "COV"},
    "score": {"winner": None, "fullTime": {"home": None, "away": None}, "halfTime": {"home": None, "away": None}},
    "odds": {"msg": "Activate Odds-Package in User-Panel to retrieve odds."},
}

_FINISHED_MATCH = {
    "area": {"id": 2072, "name": "England", "code": "ENG"},
    "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
    "season": {"id": 2403, "startDate": "2025-08-15", "endDate": "2026-05-24"},
    "id": 538164,
    "utcDate": "2026-05-24T15:00:00Z",
    "status": "FINISHED",
    "matchday": 38,
    "homeTeam": {"id": 563, "name": "West Ham United FC", "shortName": "West Ham", "tla": "WHU"},
    "awayTeam": {"id": 341, "name": "Leeds United FC", "shortName": "Leeds United", "tla": "LEE"},
    "score": {"winner": "HOME_TEAM", "fullTime": {"home": 3, "away": 0}, "halfTime": {"home": 0, "away": 0}},
    "odds": {"msg": "Activate Odds-Package in User-Panel to retrieve odds."},
}


def _mock_session(matches: list[dict], headers: dict | None = None) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.headers = headers or {}
    response.json.return_value = {"count": len(matches), "matches": matches}
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return session


def test_get_fixtures_normalizes_scheduled_match() -> None:
    session = _mock_session([_SCHEDULED_MATCH])
    client = FootballDataClient(api_key="fake-key", session=session)

    fixtures = client.get_fixtures()

    assert fixtures == [
        NormalizedMatch(
            match_id="560542", utc_date="2026-08-21T19:00:00Z", status="SCHEDULED",
            home_team="Arsenal", away_team="Coventry City", home_goals=None, away_goals=None,
        )
    ]


def test_get_results_normalizes_finished_match_with_scores() -> None:
    session = _mock_session([_FINISHED_MATCH])
    client = FootballDataClient(api_key="fake-key", session=session)

    results = client.get_results()

    assert results == [
        NormalizedMatch(
            match_id="538164", utc_date="2026-05-24T15:00:00Z", status="FINISHED",
            home_team="West Ham", away_team="Leeds United", home_goals=3, away_goals=0,
        )
    ]


def test_get_fixtures_sends_auth_header_and_status_filter() -> None:
    session = _mock_session([])
    client = FootballDataClient(api_key="my-secret-key", session=session)

    client.get_fixtures(competition_code="PL")

    call = session.get.call_args
    assert call.kwargs["headers"]["X-Auth-Token"] == "my-secret-key"
    assert call.kwargs["params"]["status"] == "SCHEDULED"
    assert "PL" in call.args[0] or "PL" in call.kwargs.get("url", "")


def test_date_range_params_included_when_provided() -> None:
    session = _mock_session([])
    client = FootballDataClient(api_key="fake-key", session=session)

    client.get_results(date_from="2026-08-01", date_to="2026-08-31")

    params = session.get.call_args.kwargs["params"]
    assert params["dateFrom"] == "2026-08-01"
    assert params["dateTo"] == "2026-08-31"
    assert params["status"] == "FINISHED"


def test_empty_matches_returns_empty_list() -> None:
    session = _mock_session([])
    client = FootballDataClient(api_key="fake-key", session=session)
    assert client.get_fixtures() == []


def test_raises_for_http_error_status() -> None:
    session = MagicMock()
    response = MagicMock()
    response.headers = {}
    response.raise_for_status.side_effect = RuntimeError("500 Server Error")
    session.get.return_value = response
    client = FootballDataClient(api_key="fake-key", session=session)

    with pytest.raises(RuntimeError):
        client.get_fixtures()


def test_client_calls_rate_limiter_before_each_request() -> None:
    session = _mock_session([])
    rate_limiter = MagicMock()
    client = FootballDataClient(api_key="fake-key", session=session, rate_limiter=rate_limiter)

    client.get_fixtures()

    rate_limiter.wait_if_needed.assert_called_once()
    rate_limiter.update_from_headers.assert_called_once()


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------

def test_rate_limiter_does_not_sleep_when_requests_available() -> None:
    sleep_fn = MagicMock()
    limiter = _RateLimiter(sleep_fn=sleep_fn, time_fn=lambda: 100.0)
    limiter.update_from_headers({"x-requests-available-minute": "5", "X-RequestCounter-Reset": "60"})

    limiter.wait_if_needed()

    sleep_fn.assert_not_called()


def test_rate_limiter_sleeps_until_reset_when_exhausted() -> None:
    sleep_fn = MagicMock()
    limiter = _RateLimiter(sleep_fn=sleep_fn, time_fn=lambda: 100.0)
    limiter.update_from_headers({"x-requests-available-minute": "0", "X-RequestCounter-Reset": "45"})

    limiter.wait_if_needed()

    sleep_fn.assert_called_once_with(45)


def test_rate_limiter_does_not_sleep_before_any_response_seen() -> None:
    """No prior response yet -- nothing to rate-limit against."""
    sleep_fn = MagicMock()
    limiter = _RateLimiter(sleep_fn=sleep_fn, time_fn=lambda: 100.0)

    limiter.wait_if_needed()

    sleep_fn.assert_not_called()
