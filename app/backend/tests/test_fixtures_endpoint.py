"""W04 support: GET /api/fixtures wraps W05's FootballDataClient directly so
the frontend has a real fixture list to render cards for -- no story yet
covers this narrowly (W09, which would populate fixtures via the cache,
is built last), so this is the minimal connective tissue W04 actually needs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app


def test_fixtures_endpoint_returns_normalized_matches():
    fake_fixtures = [
        NormalizedMatch(
            match_id="560542", utc_date="2026-08-21T19:00:00Z", status="SCHEDULED",
            home_team="Arsenal", away_team="Coventry City", home_goals=None, away_goals=None,
        ),
    ]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.return_value = fake_fixtures
        with TestClient(app) as client:
            response = client.get("/api/fixtures")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["home_team"] == "Arsenal"
    assert body[0]["away_team"] == "Coventry City"


def test_fixtures_endpoint_passes_date_range_through():
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.return_value = []
        with TestClient(app) as client:
            client.get("/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"})

    mock_client.get_fixtures.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-28")


def test_fixtures_endpoint_returns_empty_list_gracefully():
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = []
        with TestClient(app) as client:
            response = client.get("/api/fixtures")

    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# W45: wholly-past date ranges must source real historical results via
# get_results() (status=FINISHED), not get_fixtures() (status=SCHEDULED),
# which structurally can never return anything for a range that's already
# happened. `_current_real_date` is patched (not sandbox_clock) because the
# split is meant to track genuine wall-clock "today" -- independent of
# whatever date SANDBOX_MODE/SANDBOX_DATE is pretending "today" is.
# ---------------------------------------------------------------------------

_REAL_RESULT = NormalizedMatch(
    match_id="1", utc_date="2025-03-08T15:00:00Z", status="FINISHED",
    home_team="Liverpool", away_team="Man City", home_goals=2, away_goals=1,
)
_REAL_FIXTURE = NormalizedMatch(
    match_id="2", utc_date="2025-03-12T15:00:00Z", status="SCHEDULED",
    home_team="Chelsea", away_team="Arsenal", home_goals=None, away_goals=None,
)


def test_fixtures_endpoint_wholly_past_range_sources_real_results():
    """The exact W45 regression scenario: SANDBOX_DATE=2025-03-08 querying
    that same wholly-past day must surface real historical fixtures/scores
    via get_results(), and must NOT call get_fixtures() at all -- the
    endpoint's old behavior structurally returned [] here regardless of
    real data availability."""
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_client = mock_get_client.return_value
            mock_client.get_results.return_value = [_REAL_RESULT]
            mock_client.get_fixtures.return_value = []
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["home_team"] == "Liverpool"
    assert body[0]["home_goals"] == 2
    mock_client.get_results.assert_called_once_with(date_from="2025-03-08", date_to="2025-03-08")
    mock_client.get_fixtures.assert_not_called()


def test_fixtures_endpoint_wholly_future_range_still_uses_get_fixtures_only():
    """Regression guard: the existing, real (non-sandbox) forward-looking
    behavior must be unchanged when the whole requested range is today or
    later -- get_results() must not be called at all in this case."""
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 19)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_client = mock_get_client.return_value
            mock_client.get_fixtures.return_value = [_REAL_FIXTURE]
            mock_client.get_results.return_value = []
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
                )

    assert response.status_code == 200
    assert len(response.json()) == 1
    mock_client.get_fixtures.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-28")
    mock_client.get_results.assert_not_called()


def test_fixtures_endpoint_boundary_spanning_range_merges_results_and_fixtures():
    """A range spanning real 'today' must be split at the boundary: the
    past portion (date_from..yesterday) goes through get_results(), the
    future portion (today..date_to) goes through get_fixtures(), and both
    are merged back in chronological order."""
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_client = mock_get_client.return_value
            mock_client.get_results.return_value = [_REAL_RESULT]
            mock_client.get_fixtures.return_value = [_REAL_FIXTURE]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-12"}
                )

    assert response.status_code == 200
    body = response.json()
    assert [m["home_team"] for m in body] == ["Liverpool", "Chelsea"]
    mock_client.get_results.assert_called_once_with(date_from="2025-03-08", date_to="2025-03-09")
    mock_client.get_fixtures.assert_called_once_with(date_from="2025-03-10", date_to="2025-03-12")
