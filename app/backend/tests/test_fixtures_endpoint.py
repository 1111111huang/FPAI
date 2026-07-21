"""W04 support: GET /api/fixtures wraps W05's FootballDataClient directly so
the frontend has a real fixture list to render cards for -- no story yet
covers this narrowly (W09, which would populate fixtures via the cache,
is built last), so this is the minimal connective tissue W04 actually needs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
import requests
from fastapi.testclient import TestClient

from app.backend.football_data_client import NormalizedMatch
from app.backend.main import _fixture_cache, _split_fixture_date_range, app


@pytest.fixture(autouse=True)
def _clear_fixture_cache():
    """W52: the TTL cache is module-level state (mirroring the existing
    `_fixtures_client` singleton pattern) -- clear it before every test so
    identical date ranges reused across unrelated test cases in this file
    don't leak cached results between them."""
    _fixture_cache.clear()
    yield
    _fixture_cache.clear()


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
    """A range spanning real 'today' must be split at the boundary: the past
    portion (date_from..today) goes through get_results(), the future
    portion (today..date_to) goes through get_fixtures() -- today is queried
    on *both* sides, since a same-day match may already be FINISHED or still
    SCHEDULED depending on kickoff time -- and both are merged back in
    chronological order."""
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
    mock_client.get_results.assert_called_once_with(date_from="2025-03-08", date_to="2025-03-10")
    mock_client.get_fixtures.assert_called_once_with(date_from="2025-03-10", date_to="2025-03-12")


def test_fixtures_endpoint_today_only_query_checks_both_finished_and_scheduled():
    """A single-day query for exactly 'today' must check both statuses --
    an already-finished early-kickoff match must not vanish just because
    it's also technically 'today', which was a real gap found in code
    review of the first version of this fix (it previously treated 'today'
    as future-only, matching the pre-W45 behavior it was meant to close)."""
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_client = mock_get_client.return_value
            mock_client.get_results.return_value = [_REAL_RESULT]
            mock_client.get_fixtures.return_value = [_REAL_FIXTURE]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-10", "date_to": "2025-03-10"}
                )

    assert response.status_code == 200
    body = response.json()
    assert [m["home_team"] for m in body] == ["Liverpool", "Chelsea"]
    mock_client.get_results.assert_called_once_with(date_from="2025-03-10", date_to="2025-03-10")
    mock_client.get_fixtures.assert_called_once_with(date_from="2025-03-10", date_to="2025-03-10")


# ---------------------------------------------------------------------------
# W52: football-data.org's free tier (~10 req/min) was getting exhausted by
# repeated frontend navigation (Dashboard -> Match Explorer -> Bet form, each
# independently calling getFixtures() fresh on every mount against the same
# module-level `_fixtures_client` singleton), and the resulting 429 ->
# requests.exceptions.HTTPError was propagating uncaught to an unhandled 500
# instead of a clean degraded response. These tests cover both fixes: a
# short TTL cache to de-duplicate identical repeated calls, and a clean
# HTTPException instead of a raw 500 when the upstream call fails.
# ---------------------------------------------------------------------------


def test_fixtures_endpoint_deduplicates_identical_calls_within_ttl():
    """Two requests for the identical date range within the TTL window must
    only hit the underlying client once -- this is exactly the repeated-
    navigation pattern that was exhausting the shared rate-limit budget."""
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.return_value = []
        with TestClient(app) as client:
            first = client.get("/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"})
            second = client.get("/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"})

    assert first.status_code == 200
    assert second.status_code == 200
    mock_client.get_fixtures.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-28")


def test_fixtures_endpoint_upstream_http_error_returns_clean_degraded_response():
    """A requests.exceptions.HTTPError from the underlying client (e.g. the
    real 429 rate-limit error seen in the sandbox log) must not propagate as
    an unhandled 500 -- it must produce a clean, non-500 JSON error response
    the frontend can distinguish from a genuine empty-fixtures result."""
    fake_response = MagicMock()
    fake_response.status_code = 429
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.side_effect = requests.exceptions.HTTPError(
            "429 Client Error: for url: https://api.football-data.org/v4/competitions/PL/matches",
            response=fake_response,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/fixtures", params={"date_from": "2099-01-01", "date_to": "2099-01-02"})

    assert response.status_code == 503
    body = response.json()
    assert "detail" in body


class TestSplitFixtureDateRange:
    """Direct unit tests for the pure split helper -- cheaper and more
    precise than only exercising it through full HTTP round-trips."""

    def test_wholly_past_range(self):
        results, fixtures = _split_fixture_date_range("2025-03-01", "2025-03-05", date(2025, 3, 10))

        assert results == ("2025-03-01", "2025-03-05")
        assert fixtures is None

    def test_wholly_future_range(self):
        results, fixtures = _split_fixture_date_range("2025-03-15", "2025-03-20", date(2025, 3, 10))

        assert results is None
        assert fixtures == ("2025-03-15", "2025-03-20")

    def test_range_spanning_today(self):
        results, fixtures = _split_fixture_date_range("2025-03-05", "2025-03-15", date(2025, 3, 10))

        assert results == ("2025-03-05", "2025-03-10")
        assert fixtures == ("2025-03-10", "2025-03-15")

    def test_today_only(self):
        results, fixtures = _split_fixture_date_range("2025-03-10", "2025-03-10", date(2025, 3, 10))

        assert results == ("2025-03-10", "2025-03-10")
        assert fixtures == ("2025-03-10", "2025-03-10")

    def test_range_ending_exactly_on_today(self):
        results, fixtures = _split_fixture_date_range("2025-03-05", "2025-03-10", date(2025, 3, 10))

        assert results == ("2025-03-05", "2025-03-10")
        assert fixtures == ("2025-03-10", "2025-03-10")

    def test_range_starting_exactly_on_today(self):
        results, fixtures = _split_fixture_date_range("2025-03-10", "2025-03-15", date(2025, 3, 10))

        assert results == ("2025-03-10", "2025-03-10")
        assert fixtures == ("2025-03-10", "2025-03-15")

    def test_both_bounds_omitted_falls_back_to_fixtures_only(self):
        results, fixtures = _split_fixture_date_range(None, None, date(2025, 3, 10))

        assert results is None
        assert fixtures == (None, None)

    def test_one_bound_omitted_falls_back_to_fixtures_only(self):
        results, fixtures = _split_fixture_date_range("2025-03-01", None, date(2025, 3, 10))

        assert results is None
        assert fixtures == ("2025-03-01", None)

    def test_malformed_date_falls_back_to_fixtures_only(self):
        results, fixtures = _split_fixture_date_range("not-a-date", "2025-03-15", date(2025, 3, 10))

        assert results is None
        assert fixtures == ("not-a-date", "2025-03-15")

    def test_inverted_range_is_not_split_into_a_negative_window(self):
        """date_from > date_to is a malformed request the client will
        presumably reject/no-op on -- confirm the split doesn't silently
        produce a backwards or nonsensical sub-range for it."""
        results, fixtures = _split_fixture_date_range("2025-03-15", "2025-03-05", date(2025, 3, 10))

        # Wholly-past check (parsed_to < today) fires first: 2025-03-05 < 2025-03-10.
        assert results == ("2025-03-15", "2025-03-05")
        assert fixtures is None
