"""W04 support: GET /api/fixtures wraps W05's FootballDataClient directly so
the frontend has a real fixture list to render cards for -- no story yet
covers this narrowly (W09, which would populate fixtures via the cache,
is built last), so this is the minimal connective tissue W04 actually needs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
import requests
from fastapi.testclient import TestClient

from app.backend.football_data_client import NormalizedMatch
from app.backend.main import _fixture_cache, _fixture_cache_pending, _split_fixture_date_range, app


@pytest.fixture(autouse=True)
def _clear_fixture_cache():
    """W52: the TTL cache (and its in-flight-request tracking dict) is
    module-level state (mirroring the existing `_fixtures_client` singleton
    pattern) -- clear it before every test so identical date ranges reused
    across unrelated test cases in this file don't leak cached results (or
    a stuck pending entry) between them."""
    _fixture_cache.clear()
    _fixture_cache_pending.clear()
    yield
    _fixture_cache.clear()
    _fixture_cache_pending.clear()


@pytest.fixture(autouse=True)
def sweden_client_mock():
    """W57: /api/fixtures now also merges in Sweden (Allsvenskan) fixtures
    from the new Odds-API-backed SwedenFixturesClient, alongside
    football-data.org's EPL fixtures. Autouse + defaulted to empty so every
    pre-existing EPL-only test in this file keeps working completely
    unchanged (SWE contributes nothing unless a test explicitly configures
    this mock) -- mirrors the existing _clear_fixture_cache autouse pattern."""
    with patch("app.backend.main.get_sweden_fixtures_client") as mock_get_client:
        client_mock = mock_get_client.return_value
        client_mock.get_fixtures.return_value = []
        client_mock.get_results.return_value = []
        yield client_mock


@pytest.fixture(autouse=True)
def la_liga_client_mock():
    """W76: /api/fixtures now also merges in La Liga (SP1) fixtures via
    get_la_liga_fixtures_client() -- a separate accessor patched independently
    of get_fixtures_client() specifically so this mock's own default-empty
    behavior can't collide with any pre-existing E0-only test's blanket
    `.return_value` on the *E0* client mock. Autouse + defaulted to empty,
    mirroring sweden_client_mock exactly, so every pre-existing test in this
    file keeps working completely unchanged."""
    with patch("app.backend.main.get_la_liga_fixtures_client") as mock_get_client:
        client_mock = mock_get_client.return_value
        client_mock.get_fixtures.return_value = []
        client_mock.get_results.return_value = []
        yield client_mock


@pytest.fixture(autouse=True)
def _all_competitions_display_enabled():
    """SWE was flipped to display_enabled=False in the real
    config/competitions.yaml (not a major league, big-5 season starting) --
    this file's tests exercise the merge/tag/cache mechanics for all
    competitions regardless of that live toggle, so default it back to "all
    on" here (mirrors sweden_client_mock/la_liga_client_mock's own
    autouse-default-then-override pattern). The toggle's own skip-when-off
    behavior is covered separately, by
    test_fixtures_endpoint_skips_a_display_disabled_competition_entirely."""
    with patch(
        "app.backend.main.list_display_enabled_competition_ids",
        return_value=["E0", "SWE", "SP1", "I1", "D1", "F1"],
    ):
        yield


@pytest.fixture(autouse=True)
def new_leagues_client_mock():
    """W136: /api/fixtures now also merges in Serie A (I1), Bundesliga (D1),
    and Ligue 1 (F1) fixtures, each via its own thin accessor
    (get_serie_a_fixtures_client/get_bundesliga_fixtures_client/
    get_ligue1_fixtures_client) -- mirrors la_liga_client_mock exactly, one
    fixture covering all three since they share the identical shape (same
    football-data.org provider class, only the competition_code differs).
    Autouse + defaulted to empty, so every pre-existing test in this file
    keeps working completely unchanged. Returns a dict keyed by league code
    so a test can configure just the one(s) it needs."""
    mocks: dict[str, MagicMock] = {}
    with patch("app.backend.main.get_serie_a_fixtures_client") as mock_i1, \
         patch("app.backend.main.get_bundesliga_fixtures_client") as mock_d1, \
         patch("app.backend.main.get_ligue1_fixtures_client") as mock_f1:
        for code, mock_get_client in (("I1", mock_i1), ("D1", mock_d1), ("F1", mock_f1)):
            client_mock = mock_get_client.return_value
            client_mock.get_fixtures.return_value = []
            client_mock.get_results.return_value = []
            mocks[code] = client_mock
        yield mocks


def test_fixtures_endpoint_skips_a_display_disabled_competition_entirely(sweden_client_mock):
    """A competition with display_enabled=False must be skipped before its
    client is even called -- not merely filtered out of the response --
    since the whole point is to stop spending its fetch/API-quota cost too."""
    sweden_client_mock.get_fixtures.return_value = [_SWEDISH_FIXTURE]
    with patch("app.backend.main.list_display_enabled_competition_ids", return_value=["E0"]):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2027-07-01", "date_to": "2027-07-05"}
                )
    body = response.json()
    assert {m["home_team"] for m in body} == {"Chelsea"}
    sweden_client_mock.get_fixtures.assert_not_called()


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
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 19)):
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
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 19)):
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
    # Nitpick from code review: the detail must be a clean, user-facing
    # message -- not a passthrough of the raw upstream exception text (which
    # would leak implementation details and isn't something a frontend
    # error banner should show verbatim).
    assert "429" not in body["detail"]
    assert "football-data.org" not in body["detail"].lower()
    assert "temporarily unavailable" in body["detail"].lower()


def test_fixtures_endpoint_concurrent_requests_dedupe_to_a_single_upstream_call():
    """Code review follow-up (post-21e6bf9): the sequential dedup test above
    only proves a *second* request arriving after the first has already
    finished (and cached) hits the client once. It does NOT prove anything
    about two requests that are genuinely in flight *at the same time* --
    e.g. React StrictMode's double-effect-invocation in dev, or two browser
    tabs loading at once -- both racing past an empty cache before either
    has finished and populated it. This test forces that exact overlap: the
    mocked client blocks (via a real thread, since it's invoked through
    run_in_threadpool) until both requests have demonstrably started, then
    releases them together.

    Both requests are issued from their own background thread (neither from
    the main test thread) -- if either ran inline on the main thread, that
    thread would be synchronously blocked inside client.get() and could
    never reach the `release.set()` call below, deadlocking the test
    regardless of whether de-duplication works. Using two threads lets the
    main thread stay free to observe both are in flight and then unblock
    them."""
    entered = threading.Event()
    release = threading.Event()

    def slow_get_fixtures(date_from=None, date_to=None):
        entered.set()
        assert release.wait(timeout=5), "release was never signalled -- test deadlocked"
        return []

    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.side_effect = slow_get_fixtures
        with TestClient(app) as client:
            responses: dict[str, object] = {}

            def run_request(name: str):
                responses[name] = client.get(
                    "/api/fixtures", params={"date_from": "2027-05-01", "date_to": "2027-05-05"}
                )

            first_thread = threading.Thread(target=run_request, args=("first",))
            first_thread.start()

            assert entered.wait(timeout=5), "first request never reached the upstream client call"

            # A second, genuinely concurrent request for the identical key,
            # started while the first upstream call is still in flight (the
            # cache has nothing to serve yet). Without in-flight
            # de-duplication this would start its own second upstream call
            # (which would also call entered.set()/release.wait() -- either
            # way, both requests are guaranteed to be blocked on `release`
            # by the time the short sleep below elapses).
            second_thread = threading.Thread(target=run_request, args=("second",))
            second_thread.start()
            time.sleep(0.3)

            release.set()
            first_thread.join(timeout=5)
            second_thread.join(timeout=5)
            assert not first_thread.is_alive(), "first request thread never completed"
            assert not second_thread.is_alive(), "second request thread never completed"

    assert responses["first"].status_code == 200
    assert responses["second"].status_code == 200
    mock_client.get_fixtures.assert_called_once_with(date_from="2027-05-01", date_to="2027-05-05")


def test_fixtures_endpoint_cache_expires_after_ttl_and_refetches():
    """The within-TTL dedup test proves two quick, sequential requests only
    hit the client once. This proves the other half: once the TTL has
    genuinely elapsed, a third request for the same key must hit the client
    again -- the cache must not pin stale data (or suppress real calls)
    forever. Patches `_fixture_cache_now` (mirrors this file's existing
    `_current_real_date` patching convention) to advance time deterministically
    rather than a real 60-second sleep."""
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.return_value = []

        fake_now = [1_000.0]
        with patch("app.backend.main._fixture_cache_now", side_effect=lambda: fake_now[0]):
            with TestClient(app) as client:
                first = client.get(
                    "/api/fixtures", params={"date_from": "2027-06-01", "date_to": "2027-06-05"}
                )
                # Still within the 60s TTL -- must serve the cached entry.
                fake_now[0] += 30.0
                second = client.get(
                    "/api/fixtures", params={"date_from": "2027-06-01", "date_to": "2027-06-05"}
                )
                # Past the 60s TTL from the first call -- must re-fetch.
                fake_now[0] += 31.0
                third = client.get(
                    "/api/fixtures", params={"date_from": "2027-06-01", "date_to": "2027-06-05"}
                )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert mock_client.get_fixtures.call_count == 2
    mock_client.get_fixtures.assert_called_with(date_from="2027-06-01", date_to="2027-06-05")


# ---------------------------------------------------------------------------
# W57: /api/fixtures merges in Sweden (Allsvenskan) alongside EPL, sourced
# from the new SwedenFixturesClient (The Odds API) rather than
# football-data.org (which W55 confirmed doesn't cover Allsvenskan at all).
# ---------------------------------------------------------------------------

_SWEDISH_FIXTURE = NormalizedMatch(
    match_id="80dbae8461497b97aa471f4d7a4c17a2", utc_date="2026-08-25T17:00:00Z", status="SCHEDULED",
    home_team="Malmo FF", away_team="AIK", home_goals=None, away_goals=None,
)
_SWEDISH_RESULT = NormalizedMatch(
    match_id="9a15519b05f7ec514d35ca4677815355", utc_date="2025-03-08T12:00:00Z", status="FINISHED",
    home_team="Kalmar FF", away_team="Hammarby IF", home_goals=2, away_goals=1,
)


def test_fixtures_endpoint_merges_sweden_fixtures_alongside_epl(sweden_client_mock):
    sweden_client_mock.get_fixtures.return_value = [_SWEDISH_FIXTURE]
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 19)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
                )

    assert response.status_code == 200
    body = response.json()
    assert {m["home_team"] for m in body} == {"Chelsea", "Malmo FF"}
    sweden_client_mock.get_fixtures.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-28")


def test_fixtures_endpoint_tags_each_fixture_with_its_source_competition(sweden_client_mock):
    """The frontend has no way to distinguish an EPL fixture from an
    Allsvenskan one unless the endpoint tags it -- both currently come back
    as the same NormalizedMatch shape with no competition field. Tagging
    must happen at the merge point in main.py (not just inside each
    client's own normalize function), since this test -- like the existing
    W57 tests above -- mocks the clients directly and bypasses their
    internal normalize functions entirely."""
    sweden_client_mock.get_fixtures.return_value = [_SWEDISH_FIXTURE]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
        with TestClient(app) as client:
            response = client.get(
                "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
            )

    assert response.status_code == 200
    body = response.json()
    by_team = {m["home_team"]: m["competition"] for m in body}
    assert by_team == {"Chelsea": "E0", "Malmo FF": "SWE"}


def test_fixtures_endpoint_tags_epl_results_e0(sweden_client_mock):
    """Mirrors the fixtures-side test above for the get_results() (past
    date range) path -- a separate code path in get_fixtures() with its own
    merge call, so needs its own coverage."""
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = [_REAL_RESULT]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                )

    body = response.json()
    assert body[0]["competition"] == "E0"


def test_fixtures_endpoint_tags_sweden_results_swe(sweden_client_mock):
    # W71: the past-date SWE branch is sourced from
    # historical_results_from_raw_matches, not sweden_client.get_results()
    # (see the reconciliation note on the test below) -- patched here rather
    # than through sweden_client_mock.
    with patch("app.backend.main.historical_results_from_raw_matches", return_value=[_SWEDISH_RESULT]):
        with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
            with patch("app.backend.main.get_fixtures_client") as mock_get_client:
                mock_get_client.return_value.get_results.return_value = []
                with TestClient(app) as client:
                    response = client.get(
                        "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                    )

    body = response.json()
    assert body[0]["competition"] == "SWE"


def test_fixtures_endpoint_wholly_past_range_sources_swedish_results_not_football_data(sweden_client_mock):
    """The past portion of the range must go through raw_matches (via
    historical_results_from_raw_matches), never football-data.org -- W55
    confirmed football-data.org has no Allsvenskan coverage on this
    account's plan at all, so routing Sweden's results through
    FootballDataClient would silently return nothing rather than erroring.

    W71 reconciliation: this test previously asserted
    sweden_client_mock.get_results.assert_called_once_with(...) -- i.e. that
    the past-date SWE branch went through SwedenFixturesClient (The Odds
    API). That's now intentionally false: The Odds API's /scores endpoint
    structurally cannot serve an arbitrary historical date (daysFrom<=3 is a
    hard provider limit), so W71 moved this branch onto
    historical_results_from_raw_matches (backed by raw_matches, which has
    real Allsvenskan history back to 2012) instead. Updated to patch and
    assert against that function directly, and to confirm
    sweden_client.get_results() is no longer called at all for this path."""
    with patch(
        "app.backend.main.historical_results_from_raw_matches", return_value=[_SWEDISH_RESULT]
    ) as mock_historical_swe:
        with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
            with patch("app.backend.main.get_fixtures_client") as mock_get_client:
                mock_get_client.return_value.get_results.return_value = [_REAL_RESULT]
                with TestClient(app) as client:
                    response = client.get(
                        "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                    )

    assert response.status_code == 200
    body = response.json()
    assert {m["home_team"] for m in body} == {"Liverpool", "Kalmar FF"}
    mock_historical_swe.assert_called_once_with(date_from="2025-03-08", date_to="2025-03-08")
    sweden_client_mock.get_results.assert_not_called()
    sweden_client_mock.get_fixtures.assert_not_called()


def test_fixtures_endpoint_sources_historical_swe_results_from_raw_matches_not_the_odds_api(sweden_client_mock):
    """W71: the past-date SWE branch must not call the Odds-API-backed
    sweden_client.get_results() at all -- that endpoint structurally can't
    serve an arbitrary historical date. Confirms the real raw_matches-backed
    source is used instead, via a real (not mocked) query against a date
    with known real SWE data."""
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 20)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2026-05-24", "date_to": "2026-05-24"}
                )

    assert response.status_code == 200
    body = response.json()
    swe_rows = [m for m in body if m["competition"] == "SWE"]
    assert len(swe_rows) > 0
    sweden_client_mock.get_results.assert_not_called()


def test_fixtures_endpoint_sweden_fixture_cache_key_is_independent_of_epls(sweden_client_mock):
    """A cached EPL call for a date range must not accidentally serve (or be
    served by) Sweden's cache entry for the identical range -- they must be
    keyed separately even though the date range string is the same."""
    sweden_client_mock.get_fixtures.return_value = [_SWEDISH_FIXTURE]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
        with TestClient(app) as client:
            response = client.get(
                "/api/fixtures", params={"date_from": "2027-05-01", "date_to": "2027-05-05"}
            )

    body = response.json()
    assert {m["home_team"] for m in body} == {"Chelsea", "Malmo FF"}
    mock_get_client.return_value.get_fixtures.assert_called_once_with(date_from="2027-05-01", date_to="2027-05-05")
    sweden_client_mock.get_fixtures.assert_called_once_with(date_from="2027-05-01", date_to="2027-05-05")


# ---------------------------------------------------------------------------
# W76: /api/fixtures merges in La Liga (SP1) alongside E0/SWE, sourced from
# the *same* football-data.org provider as E0 (W74 live-confirmed the `PD`
# competition code covers La Liga directly) via get_la_liga_fixtures_client()
# -- a separate accessor purely for test/cache isolation, not a separate
# client class the way SWE needed one.
# ---------------------------------------------------------------------------

_LA_LIGA_FIXTURE = NormalizedMatch(
    match_id="560701", utc_date="2026-08-25T19:00:00Z", status="SCHEDULED",
    home_team="Real Madrid", away_team="Sevilla FC", home_goals=None, away_goals=None,
)
_LA_LIGA_RESULT = NormalizedMatch(
    match_id="560700", utc_date="2025-03-08T20:00:00Z", status="FINISHED",
    home_team="Barcelona", away_team="Villarreal CF", home_goals=3, away_goals=1,
)


def test_fixtures_endpoint_merges_la_liga_fixtures_alongside_e0_and_swe(la_liga_client_mock):
    la_liga_client_mock.get_fixtures.return_value = [_LA_LIGA_FIXTURE]
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 19)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
                )

    assert response.status_code == 200
    body = response.json()
    assert {m["home_team"] for m in body} == {"Chelsea", "Real Madrid"}
    la_liga_client_mock.get_fixtures.assert_called_once_with(
        competition_code="PD", date_from="2026-08-21", date_to="2026-08-28"
    )


def test_fixtures_endpoint_tags_la_liga_fixtures_sp1(la_liga_client_mock):
    la_liga_client_mock.get_fixtures.return_value = [_LA_LIGA_FIXTURE]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
        with TestClient(app) as client:
            response = client.get(
                "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
            )

    body = response.json()
    by_team = {m["home_team"]: m["competition"] for m in body}
    assert by_team == {"Chelsea": "E0", "Real Madrid": "SP1"}


def test_fixtures_endpoint_tags_la_liga_results_sp1(la_liga_client_mock):
    la_liga_client_mock.get_results.return_value = [_LA_LIGA_RESULT]
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                )

    body = response.json()
    assert body[0]["competition"] == "SP1"
    la_liga_client_mock.get_results.assert_called_once_with(
        competition_code="PD", date_from="2025-03-08", date_to="2025-03-08"
    )


def test_fixtures_endpoint_la_liga_fixture_cache_key_is_independent_of_e0s_and_swes(la_liga_client_mock, sweden_client_mock):
    """A cached E0 call for a date range must not accidentally serve (or be
    served by) La Liga's cache entry for the identical range -- keyed
    separately even though the date range string is the same, mirroring
    the existing SWE cache-key-independence test."""
    la_liga_client_mock.get_fixtures.return_value = [_LA_LIGA_FIXTURE]
    sweden_client_mock.get_fixtures.return_value = [_SWEDISH_FIXTURE]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
        with TestClient(app) as client:
            response = client.get(
                "/api/fixtures", params={"date_from": "2027-06-10", "date_to": "2027-06-15"}
            )

    body = response.json()
    assert {m["home_team"] for m in body} == {"Chelsea", "Malmo FF", "Real Madrid"}
    mock_get_client.return_value.get_fixtures.assert_called_once_with(date_from="2027-06-10", date_to="2027-06-15")
    sweden_client_mock.get_fixtures.assert_called_once_with(date_from="2027-06-10", date_to="2027-06-15")
    la_liga_client_mock.get_fixtures.assert_called_once_with(
        competition_code="PD", date_from="2027-06-10", date_to="2027-06-15"
    )


# ---------------------------------------------------------------------------
# W136: /api/fixtures merges in Serie A (I1), Bundesliga (D1), and Ligue 1
# (F1) alongside E0/SWE/SP1, each sourced from the *same* football-data.org
# provider as E0/SP1 (W134 live-confirmed SA/BL1/FL1) via their own thin
# accessors. One parametrized block covering all three, rather than
# tripling W76's own SP1 test bodies -- the shape is identical per league.
# ---------------------------------------------------------------------------

_NEW_LEAGUE_FIXTURE = NormalizedMatch(
    match_id="700001", utc_date="2026-08-25T19:00:00Z", status="SCHEDULED",
    home_team="Juventus", away_team="AC Milan", home_goals=None, away_goals=None,
)
_NEW_LEAGUE_RESULT = NormalizedMatch(
    match_id="700000", utc_date="2025-03-08T20:00:00Z", status="FINISHED",
    home_team="Bayern Munich", away_team="Borussia Dortmund", home_goals=2, away_goals=1,
)

_NEW_LEAGUE_CODES = [("I1", "SA"), ("D1", "BL1"), ("F1", "FL1")]


@pytest.mark.parametrize("league,fd_code", _NEW_LEAGUE_CODES)
def test_fixtures_endpoint_merges_new_league_fixtures_alongside_existing(
    league, fd_code, new_leagues_client_mock
):
    new_leagues_client_mock[league].get_fixtures.return_value = [_NEW_LEAGUE_FIXTURE]
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 19)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
                )

    assert response.status_code == 200
    body = response.json()
    assert {m["home_team"] for m in body} == {"Chelsea", _NEW_LEAGUE_FIXTURE.home_team}
    new_leagues_client_mock[league].get_fixtures.assert_called_once_with(
        competition_code=fd_code, date_from="2026-08-21", date_to="2026-08-28"
    )


@pytest.mark.parametrize("league,fd_code", _NEW_LEAGUE_CODES)
def test_fixtures_endpoint_tags_new_league_fixtures_correctly(league, fd_code, new_leagues_client_mock):
    new_leagues_client_mock[league].get_fixtures.return_value = [_NEW_LEAGUE_FIXTURE]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
        with TestClient(app) as client:
            response = client.get(
                "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
            )

    body = response.json()
    by_team = {m["home_team"]: m["competition"] for m in body}
    assert by_team == {"Chelsea": "E0", _NEW_LEAGUE_FIXTURE.home_team: league}


@pytest.mark.parametrize("league,fd_code", _NEW_LEAGUE_CODES)
def test_fixtures_endpoint_tags_new_league_results_correctly(league, fd_code, new_leagues_client_mock):
    new_leagues_client_mock[league].get_results.return_value = [_NEW_LEAGUE_RESULT]
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                )

    body = response.json()
    assert body[0]["competition"] == league
    new_leagues_client_mock[league].get_results.assert_called_once_with(
        competition_code=fd_code, date_from="2025-03-08", date_to="2025-03-08"
    )


def test_fixtures_endpoint_new_leagues_cache_keys_are_independent_of_each_other_and_existing(
    new_leagues_client_mock, la_liga_client_mock, sweden_client_mock
):
    """A cached E0/SP1/SWE call for a date range must not accidentally
    serve (or be served by) any of the three new leagues' cache entries for
    the identical range -- mirrors the existing SP1/SWE independence test,
    now with three more competitions in the same merge."""
    new_leagues_client_mock["I1"].get_fixtures.return_value = [_NEW_LEAGUE_FIXTURE]
    la_liga_client_mock.get_fixtures.return_value = [_LA_LIGA_FIXTURE]
    sweden_client_mock.get_fixtures.return_value = [_SWEDISH_FIXTURE]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
        with TestClient(app) as client:
            response = client.get(
                "/api/fixtures", params={"date_from": "2027-06-10", "date_to": "2027-06-15"}
            )

    body = response.json()
    assert {m["home_team"] for m in body} == {
        "Chelsea", "Malmo FF", "Real Madrid", _NEW_LEAGUE_FIXTURE.home_team,
    }
    mock_get_client.return_value.get_fixtures.assert_called_once_with(date_from="2027-06-10", date_to="2027-06-15")
    new_leagues_client_mock["I1"].get_fixtures.assert_called_once_with(
        competition_code="SA", date_from="2027-06-10", date_to="2027-06-15"
    )
    # D1/F1 contributed nothing this call (only I1 was configured above) --
    # confirms their own cache slots don't accidentally pick up I1's result.
    new_leagues_client_mock["D1"].get_fixtures.assert_called_once()
    new_leagues_client_mock["F1"].get_fixtures.assert_called_once()


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
