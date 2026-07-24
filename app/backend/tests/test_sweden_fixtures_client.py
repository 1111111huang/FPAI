"""W57: Sweden (Allsvenskan) fixtures/results client, backed by The Odds API.

W55's research found football-data.org's free tier doesn't cover Allsvenskan
at all (confirmed live: exactly 13 competitions, no Sweden), and
football-data.co.uk (the ML engine's own ingestion source,
src/ingestion/football_data/sweden_fetcher.py) is a played-results-only
historical file with zero forward-looking fixture rows -- neither can serve
as a fixtures source for Sweden. The Odds API (already integrated for odds,
W07) can serve fixtures via /events (confirmed live: 0 credits/call) and
results via /scores (confirmed live: 2 credits/call, capped at daysFrom<=3),
normalized here to the same NormalizedMatch shape FootballDataClient returns
so both sources can be treated uniformly by callers."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
from app.backend.sweden_fixtures_client import SPORT_KEY, SwedenFixturesClient

# Real event captured live 2026-07-23 (id/commence_time/team spelling, incl.
# diacritics as The Odds API actually returned them for these two teams --
# W59 found this isn't consistent across all Swedish clubs in the same
# response).
_UPCOMING_EVENT = {
    "id": "80dbae8461497b97aa471f4d7a4c17a2",
    "sport_key": "soccer_sweden_allsvenskan",
    "commence_time": "2026-07-24T17:00:00Z",
    "home_team": "Västerås SK",
    "away_team": "Örgryte IS",
}

_COMPLETED_EVENT = {
    "id": "9a15519b05f7ec514d35ca4677815355",
    "sport_key": "soccer_sweden_allsvenskan",
    "commence_time": "2026-07-20T12:00:00Z",
    "completed": True,
    "home_team": "Kalmar FF",
    "away_team": "Malmo FF",
    "scores": [{"name": "Kalmar FF", "score": "2"}, {"name": "Malmo FF", "score": "2"}],
}

_NOT_YET_COMPLETED_EVENT = {
    "id": "35e5441b0004bc8a8b8eb128c0d54996",
    "sport_key": "soccer_sweden_allsvenskan",
    "commence_time": "2026-07-27T17:00:00Z",
    "completed": False,
    "home_team": "BK Hacken",
    "away_team": "AIK",
    "scores": None,
}


def _mock_session(payload) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    session.get.return_value = response
    return session


def test_get_fixtures_normalizes_upcoming_events_as_scheduled() -> None:
    session = _mock_session([_UPCOMING_EVENT])
    client = SwedenFixturesClient(api_key="fake-key", session=session)

    fixtures = client.get_fixtures(date_from="2026-07-24", date_to="2026-07-24")

    assert fixtures == [
        NormalizedMatch(
            match_id="80dbae8461497b97aa471f4d7a4c17a2",
            utc_date="2026-07-24T17:00:00Z",
            status="SCHEDULED",
            home_team="Västerås SK",
            away_team="Örgryte IS",
            home_goals=None,
            away_goals=None,
        )
    ]


def test_get_fixtures_hits_the_free_events_endpoint_not_odds() -> None:
    session = _mock_session([])
    client = SwedenFixturesClient(api_key="fake-key", session=session)

    client.get_fixtures()

    called_url = session.get.call_args.args[0]
    assert called_url.endswith(f"/sports/{SPORT_KEY}/events")


def test_get_fixtures_filters_events_outside_the_requested_date_range() -> None:
    session = _mock_session([_UPCOMING_EVENT])
    client = SwedenFixturesClient(api_key="fake-key", session=session)

    fixtures = client.get_fixtures(date_from="2026-08-01", date_to="2026-08-31")

    assert fixtures == []


def test_get_results_normalizes_completed_events_with_scores() -> None:
    session = _mock_session([_COMPLETED_EVENT])
    client = SwedenFixturesClient(api_key="fake-key", session=session)

    results = client.get_results(date_from="2026-07-20", date_to="2026-07-20")

    assert results == [
        NormalizedMatch(
            match_id="9a15519b05f7ec514d35ca4677815355",
            utc_date="2026-07-20T12:00:00Z",
            status="FINISHED",
            home_team="Kalmar FF",
            away_team="Malmo FF",
            home_goals=2,
            away_goals=2,
        )
    ]


def test_get_results_excludes_not_yet_completed_events() -> None:
    session = _mock_session([_NOT_YET_COMPLETED_EVENT])
    client = SwedenFixturesClient(api_key="fake-key", session=session)

    results = client.get_results(date_from="2026-07-27", date_to="2026-07-27")

    assert results == []


def test_get_results_hits_the_scores_endpoint_with_days_from_clamped_to_provider_max() -> None:
    session = _mock_session([])
    client = SwedenFixturesClient(api_key="fake-key", session=session)

    client.get_results(date_from="2026-07-20", date_to="2026-07-20", days_from=30)

    called_url = session.get.call_args.args[0]
    params = session.get.call_args.kwargs["params"]
    assert called_url.endswith(f"/sports/{SPORT_KEY}/scores")
    assert params["daysFrom"] == 3  # The Odds API rejects daysFrom > 3 (confirmed live, 422)
