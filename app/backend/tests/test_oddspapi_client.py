"""A100/W198 follow-up: OddsPapi is the only vendor confirmed (agent_techspec.md
S28/W199 investigation notes) to carry live total_corners (over/under 9.5)
odds -- The Odds API doesn't offer a corners market at all. Historical pulls
already exist (scripts/pull_oddspapi_btts_corners.py) but there was no live
client. This mirrors OddsAPIClient's shape (app/backend/odds_api_client.py):
same CreditCounter, same "skip and return None over the safety margin"
degrade-gracefully behavior, same real-JSON-shape parsing already proven by
scripts/extract_oddspapi_odds_lookup.py against real downloaded snapshots."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest
import requests

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import oddspapi_client as oddspapi_client_module
from app.backend.odds_api_client import CreditCounter
from app.backend.oddspapi_client import (
    OddsPapiClient,
    OddsPapiFixture,
    _parse_corners_odds,
    _parse_fixtures,
)


@pytest.fixture(autouse=True)
def _no_real_throttle_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level rate-limit throttle (_throttle_oddspapi_request)
    uses a real time.sleep -- without this, every test calling get_fixtures/
    get_corners_odds more than once would incur real multi-second delays,
    and the module-level _last_request_at would leak between tests. Patches
    time.sleep to a no-op and resets the shared clock state before each
    test; the throttle's own interval-computation logic still runs
    unmodified, only the actual waiting is skipped."""
    monkeypatch.setattr(oddspapi_client_module.time, "sleep", lambda seconds: None)
    oddspapi_client_module._last_request_at = 0.0

# Real shape confirmed by scripts/extract_oddspapi_odds_lookup.py against
# downloaded /v4/historical-odds snapshots; W199's investigation notes confirm
# /v4/odds (the live endpoint) returns the identical shape.
_CORNERS_PAYLOAD = {
    "bookmakers": {
        "pinnacle": {
            "markets": {
                "10803": {
                    "outcomes": {
                        "10803": {"players": {"0": [{"price": 1.9}, {"price": 1.85}]}},
                        "10804": {"players": {"0": [{"price": 2.0}, {"price": 1.95}]}},
                    }
                }
            }
        }
    }
}

_FIXTURES_PAYLOAD = [
    {"fixtureId": 555, "participant1Name": "Real Madrid", "participant2Name": "Barcelona", "startTime": "2026-09-20T19:00:00Z"},
    {"fixtureId": 556, "participant1Name": "Sevilla", "participant2Name": "Valencia", "startTime": "2026-09-20T21:00:00Z"},
]


def test_parse_corners_odds_takes_the_last_tick_per_outcome() -> None:
    result = _parse_corners_odds(_CORNERS_PAYLOAD)

    assert result == {"over_9.5": 1.85, "under_9.5": 1.95}


def test_parse_corners_odds_returns_none_when_market_missing() -> None:
    result = _parse_corners_odds({"bookmakers": {"pinnacle": {"markets": {}}}})

    assert result is None


def test_parse_fixtures_reads_fixture_id_and_participant_names() -> None:
    fixtures = _parse_fixtures(_FIXTURES_PAYLOAD)

    assert fixtures == [
        OddsPapiFixture(fixture_id="555", home_team="Real Madrid", away_team="Barcelona"),
        OddsPapiFixture(fixture_id="556", home_team="Sevilla", away_team="Valencia"),
    ]


def _mock_session(payload) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    session.get.return_value = response
    return session


def test_get_fixtures_sends_correct_url_and_params_and_costs_no_credit() -> None:
    session = _mock_session(_FIXTURES_PAYLOAD)
    counter = CreditCounter()
    client = OddsPapiClient(api_key="my-key", credit_counter=counter, session=session)

    fixtures = client.get_fixtures(tournament_id=8)

    url = session.get.call_args.args[0]
    params = session.get.call_args.kwargs["params"]
    assert url == "https://api.oddspapi.io/v4/fixtures"
    assert params["apiKey"] == "my-key"
    assert params["tournamentId"] == 8
    assert params["statusId"] == 1
    assert len(fixtures) == 2
    assert counter.credits_used == 0


def test_get_fixtures_returns_empty_list_for_fixture_not_found() -> None:
    """Confirmed live (2026-09-20): a 404 with error code FIXTURE_NOT_FOUND
    means "nothing matches this query right now" (e.g. a league with no
    currently-scheduled fixtures), not a real failure -- must degrade to an
    empty list, not raise."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 404
    response.json.return_value = {"error": {"message": "No fixtures found for the specified criteria.", "code": "FIXTURE_NOT_FOUND"}}
    session.get.return_value = response
    client = OddsPapiClient(api_key="my-key", credit_counter=CreditCounter(), session=session)

    fixtures = client.get_fixtures(tournament_id=34)

    assert fixtures == []
    response.raise_for_status.assert_not_called()


def test_get_fixtures_still_raises_for_a_genuine_404_without_the_fixture_not_found_code() -> None:
    session = MagicMock()
    response = MagicMock()
    response.status_code = 404
    response.json.return_value = {"error": {"message": "Not found.", "code": "SOME_OTHER_ERROR"}}
    response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
    session.get.return_value = response
    client = OddsPapiClient(api_key="my-key", credit_counter=CreditCounter(), session=session)

    with pytest.raises(requests.exceptions.HTTPError):
        client.get_fixtures(tournament_id=34)


def test_get_corners_odds_sends_correct_url_and_params() -> None:
    session = _mock_session(_CORNERS_PAYLOAD)
    counter = CreditCounter()
    client = OddsPapiClient(api_key="my-key", credit_counter=counter, session=session)

    client.get_corners_odds(fixture_id="555")

    url = session.get.call_args.args[0]
    params = session.get.call_args.kwargs["params"]
    assert url == "https://api.oddspapi.io/v4/odds"
    assert params["apiKey"] == "my-key"
    assert params["fixtureId"] == "555"
    assert params["bookmakers"] == "pinnacle"


def test_get_corners_odds_costs_one_credit_per_call() -> None:
    session = _mock_session(_CORNERS_PAYLOAD)
    counter = CreditCounter()
    client = OddsPapiClient(api_key="my-key", credit_counter=counter, session=session)

    client.get_corners_odds(fixture_id="555")
    client.get_corners_odds(fixture_id="556")

    assert counter.credits_used == 2


def test_get_corners_odds_skips_call_when_it_would_exceed_the_safety_margin() -> None:
    session = _mock_session(_CORNERS_PAYLOAD)
    counter = CreditCounter(credits_used=241)  # 250 limit - 10 safety margin = 240 usable, already over
    client = OddsPapiClient(api_key="my-key", credit_counter=counter, session=session)

    result = client.get_corners_odds(fixture_id="555")

    assert result is None
    session.get.assert_not_called()


def test_get_corners_odds_returns_parsed_prices() -> None:
    session = _mock_session(_CORNERS_PAYLOAD)
    counter = CreditCounter()
    client = OddsPapiClient(api_key="my-key", credit_counter=counter, session=session)

    result = client.get_corners_odds(fixture_id="555")

    assert result == {"over_9.5": 1.85, "under_9.5": 1.95}


def test_throttle_sleeps_before_a_second_call_too_soon_after_the_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirmed live (2026-09-20): OddsPapi rate-limits at roughly 1
    request/1.5s per endpoint (429 RATE_LIMITED, "wait 1.52 seconds"),
    independent of the monthly credit quota. eod_batch.py's asyncio.gather()
    starts every fixture in a league at once with no await point of its own
    in add_secondary_odds(), so consecutive OddsPapi calls need their own
    throttle rather than relying on incidental delay elsewhere."""
    session = _mock_session(_FIXTURES_PAYLOAD)
    client = OddsPapiClient(api_key="my-key", credit_counter=CreditCounter(), session=session)
    sleep_calls: list[float] = []
    monkeypatch.setattr(oddspapi_client_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    client.get_fixtures(tournament_id=8)  # first call: no prior request, no sleep
    assert sleep_calls == []

    client.get_fixtures(tournament_id=8)  # second call, effectively immediately after
    assert len(sleep_calls) == 1
    assert 0 < sleep_calls[0] <= oddspapi_client_module._MIN_REQUEST_INTERVAL_SECONDS
