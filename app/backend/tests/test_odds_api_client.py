"""W07: The Odds API client with an explicit credit-usage counter.

No Odds API key was available at implementation time (2026-07-11, agreed
with the user) -- unlike W05/W06, nothing here is verified against a real
response. Event-shape fixtures below follow The Odds API's publicly
documented v4 schema (sport_key/commence_time/home_team/away_team/
bookmakers[].markets[].outcomes[]), not a live-captured payload. A live
smoke-test is still needed once a key exists -- see completion notes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.odds_api_client import (
    CreditCounter,
    FileCreditCounterStore,
    NormalizedOdds,
    OddsAPIClient,
)

_EVENT = {
    "id": "abc123",
    "sport_key": "soccer_epl",
    "commence_time": "2026-08-21T19:00:00Z",
    "home_team": "Arsenal",
    "away_team": "Coventry City",
    "bookmakers": [
        {
            "key": "bet365",
            "title": "Bet365",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 1.4},
                        {"name": "Coventry City", "price": 7.5},
                        {"name": "Draw", "price": 4.8},
                    ],
                }
            ],
        }
    ],
}


def _mock_session(events: list[dict]) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = events
    session.get.return_value = response
    return session


def test_get_odds_normalizes_h2h_market() -> None:
    session = _mock_session([_EVENT])
    counter = CreditCounter()
    client = OddsAPIClient(api_key="fake-key", credit_counter=counter, session=session)

    odds = client.get_odds()

    assert odds == [
        NormalizedOdds(
            home_team="Arsenal", away_team="Coventry City", commence_time="2026-08-21T19:00:00Z",
            home_odds=1.4, draw_odds=4.8, away_odds=7.5,
        )
    ]


def test_get_odds_sends_correct_query_params() -> None:
    session = _mock_session([])
    counter = CreditCounter()
    client = OddsAPIClient(
        api_key="my-key", credit_counter=counter, session=session, markets=("h2h",), regions=("uk",),
    )

    client.get_odds(sport_key="soccer_epl")

    params = session.get.call_args.kwargs["params"]
    assert params["apiKey"] == "my-key"
    assert params["regions"] == "uk"
    assert params["markets"] == "h2h"


def test_a_simulated_month_of_calls_matches_hand_computed_usage() -> None:
    """1 market x 1 region = 1 credit/call. 7 calls -> 7 credits used."""
    session = _mock_session([])
    counter = CreditCounter(now_fn=lambda: datetime(2026, 7, 11, tzinfo=timezone.utc))
    client = OddsAPIClient(
        api_key="fake-key", credit_counter=counter, session=session, markets=("h2h",), regions=("uk",),
    )

    for _ in range(7):
        client.get_odds()

    assert counter.credits_used == 7


def test_cost_scales_with_markets_times_regions() -> None:
    session = _mock_session([])
    counter = CreditCounter(now_fn=lambda: datetime(2026, 7, 11, tzinfo=timezone.utc))
    client = OddsAPIClient(
        api_key="fake-key", credit_counter=counter, session=session,
        markets=("h2h", "totals"), regions=("uk", "eu"),
    )

    client.get_odds()

    assert counter.credits_used == 4  # 2 markets x 2 regions


def test_call_is_skipped_once_within_safety_margin_of_limit() -> None:
    session = _mock_session([_EVENT])
    now_fn = lambda: datetime(2026, 7, 11, tzinfo=timezone.utc)
    counter = CreditCounter(now_fn=now_fn, credits_used=451, month_key="2026-07")
    client = OddsAPIClient(
        api_key="fake-key", credit_counter=counter, session=session,
        markets=("h2h",), regions=("uk",), credit_limit=500, safety_margin=50,
    )

    result = client.get_odds()

    assert result is None
    session.get.assert_not_called()
    assert counter.credits_used == 451  # unchanged -- the call never happened


def test_call_still_proceeds_just_outside_the_safety_margin() -> None:
    session = _mock_session([_EVENT])
    now_fn = lambda: datetime(2026, 7, 11, tzinfo=timezone.utc)
    counter = CreditCounter(now_fn=now_fn, credits_used=449, month_key="2026-07")
    client = OddsAPIClient(
        api_key="fake-key", credit_counter=counter, session=session,
        markets=("h2h",), regions=("uk",), credit_limit=500, safety_margin=50,
    )

    result = client.get_odds()

    assert result is not None
    session.get.assert_called_once()
    assert counter.credits_used == 450


def test_counter_resets_at_simulated_month_boundary() -> None:
    current_month = [datetime(2026, 7, 30, tzinfo=timezone.utc)]
    counter = CreditCounter(now_fn=lambda: current_month[0])
    counter.record_usage(480)
    assert counter.credits_used == 480

    current_month[0] = datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert counter.credits_used == 0

    counter.record_usage(10)
    assert counter.credits_used == 10


def test_file_credit_counter_store_round_trip(tmp_path: Path) -> None:
    now_fn = lambda: datetime(2026, 7, 11, tzinfo=timezone.utc)
    store = FileCreditCounterStore(tmp_path / "odds_credit_usage.json")

    counter = store.load(now_fn=now_fn)
    counter.record_usage(123)
    store.save(counter)

    reloaded = store.load(now_fn=now_fn)
    assert reloaded.credits_used == 123


def test_file_credit_counter_store_missing_file_starts_fresh(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "does_not_exist.json")
    counter = store.load(now_fn=lambda: datetime(2026, 7, 11, tzinfo=timezone.utc))
    assert counter.credits_used == 0
