"""W07/W25: The Odds API client with an explicit credit-usage counter.

W25 (2026-07-15): verified live against a real key. `_normalize()` needed
no changes -- a real captured response confirmed the documented v4 schema
(sport_key/commence_time/home_team/away_team/bookmakers[].markets[].outcomes[])
holds, including the literal `"Draw"` outcome name and decimal price
format. `test_normalize_matches_a_real_captured_response` below uses that
real payload (trimmed to 2 of the original 19 bookmakers) as a permanent
regression fixture, same pattern as W05/W06. Two things the real check did
surface, tracked separately rather than fixed here since they're
independent of `_normalize()`'s own correctness: (1) bookmaker ordering in
`bookmakers[]` is confirmed *not* stable across events (`_normalize()` only
ever reads `bookmakers[0]`, no fallback), and (2) the response carries
authoritative credit-usage headers (`x-requests-remaining`/`x-requests-used`)
that `CreditCounter` never reads, relying only on a client-side estimate
that happened to agree with the real count in this one check.

The other fixtures below (`_EVENT`) remain synthetic, documented-schema
constructions -- fine for exercising code paths, just not evidence the
schema itself is real."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.odds_api_client import (
    CreditCounter,
    FileCreditCounterStore,
    NormalizedOdds,
    NormalizedSecondaryOdds,
    OddsAPIClient,
    _normalize,
    _normalize_secondary,
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

# A real event captured 2026-07-15 from a live GET /v4/sports/soccer_epl/odds
# call (W25), trimmed from its original 19 bookmakers down to 2 -- the exact
# real fixture (Arsenal v Coventry City, 2026-08-21) later also used for the
# W17/W23 live checks.
_REAL_CAPTURED_EVENT = {
    "id": "eb2553d10d63dc912b99f8fd0d675721",
    "sport_key": "soccer_epl",
    "sport_title": "EPL",
    "commence_time": "2026-08-21T19:00:00Z",
    "home_team": "Arsenal",
    "away_team": "Coventry City",
    "bookmakers": [
        {
            "key": "betfair_sb_uk",
            "title": "Betfair Sportsbook",
            "last_update": "2026-07-13T01:45:54Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2026-07-13T01:45:54Z",
                    "outcomes": [
                        {"name": "Arsenal", "price": 1.15},
                        {"name": "Coventry City", "price": 17.0},
                        {"name": "Draw", "price": 7.0},
                    ],
                }
            ],
        },
        {
            "key": "betfred_uk",
            "title": "Betfred (UK)",
            "last_update": "2026-07-13T01:46:27Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2026-07-13T01:46:27Z",
                    "outcomes": [
                        {"name": "Arsenal", "price": 1.17},
                        {"name": "Coventry City", "price": 17.0},
                        {"name": "Draw", "price": 7.0},
                    ],
                }
            ],
        },
    ],
}


def test_normalize_matches_a_real_captured_response() -> None:
    """W25: proves _normalize() against a real API payload, not just a
    synthetic one built from the docs."""
    result = _normalize(_REAL_CAPTURED_EVENT)
    assert result == NormalizedOdds(
        home_team="Arsenal", away_team="Coventry City", commence_time="2026-08-21T19:00:00Z",
        home_odds=1.15, draw_odds=7.0, away_odds=17.0,
        event_id="eb2553d10d63dc912b99f8fd0d675721",
    )


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
            home_odds=1.4, draw_odds=4.8, away_odds=7.5, event_id="abc123",
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


def test_normalize_secondary_reads_totals_at_the_2_5_line_and_btts() -> None:
    """W164: totals/btts come from the per-event endpoint's own event-odds
    payload shape (bookmakers[].markets[].outcomes[] with a `point` field
    for totals) -- confirmed live against a real key, not a guess."""
    payload = {
        "bookmakers": [
            {"key": "betfred_uk", "markets": [{"key": "h2h", "outcomes": []}]},  # no totals/btts here
            {"key": "williamhill", "markets": [
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": 1.9, "point": 2.5},
                    {"name": "Under", "price": 1.95, "point": 2.5},
                ]},
            ]},
            {"key": "coral", "markets": [
                {"key": "btts", "outcomes": [{"name": "Yes", "price": 1.7}, {"name": "No", "price": 2.1}]},
            ]},
        ],
    }

    result = _normalize_secondary(payload)

    assert result == NormalizedSecondaryOdds(
        total_goals={"over_2.5": 1.9, "under_2.5": 1.95}, btts={"yes": 1.7, "no": 2.1},
    )


def test_normalize_secondary_ignores_non_2_5_totals_lines() -> None:
    """A totals market can carry multiple lines (alternate_totals-style) --
    only the 2.5 line matches this codebase's existing fixed-line
    convention (schema.py's over_2.5/under_2.5)."""
    payload = {"bookmakers": [{"markets": [{"key": "totals", "outcomes": [
        {"name": "Over", "price": 1.4, "point": 1.5},
        {"name": "Under", "price": 3.0, "point": 1.5},
    ]}]}]}

    result = _normalize_secondary(payload)

    assert result.total_goals is None


def test_normalize_secondary_none_when_no_bookmaker_has_the_market() -> None:
    payload = {"bookmakers": [{"markets": [{"key": "h2h", "outcomes": []}]}]}

    result = _normalize_secondary(payload)

    assert result == NormalizedSecondaryOdds(total_goals=None, btts=None)


def _mock_event_odds_session(payload: dict) -> MagicMock:
    """Like _mock_session, but for the per-event endpoint -- which returns a
    single event dict, not a list of events like the bulk endpoint does."""
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    session.get.return_value = response
    return session


def test_get_event_odds_sends_correct_url_and_params() -> None:
    session = _mock_event_odds_session({"bookmakers": []})
    counter = CreditCounter()
    client = OddsAPIClient(api_key="my-key", credit_counter=counter, session=session)

    client.get_event_odds(sport_key="soccer_epl", event_id="evt123")

    url = session.get.call_args.args[0]
    params = session.get.call_args.kwargs["params"]
    assert url == "https://api.the-odds-api.com/v4/sports/soccer_epl/events/evt123/odds"
    assert params["apiKey"] == "my-key"
    assert params["markets"] == "totals,btts"


def test_get_event_odds_costs_markets_times_regions() -> None:
    session = _mock_event_odds_session({"bookmakers": []})
    counter = CreditCounter(now_fn=lambda: datetime(2026, 7, 11, tzinfo=timezone.utc))
    client = OddsAPIClient(api_key="fake-key", credit_counter=counter, session=session, regions=("uk", "eu"))

    client.get_event_odds(sport_key="soccer_epl", event_id="evt123")

    assert counter.credits_used == 4  # 2 markets (totals, btts) x 2 regions


def test_get_event_odds_skips_call_when_it_would_exceed_the_safety_margin() -> None:
    session = _mock_event_odds_session({"bookmakers": []})
    counter = CreditCounter(credits_used=495)  # 500 limit - 50 safety margin = 450 usable, already over
    client = OddsAPIClient(api_key="fake-key", credit_counter=counter, session=session)

    result = client.get_event_odds(sport_key="soccer_epl", event_id="evt123")

    assert result is None
    session.get.assert_not_called()


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


def _save_july_state(store: FileCreditCounterStore) -> None:
    counter_a = store.load(now_fn=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc))
    counter_a.record_usage(480)
    store.save(counter_a)


def test_credit_counter_rolls_over_on_first_check_after_a_month_boundary_restart(tmp_path: Path) -> None:
    """Realistic restart sequence: an instance accumulates usage and saves
    in July; a fresh instance ('process restart') loads that same file in
    August -- its very first real usage check must correctly roll over
    rather than carrying the stale July count forward. Same 'two separate
    process instances sharing the same on-disk file' pattern W21 used for
    the scheduler (test_scheduler_integration.py).

    credits_used/would_exceed/record_usage each call their own
    _roll_month_if_needed() independently -- a fresh store.load() is used
    per method below (not one shared instance) so an earlier call's side
    effect can't mask a later method's own rollover check silently being
    broken. would_exceed() in particular is the actual first call made in
    production (OddsAPIClient.request()), so it must be proven on its own,
    not merely alongside credits_used."""
    counter_path = tmp_path / "odds_credit_usage.json"
    store = FileCreditCounterStore(counter_path)
    august_now = lambda: datetime(2026, 8, 1, tzinfo=timezone.utc)

    _save_july_state(store)
    counter_would_exceed = store.load(now_fn=august_now)
    assert not counter_would_exceed.would_exceed(cost=1, limit=500, safety_margin=50)

    _save_july_state(store)
    counter_credits_used = store.load(now_fn=august_now)
    assert counter_credits_used.credits_used == 0

    _save_july_state(store)
    counter_record_usage = store.load(now_fn=august_now)
    counter_record_usage.record_usage(10)
    assert counter_record_usage.credits_used == 10

    # confirm the rollover was persisted, not just held in memory
    store.save(counter_record_usage)
    reloaded = json.loads(counter_path.read_text())
    assert reloaded == {"credits_used": 10, "month_key": "2026-08"}


def test_file_credit_counter_store_missing_file_starts_fresh(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "does_not_exist.json")
    counter = store.load(now_fn=lambda: datetime(2026, 7, 11, tzinfo=timezone.utc))
    assert counter.credits_used == 0
