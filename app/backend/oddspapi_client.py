"""Live OddsPapi client for the total_corners market (over/under 9.5) --
the one market The Odds API doesn't carry at all (confirmed live,
agent_techspec.md S28/A69). Every prior OddsPapi integration in this
codebase (scripts/pull_oddspapi_btts_corners.py, extract_oddspapi_odds_lookup.py)
was a one-off historical pull into a static JSON file for backtesting only;
this is the first live client, used by eod_batch.py's add_secondary_odds()
alongside the existing OddsAPIClient calls.

Market ID/line and the historical-odds JSON shape (bookmakers.pinnacle.markets
[id].outcomes[id].players["0"] -> price ticks, last tick = current price) are
copied from scripts/extract_oddspapi_odds_lookup.py's already-proven parsing
(real downloaded snapshots) -- W199's investigation notes confirm the live
/v4/odds endpoint returns the identical shape. Credit gating mirrors
OddsAPIClient's CreditCounter pattern exactly (app/backend/odds_api_client.py),
just against OddsPapi's own 250-req/month free-tier budget instead of The
Odds API's 500.

W236 follow-up: multi-key fallback (ODDSPAPI_API_KEY_2/_3), mirroring
OddsAPIClient's own ODDS_API_KEY_2/_3 chain, lives in scheduler_wiring.py's
FallbackOddsPapiClient -- this module itself (OddsPapiClient) still only
ever knows about one key; the fallback is a wrapper around several
instances of it, same separation OddsAPIClient/FallbackOddsClient keep.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import requests

from app.backend.odds_api_client import CreditCounter
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

BASE_URL = "https://api.oddspapi.io"

# Confirmed live (2026-09-20): OddsPapi enforces a real per-endpoint rate
# limit, completely independent of the monthly credit quota -- a 429 with
# {"code": "RATE_LIMITED", "details": "Please wait 1.52 seconds before
# making another request to /v4/fixtures.", "retryMs": 1520}. eod_batch.py's
# asyncio.gather() starts every fixture in a league at once, and
# add_secondary_odds() has no await point of its own, so its OddsPapi calls
# previously fired back-to-back with zero delay between them -- easily
# faster than the vendor allows, tripping the limit even with plenty of
# monthly credits left. The one-off historical backfill script
# (scripts/pull_oddspapi_btts_corners.py) already knew to add its own
# time.sleep(5.0) between every call for this exact reason; that protection
# was never carried into this live path.
#
# Global, not per-key: the vendor's error message doesn't say whether the
# limit is per-key or per-IP, and a global throttle is correct either way
# (just possibly more conservative than strictly required if it's per-key).
# A small margin over the vendor's own observed 1.52s.
_MIN_REQUEST_INTERVAL_SECONDS = 1.6
_throttle_lock = threading.Lock()
_last_request_at = 0.0


def _throttle_oddspapi_request() -> None:
    global _last_request_at
    with _throttle_lock:
        wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()

# Confirmed live via /v4/tournaments?sportId=10 (scripts/pull_oddspapi_btts_corners.py).
LEAGUE_TOURNAMENT_IDS = {
    "E0": 17,
    "SP1": 8,
    "I1": 23,
    "F1": 34,
    "D1": 35,
}

# Fixed line, matching total_goals'/home_goals'/away_goals' own fixed-line
# convention -- confirmed via corners_line_map.json: 99.9% real-tick coverage
# across all 861 resolved matches, the highest of any line.
CORNERS_LINE = 9.5
CORNERS_MARKET_ID = "10803"
CORNERS_OUTCOME_OVER = "10803"
CORNERS_OUTCOME_UNDER = "10804"


@dataclass(frozen=True)
class OddsPapiFixture:
    fixture_id: str
    home_team: str
    away_team: str


def _last_price(outcomes: dict, outcome_id: str) -> float | None:
    outcome = outcomes.get(outcome_id)
    if not outcome:
        return None
    ticks = outcome.get("players", {}).get("0", [])
    if not ticks:
        return None
    return ticks[-1].get("price")


def _parse_corners_odds(payload: dict) -> dict[str, float] | None:
    outcomes = payload.get("bookmakers", {}).get("pinnacle", {}).get("markets", {}).get(CORNERS_MARKET_ID, {}).get("outcomes", {})
    over = _last_price(outcomes, CORNERS_OUTCOME_OVER)
    under = _last_price(outcomes, CORNERS_OUTCOME_UNDER)
    if over is None or under is None:
        return None
    return {f"over_{CORNERS_LINE}": over, f"under_{CORNERS_LINE}": under}


def _parse_fixtures(payload: list[dict]) -> list[OddsPapiFixture]:
    return [
        OddsPapiFixture(
            fixture_id=str(f["fixtureId"]), home_team=f["participant1Name"], away_team=f["participant2Name"],
        )
        for f in payload
    ]


class OddsPapiClient:
    """Client for OddsPapi, scoped to the total_corners market only. Refuses
    to call get_corners_odds() once within safety_margin credits of
    credit_limit for the current month, returning None (caller keeps
    corners omitted, same as no-odds-found today) instead of discovering the
    cutoff via a failed request -- same contract as OddsAPIClient."""

    def __init__(
        self,
        api_key: str,
        credit_counter: CreditCounter,
        session: requests.Session | None = None,
        credit_limit: int = 250,
        safety_margin: int = 10,
    ) -> None:
        self._api_key = api_key
        self._credit_counter = credit_counter
        self._session = session or requests.Session()
        self._credit_limit = credit_limit
        self._safety_margin = safety_margin

    def get_fixtures(self, tournament_id: int, status_id: int = 0) -> list[OddsPapiFixture]:
        # W200 (2026-09-22): default was statusId=1, which OddsPapi's live
        # API returns literally zero rows for, for every league, always
        # (confirmed via a direct /v4/fixtures call, statusId in {0,1,2,3}
        # -- 0 alone returned real fixtures, all with statusName="Pre-Game").
        # 1 is presumably "Live/In-Play" -- never true for tomorrow's
        # not-yet-started fixtures, which is the only thing eod_batch.py
        # ever asks this for. This silently zeroed every OddsPapi fixture
        # lookup since W236 shipped (2026-09-19): total_corners has had no
        # real live odds this whole time, not just "thin coverage" as W236's
        # own credit-budget caveat assumed. statusId=2/3 (used correctly by
        # scripts/pull_oddspapi_btts_corners.py for historical fixtures) are
        # unaffected by this default and untouched here.
        #
        # Fixture discovery is (almost) free -- confirmed live,
        # scripts/pull_oddspapi_btts_corners.py's own investigation notes --
        # so it isn't gated by CreditCounter like get_corners_odds() below.
        _throttle_oddspapi_request()
        response = self._session.get(
            f"{BASE_URL}/v4/fixtures",
            params={"apiKey": self._api_key, "tournamentId": tournament_id, "statusId": status_id},
            timeout=10,
        )
        # Confirmed live (2026-09-20): a 404 here can genuinely mean "no
        # fixtures match this query right now" (error code FIXTURE_NOT_FOUND,
        # e.g. a league with nothing currently scheduled), not a real
        # failure -- that's a normal, expected result, not an exception.
        if response.status_code == 404:
            try:
                error_code = response.json().get("error", {}).get("code")
            except ValueError:
                error_code = None
            if error_code == "FIXTURE_NOT_FOUND":
                return []
        response.raise_for_status()
        return _parse_fixtures(response.json())

    def get_corners_odds(self, fixture_id: str) -> dict[str, float] | None:
        cost = 1  # one /v4/odds call returns every market for the fixture, not per-market billing

        if self._credit_counter.would_exceed(cost, self._credit_limit, self._safety_margin):
            LOGGER.warning(
                "OddsPapiClient.get_corners_odds: skipping fixture_id=%s, would cross safety margin "
                "(used=%d cost=%d limit=%d safety_margin=%d).",
                fixture_id, self._credit_counter.credits_used, cost, self._credit_limit, self._safety_margin,
            )
            return None

        _throttle_oddspapi_request()
        response = self._session.get(
            f"{BASE_URL}/v4/odds",
            params={"apiKey": self._api_key, "fixtureId": fixture_id, "bookmakers": "pinnacle"},
            timeout=10,
        )
        response.raise_for_status()
        self._credit_counter.record_usage(cost)

        return _parse_corners_odds(response.json())
