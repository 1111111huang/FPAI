"""The Odds API client (W07, D2b) -- fetches 1X2 (h2h) odds for E0 fixtures,
scoped to a single market and single region to stay within the free tier's
~500-credit/month budget.

Verified live against a real key (W25, 2026-07-15): _normalize() correctly
parses a real response, matching the publicly documented v4 schema
(sport_key/commence_time/home_team/away_team/bookmakers[].markets[].outcomes[])
it was originally built against -- see test_odds_api_client.py's
test_normalize_matches_a_real_captured_response for the real fixture.

W164: two things a live check previously surfaced here are now addressed.
(1) bookmakers[] ordering is confirmed *not* stable across events, and not
every bookmaker in the list carries every market -- confirmed live (W164)
against a real key that of 21 UK bookmakers on one fixture, only 4 carried
`totals`, and bookmakers[0] specifically never did. _first_priced_outcomes()
(below) scans for the first bookmaker that actually has the requested
market, used by both _normalize() (h2h) and _normalize_secondary()
(totals/btts) -- one fix point instead of two copies of the same bug.
(2) real responses carry authoritative credit-usage headers
(x-requests-remaining/x-requests-used) that CreditCounter still never
reads, relying only on a client-side cost estimate -- unaddressed, tracked.

Separately (BUG-015, same live check): The Odds API and football-data.org
spell a number of clubs differently ("Man United" vs "Manchester United",
"Nottingham" vs "Nottingham Forest", etc.) -- fixed at the fixture-matching
layer (app/backend/eod_batch.py's odds_lookup()/match_odds()), not here;
this client's own output (NormalizedOdds.home_team/away_team) is left as
The Odds API's own spelling, matching what it actually returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Callable

import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"


@dataclass(frozen=True)
class NormalizedOdds:
    home_team: str
    away_team: str
    commence_time: str
    home_odds: float | None
    draw_odds: float | None
    away_odds: float | None
    # W164: needed to call get_event_odds() (the per-event endpoint, the
    # only one The Odds API serves totals/btts on) for this exact fixture
    # without a second lookup. "" for any caller/test predating this field
    # (e.g. a hand-built NormalizedOdds with no real event behind it).
    event_id: str = ""


def _first_priced_outcomes(bookmakers: list[dict], market_key: str) -> list[dict] | None:
    """Scans bookmakers in order for the first one that actually carries
    `market_key`, returning its raw `outcomes` list -- not bookmakers[0]
    blindly, since bookmakers[] ordering isn't stable and most don't carry
    every market (see module docstring, W164)."""
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") == market_key:
                return market.get("outcomes", [])
    return None


def _normalize(event: dict) -> NormalizedOdds:
    home_team = event["home_team"]
    away_team = event["away_team"]
    home_odds = draw_odds = away_odds = None

    outcomes = _first_priced_outcomes(event.get("bookmakers") or [], "h2h")
    for outcome in outcomes or []:
        name, price = outcome.get("name"), outcome.get("price")
        if name == home_team:
            home_odds = price
        elif name == away_team:
            away_odds = price
        elif name == "Draw":
            draw_odds = price

    return NormalizedOdds(
        home_team=home_team, away_team=away_team, commence_time=event["commence_time"],
        home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds,
        event_id=event.get("id", ""),
    )


# W164: fixed reference lines, matching this codebase's own existing
# convention for these exact markets -- graph.py's prompt already speaks of
# total_goals as "over/under 2.5" (raw_matches' over25_odds/under25_odds
# columns, backtest-only); schema.py's _ELIGIBLE_CONDITIONAL_MARKETS already
# hardcodes ("home_corners"/"away_corners", "over_2.5"). Not reinventing a
# line convention, just finally supplying live odds for the one that
# already existed.
_TOTAL_GOALS_LINE = 2.5

# W199: team_totals per-team over/under line, same fixed-line convention as
# _TOTAL_GOALS_LINE above.
_TEAM_GOALS_LINE = 1.5


@dataclass(frozen=True)
class NormalizedSecondaryOdds:
    """totals/btts/team_totals odds for one fixture, fetched via the
    per-event endpoint (W164/W199) -- The Odds API doesn't serve these on
    the bulk /odds endpoint (confirmed live: 422 "Markets not supported by
    this endpoint"). None per field when no bookmaker priced that market
    for this fixture."""
    total_goals: dict[str, float] | None  # {"over_2.5": .., "under_2.5": ..}
    btts: dict[str, float] | None  # {"yes": .., "no": ..}
    home_goals: dict[str, float] | None = None  # {"over_1.5": .., "under_1.5": ..}
    away_goals: dict[str, float] | None = None
    # BUG-096: The Odds API's "spreads" market (soccer's Asian handicap).
    # {"line": .., "home": .., "away": ..} -- "line" is the HOME side's own
    # handicap point, which is this project's MKT_AH_LINE convention
    # (football-data.co.uk's AHh: negative = home favored). Feeds the
    # MKT_LAMBDA_* goal-split features, which need a real line per match.
    asian_handicap: dict[str, float] | None = None


def _parse_asian_handicap(
    bookmakers: list[dict], home_team: str | None, away_team: str | None,
) -> dict[str, float] | None:
    """BUG-096: the first bookmaker's main spreads line, as
    {"line", "home", "away"}. A book can list more than one line; the pair
    whose two prices are closest to each other is the main (balanced) line,
    the same "AHh" football-data.co.uk ships for training data. None when no
    book priced spreads, or the outcomes can't be matched to both teams."""
    if not home_team or not away_team:
        return None
    outcomes = _first_priced_outcomes(bookmakers, "spreads")
    if not outcomes:
        return None
    home_by_point: dict[float, float] = {}
    away_by_point: dict[float, float] = {}
    for outcome in outcomes:
        name, price, point = outcome.get("name"), outcome.get("price"), outcome.get("point")
        if price is None or point is None:
            continue
        if name == home_team:
            home_by_point[float(point)] = float(price)
        elif name == away_team:
            away_by_point[float(point)] = float(price)
    # A home line of -0.5 pairs with an away line of +0.5.
    pairs = [
        (home_point, home_price, away_by_point[-home_point])
        for home_point, home_price in home_by_point.items()
        if -home_point in away_by_point
    ]
    if not pairs:
        return None
    line, home_price, away_price = min(pairs, key=lambda pair: abs(pair[1] - pair[2]))
    return {"line": line, "home": home_price, "away": away_price}


def _normalize_secondary(
    payload: dict, home_team: str | None = None, away_team: str | None = None,
) -> NormalizedSecondaryOdds:
    bookmakers = payload.get("bookmakers") or []

    total_goals = None
    totals_outcomes = _first_priced_outcomes(bookmakers, "totals")
    for outcome in totals_outcomes or []:
        if outcome.get("point") != _TOTAL_GOALS_LINE:
            continue
        name, price = outcome.get("name"), outcome.get("price")
        if name == "Over":
            total_goals = {**(total_goals or {}), "over_2.5": price}
        elif name == "Under":
            total_goals = {**(total_goals or {}), "under_2.5": price}

    btts = None
    btts_outcomes = _first_priced_outcomes(bookmakers, "btts")
    for outcome in btts_outcomes or []:
        name, price = outcome.get("name"), outcome.get("price")
        if name == "Yes":
            btts = {**(btts or {}), "yes": price}
        elif name == "No":
            btts = {**(btts or {}), "no": price}

    home_goals = away_goals = None
    if home_team and away_team:
        team_totals_outcomes = _first_priced_outcomes(bookmakers, "team_totals")
        for outcome in team_totals_outcomes or []:
            if outcome.get("point") != _TEAM_GOALS_LINE:
                continue
            name, price, team = outcome.get("name"), outcome.get("price"), outcome.get("description")
            key = "over_1.5" if name == "Over" else "under_1.5" if name == "Under" else None
            if key is None:
                continue
            if team == home_team:
                home_goals = {**(home_goals or {}), key: price}
            elif team == away_team:
                away_goals = {**(away_goals or {}), key: price}

    asian_handicap = _parse_asian_handicap(bookmakers, home_team, away_team)

    return NormalizedSecondaryOdds(
        total_goals=total_goals, btts=btts, home_goals=home_goals, away_goals=away_goals,
        asian_handicap=asian_handicap,
    )


def _month_key(moment: datetime) -> str:
    return moment.strftime("%Y-%m")


class CreditCounter:
    """Monthly credit tally, computed client-side from the provider's own
    documented cost formula (markets x regions per call) -- not read from any
    response header, so nothing here depends on unverified API behavior.
    Resets automatically the first time it's consulted in a new month."""

    def __init__(
        self,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        credits_used: int = 0,
        month_key: str | None = None,
        unavailable_until: datetime | None = None,
        unavailable_reason: str | None = None,
    ) -> None:
        self._now_fn = now_fn
        self._credits_used = credits_used
        self._month_key = month_key or _month_key(now_fn())
        self._unavailable_until = unavailable_until
        self._unavailable_reason = unavailable_reason

    def _roll_month_if_needed(self) -> None:
        current = _month_key(self._now_fn())
        if current != self._month_key:
            self._month_key = current
            self._credits_used = 0

    @property
    def credits_used(self) -> int:
        self._roll_month_if_needed()
        return self._credits_used

    def would_exceed(self, cost: int, limit: int, safety_margin: int) -> bool:
        self._roll_month_if_needed()
        return self._credits_used + cost > (limit - safety_margin)

    def record_usage(self, cost: int) -> None:
        self._roll_month_if_needed()
        self._credits_used += cost

    # W254: direct user request -- a key rejected with 401 (revoked/invalid,
    # confirmed live BUG-056) or 429 (rate-limited) stayed in the fallback
    # rotation's front position forever, so every single call re-tried it
    # first and paid for its failure before reaching a key that might
    # actually work. mark_unavailable()/is_available() track a per-key
    # cooldown, persisted the same way credits_used already is, so a
    # known-bad key is deprioritized across calls *and* restarts instead of
    # being retried from scratch every time.
    def mark_unavailable(self, until: datetime, reason: str) -> None:
        self._unavailable_until = until
        self._unavailable_reason = reason

    def is_available(self) -> bool:
        return self._unavailable_until is None or self._now_fn() >= self._unavailable_until

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.is_available() else self._unavailable_reason

    def to_dict(self) -> dict:
        return {
            "credits_used": self._credits_used,
            "month_key": self._month_key,
            "unavailable_until": self._unavailable_until.isoformat() if self._unavailable_until else None,
            "unavailable_reason": self._unavailable_reason,
        }

    @classmethod
    def from_dict(
        cls, data: dict, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> "CreditCounter":
        unavailable_until_raw = data.get("unavailable_until")
        return cls(
            now_fn=now_fn,
            credits_used=data.get("credits_used", 0),
            month_key=data.get("month_key"),
            unavailable_until=datetime.fromisoformat(unavailable_until_raw) if unavailable_until_raw else None,
            unavailable_reason=data.get("unavailable_reason"),
        )


class FileCreditCounterStore:
    """Persists a CreditCounter's state to a JSON file, so an app restart
    doesn't lose track of usage within the current billing month."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> CreditCounter:
        if not self._path.exists():
            return CreditCounter(now_fn=now_fn)
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return CreditCounter.from_dict(data, now_fn=now_fn)

    def save(self, counter: CreditCounter) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(counter.to_dict()), encoding="utf-8")


class OddsAPIClient:
    """Client for The Odds API, scoped to a single market/region pair. Refuses
    to call once within safety_margin credits of credit_limit for the current
    month, returning None (caller should keep last-known odds) instead of
    discovering the cutoff via a failed request."""

    def __init__(
        self,
        api_key: str,
        credit_counter: CreditCounter,
        session: requests.Session | None = None,
        markets: tuple[str, ...] = ("h2h",),
        regions: tuple[str, ...] = ("uk",),
        credit_limit: int = 500,
        safety_margin: int = 50,
    ) -> None:
        self._api_key = api_key
        self._credit_counter = credit_counter
        self._session = session or requests.Session()
        self._markets = markets
        self._regions = regions
        self._credit_limit = credit_limit
        self._safety_margin = safety_margin

    def get_odds(self, sport_key: str = "soccer_epl", date: str | None = None) -> list[NormalizedOdds] | None:
        # date is unused here -- accepted only for interface parity with
        # HistoricalOddsClient.get_odds() (W54): this is a live-current-odds
        # call with no historical-date concept, so callers that duck-type
        # across both client types (run_eod_batch) can pass it unconditionally.
        cost = len(self._markets) * len(self._regions)

        if self._credit_counter.would_exceed(cost, self._credit_limit, self._safety_margin):
            # W250: this class has no idea whether it's called directly (a
            # single-key deployment, where "keeping last-known odds" is
            # accurate) or wrapped by FallbackOddsClient (scheduler_wiring.py),
            # which tries the next configured key immediately on a None
            # return -- claiming finality here was misleading in the
            # multi-key case (found live: read as "the backend gave up"
            # when it was actually one hop into a working fallback chain).
            # FallbackOddsClient._try_each_client logs the real outcome.
            LOGGER.warning(
                "OddsAPIClient: skipping call for this key, would cross safety margin "
                "(used=%d cost=%d limit=%d safety_margin=%d).",
                self._credit_counter.credits_used, cost, self._credit_limit, self._safety_margin,
            )
            return None

        response = self._session.get(
            f"{BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": self._api_key,
                "regions": ",".join(self._regions),
                "markets": ",".join(self._markets),
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        response.raise_for_status()
        self._credit_counter.record_usage(cost)

        return [_normalize(event) for event in response.json()]

    # W164: totals/btts require the per-event endpoint -- confirmed live,
    # the bulk endpoint 422s ("Markets not supported by this endpoint")
    # for either. Cost is markets x regions same as get_odds(), but per
    # fixture rather than per league -- gated by the same CreditCounter so
    # it degrades the same way (returns None, keep last-known/no secondary
    # odds) rather than a separate budget with its own failure mode.
    # W199: `regions` overrides self._regions for this call only (team_totals
    # needs a wider region set, e.g. us/us2, than totals/btts do). `home_team`/
    # `away_team` are optional and only matter together -- passing both is what
    # activates team_totals parsing in _normalize_secondary().
    def get_event_odds(
        self, sport_key: str, event_id: str, markets: tuple[str, ...] = ("totals", "btts"),
        regions: tuple[str, ...] | None = None, home_team: str | None = None, away_team: str | None = None,
    ) -> NormalizedSecondaryOdds | None:
        effective_regions = regions if regions is not None else self._regions
        cost = len(markets) * len(effective_regions)

        if self._credit_counter.would_exceed(cost, self._credit_limit, self._safety_margin):
            LOGGER.warning(
                "OddsAPIClient.get_event_odds: skipping event_id=%s, would cross safety margin "
                "(used=%d cost=%d limit=%d safety_margin=%d).",
                event_id, self._credit_counter.credits_used, cost, self._credit_limit, self._safety_margin,
            )
            return None

        response = self._session.get(
            f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds",
            params={
                "apiKey": self._api_key,
                "regions": ",".join(effective_regions),
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        response.raise_for_status()
        self._credit_counter.record_usage(cost)

        return _normalize_secondary(response.json(), home_team=home_team, away_team=away_team)
