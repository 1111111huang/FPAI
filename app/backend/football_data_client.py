"""football-data.org client (W05) -- fetches upcoming fixtures and completed
results for supported leagues/date ranges. Odds are explicitly NOT sourced
here (football-data.org's free tier has none -- see D1a/D2b, W07 is the odds
provider). Returns a normalized internal shape independent of the provider's
own field names, respecting the free tier's ~10-requests/minute rate limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING, Callable

import requests

if TYPE_CHECKING:
    # W213: only needed for the type hint below -- results_cache.py imports
    # NormalizedMatch from this module, so a real (non-guarded) import here
    # would be circular.
    from app.backend.results_cache import ResultsCache

BASE_URL = "https://api.football-data.org/v4"


def _utc_today_isoformat() -> str:
    """Genuine wall-clock UTC 'today' -- split out (not inlined in
    get_results() below) so tests can monkeypatch it, mirroring main.py's
    own _current_real_date()/_fixture_cache_now() pattern for exactly the
    same reason: exercising "is this date today" deterministically without
    the test's pass/fail depending on which real calendar day it happens to
    run on."""
    return datetime.now(timezone.utc).date().isoformat()


@dataclass(frozen=True)
class NormalizedMatch:
    match_id: str
    utc_date: str
    status: str
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    # W64: which competition this fixture belongs to -- "E0" or "SWE" today.
    # Defaults to "E0" so every existing construction site across the
    # codebase (tests included) keeps working unchanged; only
    # get_fixtures()'s merge logic in main.py sets it explicitly per source.
    competition: str = "E0"


def _normalize(raw: dict) -> NormalizedMatch:
    full_time = (raw.get("score") or {}).get("fullTime") or {}
    return NormalizedMatch(
        match_id=str(raw["id"]),
        utc_date=raw["utcDate"],
        status=raw["status"],
        home_team=raw["homeTeam"]["shortName"],
        away_team=raw["awayTeam"]["shortName"],
        home_goals=full_time.get("home"),
        away_goals=full_time.get("away"),
    )


class _RateLimiter:
    """Tracks the free tier's ~10-requests/minute budget from the provider's
    own response headers (x-requests-available-minute, X-RequestCounter-Reset)
    and proactively sleeps until the window resets once exhausted, rather than
    waiting to be rejected with a 429."""

    def __init__(
        self,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn
        self._remaining: int | None = None
        self._reset_at: float | None = None

    def update_from_headers(self, headers) -> None:
        remaining = headers.get("x-requests-available-minute")
        reset_seconds = headers.get("X-RequestCounter-Reset")
        if remaining is not None:
            self._remaining = int(remaining)
        if reset_seconds is not None:
            self._reset_at = self._time_fn() + int(reset_seconds)

    def wait_if_needed(self) -> None:
        if self._remaining is None or self._remaining > 0 or self._reset_at is None:
            return
        wait_seconds = self._reset_at - self._time_fn()
        if wait_seconds > 0:
            self._sleep_fn(wait_seconds)

    def would_block(self) -> bool:
        """True if a call right now would hit wait_if_needed()'s blocking
        sleep -- lets a caller on a user-facing request path (e.g.
        auto-settle-on-log, below) decide to skip instead of blocking,
        without duplicating wait_if_needed()'s own exhaustion logic."""
        if self._remaining is None or self._remaining > 0 or self._reset_at is None:
            return False
        return self._reset_at - self._time_fn() > 0


class RateLimitWouldBlock(requests.exceptions.RequestException):
    """Raised by _get_matches() when blocking=False and the rate limiter's
    budget is exhausted, instead of sleeping for up to a minute. Subclasses
    RequestException so every existing per-competition try/except in
    settlement.py's settle_open_bets() already catches it -- no new except
    clause needed anywhere that already tolerates a transient results-fetch
    failure."""


class FootballDataClient:
    """Typed wrapper around football-data.org's fixtures/results API."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        rate_limiter: _RateLimiter | None = None,
        results_cache: ResultsCache | None = None,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._rate_limiter = rate_limiter or _RateLimiter()
        # W213: optional -- every existing construction site (tests
        # included) omits this and keeps calling the live API on every
        # get_results(), completely unchanged.
        self._results_cache = results_cache

    def get_fixtures(
        self, competition_code: str = "PL", date_from: str | None = None, date_to: str | None = None,
    ) -> list[NormalizedMatch]:
        # Found live (2026-08-14): a bare status=SCHEDULED filter is honored
        # inconsistently per competition by football-data.org -- PL returned
        # both TIMED and SCHEDULED matches under it, but PD strictly excluded
        # every TIMED (confirmed-kickoff-time) match, silently dropping La
        # Liga's entire next ~4 weeks of fixtures while EPL's identically-shaped
        # near-term matches happened to survive. Comma-separated works
        # reliably against the real API and doesn't depend on that quirk.
        #
        # IN_PLAY/PAUSED added (2026-08-15): a match currently being played is
        # neither SCHEDULED/TIMED (kickoff already happened) nor FINISHED (not
        # over yet) -- without these, a live match is invisible to this call
        # for the entire window it's actually being played, then reappears
        # once it's FINISHED. get_results() deliberately stays FINISHED-only
        # (a live match isn't a result yet either).
        #
        # LIVE added (2026-08-16): found live -- PD's own in-progress matches
        # report status="LIVE" verbatim (confirmed via a direct single-match
        # fetch), not "IN_PLAY" as the above assumed. IN_PLAY/PAUSED is left
        # in rather than replaced -- still real, documented football-data.org
        # values (used elsewhere, e.g. by other competitions/seasons) -- this
        # is additive coverage for a second live-match spelling, not a
        # correction of the first. Same invisibility bug either way: a status
        # string this filter doesn't name is silently dropped, not erred on.
        return self._get_matches(competition_code, "SCHEDULED,TIMED,IN_PLAY,PAUSED,LIVE", date_from, date_to)

    def get_results(
        self, competition_code: str = "PL", date_from: str | None = None, date_to: str | None = None,
        blocking: bool = True,
    ) -> list[NormalizedMatch]:
        # blocking=False (settlement.py's settle_open_bets, threaded down
        # from main.py's auto-settle-on-log path only -- see
        # RateLimitWouldBlock's own docstring): raises instead of sleeping
        # through _RateLimiter.wait_if_needed()'s up-to-a-minute blocking
        # wait when the budget's exhausted. A bet-logging request
        # shouldn't hang on an external API's rate limit just because it
        # also tries to auto-settle -- the bet is already saved open either
        # way (_settle_if_already_decided's own try/except), so skipping
        # cleanly here just means it settles later via the normal Settle
        # open bets flow (still blocking=True, a deliberate user-triggered
        # wait) instead of immediately.
        #
        # W213: caching only applies to the single-exact-day shape every
        # real caller actually uses (settlement.py/recommendation_outcomes.py
        # both pass date_from == date_to) -- a genuine multi-day range
        # bypasses the cache entirely rather than splitting it into daily
        # entries, since no caller needs that.
        #
        # BUG (found live, 2026-09-15): the cache's whole premise --
        # ResultsCache's own module docstring -- is that a FINISHED result
        # is immutable once set, so caching forever (no TTL) is safe. True
        # for a genuinely past day; false for *today*, which keeps
        # producing newly-finished matches as the day goes on. The first
        # same-day call (e.g. GET /api/fixtures?date_from=date_to=today,
        # or a same-day settlement check, W212's auto-settle-on-log
        # included) permanently cached whatever had finished *so far* --
        # every match that finished after that first call became invisible
        # to every later same-day caller, forever, since this cache never
        # expires anything. Today is the one date that must always bypass
        # the persistent cache and ask the live API fresh.
        is_today = date_from is not None and date_from == date_to and date_from == _utc_today_isoformat()
        if self._results_cache is not None and date_from is not None and date_from == date_to and not is_today:
            cached = self._results_cache.get(competition_code, date_from)
            if cached is not None:
                return cached
            matches = self._get_matches(competition_code, "FINISHED", date_from, date_to, blocking=blocking)
            self._results_cache.store(competition_code, date_from, matches)
            return matches
        return self._get_matches(competition_code, "FINISHED", date_from, date_to, blocking=blocking)

    def _get_matches(
        self, competition_code: str, status: str, date_from: str | None, date_to: str | None,
        blocking: bool = True,
    ) -> list[NormalizedMatch]:
        params: dict[str, str] = {"status": status}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        if not blocking and self._rate_limiter.would_block():
            raise RateLimitWouldBlock(
                f"Rate limit exhausted for {competition_code}; skipping non-blocking call rather than waiting."
            )
        self._rate_limiter.wait_if_needed()
        response = self._session.get(
            f"{BASE_URL}/competitions/{competition_code}/matches",
            headers={"X-Auth-Token": self._api_key},
            params=params,
            timeout=10,
        )
        self._rate_limiter.update_from_headers(response.headers)
        response.raise_for_status()

        payload = response.json()
        return [_normalize(match) for match in payload.get("matches", [])]
