"""football-data.org client (W05) -- fetches upcoming fixtures and completed
results for supported leagues/date ranges. Odds are explicitly NOT sourced
here (football-data.org's free tier has none -- see D1a/D2b, W07 is the odds
provider). Returns a normalized internal shape independent of the provider's
own field names, respecting the free tier's ~10-requests/minute rate limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import requests

BASE_URL = "https://api.football-data.org/v4"


@dataclass(frozen=True)
class NormalizedMatch:
    match_id: str
    utc_date: str
    status: str
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None


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


class FootballDataClient:
    """Typed wrapper around football-data.org's fixtures/results API."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        rate_limiter: _RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._rate_limiter = rate_limiter or _RateLimiter()

    def get_fixtures(
        self, competition_code: str = "PL", date_from: str | None = None, date_to: str | None = None,
    ) -> list[NormalizedMatch]:
        return self._get_matches(competition_code, "SCHEDULED", date_from, date_to)

    def get_results(
        self, competition_code: str = "PL", date_from: str | None = None, date_to: str | None = None,
    ) -> list[NormalizedMatch]:
        return self._get_matches(competition_code, "FINISHED", date_from, date_to)

    def _get_matches(
        self, competition_code: str, status: str, date_from: str | None, date_to: str | None,
    ) -> list[NormalizedMatch]:
        params: dict[str, str] = {"status": status}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

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
