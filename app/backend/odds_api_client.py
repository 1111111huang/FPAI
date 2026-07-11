"""The Odds API client (W07, D2b) -- fetches 1X2 (h2h) odds for E0 fixtures,
scoped to a single market and single region to stay within the free tier's
~500-credit/month budget.

NOTE on verification status: no Odds API key was available at implementation
time (2026-07-11) -- agreed with the user to build this against mocks and do
a live smoke-test later, the same pattern used once already for W06. The
event-shape normalization in _normalize() follows The Odds API's publicly
documented v4 schema (sport_key/commence_time/home_team/away_team/
bookmakers[].markets[].outcomes[]), not a live-captured response. Treat this
as an assumption to verify, not a confirmed fact, before trusting it in
production -- unlike football_data_client.py, which was verified live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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


def _normalize(event: dict) -> NormalizedOdds:
    home_team = event["home_team"]
    away_team = event["away_team"]
    home_odds = draw_odds = away_odds = None

    bookmakers = event.get("bookmakers") or []
    if bookmakers:
        for market in bookmakers[0].get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
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
    ) -> None:
        self._now_fn = now_fn
        self._credits_used = credits_used
        self._month_key = month_key or _month_key(now_fn())

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

    def to_dict(self) -> dict:
        return {"credits_used": self._credits_used, "month_key": self._month_key}

    @classmethod
    def from_dict(
        cls, data: dict, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> "CreditCounter":
        return cls(now_fn=now_fn, credits_used=data.get("credits_used", 0), month_key=data.get("month_key"))


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

    def get_odds(self, sport_key: str = "soccer_epl") -> list[NormalizedOdds] | None:
        cost = len(self._markets) * len(self._regions)

        if self._credit_counter.would_exceed(cost, self._credit_limit, self._safety_margin):
            LOGGER.warning(
                "OddsAPIClient: skipping call, would cross safety margin "
                "(used=%d cost=%d limit=%d safety_margin=%d) -- keeping last-known odds.",
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
