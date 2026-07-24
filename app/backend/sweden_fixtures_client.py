"""W57: Sweden (Allsvenskan) fixtures/results client, backed by The Odds API
instead of football-data.org.

W55's research found football-data.org's free tier doesn't cover Allsvenskan
at all -- confirmed live against the project's real key: exactly 13
competitions returned by /v4/competitions, no Sweden. football-data.co.uk
(the ML engine's own ingestion source, src/ingestion/football_data/
sweden_fetcher.py) is a played-results-only historical file (verified live:
3,489 rows, zero with a blank score) -- structurally unusable as a fixtures
source regardless of provider choice. The Odds API (already integrated for
odds, W07/odds_api_client.py) turns out to cover both jobs for Sweden:
/events returns upcoming fixtures (confirmed live: 0 credits/call) and
/scores returns completed results (confirmed live: 2 credits/call, capped at
daysFrom<=3 -- the provider rejects anything higher with a 422). Both are
normalized here to the same NormalizedMatch shape FootballDataClient already
returns, so main.py/settlement.py can treat both sources uniformly.

EPL is unaffected -- it stays on FootballDataClient/football-data.org
entirely, unchanged."""

from __future__ import annotations

import requests

from app.backend.football_data_client import NormalizedMatch

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_sweden_allsvenskan"


def _extract_goals(scores: list[dict] | None, team: str) -> int | None:
    if not scores:
        return None
    for entry in scores:
        if entry.get("name") == team:
            try:
                return int(entry["score"])
            except (TypeError, ValueError):
                return None
    return None


def _in_range(commence_time: str, date_from: str | None, date_to: str | None) -> bool:
    day = commence_time[:10]
    if date_from and day < date_from:
        return False
    if date_to and day > date_to:
        return False
    return True


def _normalize_event(event: dict, status: str) -> NormalizedMatch:
    scores = event.get("scores") if status == "FINISHED" else None
    return NormalizedMatch(
        match_id=str(event["id"]),
        utc_date=event["commence_time"],
        status=status,
        home_team=event["home_team"],
        away_team=event["away_team"],
        home_goals=_extract_goals(scores, event["home_team"]),
        away_goals=_extract_goals(scores, event["away_team"]),
    )


class SwedenFixturesClient:
    """Fixtures/results for Swedish Allsvenskan (SWE), sourced from The Odds
    API. Duck-type compatible with FootballDataClient's get_fixtures/
    get_results signature (minus competition_code -- this client is
    inherently scoped to one sport_key)."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        sport_key: str = SPORT_KEY,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._sport_key = sport_key

    def get_fixtures(
        self, date_from: str | None = None, date_to: str | None = None,
    ) -> list[NormalizedMatch]:
        response = self._session.get(
            f"{BASE_URL}/sports/{self._sport_key}/events",
            params={"apiKey": self._api_key},
            timeout=10,
        )
        response.raise_for_status()
        return [
            _normalize_event(event, "SCHEDULED")
            for event in response.json()
            if _in_range(event["commence_time"], date_from, date_to)
        ]

    def get_results(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        days_from: int = 3,
    ) -> list[NormalizedMatch]:
        # The Odds API's /scores endpoint rejects daysFrom outside [1, 3]
        # (confirmed live: daysFrom=5 -> 422 INVALID_SCORES_DAYS_FROM).
        clamped_days_from = max(1, min(3, days_from))
        response = self._session.get(
            f"{BASE_URL}/sports/{self._sport_key}/scores",
            params={"apiKey": self._api_key, "daysFrom": clamped_days_from},
            timeout=10,
        )
        response.raise_for_status()
        return [
            _normalize_event(event, "FINISHED")
            for event in response.json()
            if event.get("completed") and _in_range(event["commence_time"], date_from, date_to)
        ]
