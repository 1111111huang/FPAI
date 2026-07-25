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

import pandas as pd
import requests

from app.backend.football_data_client import NormalizedMatch
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_sweden_allsvenskan"

_LOG = get_logger(__name__)


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


def historical_results_from_raw_matches(date_from: str | None, date_to: str | None) -> list[NormalizedMatch]:
    """W71: The Odds API's /scores endpoint (get_results, below) can only
    ever see the last few real days (daysFrom<=3, a hard provider limit) --
    it structurally cannot serve an arbitrary historical date the way
    football-data.org's get_results() does for E0 (W45). raw_matches
    already has real Allsvenskan history back to 2012 (the ML engine's own
    ingestion target, src/ingestion/football_data/sweden_fetcher.py), so
    historical SWE fixtures are sourced from there instead for any
    already-past date range. get_fixtures() (future dates) is unaffected --
    the Odds API's /events endpoint serves that correctly.

    Never raises: DuckDBManager()/load_settings() can raise (missing/invalid
    config.yaml) and conn.execute() can raise a duckdb.Error (e.g. no
    raw_matches table in this environment yet) -- neither should take down
    the whole /api/fixtures response (which may already have real E0 results
    merged in from a separate branch); a lookup failure here degrades to an
    empty SWE result set instead, mirroring
    recommendations._lookup_corpus_match_id's identical never-raises
    contract for the same class of DB-availability failure.

    W71 known limitation (match_id id-space discontinuity, out of scope to
    fix here): this returns raw_matches.match_id, a different id space than
    SwedenFixturesClient.get_fixtures()'s Odds-API event ids (see this
    module's get_fixtures/get_results above) -- mirrors the same disjoint-
    id-space situation settlement.py already documents for football-data.org
    vs The Odds API match ids. In live/production mode this means a SWE
    match's cached recommendation (keyed by match_id) can become unreachable
    once the match transitions from "upcoming" (Odds API event id) to
    "finished" (raw_matches match_id) mid-session. This only matters outside
    sandbox mode -- a sandbox session's SANDBOX_DATE is static, so a fixture
    never transitions status within one session."""
    try:
        db = DuckDBManager()
        query = "SELECT match_id, date, home_team, away_team, fthg, ftag FROM raw_matches WHERE league = 'SWE'"
        params: list[str] = []
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)
        query += " ORDER BY date"
        with db.connection(read_only=True) as conn:
            rows = conn.execute(query, params).fetchdf()
    except Exception:
        _LOG.warning(
            "historical_swe_results_lookup_failed | date_from=%s | date_to=%s",
            date_from, date_to, exc_info=True,
        )
        return []

    return [
        NormalizedMatch(
            match_id=row["match_id"],
            utc_date=row["date"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            status="FINISHED",
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_goals=None if row["fthg"] is None or pd.isna(row["fthg"]) else int(row["fthg"]),
            away_goals=None if row["ftag"] is None or pd.isna(row["ftag"]) else int(row["ftag"]),
            competition="SWE",
        )
        for _, row in rows.iterrows()
    ]
