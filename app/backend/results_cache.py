"""W213: persistent cache for FINISHED match results, sourced from
football-data.org (D1a) via `FootballDataClient.get_results()`. Motivated
by direct user feedback (2026-09-14): every settlement attempt -- now
including the auto-settle-on-log path (W212) -- hit the live API fresh,
rate-limited to ~10 requests/minute with a blocking sleep once exhausted.
A finished match's score is immutable once set, so unlike `get_fixtures()`'s
existing 60s TTL cache (`main.py`'s `_fixture_cache`, upcoming fixtures
genuinely change before kickoff), a settled result can be cached forever --
no expiry logic needed at all.

Keyed by (competition_code, date), not match_id: `get_results()` is always
called for one exact day at a time in practice (settlement.py,
recommendation_outcomes.py both pass date_from == date_to), so caching at
that granularity means a repeat query for the same day+competition -- the
common case, since settlement re-checks the same recent dates repeatedly
until every bet on them resolves -- costs zero live calls after the first.
A day with genuinely zero finished matches (a rest day, a midweek gap) is
still marked fetched (`fetched_queries`, separate from `match_results`) so
it isn't re-queried forever waiting for results that will never appear.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3

from app.backend.football_data_client import NormalizedMatch

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "match_results_cache.db"

# Deliberate, documented scope boundary, not a strict floor enforced
# anywhere else in the app -- every current fixture-search window (W211,
# 30 days back from "today") never reaches earlier than this in practice.
# A query for an older date is always a cache miss, falling straight
# through to the live API unchanged from pre-cache behavior -- correctness
# is never at risk, only that older dates don't get the speed benefit.
SUPPORTED_SINCE = "2026-08-01"

# BUG (found live, 2026-10-11): an empty FINISHED-matches answer got cached
# here permanently (no TTL, by design -- see module docstring) before
# football-data.org had actually posted that day's results, freezing a
# wrong "0 matches" answer forever even once the real (non-empty) results
# existed upstream. A non-empty answer is unambiguous -- real FINISHED
# matches are definitionally final -- but an empty one can mean either
# "genuinely no matches" or "not reported yet", and this cache has no way
# to tell those apart at write time. Checking at *read* time instead,
# against the already-stored `fetched_at`, means an old poisoned row
# self-heals the moment it's read again with enough real time having
# passed -- no manual purge of already-cached bad rows required.
# ponytail: calendar-day heuristic, not a provider-confirmed signal. Raise
# this (or check the provider's own fixture status) if lag beyond 2 days
# is ever observed.
EMPTY_RESULT_TRUST_DELAY_DAYS = 2


def _utc_now() -> datetime:
    """Split out so tests can monkeypatch it -- mirrors football_data_client
    .py's own _utc_today_isoformat() convention (deterministic "how much
    time has passed" assertions shouldn't depend on which real calendar
    day the test happens to run on)."""
    return datetime.now(timezone.utc)


def _empty_result_is_trustworthy(match_date: str, fetched_at: str) -> bool:
    fetched_date = datetime.fromisoformat(fetched_at).date()
    return (fetched_date - date.fromisoformat(match_date)).days >= EMPTY_RESULT_TRUST_DELAY_DAYS


class ResultsCache:
    """SQLite-backed cache of FINISHED match results. See module docstring
    for the caching strategy."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS match_results (
                    match_id TEXT PRIMARY KEY,
                    competition_code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    utc_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fetched_queries (
                    competition_code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (competition_code, date)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_match_results_lookup ON match_results (competition_code, date)"
            )

    def get(self, competition_code: str, date: str) -> list[NormalizedMatch] | None:
        """Cached results for this (competition_code, date), or None on a
        cache miss (the caller must fetch live, then call `store()`). A hit
        can be an empty list -- a genuine no-finished-matches day is a valid,
        cached answer, not a miss -- but only once it was fetched long enough
        after `date` to trust (see EMPTY_RESULT_TRUST_DELAY_DAYS); an empty
        answer fetched too soon after `date` is treated as a miss instead,
        so the caller re-fetches live and overwrites it via `store()`.
        Always a miss before SUPPORTED_SINCE."""
        if date < SUPPORTED_SINCE:
            return None
        with self._connect() as conn:
            fetched = conn.execute(
                "SELECT fetched_at FROM fetched_queries WHERE competition_code = ? AND date = ?",
                (competition_code, date),
            ).fetchone()
            if fetched is None:
                return None
            rows = conn.execute(
                "SELECT match_id, utc_date, status, home_team, away_team, home_goals, away_goals "
                "FROM match_results WHERE competition_code = ? AND date = ?",
                (competition_code, date),
            ).fetchall()
        if not rows and not _empty_result_is_trustworthy(date, fetched[0]):
            return None
        # `competition` deliberately left at its dataclass default here, not
        # set to `competition_code` -- the live (uncached) path through
        # FootballDataClient._normalize() never sets it either (it's an
        # E0-specific field main.py's /api/fixtures merge logic populates
        # separately), so leaving it unset keeps a cached result identical
        # to what an uncached call would have returned. Neither settlement.py
        # nor recommendation_outcomes.py reads this field.
        return [
            NormalizedMatch(
                match_id=row[0], utc_date=row[1], status=row[2],
                home_team=row[3], away_team=row[4], home_goals=row[5], away_goals=row[6],
            )
            for row in rows
        ]

    def store(self, competition_code: str, date: str, matches: list[NormalizedMatch]) -> None:
        """Persists a live result (possibly empty) for (competition_code,
        date) and marks it fetched. No-op before SUPPORTED_SINCE, so an
        out-of-scope date is never written."""
        if date < SUPPORTED_SINCE:
            return
        fetched_at = _utc_now().replace(microsecond=0).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO fetched_queries (competition_code, date, fetched_at) VALUES (?, ?, ?)",
                (competition_code, date, fetched_at),
            )
            for match in matches:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO match_results
                    (match_id, competition_code, date, utc_date, status, home_team, away_team,
                     home_goals, away_goals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (match.match_id, competition_code, date, match.utc_date, match.status,
                     match.home_team, match.away_team, match.home_goals, match.away_goals),
                )
