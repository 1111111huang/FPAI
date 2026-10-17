"""BUG-086: persistent backing store for fixture_cache.py's per-day cache
-- fixture_cache._cache is a plain in-memory dict, wiped on every process
restart (every Railway redeploy), forcing a full re-fetch storm against
football-data.org's shared rate-limited budget on every single deploy.
Mirrors results_cache.py's ResultsCache exactly -- same dedicated SQLite
file under data/, same stdlib sqlite3 usage (no new dependency), same
fresh-connection-per-call pattern (safe across the two concurrency domains
fixture_cache.py's own module docstring already documents: the HTTP event
loop and the reconciliation job's own thread).

Unlike ResultsCache, which caches only immutable FINISHED results forever,
this store covers fixture_cache's full mix (including SCHEDULED/TIMED/LIVE
days that genuinely do change) -- so entries aren't kept forever:
prune_older_than() deletes anything whose day has passed the given cutoff,
called once at boot (fixture_cache.load_from_disk()) before the persisted
data is loaded into the in-memory cache, so the table never accumulates
more than a bounded retention window's worth of history regardless of how
long the process stays up between redeploys."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from app.backend.football_data_client import NormalizedMatch

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fixture_cache.db"


class FixtureCacheStore:
    """SQLite-backed persistence for fixture_cache.py's (call_type, day) ->
    matches entries. See module docstring for why/how."""

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
                CREATE TABLE IF NOT EXISTS cached_days (
                    call_type TEXT NOT NULL,
                    day TEXT NOT NULL,
                    PRIMARY KEY (call_type, day)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_matches (
                    call_type TEXT NOT NULL,
                    day TEXT NOT NULL,
                    match_id TEXT NOT NULL,
                    utc_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    PRIMARY KEY (call_type, day, match_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cached_matches_lookup ON cached_matches (call_type, day)"
            )

    def store_day(self, call_type: str, day: str, matches: list[NormalizedMatch]) -> None:
        """Persists one (call_type, day)'s matches (possibly empty),
        replacing anything previously stored for that key."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cached_days (call_type, day) VALUES (?, ?)",
                (call_type, day),
            )
            conn.execute(
                "DELETE FROM cached_matches WHERE call_type = ? AND day = ?",
                (call_type, day),
            )
            for match in matches:
                conn.execute(
                    """
                    INSERT INTO cached_matches
                    (call_type, day, match_id, utc_date, status, home_team, away_team, home_goals, away_goals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (call_type, day, match.match_id, match.utc_date, match.status,
                     match.home_team, match.away_team, match.home_goals, match.away_goals),
                )

    def prune_older_than(self, cutoff_day: str) -> None:
        """Deletes every cached (call_type, day) entry strictly before
        cutoff_day (an ISO date string, both tables)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cached_matches WHERE day < ?", (cutoff_day,))
            conn.execute("DELETE FROM cached_days WHERE day < ?", (cutoff_day,))

    def load_all(self) -> dict[tuple[str, str], list[NormalizedMatch]]:
        """Everything currently persisted, keyed by (call_type, day) -- a
        key with zero matches still appears (as an empty list) if it has a
        cached_days row, distinguishing a genuinely empty day from one
        never stored at all."""
        with self._connect() as conn:
            days = conn.execute("SELECT call_type, day FROM cached_days").fetchall()
            rows = conn.execute(
                "SELECT call_type, day, match_id, utc_date, status, home_team, away_team, home_goals, away_goals "
                "FROM cached_matches"
            ).fetchall()
        result: dict[tuple[str, str], list[NormalizedMatch]] = {
            (call_type, day): [] for call_type, day in days
        }
        for call_type, day, match_id, utc_date, status, home_team, away_team, home_goals, away_goals in rows:
            result.setdefault((call_type, day), []).append(
                NormalizedMatch(
                    match_id=match_id, utc_date=utc_date, status=status,
                    home_team=home_team, away_team=away_team,
                    home_goals=home_goals, away_goals=away_goals,
                )
            )
        return result
