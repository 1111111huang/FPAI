"""Shared per-unit isolation primitive for the background/async surface.

run_isolated()/run_isolated_async() wrap a single unit of work (one
fixture, one league, one notify call) so a caught exception never
propagates to the caller's loop -- the loop always continues to the next
unit. Every caught failure is written to FailuresStore (sqlite3, same
fresh-connection-per-call pattern as scheduler.py's JobRunLog) instead of
just a log line, and alerted (debounced) via the existing slack_alerts
webhook path on first occurrence within the cooldown window.

2026-11-02 design: docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import traceback as traceback_module
from typing import Any, Awaitable, Callable, TypeVar

from app.backend.slack_alerts import post_slack_alert
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

T = TypeVar("T")

# W163-style convention: repo-root data/, not app/data/ -- matches every
# other per-concern store in this codebase (job_runs.db, recommendation_cache.db).
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "failures.db"

_FAILURES_COLUMNS = ("id", "ts", "subsystem", "unit_id", "league", "exc_type", "message", "traceback", "retryable", "alerted")


class FailuresStore:
    """Persists one row per run_isolated-caught failure. retryable is
    reserved (always NULL today) -- see the design spec's own note on why
    automatic transient/deterministic classification isn't part of this
    primitive."""

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
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    league TEXT,
                    exc_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    traceback TEXT NOT NULL,
                    retryable INTEGER,
                    alerted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_subsystem_ts ON failures (subsystem, ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_league_ts ON failures (league, ts)"
            )

    def record(
        self, subsystem: str, unit_id: str, exc: BaseException, alerted: bool, league: str | None = None,
    ) -> None:
        tb = "".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO failures (ts, subsystem, unit_id, league, exc_type, message, traceback, retryable, alerted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    subsystem,
                    unit_id,
                    league,
                    type(exc).__name__,
                    str(exc),
                    tb,
                    1 if alerted else 0,
                ),
            )

    def list_recent(
        self, since: str | None = None, subsystem: str | None = None, league: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT " + ", ".join(_FAILURES_COLUMNS) + " FROM failures WHERE 1=1"
        params: list[Any] = []
        if since is not None:
            query += " AND ts >= ?"
            params.append(since)
        if subsystem is not None:
            query += " AND subsystem = ?"
            params.append(subsystem)
        if league is not None:
            query += " AND league = ?"
            params.append(league)
        query += " ORDER BY ts DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(zip(_FAILURES_COLUMNS, row)) for row in rows]


_singleton_store: FailuresStore | None = None


def get_failures_store() -> FailuresStore:
    """Module-level singleton, matching bets.get_bet_tracker's own pattern
    (main.py) -- overridden in tests by resetting this module attribute."""
    global _singleton_store
    if _singleton_store is None:
        # Pass DEFAULT_DB_PATH explicitly (rather than relying on
        # FailuresStore.__init__'s default parameter) so a test's
        # monkeypatch.setattr on this module's DEFAULT_DB_PATH is honored --
        # a default arg value is bound once at class-definition time, before
        # any monkeypatch runs.
        _singleton_store = FailuresStore(DEFAULT_DB_PATH)
    return _singleton_store


def _record_failure(subsystem: str, unit_id: str, exc: BaseException, league: str | None) -> None:
    """Never raises -- a failure to record a failure must degrade to a log
    line, not become a second unhandled exception (same contract
    slack_alerts.post_slack_alert already holds itself to)."""
    try:
        alerted = post_slack_alert(
            f":warning: {subsystem} failed for {unit_id}: {type(exc).__name__}: {exc}",
            debounce_key=f"run_isolated:{subsystem}:{type(exc).__name__}",
        )
        get_failures_store().record(subsystem, unit_id, exc, alerted=alerted, league=league)
    except Exception:
        LOGGER.exception(
            "run_isolated: failed to record failure for subsystem=%s unit_id=%s -- logging only.",
            subsystem, unit_id,
        )


def run_isolated(subsystem: str, unit_id: str, fn: Callable[[], T], league: str | None = None) -> T | None:
    """Runs fn(), catching any exception. On failure: logs, alerts
    (debounced, first occurrence of this subsystem+exc_type pair within the
    cooldown window), records a failures row, and returns None instead of
    raising -- the caller's loop always continues to the next unit.
    `league`, when known at the call site, is stored so the /ops dashboard
    can filter by it -- NULL for failures not scoped to one league (e.g. a
    shared-builder failure, BUG-101's class)."""
    try:
        return fn()
    except Exception as exc:
        LOGGER.warning("run_isolated: %s/%s failed: %s", subsystem, unit_id, exc, exc_info=exc)
        _record_failure(subsystem, unit_id, exc, league)
        return None


async def run_isolated_async(
    subsystem: str, unit_id: str, fn: Callable[[], Awaitable[T]], league: str | None = None,
) -> T | None:
    """Same contract as run_isolated, for an async unit of work (an
    asyncio.to_thread-wrapped vendor/LLM call)."""
    try:
        return await fn()
    except Exception as exc:
        LOGGER.warning("run_isolated_async: %s/%s failed: %s", subsystem, unit_id, exc, exc_info=exc)
        _record_failure(subsystem, unit_id, exc, league)
        return None
