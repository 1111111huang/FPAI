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
