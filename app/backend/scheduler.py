"""W08: scheduler infrastructure -- an in-process APScheduler running
alongside FastAPI, timezone-safe via zoneinfo("America/New_York") (correctly
handles EST/EDT, not a fixed UTC offset).

Job state must be re-derivable from persisted fixture/kickoff data on
startup, so a backend restart doesn't silently drop a pending job or miss
a window without catching up. APScheduler's own default (in-memory)
jobstore loses all scheduled fire times across a restart, so this module
does not rely on it for correctness: JobRunLog persists a durable
'job_id/run_key already ran' marker (SQLite, same pattern as W11/W12's
stores), and RecoverableScheduler checks it at registration time -- every
schedule_daily()/schedule_once() call immediately runs the job if its
trigger time has already passed and it hasn't been marked as run, rather
than waiting for APScheduler's own trigger to fire (which, after a
restart, may never come again for a missed one-off job, or only comes
tomorrow for a missed daily job).

W09 (EOD batch generation) uses schedule_daily(); W10 (per-fixture T-30
refresh) uses schedule_once() once per fixture, re-derived from live
fixture data on each startup/EOD run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
import threading
from typing import Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

NY_TZ = ZoneInfo("America/New_York")

# W163: repo-root data/, not app/data/ -- see recommendation_cache.py's W163
# note. Production's own RecoverableScheduler(run_log=None) falls back to
# JobRunLog() (this default) -- only sandbox mode ever overrode it
# (main.py's _SANDBOX_JOB_RUNS_DB_PATH, via sandbox_clock.sandbox_scoped_path,
# already correctly repo-root-relative), so this exact bug -- the one
# sandbox_scoped_path()'s own docstring names as "already caused one real
# bug (W29)" -- was still live for every real production JobRunLog the
# whole time, just masked by sandbox mode's separate, already-correct path.
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "job_runs.db"


class JobRunLog:
    """Persists 'job_id ran for run_key' markers so a restart can detect a
    missed run instead of silently skipping it."""

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
                CREATE TABLE IF NOT EXISTS job_runs (
                    job_id TEXT NOT NULL,
                    run_key TEXT NOT NULL,
                    ran_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, run_key)
                )
                """
            )

    def mark_ran(self, job_id: str, run_key: str) -> None:
        ran_at = datetime.now(NY_TZ).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO job_runs (job_id, run_key, ran_at) VALUES (?, ?, ?)",
                (job_id, run_key, ran_at),
            )

    def has_run(self, job_id: str, run_key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM job_runs WHERE job_id = ? AND run_key = ?", (job_id, run_key)
            ).fetchone()
        return row is not None


class RecoverableScheduler:
    """Wraps APScheduler's BackgroundScheduler with a restart-safe
    catch-up check, backed by a JobRunLog. Registering a job (daily or
    one-off) both schedules its future APScheduler trigger and immediately
    runs it now if its trigger time has already passed and it hasn't run
    yet -- covering both a fresh-process catch-up and the normal case of
    registering a job whose time simply hasn't come yet."""

    def __init__(
        self,
        run_log: JobRunLog | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(NY_TZ),
        timezone: ZoneInfo = NY_TZ,
    ) -> None:
        self.run_log = run_log if run_log is not None else JobRunLog()
        self._now_fn = now_fn
        self.timezone = timezone
        self._scheduler = BackgroundScheduler(timezone=timezone)

    def schedule_daily(self, job_id: str, fn: Callable[[], None], hour: int, minute: int) -> None:
        self._scheduler.add_job(
            lambda: self._run_and_mark(job_id, fn, run_key=self._now_fn().date().isoformat()),
            trigger=CronTrigger(hour=hour, minute=minute, timezone=self.timezone),
            id=job_id,
            replace_existing=True,
        )
        now = self._now_fn()
        trigger_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        run_key = now.date().isoformat()
        if now >= trigger_today and not self.run_log.has_run(job_id, run_key):
            self._run_and_mark(job_id, fn, run_key, wait=False)

    def schedule_once(self, job_id: str, fn: Callable[[], None], run_at: datetime) -> None:
        # Keyed on (job_id, run_at) rather than a constant: job_id alone is
        # not enough, since a postponed/rescheduled fixture re-registers
        # the *same* job_id (f"t30_{match_id}") with a *new* run_at -- a
        # constant run_key would let the old marker permanently block the
        # new time from ever firing (W33).
        run_key = run_at.isoformat()
        self._scheduler.add_job(
            lambda: self._run_and_mark(job_id, fn, run_key=run_key),
            trigger=DateTrigger(run_date=run_at, timezone=self.timezone),
            id=job_id,
            replace_existing=True,
        )
        if self._now_fn() >= run_at and not self.run_log.has_run(job_id, run_key):
            self._run_and_mark(job_id, fn, run_key, wait=False)

    def schedule_weekly(self, job_id: str, fn: Callable[[], None], day_of_week: int, hour: int, minute: int) -> None:
        """Same restart-safe catch-up contract as schedule_daily(), scoped to
        one weekday instead of every day. day_of_week: 0=Monday..6=Sunday --
        Python's own datetime.weekday() convention, which APScheduler's
        CronTrigger day_of_week integer form also uses, so the catch-up
        check below can compare them directly with no translation.

        Without the day_of_week condition in the catch-up check, a restart
        on any day past `hour:minute` would incorrectly look identical to
        schedule_daily()'s own "missed today's run" case and fire
        immediately regardless of which weekday it actually is."""
        self._scheduler.add_job(
            lambda: self._run_and_mark(job_id, fn, run_key=self._now_fn().date().isoformat()),
            trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute, timezone=self.timezone),
            id=job_id,
            replace_existing=True,
        )
        now = self._now_fn()
        trigger_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        run_key = now.date().isoformat()
        if now.weekday() == day_of_week and now >= trigger_today and not self.run_log.has_run(job_id, run_key):
            self._run_and_mark(job_id, fn, run_key, wait=False)

    def _run_and_mark(self, job_id: str, fn: Callable[[], None], run_key: str, wait: bool = True) -> None:
        """Never lets a job's own exception propagate -- an unguarded failure
        here would crash whoever calls this, at worst the whole app's
        startup. A failed run is also not marked as done, so the next
        registration retries it rather than silently treating a failure as
        a success.

        Runs fn() on a dedicated thread rather than directly on the calling
        thread -- guaranteed loop-free regardless of the caller's own
        context, since fn() may itself call asyncio.run() internally (e.g.
        scheduler_wiring.py's _eod_job). Found live: register_eod_job() is
        called from FastAPI's async lifespan startup; when the immediate
        catch-up path fired there (app restarted after today's scheduled
        hour) and fn() ran directly on lifespan's own thread, which already
        had a running event loop attached, asyncio.run() raised "cannot be
        called from a running event loop" (W159).

        wait=True (the default; APScheduler's own later trigger fires,
        already isolated on their own background thread, and any other
        caller with no reason to return early) blocks until fn() finishes,
        matching the original synchronous contract.

        wait=False (schedule_daily()/schedule_once()'s own immediate
        catch-up path specifically) returns immediately without waiting.
        Found live in production, a second incident right after W159:
        blocking here can hang the calling async context indefinitely --
        register_eod_job()'s catch-up now actually *runs* the EOD job
        (rather than failing fast the way the pre-W159 bug did), and a
        large catch-up backlog (real fixture/odds/LLM calls, potentially
        for every league) took long enough that FastAPI's lifespan startup
        never reached its own `yield`, so the app never finished starting
        and returned 502 for every request indefinitely. The pre-W159 bug
        had accidentally "protected" against this by always failing
        instantly; fixing it to actually succeed exposed this separate,
        real blocking-startup risk."""
        def _target() -> None:
            try:
                fn()
            except Exception:
                LOGGER.exception("RecoverableScheduler: job %r (run_key=%r) failed.", job_id, run_key)
                return
            self.run_log.mark_ran(job_id, run_key)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        if wait:
            thread.join()

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self) -> None:
        self._scheduler.shutdown()
