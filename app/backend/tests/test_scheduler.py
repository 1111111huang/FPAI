"""W08: scheduler infrastructure -- an in-process APScheduler running
alongside FastAPI, timezone-safe via zoneinfo("America/New_York"). Job
state must be re-derivable from persisted fixture/kickoff data on startup
(here: a SQLite JobRunLog of 'this job_id/run_key already ran' markers),
so a backend restart doesn't silently drop a pending job or miss a window
without catching up -- confirmed via simulating a restart mid-day."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Polls predicate() until it's true or timeout elapses. W159 follow-up:
    the immediate catch-up path (schedule_daily()/schedule_once()) now runs
    on a background thread without waiting (a hang-on-startup fix -- see
    scheduler.py's _run_and_mark docstring), so its side effects are no
    longer guaranteed visible the instant the call returns. Condition-based
    waiting instead of a fixed sleep: fast when the thread finishes
    quickly (the common case in these tests), still correct if it's ever
    slower."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_job_run_log_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "job_runs.db"
    JobRunLog(db_path=db_path).mark_ran("daily_eod", "2026-07-12")

    reloaded = JobRunLog(db_path=db_path)
    assert reloaded.has_run("daily_eod", "2026-07-12")
    assert not reloaded.has_run("daily_eod", "2026-07-13")
    assert not reloaded.has_run("other_job", "2026-07-12")


def test_daily_job_catches_up_immediately_when_trigger_time_already_passed(tmp_path: Path) -> None:
    """Simulates a restart mid-day: 'now' is past today's trigger time and
    the job hasn't run yet today -- must run immediately, not wait for
    tomorrow's cron fire."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 7, 12, 23, 30, tzinfo=NY_TZ)  # 30 min past a 23:00 trigger

    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    scheduler.schedule_daily("daily_eod", lambda: calls.append("ran"), hour=23, minute=0)

    assert _wait_until(lambda: run_log.has_run("daily_eod", "2026-07-12"))
    assert calls == ["ran"]


def test_daily_job_does_not_rerun_once_already_run_today(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    run_log.mark_ran("daily_eod", "2026-07-12")
    calls = []
    now = datetime(2026, 7, 12, 23, 30, tzinfo=NY_TZ)

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_daily(
        "daily_eod", lambda: calls.append("ran"), hour=23, minute=0
    )

    assert calls == []


def test_daily_job_not_triggered_early(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 7, 12, 10, 0, tzinfo=NY_TZ)  # well before 23:00

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_daily(
        "daily_eod", lambda: calls.append("ran"), hour=23, minute=0
    )

    assert calls == []


def test_schedule_once_catches_up_when_run_at_already_passed(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 22, 15, 5, tzinfo=NY_TZ)
    run_at = now - timedelta(minutes=5)  # scheduled 5 min ago -- missed

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_once(
        "t30_m1", lambda: calls.append("ran"), run_at=run_at
    )

    assert _wait_until(lambda: run_log.has_run("t30_m1", run_at.isoformat()))
    assert calls == ["ran"]


def test_schedule_once_does_not_rerun_once_marked_ran(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 8, 22, 15, 5, tzinfo=NY_TZ)
    run_at = now - timedelta(minutes=5)
    run_log.mark_ran("t30_m1", run_at.isoformat())
    calls = []

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_once(
        "t30_m1", lambda: calls.append("ran"), run_at=run_at
    )

    assert calls == []


def test_schedule_once_not_triggered_early(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 22, 14, 0, tzinfo=NY_TZ)
    run_at = now + timedelta(minutes=30)  # still in the future

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_once(
        "t30_m1", lambda: calls.append("ran"), run_at=run_at
    )

    assert calls == []


def test_weekly_job_catches_up_when_trigger_day_and_time_already_passed(tmp_path: Path) -> None:
    """2026-08-23 is a Sunday. 'now' is past that Sunday's 09:00 trigger and
    the job hasn't run yet this week -- must run immediately, not wait for
    next Sunday's cron fire."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 23, 9, 30, tzinfo=NY_TZ)

    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    scheduler.schedule_weekly("weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0)

    assert _wait_until(lambda: run_log.has_run("weekly_review", "2026-08-23"))
    assert calls == ["ran"]


def test_weekly_job_does_not_catch_up_on_a_different_weekday_even_past_the_trigger_time(tmp_path: Path) -> None:
    """2026-08-24 is a Monday -- 09:30 is past 09:00, but this isn't the
    target weekday (Sunday=6), so schedule_daily()'s own 'hour:minute
    already passed today' catch-up logic must NOT fire here."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 24, 9, 30, tzinfo=NY_TZ)

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_weekly(
        "weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0
    )

    assert calls == []


def test_weekly_job_not_triggered_early_on_the_target_weekday(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 23, 3, 0, tzinfo=NY_TZ)  # Sunday, well before 09:00

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_weekly(
        "weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0
    )

    assert calls == []


def test_weekly_job_does_not_rerun_once_already_run_this_week(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    run_log.mark_ran("weekly_review", "2026-08-23")
    calls = []
    now = datetime(2026, 8, 23, 9, 30, tzinfo=NY_TZ)

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_weekly(
        "weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0
    )

    assert calls == []


def test_restart_mid_day_detects_and_runs_a_missed_job_only_once(tmp_path: Path) -> None:
    """Explicit restart narrative: process A registers today's daily job
    before its trigger time (no catch-up yet); process A then crashes
    (simulated by simply discarding it, since only JobRunLog persists
    across a real restart); process B starts later, past the trigger
    time -- it must detect the job never ran and run it now, exactly
    once, not skip it and not double-run it on a third registration."""
    db_path = tmp_path / "job_runs.db"
    run_log_a = JobRunLog(db_path=db_path)
    calls = []

    before_trigger = datetime(2026, 7, 12, 10, 0, tzinfo=NY_TZ)
    RecoverableScheduler(run_log=run_log_a, now_fn=lambda: before_trigger).schedule_daily(
        "daily_eod", lambda: calls.append("ran"), hour=23, minute=0
    )
    assert calls == []  # process A: too early, nothing ran yet

    # "restart": a fresh JobRunLog instance reading the same on-disk db,
    # a fresh RecoverableScheduler, later in the day
    run_log_b = JobRunLog(db_path=db_path)
    after_trigger = datetime(2026, 7, 12, 23, 45, tzinfo=NY_TZ)
    RecoverableScheduler(run_log=run_log_b, now_fn=lambda: after_trigger).schedule_daily(
        "daily_eod", lambda: calls.append("ran"), hour=23, minute=0
    )
    # Waits on run_log_b.has_run() specifically, not just calls -- that's
    # the actual condition the third registration below depends on; waiting
    # on calls alone leaves a real window between calls.append() and
    # run_log.mark_ran() where the third registration can race in and spawn
    # its own (duplicate) catch-up thread.
    assert _wait_until(lambda: run_log_b.has_run("daily_eod", "2026-07-12"))
    assert calls == ["ran"]  # process B: detected the miss, ran it once

    # a third registration (e.g. another restart) must not double-run it
    run_log_c = JobRunLog(db_path=db_path)
    RecoverableScheduler(run_log=run_log_c, now_fn=lambda: after_trigger).schedule_daily(
        "daily_eod", lambda: calls.append("ran"), hour=23, minute=0
    )
    assert calls == ["ran"]  # unchanged: run_log_c sees it already ran, no catch-up thread spawned at all


def test_scheduler_uses_america_new_york_timezone(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    scheduler = RecoverableScheduler(run_log=run_log)
    assert scheduler.timezone == ZoneInfo("America/New_York")


def test_a_failing_catchup_job_does_not_raise_and_is_not_marked_as_run(tmp_path: Path) -> None:
    """A job failing during the immediate catch-up path (e.g. a real network
    error) must not propagate -- schedule_daily()/schedule_once() run
    synchronously in the caller's own thread (unlike APScheduler's own
    later trigger fires, which run on its background thread), so an
    unguarded exception here would crash whoever is registering the job --
    at worst, the whole app's startup. It also must not be marked as run,
    so the next registration retries it."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 7, 12, 23, 30, tzinfo=NY_TZ)

    def _boom() -> None:
        raise RuntimeError("simulated network failure")

    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    scheduler.schedule_daily("daily_eod", _boom, hour=23, minute=0)  # must not raise

    assert not run_log.has_run("daily_eod", "2026-07-12")


def test_catchup_job_whose_body_calls_asyncio_run_works_from_inside_a_running_loop(tmp_path: Path) -> None:
    """Regression, found live in production: register_eod_job() (app/backend/
    scheduler_wiring.py) is called from FastAPI's async lifespan startup, and
    its job body (_eod_job) calls asyncio.run() internally, once per league.
    Before this fix, the catch-up path ran fn() directly on the calling
    thread -- when the app restarted after today's scheduled EOD hour (the
    catch-up condition), fn() ran on lifespan's own thread, which already
    had a running event loop attached, and asyncio.run() raised "cannot be
    called from a running event loop". The failure was caught and logged
    (not a startup crash) but the job was never marked as run -- meaning
    every restart after the scheduled hour silently failed to regenerate
    recommendations, indefinitely, until a restart happened to land before
    the scheduled hour instead. Reproduces the exact shape: schedule_daily()
    invoked from inside a running event loop, whose job body itself calls
    asyncio.run().

    Follow-up (same production incident, second bug): once fixed to
    actually succeed, the catch-up path could then block the calling
    async context for as long as the job body took to run -- for a real
    EOD catch-up backlog, long enough that FastAPI's lifespan startup
    never completed and the app returned 502 for every request
    indefinitely. schedule_daily()'s catch-up path now runs the job body
    without waiting for it -- asserted here via condition-based polling
    (_wait_until) rather than immediately after schedule_daily() returns,
    since the whole point is that it no longer blocks."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 7, 12, 23, 30, tzinfo=NY_TZ)
    calls = []

    async def _async_work() -> None:
        calls.append("ran")

    def _job_like_eod_job() -> None:
        asyncio.run(_async_work())

    async def _register_from_inside_a_running_loop() -> None:
        scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
        scheduler.schedule_daily("daily_eod", _job_like_eod_job, hour=23, minute=0)
        # the whole point of this test: registering the job (and its
        # catch-up firing) must not block this coroutine -- returning here
        # promptly, before the background thread necessarily finishes, is
        # correct, not a race to work around.

    asyncio.run(_register_from_inside_a_running_loop())

    assert _wait_until(lambda: run_log.has_run("daily_eod", "2026-07-12"))
    assert calls == ["ran"]
