"""W08: scheduler infrastructure -- an in-process APScheduler running
alongside FastAPI, timezone-safe via zoneinfo("America/New_York"). Job
state must be re-derivable from persisted fixture/kickoff data on startup
(here: a SQLite JobRunLog of 'this job_id/run_key already ran' markers),
so a backend restart doesn't silently drop a pending job or miss a window
without catching up -- confirmed via simulating a restart mid-day."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler


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

    assert calls == ["ran"]
    assert run_log.has_run("daily_eod", "2026-07-12")


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

    assert calls == ["ran"]
    assert run_log.has_run("t30_m1", "once")


def test_schedule_once_does_not_rerun_once_marked_ran(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    run_log.mark_ran("t30_m1", "once")
    calls = []
    now = datetime(2026, 8, 22, 15, 5, tzinfo=NY_TZ)
    run_at = now - timedelta(minutes=5)

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
    assert calls == ["ran"]  # process B: detected the miss, ran it once

    # a third registration (e.g. another restart) must not double-run it
    run_log_c = JobRunLog(db_path=db_path)
    RecoverableScheduler(run_log=run_log_c, now_fn=lambda: after_trigger).schedule_daily(
        "daily_eod", lambda: calls.append("ran"), hour=23, minute=0
    )
    assert calls == ["ran"]


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
