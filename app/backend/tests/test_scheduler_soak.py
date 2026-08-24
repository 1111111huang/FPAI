"""W33: scheduler multi-day soak test + DST correctness. Extends W21's
single-instant restart tests: (1) a rescheduled T-30 job (postponed
fixture, same match_id, new kickoff) must fire at its new time rather than
being silently blocked by an old, same-job-id 'already ran' marker; (2) the
daily EOD job fires exactly once per calendar day across a multi-day
simulated run; (3) a real DST transition (EST<->EDT) doesn't break the
23:00-local trigger comparison, since NY_TZ is a real zoneinfo timezone."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Polls predicate() until it's true or timeout elapses. W159 follow-up:
    the immediate catch-up path now runs the job body on a background
    thread without waiting, so its side effects are no longer guaranteed
    visible the instant schedule_daily()/schedule_once() returns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_rescheduled_t30_job_fires_at_its_new_time_after_a_postponement(tmp_path: Path) -> None:
    """A fixture's original T-30 job fires and is marked ran. The fixture is
    then postponed to a later kickoff and the *same* job_id ('t30_m1') is
    re-registered with a new run_at. The old 'already ran' marker must not
    block the new time from ever firing."""
    db_path = tmp_path / "job_runs.db"
    run_log = JobRunLog(db_path=db_path)
    calls = []

    original_run_at = datetime(2026, 8, 22, 14, 30, tzinfo=NY_TZ)
    now_at_original = datetime(2026, 8, 22, 14, 35, tzinfo=NY_TZ)
    RecoverableScheduler(run_log=run_log, now_fn=lambda: now_at_original).schedule_once(
        "t30_m1", lambda: calls.append("original"), run_at=original_run_at
    )
    # Waits on run_log.has_run() specifically, not just calls -- see
    # test_daily_eod_job_fires_exactly_once... for why calls alone leaves a
    # real race window against the next registration's own catch-up check.
    assert _wait_until(lambda: run_log.has_run("t30_m1", original_run_at.isoformat()))
    assert calls == ["original"]

    # fixture postponed a day later -- same job_id, new run_at
    new_run_at = datetime(2026, 8, 23, 14, 30, tzinfo=NY_TZ)
    now_at_new = datetime(2026, 8, 23, 14, 35, tzinfo=NY_TZ)
    RecoverableScheduler(run_log=run_log, now_fn=lambda: now_at_new).schedule_once(
        "t30_m1", lambda: calls.append("rescheduled"), run_at=new_run_at
    )

    assert _wait_until(lambda: run_log.has_run("t30_m1", new_run_at.isoformat()))
    assert calls == ["original", "rescheduled"]

    # re-registering the *same* new run_at again must still not double-fire
    RecoverableScheduler(run_log=run_log, now_fn=lambda: now_at_new).schedule_once(
        "t30_m1", lambda: calls.append("rescheduled_again"), run_at=new_run_at
    )
    assert calls == ["original", "rescheduled"]  # unchanged: run_log already shows it ran, no thread spawned


def test_daily_eod_job_fires_exactly_once_per_calendar_day_across_a_multi_day_run(tmp_path: Path) -> None:
    """Simulates N days of continuous operation: each day, a fresh
    RecoverableScheduler is (re)constructed at a moment past the 23:00
    trigger (as a real long-running process's own CronTrigger fire, or a
    same-day restart, would both do) and registers the same daily job_id.
    Exactly one fire per day -- never zero, never twice, even if the same
    day's registration happens more than once (e.g. two restarts on the
    same day)."""
    db_path = tmp_path / "job_runs.db"
    run_log = JobRunLog(db_path=db_path)
    calls = []
    base = datetime(2026, 7, 1, 23, 30, tzinfo=NY_TZ)

    for day_offset in range(10):
        now = base + timedelta(days=day_offset)
        RecoverableScheduler(run_log=run_log, now_fn=lambda now=now: now).schedule_daily(
            "eod_batch_generation", lambda day=day_offset: calls.append(day), hour=23, minute=0
        )
        # W159 follow-up: the catch-up fire above runs on a background
        # thread without waiting. Wait on run_log.has_run() specifically,
        # not just calls -- that's the actual condition the next
        # registration's own catch-up check depends on; waiting on calls
        # alone leaves a real window between calls.append() and
        # run_log.mark_ran() where a second registration can race in and
        # spawn its own (duplicate) catch-up thread.
        assert _wait_until(lambda now=now: run_log.has_run("eod_batch_generation", now.date().isoformat()))
        # a same-day restart re-registering must not double-fire today
        RecoverableScheduler(run_log=run_log, now_fn=lambda now=now: now).schedule_daily(
            "eod_batch_generation", lambda day=day_offset: calls.append(day), hour=23, minute=0
        )

    assert calls == list(range(10))  # exactly one fire per day, in order


def test_daily_job_fires_correctly_at_local_2300_on_both_sides_of_a_dst_transition(tmp_path: Path) -> None:
    """2026-03-08 is a real US spring-forward DST transition (clocks jump
    02:00 EST -> 03:00 EDT). A naive fixed-UTC-offset trigger comparison
    would drift by an hour across it; NY_TZ is a real zoneinfo timezone, so
    local 23:00 must resolve correctly on both the day before and the day
    of the transition."""
    db_path = tmp_path / "job_runs.db"
    run_log = JobRunLog(db_path=db_path)
    calls = []

    before_transition = datetime(2026, 3, 7, 23, 30, tzinfo=NY_TZ)  # still EST (UTC-5)
    RecoverableScheduler(run_log=run_log, now_fn=lambda: before_transition).schedule_daily(
        "eod_batch_generation", lambda: calls.append("2026-03-07"), hour=23, minute=0
    )
    # Waits on run_log.has_run() specifically, not just calls -- that's the
    # actual condition the next registration's own catch-up check depends
    # on (see test_daily_eod_job_fires_exactly_once... for why calls alone
    # leaves a real race window).
    assert _wait_until(lambda: run_log.has_run("eod_batch_generation", "2026-03-07"))
    assert calls == ["2026-03-07"]
    assert before_transition.utcoffset() == timedelta(hours=-5)

    on_transition_day = datetime(2026, 3, 8, 23, 30, tzinfo=NY_TZ)  # now EDT (UTC-4)
    RecoverableScheduler(run_log=run_log, now_fn=lambda: on_transition_day).schedule_daily(
        "eod_batch_generation", lambda: calls.append("2026-03-08"), hour=23, minute=0
    )
    assert _wait_until(lambda: run_log.has_run("eod_batch_generation", "2026-03-08"))
    assert calls == ["2026-03-07", "2026-03-08"]
    assert on_transition_day.utcoffset() == timedelta(hours=-4)

    # a same-day re-registration on the transition day itself must not double-fire
    RecoverableScheduler(run_log=run_log, now_fn=lambda: on_transition_day).schedule_daily(
        "eod_batch_generation", lambda: calls.append("2026-03-08"), hour=23, minute=0
    )
    assert calls == ["2026-03-07", "2026-03-08"]
