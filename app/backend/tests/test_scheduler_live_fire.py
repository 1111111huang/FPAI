"""W41: scheduler live-fire smoke test for CronTrigger. W21/W33's own
scheduler tests are thorough about JobRunLog's restart/catch-up logic, but
every one of them only ever exercises the *immediate catch-up* path
(RecoverableScheduler._run_and_mark's own `now >= trigger_today` comparison)
-- never APScheduler's own live CronTrigger actually firing on a real
background thread during continuous operation. These are genuinely
different code paths: catch-up is our own manual datetime comparison, live
firing is APScheduler's internal trigger-scheduling machinery -- a bug
specific to how CronTrigger interprets hour/minute/timezone would not be
caught by anything else in this suite.

Registers a job with a trigger time in the past couple of seconds
relative to registration (so the catch-up check at registration time sees
a trigger that hasn't happened *yet*, and cannot be what fires it), then
does a genuine, short wall-clock wait for APScheduler's own background
thread to fire it live. Marked @pytest.mark.live per conftest.py's opt-out
convention -- this is the one test in the suite that needs real wall-clock
time, not network, but the same "excluded from the default fast run"
opt-out mechanism applies.

RecoverableScheduler.schedule_daily()'s own public signature only takes
hour/minute (no seconds), which would need waiting up to a real minute for
a live CronTrigger fire -- correct but needlessly slow for a smoke test.
Registers a second-granularity CronTrigger directly against the wrapped
BackgroundScheduler instead (RecoverableScheduler._scheduler, the same
object schedule_daily()/schedule_once() drive) -- this exercises the exact
same live-firing machinery this story is about, just without waiting on
the public wrapper's coarser catch-up-comparison path at all (irrelevant
here: firing here is not driven by a "trigger time already passed"
comparison, it is APScheduler's own live cron mechanism)."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest
from apscheduler.triggers.cron import CronTrigger

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.mark.live
def test_cron_trigger_fires_live_via_the_real_background_scheduler(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    scheduler = RecoverableScheduler(run_log=run_log)
    calls: list[str] = []

    try:
        scheduler.start()
        scheduler._scheduler.add_job(
            lambda: calls.append("fired"),
            trigger=CronTrigger(second="*/2", timezone=NY_TZ),
            id="w41_live_fire_probe",
        )
        assert _wait_until(lambda: len(calls) >= 1, timeout=5.0), (
            "CronTrigger never fired live within 5s -- the live-fire path itself is broken, "
            "not just uncovered by tests."
        )
    finally:
        scheduler.shutdown()
