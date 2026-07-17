"""W39: does the still-disconnected weekly data-refresh scheduler (US#109)
coexist safely with W08's RecoverableScheduler if ever run in the same
process? Each constructs its own independent BackgroundScheduler instance
(data_refresh_scheduler.py's has no explicit timezone, unlike
RecoverableScheduler's NY_TZ-aware one) -- there is no shared jobstore for
a job-id collision to occur in; this test confirms both fire independently,
with no exception and no cross-registration, as a forward-looking check in
case they're ever wired together.

Scrutiny note: an isdisjoint() comparison of {job.id for job in
recoverable_jobs} against {job.id for job in weekly_jobs} was verified
(empirically, against real BackgroundScheduler/jobstore instances -- not
just by inspection) to *also* correctly catch a genuinely shared jobstore:
get_jobs() reflects the live contents of whatever store is registered to
that instance, so a real collision does make the two ID sets overlap, not
just the two literal strings differ. isdisjoint isn't a trap after all --
it's kept below as a cheap sanity check. The cross-instance get_job()
lookups (each job id queried THROUGH THE OTHER scheduler's own get_job())
are a more direct phrasing of the same "genuinely separate stores"
property, not a fix for a gap the isdisjoint check actually had.
Likewise, merely calling `scheduler.get_job(...).func()` and checking the
call log is already covered in isolation by
tests/test_data_refresh_scheduler.py::test_scheduled_job_invokes_the_injected_refresh_fn
-- it says nothing about coexistence. So here, firing one scheduler's job
is followed by asserting the *other* scheduler's call log and job registry
are untouched, which is the actual coexistence property in question.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("apscheduler")

from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler
from src.scheduling.data_refresh_scheduler import build_weekly_refresh_scheduler


def test_both_schedulers_construct_and_start_in_the_same_process_without_interfering(tmp_path: Path) -> None:
    recoverable_calls: list[str] = []
    weekly_calls: list[str] = []

    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    recoverable = RecoverableScheduler(
        run_log=run_log, now_fn=lambda: datetime(2026, 7, 12, 10, 0, tzinfo=NY_TZ),
    )
    recoverable.schedule_daily(
        "eod_batch_generation", lambda: recoverable_calls.append("eod"), hour=23, minute=0
    )

    weekly = build_weekly_refresh_scheduler(refresh_fn=lambda: weekly_calls.append("weekly"))

    recoverable.start()
    weekly.start()
    try:
        assert recoverable._scheduler.running is True
        assert weekly.running is True

        recoverable_job_ids = {job.id for job in recoverable._scheduler.get_jobs()}
        weekly_job_ids = {job.id for job in weekly.get_jobs()}
        assert recoverable_job_ids == {"eod_batch_generation"}
        assert weekly_job_ids == {"weekly_data_refresh"}
        # Also a genuine no-shared-jobstore proof (verified empirically,
        # see module docstring), not just a cheap string-inequality check.
        assert recoverable_job_ids.isdisjoint(weekly_job_ids)

        # A more direct phrasing of the same "genuinely separate stores"
        # property: look each scheduler's job id up THROUGH THE OTHER
        # scheduler's own get_job().
        assert recoverable._scheduler.get_job("weekly_data_refresh") is None
        assert weekly.get_job("eod_batch_generation") is None

        # Too early for the EOD cron trigger -- confirms starting didn't
        # spuriously fire anything via APScheduler's own trigger, and that
        # starting `weekly` didn't somehow also fire `recoverable`'s job.
        assert recoverable_calls == []
        assert weekly_calls == []

        # Firing recoverable's job directly must leave weekly's call log and
        # job registry untouched -- this is the actual coexistence property
        # under test, unlike re-invoking weekly's own job in isolation
        # (already covered elsewhere).
        recoverable._scheduler.get_job("eod_batch_generation").func()
        assert recoverable_calls == ["eod"]
        assert weekly_calls == []
        assert {job.id for job in weekly.get_jobs()} == {"weekly_data_refresh"}

        # And the reverse direction: firing weekly's job must leave
        # recoverable's call log and job registry untouched.
        weekly.get_job("weekly_data_refresh").func()
        assert weekly_calls == ["weekly"]
        assert recoverable_calls == ["eod"]
        assert {job.id for job in recoverable._scheduler.get_jobs()} == {"eod_batch_generation"}
    finally:
        recoverable.shutdown()
        weekly.shutdown()


def test_shutting_down_one_scheduler_does_not_stop_the_others_background_thread(tmp_path: Path) -> None:
    """Each scheduler owns its own BackgroundScheduler instance (and thus its
    own executor/background thread). Confirms that isn't secretly shared --
    if it were, shutting down one would silently stop the other too."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    recoverable = RecoverableScheduler(
        run_log=run_log, now_fn=lambda: datetime(2026, 7, 12, 10, 0, tzinfo=NY_TZ),
    )
    recoverable.schedule_daily("eod_batch_generation", lambda: None, hour=23, minute=0)
    weekly = build_weekly_refresh_scheduler(refresh_fn=lambda: None)

    recoverable.start()
    weekly.start()
    try:
        recoverable.shutdown()
        assert weekly.running is True
    finally:
        weekly.shutdown()
