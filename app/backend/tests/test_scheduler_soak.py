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

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler


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
    assert calls == ["original"]

    # fixture postponed a day later -- same job_id, new run_at
    new_run_at = datetime(2026, 8, 23, 14, 30, tzinfo=NY_TZ)
    now_at_new = datetime(2026, 8, 23, 14, 35, tzinfo=NY_TZ)
    RecoverableScheduler(run_log=run_log, now_fn=lambda: now_at_new).schedule_once(
        "t30_m1", lambda: calls.append("rescheduled"), run_at=new_run_at
    )

    assert calls == ["original", "rescheduled"]

    # re-registering the *same* new run_at again must still not double-fire
    RecoverableScheduler(run_log=run_log, now_fn=lambda: now_at_new).schedule_once(
        "t30_m1", lambda: calls.append("rescheduled_again"), run_at=new_run_at
    )
    assert calls == ["original", "rescheduled"]
