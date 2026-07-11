"""Regression tests for US#109: weekly scheduled refresh-data job. Covers only
the scheduling wrapper itself -- run_refresh_data's own idempotency (off-season
no-op, measurable MAX(date) advance during active season) is existing,
already-tested pipeline behavior, not re-tested here."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("apscheduler")

from apscheduler.triggers.cron import CronTrigger

from src.scheduling.data_refresh_scheduler import build_weekly_refresh_scheduler, run_refresh_job


def test_build_weekly_refresh_scheduler_registers_one_weekly_job() -> None:
    scheduler = build_weekly_refresh_scheduler(refresh_fn=lambda: None)
    jobs = scheduler.get_jobs()

    assert len(jobs) == 1
    assert jobs[0].id == "weekly_data_refresh"
    assert isinstance(jobs[0].trigger, CronTrigger)


def test_build_weekly_refresh_scheduler_uses_given_schedule_params() -> None:
    scheduler = build_weekly_refresh_scheduler(refresh_fn=lambda: None, day_of_week="mon", hour=4, minute=30)
    trigger = scheduler.get_job("weekly_data_refresh").trigger

    fields = {f.name: str(f) for f in trigger.fields}
    assert fields["day_of_week"] == "mon"
    assert fields["hour"] == "4"
    assert fields["minute"] == "30"


def test_scheduled_job_invokes_the_injected_refresh_fn() -> None:
    """The job registered in the scheduler must actually call the wired-up
    refresh function when it fires -- not just look right on paper."""
    calls: list[str] = []
    scheduler = build_weekly_refresh_scheduler(refresh_fn=lambda: calls.append("ran"))

    scheduler.get_job("weekly_data_refresh").func()

    assert calls == ["ran"]


def test_run_refresh_job_calls_the_refresh_function() -> None:
    calls: list[str] = []
    run_refresh_job(lambda: calls.append("ran"))
    assert calls == ["ran"]


def test_run_refresh_job_logs_and_reraises_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Acceptance: failures are logged somewhere visible, not silently swallowed."""
    def _boom() -> None:
        raise RuntimeError("scrape source unreachable")

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="scrape source unreachable"):
            run_refresh_job(_boom)

    assert any("refresh-data" in record.message.lower() for record in caplog.records)
