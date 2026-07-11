"""Weekly scheduled refresh of all scraped data (US#109).

Standalone and independent of the web app's own scheduler infrastructure
(app_user_stories.md W08, not yet built) -- which process actually owns
running this is deliberately left open by the story; this module just
establishes that the weekly cadence exists and can be wired into whichever
process ends up running it (a cron job, this module's own CLI entry point,
or later folded into W08's scheduler).
"""

from __future__ import annotations

from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

JOB_ID = "weekly_data_refresh"


def run_refresh_job(refresh_fn: Callable[[], None]) -> None:
    """Run one refresh-data pass, logging (not swallowing) any failure.

    Off-season no-op behavior (no new matches available) and measurable
    MAX(date) advancement during an active season are properties of
    refresh_fn itself (run_refresh_data's existing scrape/ingest pipeline),
    not of this wrapper.
    """
    try:
        refresh_fn()
    except Exception:
        LOGGER.exception("Scheduled refresh-data run failed.")
        raise


def _default_refresh_fn(league: str) -> Callable[[], None]:
    def _refresh() -> None:
        from main import run_refresh_data
        from src.utils import DuckDBManager
        from src.utils.config_loader import settings

        run_refresh_data(settings, DuckDBManager(), league=league)

    return _refresh


def build_weekly_refresh_scheduler(
    refresh_fn: Callable[[], None] | None = None,
    day_of_week: str = "sun",
    hour: int = 3,
    minute: int = 0,
    league: str = "E0",
) -> BackgroundScheduler:
    """Build (but do not start) a scheduler with the weekly refresh job registered.

    refresh_fn defaults to the real run_refresh_data pipeline; tests inject a
    fake to avoid real network/scrape calls.
    """
    effective_refresh_fn = refresh_fn if refresh_fn is not None else _default_refresh_fn(league)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: run_refresh_job(effective_refresh_fn),
        trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
        id=JOB_ID,
        replace_existing=True,
    )
    return scheduler
