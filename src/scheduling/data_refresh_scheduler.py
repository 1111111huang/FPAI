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

# US#132: Sweden (Allsvenskan) gets its own job id/cadence, distinct from
# EPL's weekly Sunday job. Live investigation (2026-07-21, football-data.co.uk
# new/SWE.csv) found the 2026 season includes a genuine midweek round (Wed
# 2026-04-22 / Thu 2026-04-23, 4 matches each) landing only 3-4 days after
# the previous weekend round and 3 days before the next -- closer together
# than EPL's rounds, which are reliably weekend-only bar rare one-off
# fixtures. A Sunday-only weekly refresh would leave a midweek round like
# that uningested for a full week, spanning past the *next* round's kickoff
# (the exact "gap spans an entire round" risk this story asks to guard
# against). "tue,fri" keeps worst-case staleness to ~3 days: Tuesday's run
# catches any Fri-Mon weekend round, Friday's run catches any Tue-Thu
# midweek round, before the following round starts.
SWEDEN_JOB_ID = "sweden_data_refresh"
DEFAULT_SWEDEN_DAY_OF_WEEK = "tue,fri"


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


def _default_sweden_refresh_fn() -> Callable[[], None]:
    def _refresh() -> None:
        from main import run_refresh_data
        from src.utils import DuckDBManager
        from src.utils.config_loader import settings

        run_refresh_data(settings, DuckDBManager(), league="SWE")

    return _refresh


def build_sweden_refresh_scheduler(
    refresh_fn: Callable[[], None] | None = None,
    day_of_week: str = DEFAULT_SWEDEN_DAY_OF_WEEK,
    hour: int = 3,
    minute: int = 0,
) -> BackgroundScheduler:
    """Build (but do not start) a scheduler with Sweden's refresh job registered.

    US#132: kept as a separate scheduler/job (not folded into
    build_weekly_refresh_scheduler's single job) because Sweden's justified
    cadence (twice weekly, see DEFAULT_SWEDEN_DAY_OF_WEEK) genuinely differs
    from EPL's weekly one -- see module-level comment by SWEDEN_JOB_ID for
    the live-data evidence. refresh_fn defaults to run_refresh_data(league=
    "SWE"); tests inject a fake to avoid real network/scrape calls.
    """
    effective_refresh_fn = refresh_fn if refresh_fn is not None else _default_sweden_refresh_fn()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: run_refresh_job(effective_refresh_fn),
        trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
        id=SWEDEN_JOB_ID,
        replace_existing=True,
    )
    return scheduler
