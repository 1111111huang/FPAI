"""Shared week-bucketing for the /ops dashboard's trend endpoints
(odds-availability-trend, feature-health-trend, performance-trend). Weeks run
Thursday-to-Wednesday (direct user request, 2026-11-03) -- the newest bucket
is always "today"'s own week, necessarily partial until the following
Wednesday; shown as-is, same as any other in-progress aggregation elsewhere in
this codebase, not specially called out."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def thursday_week_start(date_str: str) -> str:
    """ISO date (YYYY-MM-DD) -> the ISO date of the Thursday that starts its
    week bucket. date.weekday() is Mon=0..Sun=6, so Thursday=3; the modulo
    handles every day of the week in one formula, including Thursday itself
    (maps to 0 days back)."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    days_since_thursday = (d.weekday() - 3) % 7
    return (d - timedelta(days=days_since_thursday)).isoformat()


def recent_week_starts(today: date, weeks: int) -> list[str]:
    """Oldest-to-newest list of `weeks` Thursday week_start ISO dates ending
    at `today`'s own (possibly partial) week. `weeks` must be >= 1."""
    current_week_start = datetime.strptime(thursday_week_start(today.isoformat()), "%Y-%m-%d").date()
    return [(current_week_start - timedelta(weeks=n)).isoformat() for n in range(weeks - 1, -1, -1)]
