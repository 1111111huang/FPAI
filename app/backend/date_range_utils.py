"""Calendar-day range helpers shared by FootballDataClient's per-day
ResultsCache reassembly (W237) and fixture_cache.py's per-day
_fixture_cache (W249) -- extracted here (out of football_data_client.py,
where both functions originated) so both call sites use the exact same
date-splitting logic instead of two independent copies that could drift
apart. Pure stdlib, no app dependencies, deliberately -- this sits below
every other backend module in the import graph."""

from __future__ import annotations

from datetime import datetime, timedelta


def date_range(date_from: str, date_to: str) -> list[str]:
    """Every calendar day from date_from to date_to, inclusive."""
    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date()
    days = []
    day = start
    while day <= end:
        days.append(day.isoformat())
        day += timedelta(days=1)
    return days


def contiguous_date_ranges(days: list[str]) -> list[tuple[str, str]]:
    """Groups a (sorted, deduped) list of day strings into the fewest
    contiguous (start, end) spans -- e.g. ["08-01", "08-02", "08-04"] ->
    [("08-01", "08-02"), ("08-04", "08-04")]. Used so a cache-miss day
    range with gaps still costs one upstream call per contiguous run, not
    one per missing day."""
    if not days:
        return []
    ranges: list[tuple[str, str]] = []
    start = prev = days[0]
    for day in days[1:]:
        if datetime.strptime(day, "%Y-%m-%d").date() - datetime.strptime(prev, "%Y-%m-%d").date() == timedelta(days=1):
            prev = day
        else:
            ranges.append((start, prev))
            start = prev = day
    ranges.append((start, prev))
    return ranges
