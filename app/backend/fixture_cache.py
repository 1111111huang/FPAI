"""Per-day fixture/result cache shared by main.py's GET /api/fixtures route
and scheduler_wiring.py's fixture-reconciliation job (W249) -- extracted
out of main.py specifically so the reconciliation job can warm/invalidate
it without creating a circular import (scheduler_wiring.py is already
imported BY main.py; this module depends on neither).

TTL is computed per day from the matches that day actually contains,
instead of a blind constant -- see compute_ttl_seconds's own docstring.
Two different concurrency domains touch this module's module-level state:
the HTTP route (always the same uvicorn event loop) and the reconciliation
job (APScheduler's own background thread, via asyncio.run() -- a
*different* event loop). See get_range()/force_refresh_range()'s own
docstrings for how each is made safe."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import threading
import time
from typing import Callable

from starlette.concurrency import run_in_threadpool

from app.backend.date_range_utils import contiguous_date_ranges, date_range
from app.backend.football_data_client import NormalizedMatch

TTL_FLOOR_SECONDS = 60.0
TTL_CEILING_SECONDS = 4 * 3600.0
TTL_DEFAULT_SECONDS = 15 * 60.0
_LIVE_STATUSES = frozenset({"LIVE", "IN_PLAY", "PAUSED"})
_UPCOMING_STATUSES = frozenset({"SCHEDULED", "TIMED"})


def compute_ttl_seconds(matches: list[NormalizedMatch], now: datetime) -> float:
    """How long a single day's cached fixture/result list stays valid,
    computed from the matches it actually contains:

    - Empty day (nothing fetched for it): the plain default -- nothing to
      compute a kickoff-driven expiry from, but not worth the ceiling
      either in case a fixture gets added before the reconciliation job's
      own next cycle notices.
    - Any live/in-progress match (LIVE/IN_PLAY/PAUSED): the floor --
      scores update continuously, this cache layer must not sit on stale
      data for long.
    - A not-yet-kicked-off SCHEDULED/TIMED match expires at its own
      kickoff time (when it may flip to LIVE) -- the EARLIEST such
      kickoff among the day's matches, clamped to [floor, ceiling]. A
      SCHEDULED match whose kickoff has already passed (a stale fetch
      snapshot, mirrors eod_batch.has_kicked_off's own "status reflects
      whenever the fixture was fetched, not right now" reasoning) is
      treated the same as FINISHED, not as "about to kick off".
    - Everything else (only FINISHED matches, or an unparseable utc_date)
      is effectively immutable from this cache's point of view -- the
      ceiling. Postponement/reschedule is NOT covered by this TTL at all;
      that is register_fixture_reconciliation_job's job (a periodic live
      re-check independent of this cache's own expiry), not this
      function's."""
    if not matches:
        return TTL_DEFAULT_SECONDS

    if any(m.status in _LIVE_STATUSES for m in matches):
        return TTL_FLOOR_SECONDS

    upcoming_kickoffs: list[datetime] = []
    for m in matches:
        if m.status not in _UPCOMING_STATUSES:
            continue
        try:
            kickoff = datetime.fromisoformat(m.utc_date.replace("Z", "+00:00"))
        except ValueError:
            continue
        if kickoff > now:
            upcoming_kickoffs.append(kickoff)

    if not upcoming_kickoffs:
        return TTL_CEILING_SECONDS

    seconds_to_earliest = (min(upcoming_kickoffs) - now).total_seconds()
    return max(TTL_FLOOR_SECONDS, min(TTL_CEILING_SECONDS, seconds_to_earliest))


# Call-type naming convention: "{results|fixtures}{_<league suffix>}" -- E0
# has no suffix (the original, pre-multi-league naming, kept so an
# existing cached E0 key shape isn't silently orphaned), every other
# league suffixes its own lowercase code. Shared here (not duplicated as
# inline literals in main.py and scheduler_wiring.py both) so the two
# call sites can never drift onto different strings for the same league.
_CALL_TYPE_SUFFIX_BY_LEAGUE: dict[str, str] = {
    "E0": "", "SWE": "_swe", "SP1": "_sp1", "I1": "_i1", "D1": "_d1", "F1": "_f1",
}


def results_call_type(league: str) -> str:
    return f"results{_CALL_TYPE_SUFFIX_BY_LEAGUE[league]}"


def fixtures_call_type(league: str) -> str:
    return f"fixtures{_CALL_TYPE_SUFFIX_BY_LEAGUE[league]}"


# (call_type, day) -> (monotonic expiry, that day's matches). Guarded by
# _cache_lock -- see this module's own docstring for why: the
# reconciliation job (scheduler_wiring.py) writes here from APScheduler's
# background thread via its own asyncio.run() (a *different* event loop
# than the one serving HTTP requests), so plain dict access here is a real
# cross-thread race, unlike every other module-level cache in this
# codebase (all HTTP-route-only, single event loop, no lock needed).
_cache: dict[tuple[str, str], tuple[float, list[NormalizedMatch]]] = {}
_cache_lock = threading.Lock()

# call_type+span -> in-flight asyncio.Task. Deliberately NOT lock-guarded
# -- unlike _cache above, this dict is only ever touched by get_range()
# (the HTTP-route path), never by force_refresh_range() (the
# reconciliation path), so it only ever sees one event loop's own
# cooperative scheduling, the same single-threaded-safe reasoning
# main.py's original _fixture_cache_pending relied on.
_pending: dict[tuple[str, str, str], "asyncio.Task[None]"] = {}


def now_monotonic() -> float:
    """Split out so tests can monkeypatch it -- mirrors main.py's existing
    _fixture_cache_now()/_current_real_date() patchable-function
    convention. Monotonic (not wall-clock) specifically so TTL bookkeeping
    is immune to a wall-clock jump (DST, NTP correction)."""
    return time.monotonic()


def wall_clock_now() -> datetime:
    """Genuine wall-clock UTC now -- deliberately not sandbox-aware
    (mirrors main.py's _current_real_date()'s own reasoning:
    football-data.org's real-world match statuses move in real time
    regardless of what SANDBOX_DATE is pretending 'today' is). Split out
    so tests can monkeypatch it, same convention as now_monotonic()."""
    return datetime.now(timezone.utc)


def clear() -> None:
    """Test-only reset -- mirrors main.py's existing module-level-cache
    test fixtures (_fixture_cache.clear() etc.)."""
    with _cache_lock:
        _cache.clear()
    _pending.clear()


def store_day(call_type: str, day: str, matches: list[NormalizedMatch]) -> None:
    ttl = compute_ttl_seconds(matches, wall_clock_now())
    with _cache_lock:
        _cache[(call_type, day)] = (now_monotonic() + ttl, matches)


def _read_days(call_type: str, days: list[str]) -> list[NormalizedMatch]:
    matches: list[NormalizedMatch] = []
    now = now_monotonic()
    with _cache_lock:
        for day in days:
            entry = _cache.get((call_type, day))
            if entry is not None and entry[0] > now:
                matches.extend(entry[1])
    return matches


def _missing_days(call_type: str, days: list[str]) -> list[str]:
    now = now_monotonic()
    with _cache_lock:
        return [d for d in days if (call_type, d) not in _cache or _cache[(call_type, d)][0] <= now]


async def _fetch_and_store(
    call_type: str, sub_from: str, sub_to: str,
    fetch: Callable[..., list[NormalizedMatch]], fetch_kwargs: dict,
) -> None:
    matches = await run_in_threadpool(fetch, date_from=sub_from, date_to=sub_to, **fetch_kwargs)
    by_day: dict[str, list[NormalizedMatch]] = {day: [] for day in date_range(sub_from, sub_to)}
    for match in matches:
        by_day.setdefault(match.utc_date[:10], []).append(match)
    for day, day_matches in by_day.items():
        store_day(call_type, day, day_matches)


async def get_range(
    call_type: str,
    fetch: Callable[..., list[NormalizedMatch]],
    date_from: str | None,
    date_to: str | None,
    **fetch_kwargs: str,
) -> list[NormalizedMatch]:
    """Looks up [date_from, date_to] day by day; any day missing/expired is
    (re)fetched -- grouped into the fewest contiguous spans, one upstream
    call per span -- and written back per day via store_day(). Two
    requests with *different* outer ranges that overlap on some days
    share every day they have in common (the W249 fix -- previously each
    outer range was its own, independently-TTL'd cache entry).

    date_from/date_to of None is the pre-existing "unbounded, single call"
    shape -- never cached at all, matching the original pre-cache
    behavior for that case exactly (an unbounded range has no "day" to
    key on).

    In-flight de-dup (via _pending) only coordinates calls on the SAME
    event loop -- see force_refresh_range()'s own docstring for why the
    reconciliation job's calls don't participate in it."""
    if date_from is None or date_to is None:
        return await run_in_threadpool(fetch, date_from=date_from, date_to=date_to, **fetch_kwargs)

    days = date_range(date_from, date_to)
    missing = _missing_days(call_type, days)

    for sub_from, sub_to in contiguous_date_ranges(missing):
        key = (call_type, sub_from, sub_to)
        task = _pending.get(key)
        if task is None:
            task = asyncio.ensure_future(_fetch_and_store(call_type, sub_from, sub_to, fetch, fetch_kwargs))
            _pending[key] = task
        try:
            await task
        finally:
            if _pending.get(key) is task:
                del _pending[key]

    return _read_days(call_type, days)


async def force_refresh_range(
    call_type: str,
    fetch: Callable[..., list[NormalizedMatch]],
    date_from: str,
    date_to: str,
    **fetch_kwargs: str,
) -> list[NormalizedMatch]:
    """Unconditionally live-fetches [date_from, date_to] (ignoring whatever
    is currently cached/its TTL) and overwrites every day's cache entry --
    used by scheduler_wiring.py's fixture-reconciliation job (W249) to
    both catch a postponement/reschedule (a change get_range()'s own TTL
    has no way to know about) and warm the hot window for every page's
    subsequent get_range() call.

    Deliberately bypasses _pending (the asyncio.Task in-flight dedup) --
    that dict is only safe within a single event loop (see this module's
    own docstring on cross-thread safety), and this function is called
    from the reconciliation job's OWN event loop (a different thread via
    asyncio.run()), never concurrently with itself for the same call_type
    (APScheduler runs one instance of a given job_id at a time). _cache
    writes still go through the thread-safe store_day()/_cache_lock.

    Does NOT guarantee "cached data is fresh once this returns": a
    get_range() fetch that started before this call and finishes after it
    will overwrite the just-reconciled day with its own older snapshot.
    Benign (last-write-wins under _cache_lock, never a corrupt entry) and
    self-healing at the next reconciliation cycle -- but don't read this
    function as a barrier."""
    await _fetch_and_store(call_type, date_from, date_to, fetch, fetch_kwargs)
    return _read_days(call_type, date_range(date_from, date_to))
