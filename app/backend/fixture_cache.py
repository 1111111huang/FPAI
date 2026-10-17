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
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Callable

from starlette.concurrency import run_in_threadpool

from app.backend.date_range_utils import contiguous_date_ranges, date_range
from app.backend.fixture_cache_store import FixtureCacheStore
from app.backend.football_data_client import NormalizedMatch
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

TTL_FLOOR_SECONDS = 60.0
TTL_CEILING_SECONDS = 4 * 3600.0
TTL_DEFAULT_SECONDS = 15 * 60.0
# BUG-086: how far back persisted (call_type, day) entries are kept --
# fixtures aren't immutable the way ResultsCache's FINISHED results are
# (a postponement/reschedule, or simply a day far enough in the past that
# nothing will ever query it through this cache again), so this table
# can't just grow forever the way that one safely does. Pruned once per
# boot (load_from_disk(), below) rather than on a dedicated schedule --
# this app redeploys often enough in practice for that to be sufficient
# (direct user decision, 2026-10-17); revisit with a scheduled prune job
# (mirroring register_recommendation_cache_prune_job) only if a single
# deployment is ever expected to stay up for weeks uninterrupted.
RETENTION_DAYS = 14
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


# BUG-086 follow-up (found in final review): constructing eagerly at
# import time meant ANY import of this module -- not just the
# ENABLE_SCHEDULER-gated boot path -- touched the real data/fixture_cache.db
# as a side effect (confirmed live: a standalone CLI tool that explicitly
# documents "never writes into the app's cache" recreated the file just by
# importing scheduler_wiring.py, which imports this module transitively),
# and a corrupted/unreadable db file would crash that import outright
# (sqlite3.DatabaseError from _init_schema()) rather than failing only
# within the gated code path that actually uses it. Lazy construction
# (mirroring main.py's own get_fixtures_client() pattern) confines both
# risks to only the call sites that actually need a store: load_from_disk()
# (ENABLE_SCHEDULER-gated, never test-exercised) and store_day() (only
# reached by a real cache write). Tests still monkeypatch the module-level
# `_store` attribute directly (see test_fixture_cache.py's _clear_cache
# fixture) -- _get_store() only constructs when `_store` is still None, so
# a monkeypatched instance is used as-is, unchanged from before.
_store: FixtureCacheStore | None = None


def _get_store() -> FixtureCacheStore:
    global _store
    if _store is None:
        _store = FixtureCacheStore()
    return _store

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
    test fixtures (_fixture_cache.clear() etc.). Clears only the
    in-memory cache, not the persisted store -- tests that care about the
    store's own contents manage it directly (see test_fixture_cache.py's
    _clear_cache fixture, which points _store at a fresh tmp_path file per
    test instead)."""
    with _cache_lock:
        _cache.clear()
    _pending.clear()


def load_from_disk() -> None:
    """Boot-time hydration (BUG-086): prunes anything older than
    RETENTION_DAYS, then seeds the in-memory cache with everything that
    remains, each entry marked already-expired (now_monotonic() - 1) --
    not a real monotonic-time TTL carried across the restart (a monotonic
    clock means nothing across a process boundary), just enough to make
    _missing_days() attempt a background refresh on first use while
    _read_days() (BUG-084) still serves it immediately in the meantime.
    Called once from main.py's lifespan startup, gated behind
    ENABLE_SCHEDULER same as that function's other boot-time hooks (see
    its own comment there) -- never exercised by the test suite."""
    cutoff = (wall_clock_now().date() - timedelta(days=RETENTION_DAYS)).isoformat()
    store = _get_store()
    store.prune_older_than(cutoff)
    persisted = store.load_all()
    now = now_monotonic()
    with _cache_lock:
        for (call_type, day), matches in persisted.items():
            _cache[(call_type, day)] = (now - 1, matches)


def store_day(call_type: str, day: str, matches: list[NormalizedMatch]) -> None:
    ttl = compute_ttl_seconds(matches, wall_clock_now())
    with _cache_lock:
        _cache[(call_type, day)] = (now_monotonic() + ttl, matches)
    _get_store().store_day(call_type, day, matches)


def _read_days(call_type: str, days: list[str]) -> list[NormalizedMatch]:
    """Serves whatever is cached for each day, regardless of whether its TTL
    has expired -- BUG-084: staleness only governs whether to ATTEMPT a
    refetch (_missing_days, below), not whether to return what's already
    there. A successful refetch already overwrites the entry (via
    store_day()) before this is ever called, so in the common case this
    never actually serves stale data -- it only matters when get_range()'s
    own refetch attempt for an expired day FAILED (e.g. an upstream rate
    limit), where serving the last-known-good list beats the match silently
    vanishing from the response on every reload until the upstream budget
    recovers."""
    matches: list[NormalizedMatch] = []
    with _cache_lock:
        for day in days:
            entry = _cache.get((call_type, day))
            if entry is not None:
                matches.extend(entry[1])
    return matches


def _missing_days(call_type: str, days: list[str]) -> list[str]:
    now = now_monotonic()
    with _cache_lock:
        return [d for d in days if (call_type, d) not in _cache or _cache[(call_type, d)][0] <= now]


def _any_day_cached(call_type: str, days: list[str]) -> bool:
    """True if at least one of `days` already has a cached entry for
    call_type, fresh or expired -- used by get_range() to decide whether a
    failed refetch should degrade to serving that stale data (BUG-084) or
    propagate the failure as before (a genuinely cold cache, nothing to
    fall back on)."""
    with _cache_lock:
        return any((call_type, day) in _cache for day in days)


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
    **fetch_kwargs: str | bool,
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

    # BUG-085 (found live 2026-10-17): a wide range (e.g. Dashboard's own
    # 90-day-forward window) can split into several missing spans -- a
    # near one already warmed by an earlier request, a far one still cold.
    # Previously, a cold span's failure `raise`d immediately and aborted
    # this whole function, discarding the near span's perfectly good,
    # already-cached data along with it (confirmed live: a league with
    # today's matches already cached came back completely empty on a
    # fresh page load, because *tomorrow+*'s span happened to fail). Each
    # span's failure is now handled independently -- a hard failure (no
    # fallback at all for that span, BUG-084's own condition) is recorded
    # but doesn't stop the loop, so every other span still gets its chance
    # to populate the cache and contribute to the final result.
    hard_failure: Exception | None = None
    for sub_from, sub_to in contiguous_date_ranges(missing):
        key = (call_type, sub_from, sub_to)
        task = _pending.get(key)
        if task is None:
            task = asyncio.ensure_future(_fetch_and_store(call_type, sub_from, sub_to, fetch, fetch_kwargs))
            _pending[key] = task
        try:
            await task
        except Exception as exc:
            # BUG-084: don't let a failed refetch (e.g. upstream rate-limit)
            # blank out a day this cache already has stale-but-real data
            # for -- _read_days() below now serves whatever's present
            # regardless of TTL, so logging and continuing here just means
            # that stale data gets returned instead of nothing.
            if _any_day_cached(call_type, date_range(sub_from, sub_to)):
                LOGGER.warning(
                    "fixture_cache: live fetch failed for %s [%s..%s] -- serving stale cached data instead.",
                    call_type, sub_from, sub_to, exc_info=True,
                )
            else:
                hard_failure = exc
                LOGGER.warning(
                    "fixture_cache: live fetch failed for %s [%s..%s] -- no cached fallback for this span.",
                    call_type, sub_from, sub_to, exc_info=True,
                )
        finally:
            if _pending.get(key) is task:
                del _pending[key]

    result = _read_days(call_type, days)
    # Only propagate a hard failure (and only the most recent one -- any
    # earlier hard failure in this same call already logged its own
    # warning above) when the whole requested range ends up with nothing
    # at all to show -- same "genuinely cold, nothing to fall back on"
    # contract BUG-084 preserved for the single-span case, now extended to
    # the multi-span one: a caller with *some* real data (even if a
    # distant, less important span of the range failed) should get that
    # data back, not an exception that discards all of it.
    if hard_failure is not None and not result:
        raise hard_failure
    return result


async def force_refresh_range(
    call_type: str,
    fetch: Callable[..., list[NormalizedMatch]],
    date_from: str,
    date_to: str,
    **fetch_kwargs: str | bool,
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
