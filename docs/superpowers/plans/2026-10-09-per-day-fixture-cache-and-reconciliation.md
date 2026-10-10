# Per-Day Fixture Cache + Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `main.py`'s flat-60s, whole-range-keyed fixture cache with a per-day, event-computed-TTL cache shared across any two overlapping requests, and add a periodic scheduler job that both re-syncs T-30 jobs for postponed/rescheduled fixtures and keeps the ±2-day window warm.

**Architecture:** Two new backend modules (`date_range_utils.py` for pure day-splitting helpers, `fixture_cache.py` for the per-day cache + TTL logic) sit below both `main.py` and `scheduler_wiring.py` so neither has to import the other circularly. `main.py`'s `/api/fixtures` route and a new `register_fixture_reconciliation_job` in `scheduler_wiring.py` both call into `fixture_cache.py`. Cross-thread safety matters here: the reconciliation job runs on APScheduler's own background thread via `asyncio.run()` (a *different* event loop than the one serving HTTP requests), so cache-dict mutations are guarded by a plain `threading.Lock` (mirroring `FootballDataClient._request_lock`'s existing precedent for the exact same cross-thread-shared-state problem) while the HTTP-path's in-flight-request de-dup (`asyncio.Task`-based) stays scoped to the single event loop it already safely worked on.

**Tech Stack:** Python, FastAPI, Starlette's `run_in_threadpool`, APScheduler (via `RecoverableScheduler`), pytest.

---

## Context for the engineer

This plan replaces part of `app/backend/main.py`'s existing fixture-caching mechanism. Read these first:

- `app/backend/main.py` lines ~160-245 (`_fixture_cache`, `_fetch_and_cache_fixtures`, `_cached_fixture_call`) — the code being replaced.
- `app/backend/main.py` lines ~899-1029 (`get_fixtures` route) — the caller, whose body changes only slightly (cache-key tuples become plain strings).
- `app/backend/football_data_client.py` lines ~56-80 (`_date_range`, `_contiguous_date_ranges`) and lines ~265-321 (`_get_results_range_cached`) — the existing per-day-cache pattern this plan generalizes one layer up. `_date_range`/`_contiguous_date_ranges` move out into a new shared module; everything else in this file is unchanged.
- `app/backend/scheduler_wiring.py` lines ~54-100 (job ID/hour constants), ~333-355 (`t30_run_at`/`build_schedule_t30`), ~358-402 (`_fetch_fixtures_for_league`), ~405-470 (`register_eod_job`, the closest existing precedent for the new job's shape) — read the whole file's module docstring too, it explains why scheduler wiring lives outside `main.py`.
- `app/backend/eod_batch.py` lines ~150-164 (`has_kicked_off`) — reused as-is, not reimplemented.
- `app/backend/tests/test_fixtures_endpoint.py` — the existing test suite for the route being changed. Several tests pass unchanged; a few are rewritten (noted per-task below).
- `app/backend/tests/test_scheduler_wiring.py` — read `test_register_eod_job_generates_recommendations_and_schedules_t30` (around line 361) for the exact mocking/timing conventions (`_FUTURE_DAY` anchored to real wall-clock + 2 days, `_wait_until`, `RecoverableScheduler(run_log=..., now_fn=...)`) the new job's tests must follow.

**Why the cross-thread lock matters (don't skip this):** before this plan, `_fixture_cache` was *only* ever touched by HTTP request handlers (always the same uvicorn event loop). The new reconciliation job runs via APScheduler's `BackgroundScheduler` on its own OS thread, calling `asyncio.run()` — a *second*, independent event loop. Two different event loops on two different threads touching the same plain dict concurrently is a real race (Python dicts aren't atomic for compound ops, and `asyncio.Task` objects aren't safely shared across loops). `FootballDataClient._request_lock` already solves this exact class of problem with a `threading.Lock` — `fixture_cache.py` does the same for its own dict.

---

### Task 1: Extract `date_range_utils.py`

**Files:**
- Create: `app/backend/date_range_utils.py`
- Modify: `app/backend/football_data_client.py:56-80` (replace local defs with imports), `:293`, `:309`, `:311` (call sites, unchanged by name)
- Test: `app/backend/tests/test_date_range_utils.py`

- [ ] **Step 1: Write the new module**

```python
# app/backend/date_range_utils.py
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
```

- [ ] **Step 2: Write direct unit tests for the new module**

```python
# app/backend/tests/test_date_range_utils.py
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.date_range_utils import contiguous_date_ranges, date_range


def test_date_range_single_day():
    assert date_range("2026-08-21", "2026-08-21") == ["2026-08-21"]


def test_date_range_multi_day_inclusive():
    assert date_range("2026-08-21", "2026-08-23") == ["2026-08-21", "2026-08-22", "2026-08-23"]


def test_contiguous_date_ranges_empty_list():
    assert contiguous_date_ranges([]) == []


def test_contiguous_date_ranges_single_span():
    assert contiguous_date_ranges(["2026-08-21", "2026-08-22", "2026-08-23"]) == [("2026-08-21", "2026-08-23")]


def test_contiguous_date_ranges_splits_on_a_gap():
    assert contiguous_date_ranges(["2026-08-21", "2026-08-22", "2026-08-24"]) == [
        ("2026-08-21", "2026-08-22"), ("2026-08-24", "2026-08-24"),
    ]
```

- [ ] **Step 3: Run the new tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_date_range_utils.py -v`
Expected: 5 passed (this is pure extraction of already-working logic, not new behavior).

- [ ] **Step 4: Update `football_data_client.py` to import from the new module**

Delete lines 56-80 (the local `_date_range`/`_contiguous_date_ranges` defs) and add an import instead. The three call sites (`_date_range(date_from, date_to)` at line 293, `_contiguous_date_ranges(missing_days)` at line 309, `_date_range(sub_from, sub_to)` at line 311) are unchanged -- only the import changes, via aliasing:

```python
from app.backend.date_range_utils import contiguous_date_ranges as _contiguous_date_ranges, date_range as _date_range
```

Add this import near the top of `football_data_client.py`, alongside the existing `import requests` line.

- [ ] **Step 5: Run the full football_data_client test suite to verify no regression**

Run: `python -m pytest app/backend/tests/test_football_data_client.py -v`
Expected: all 31 tests still pass (pure refactor, no behavior change).

- [ ] **Step 6: Commit**

```bash
git add app/backend/date_range_utils.py app/backend/tests/test_date_range_utils.py app/backend/football_data_client.py
git commit -m "refactor(backend): extract date_range/contiguous_date_ranges into a shared module"
```

---

### Task 2: `fixture_cache.py` — TTL computation

**Files:**
- Create: `app/backend/fixture_cache.py`
- Test: `app/backend/tests/test_fixture_cache.py`

- [ ] **Step 1: Write the failing tests for `compute_ttl_seconds`**

```python
# app/backend/tests/test_fixture_cache.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.fixture_cache import (
    TTL_CEILING_SECONDS, TTL_DEFAULT_SECONDS, TTL_FLOOR_SECONDS, compute_ttl_seconds,
)
from app.backend.football_data_client import NormalizedMatch

_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def _match(status: str, hours_from_now: float) -> NormalizedMatch:
    kickoff = _NOW + timedelta(hours=hours_from_now)
    return NormalizedMatch(
        match_id="m1", utc_date=kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"), status=status,
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )


def test_empty_day_gets_the_plain_default():
    assert compute_ttl_seconds([], _NOW) == TTL_DEFAULT_SECONDS


def test_a_live_match_gets_the_floor():
    assert compute_ttl_seconds([_match("LIVE", -0.5)], _NOW) == TTL_FLOOR_SECONDS


def test_in_play_and_paused_also_get_the_floor():
    assert compute_ttl_seconds([_match("IN_PLAY", -0.2)], _NOW) == TTL_FLOOR_SECONDS
    assert compute_ttl_seconds([_match("PAUSED", -0.2)], _NOW) == TTL_FLOOR_SECONDS


def test_only_finished_matches_get_the_ceiling():
    matches = [_match("FINISHED", -5), _match("FINISHED", -3)]
    assert compute_ttl_seconds(matches, _NOW) == TTL_CEILING_SECONDS


def test_scheduled_match_expires_at_its_own_kickoff():
    # 2 hours out -> 7200s, comfortably inside [floor, ceiling].
    assert compute_ttl_seconds([_match("SCHEDULED", 2)], _NOW) == 7200.0


def test_scheduled_match_kickoff_ttl_is_clamped_to_the_floor():
    # 10 seconds out would compute to 10s -- clamped up to the 60s floor so
    # the cache doesn't thrash on a match seconds from kickoff.
    assert compute_ttl_seconds([_match("SCHEDULED", 10 / 3600)], _NOW) == TTL_FLOOR_SECONDS


def test_scheduled_match_kickoff_ttl_is_clamped_to_the_ceiling():
    # 30 days out would compute to a huge number -- clamped down to the
    # 4h ceiling so the reconciliation job's own cadence is the real backstop.
    assert compute_ttl_seconds([_match("SCHEDULED", 30 * 24)], _NOW) == TTL_CEILING_SECONDS


def test_earliest_upcoming_kickoff_wins_among_several_scheduled_matches():
    matches = [_match("SCHEDULED", 3), _match("SCHEDULED", 1), _match("SCHEDULED", 5)]
    assert compute_ttl_seconds(matches, _NOW) == 3600.0


def test_a_scheduled_match_whose_kickoff_already_passed_is_ignored_like_finished():
    # Found live risk: a SCHEDULED fixture snapshot can be stale by the time
    # this runs (mirrors has_kicked_off's own docstring reasoning) -- a
    # "kickoff" in the past must not count as "upcoming" and drag the TTL
    # down to near-zero.
    matches = [_match("SCHEDULED", -1), _match("FINISHED", -2)]
    assert compute_ttl_seconds(matches, _NOW) == TTL_CEILING_SECONDS


def test_an_unparseable_utc_date_is_ignored_rather_than_raising():
    bad = NormalizedMatch(
        match_id="m1", utc_date="not-a-date", status="SCHEDULED",
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )
    assert compute_ttl_seconds([bad], _NOW) == TTL_CEILING_SECONDS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_fixture_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.fixture_cache'`

- [ ] **Step 3: Write `fixture_cache.py` with just the TTL function**

```python
# app/backend/fixture_cache.py
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

from datetime import datetime, timezone

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_fixture_cache.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add app/backend/fixture_cache.py app/backend/tests/test_fixture_cache.py
git commit -m "feat(backend): add event-computed per-day fixture cache TTL"
```

---

### Task 3: `fixture_cache.py` — storage, locking, `get_range`, `force_refresh_range`

**Files:**
- Modify: `app/backend/fixture_cache.py`
- Test: `app/backend/tests/test_fixture_cache.py`

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/tests/test_fixture_cache.py`:

```python
import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from app.backend.fixture_cache import clear, fixtures_call_type, force_refresh_range, get_range, results_call_type


@pytest.fixture(autouse=True)
def _clear_cache():
    clear()
    yield
    clear()


def _future_match(day: str, hour: int = 15) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=f"m-{day}-{hour}", utc_date=f"{day}T{hour:02d}:00:00Z", status="SCHEDULED",
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )


def test_get_range_fetches_once_on_a_cold_cache():
    fetch = MagicMock(return_value=[_future_match("2026-08-21")])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        result = asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
    assert len(result) == 1
    fetch.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-21")


def test_get_range_second_call_within_ttl_does_not_refetch():
    fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=20)])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
    fetch.assert_called_once()


def test_get_range_shares_overlapping_days_across_two_different_outer_ranges():
    """The original motivating bug: Dashboard's today..+90 and Match
    Explorer's today-30..+90 must share every day they have in common, not
    just the lucky case where their ranges split identically."""
    fetch = MagicMock(return_value=[_future_match("2026-08-22")])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-22", date_to="2026-08-24"))
        fetch.reset_mock()
        fetch.return_value = []
        # A second, differently-shaped range that fully overlaps the first
        # -- every one of its days is already cached, so no new call.
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-20", date_to="2026-08-24"))
    fetch.assert_not_called()


def test_get_range_only_fetches_the_missing_days_as_one_contiguous_call():
    fetch = MagicMock(return_value=[])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-22", date_to="2026-08-22"))
        fetch.reset_mock()
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-20", date_to="2026-08-24"))
    # 08-22 already cached -- the two missing spans are 08-20..08-21 and 08-23..08-24.
    assert fetch.call_count == 2
    fetch.assert_any_call(date_from="2026-08-20", date_to="2026-08-21")
    fetch.assert_any_call(date_from="2026-08-23", date_to="2026-08-24")


def test_get_range_passes_through_extra_kwargs_like_competition_code():
    fetch = MagicMock(return_value=[])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range(
            "fixtures_sp1", fetch, date_from="2026-08-21", date_to="2026-08-21", competition_code="PD",
        ))
    fetch.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-21", competition_code="PD")


def test_get_range_unbounded_range_bypasses_the_cache_entirely():
    fetch = MagicMock(return_value=[])
    asyncio.run(get_range("fixtures", fetch, date_from=None, date_to=None))
    asyncio.run(get_range("fixtures", fetch, date_from=None, date_to=None))
    assert fetch.call_count == 2


def test_get_range_concurrent_identical_calls_dedupe_to_one_fetch():
    entered = threading.Event()
    release = threading.Event()

    def slow_fetch(date_from=None, date_to=None):
        entered.set()
        assert release.wait(timeout=5)
        return []

    results: dict[str, object] = {}

    def run(name: str):
        with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
            results[name] = asyncio.run(get_range("fixtures", slow_fetch, date_from="2026-08-21", date_to="2026-08-21"))

    # Both calls must share the SAME event loop for the asyncio.Task-based
    # dedup to apply -- run them as two tasks on one loop via asyncio.run
    # over a gathering coroutine, not two OS threads (that would be the
    # fixture_cache force_refresh_range / cross-thread case, covered
    # separately below -- this test is specifically the single-event-loop
    # in-flight dedup path).
    async def _both():
        await asyncio.gather(
            get_range("fixtures", slow_fetch, date_from="2026-08-21", date_to="2026-08-21"),
            get_range("fixtures", slow_fetch, date_from="2026-08-21", date_to="2026-08-21"),
        )

    call_count = {"n": 0}

    def counting_fetch(date_from=None, date_to=None):
        call_count["n"] += 1
        entered.set()
        assert release.wait(timeout=5)
        return []

    async def _runner():
        task = asyncio.ensure_future(asyncio.gather(
            get_range("fixtures", counting_fetch, date_from="2026-08-21", date_to="2026-08-21"),
            get_range("fixtures", counting_fetch, date_from="2026-08-21", date_to="2026-08-21"),
        ))
        await asyncio.sleep(0.05)
        release.set()
        await task

    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(_runner())
    assert call_count["n"] == 1


def test_force_refresh_range_always_fetches_even_if_cached():
    fetch = MagicMock(return_value=[])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
        fetch.reset_mock()
        asyncio.run(force_refresh_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
    fetch.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-21")


def test_force_refresh_range_overwrites_the_cached_day_and_warms_it_for_get_range():
    stale_fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=10)])
    fresh_fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=18)])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", stale_fetch, date_from="2026-08-21", date_to="2026-08-21"))
        asyncio.run(force_refresh_range("fixtures", fresh_fetch, date_from="2026-08-21", date_to="2026-08-21"))
        # A plain get_range() afterward must see the force-refreshed data,
        # not re-fetch (the warming half of W249).
        never_called = MagicMock()
        result = asyncio.run(get_range("fixtures", never_called, date_from="2026-08-21", date_to="2026-08-21"))
    never_called.assert_not_called()
    assert result[0].match_id == "m-2026-08-21-18"


def test_results_call_type_and_fixtures_call_type_naming():
    assert results_call_type("E0") == "results"
    assert fixtures_call_type("E0") == "fixtures"
    assert results_call_type("SP1") == "results_sp1"
    assert fixtures_call_type("SWE") == "fixtures_swe"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_fixture_cache.py -v`
Expected: FAIL with `ImportError` (clear/get_range/force_refresh_range/results_call_type/fixtures_call_type don't exist yet)

- [ ] **Step 3: Implement storage, locking, `get_range`, `force_refresh_range`**

Append to `app/backend/fixture_cache.py` (after `compute_ttl_seconds`):

```python
import asyncio
import logging
import threading
import time
from typing import Callable

from starlette.concurrency import run_in_threadpool

from app.backend.date_range_utils import contiguous_date_ranges, date_range

LOGGER = logging.getLogger(__name__)

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
    writes still go through the thread-safe store_day()/_cache_lock."""
    await _fetch_and_store(call_type, date_from, date_to, fetch, fetch_kwargs)
    return _read_days(call_type, date_range(date_from, date_to))
```

Also add `from datetime import datetime, timezone` to the top imports (alongside the existing `from datetime import datetime, timezone` already added in Task 2 -- just make sure both names are present).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_fixture_cache.py -v`
Expected: all tests passed (10 from Task 2 + the new ones from this task)

- [ ] **Step 5: Commit**

```bash
git add app/backend/fixture_cache.py app/backend/tests/test_fixture_cache.py
git commit -m "feat(backend): add per-day fixture cache storage, dedup, and force-refresh"
```

---

### Task 4: Rewire `main.py`'s `/api/fixtures` route onto `fixture_cache.py`

**Files:**
- Modify: `app/backend/main.py:169-245` (delete old cache code), `:943-1024` (call sites), `:1025` (gather error handling)
- Modify: `app/backend/tests/test_fixtures_endpoint.py` (several tests rewritten, see below)

- [ ] **Step 1: Delete the old cache implementation**

In `app/backend/main.py`, delete lines 169-245 entirely: `_FIXTURE_CACHE_TTL_SECONDS`, `_fixture_cache`, `_fixture_cache_pending`, `_fixture_cache_now`, `_fetch_and_cache_fixtures`, `_cached_fixture_call`. Add this import near the top of the file, alongside the other `app.backend` imports:

```python
from app.backend import fixture_cache
```

- [ ] **Step 2: Rewrite the `/api/fixtures` route's call sites**

In the `get_fixtures` route (around what was line 943-1024), replace every `_cached_fixture_call((cache_key_tuple), fetch, **kwargs)` call with `fixture_cache.get_range(call_type_string, fetch, **kwargs)`, using the new `results_call_type`/`fixtures_call_type` helpers instead of hardcoded literals. The full block becomes:

```python
    competitions: list[str] = []
    calls: list[Any] = []
    if results_range is not None:
        past_from, past_to = results_range
        if "E0" in enabled:
            competitions.append("E0")
            calls.append(fixture_cache.get_range(
                fixture_cache.results_call_type("E0"), client.get_results, date_from=past_from, date_to=past_to
            ))
        if "SWE" in enabled:
            competitions.append("SWE")
            calls.append(fixture_cache.get_range(
                fixture_cache.results_call_type("SWE"),
                historical_results_from_raw_matches, date_from=past_from, date_to=past_to,
            ))
        if "SP1" in enabled:
            competitions.append("SP1")
            calls.append(fixture_cache.get_range(
                fixture_cache.results_call_type("SP1"), la_liga_client.get_results,
                competition_code=LA_LIGA_COMPETITION_CODE, date_from=past_from, date_to=past_to,
            ))
        if "I1" in enabled:
            competitions.append("I1")
            calls.append(fixture_cache.get_range(
                fixture_cache.results_call_type("I1"), serie_a_client.get_results,
                competition_code=SERIE_A_COMPETITION_CODE, date_from=past_from, date_to=past_to,
            ))
        if "D1" in enabled:
            competitions.append("D1")
            calls.append(fixture_cache.get_range(
                fixture_cache.results_call_type("D1"), bundesliga_client.get_results,
                competition_code=BUNDESLIGA_COMPETITION_CODE, date_from=past_from, date_to=past_to,
            ))
        if "F1" in enabled:
            competitions.append("F1")
            calls.append(fixture_cache.get_range(
                fixture_cache.results_call_type("F1"), ligue1_client.get_results,
                competition_code=LIGUE_1_COMPETITION_CODE, date_from=past_from, date_to=past_to,
            ))
    if fixtures_range is not None:
        future_from, future_to = fixtures_range
        if "E0" in enabled:
            competitions.append("E0")
            calls.append(fixture_cache.get_range(
                fixture_cache.fixtures_call_type("E0"), client.get_fixtures, date_from=future_from, date_to=future_to
            ))
        if "SWE" in enabled:
            competitions.append("SWE")
            calls.append(fixture_cache.get_range(
                fixture_cache.fixtures_call_type("SWE"), sweden_client.get_fixtures,
                date_from=future_from, date_to=future_to,
            ))
        if "SP1" in enabled:
            competitions.append("SP1")
            calls.append(fixture_cache.get_range(
                fixture_cache.fixtures_call_type("SP1"), la_liga_client.get_fixtures,
                competition_code=LA_LIGA_COMPETITION_CODE, date_from=future_from, date_to=future_to,
            ))
        if "I1" in enabled:
            competitions.append("I1")
            calls.append(fixture_cache.get_range(
                fixture_cache.fixtures_call_type("I1"), serie_a_client.get_fixtures,
                competition_code=SERIE_A_COMPETITION_CODE, date_from=future_from, date_to=future_to,
            ))
        if "D1" in enabled:
            competitions.append("D1")
            calls.append(fixture_cache.get_range(
                fixture_cache.fixtures_call_type("D1"), bundesliga_client.get_fixtures,
                competition_code=BUNDESLIGA_COMPETITION_CODE, date_from=future_from, date_to=future_to,
            ))
        if "F1" in enabled:
            competitions.append("F1")
            calls.append(fixture_cache.get_range(
                fixture_cache.fixtures_call_type("F1"), ligue1_client.get_fixtures,
                competition_code=LIGUE_1_COMPETITION_CODE, date_from=future_from, date_to=future_to,
            ))

    try:
        results = await asyncio.gather(*calls)
    except requests.exceptions.HTTPError as exc:
        LOGGER.warning("Upstream fixture provider call failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                "Fixture data is temporarily unavailable (the upstream provider is rate-limited "
                "or unreachable). Please try again in a minute."
            ),
        ) from exc
    matches: list[NormalizedMatch] = []
    for competition, result in zip(competitions, results):
        matches += _tag(result, competition)
    return matches
```

- [ ] **Step 3: Update `test_fixtures_endpoint.py`'s fixture/import setup**

Replace the import line:

```python
from app.backend.main import _split_fixture_date_range, app
```

(drops `_fixture_cache, _fixture_cache_pending` -- they no longer exist in `main.py`). Add a new import:

```python
from app.backend import fixture_cache
```

Replace the `_clear_fixture_cache` autouse fixture body:

```python
@pytest.fixture(autouse=True)
def _clear_fixture_cache():
    """W52/W249: the TTL cache (and its in-flight-request tracking dict)
    is module-level state in fixture_cache.py -- clear it before every
    test so identical date ranges reused across unrelated test cases in
    this file don't leak cached results (or a stuck pending entry)
    between them."""
    fixture_cache.clear()
    yield
    fixture_cache.clear()
```

- [ ] **Step 4: Run the existing test suite to see what still passes unchanged**

Run: `python -m pytest app/backend/tests/test_fixtures_endpoint.py -v 2>&1 | tail -60`

Expected: most tests pass unchanged (a cold cache's single contiguous missing span produces the exact same single upstream call as before). `test_fixtures_endpoint_cache_expires_after_ttl_and_refetches` fails -- it hardcodes the old flat-60s TTL assumption, rewritten next. The same 8 pre-existing environment-gap failures from before this plan (`test_fixtures_endpoint_tags_each_fixture_with_its_source_competition` and friends -- real-wall-clock-dependent fixture dates that have rotted past their original "future" dates) are expected and unrelated; confirm via `git stash` that they fail identically on the pre-plan code if in doubt.

- [ ] **Step 5: Rewrite `test_fixtures_endpoint_cache_expires_after_ttl_and_refetches`**

Replace it with a version that exercises the new per-day, event-computed TTL instead of the old flat constant:

```python
def test_fixtures_endpoint_cache_expires_at_a_matchs_own_kickoff_and_refetches():
    """The within-TTL dedup test above proves two quick, sequential
    requests only hit the client once. This proves the other half: once a
    cached day's own computed TTL has genuinely elapsed (here, because the
    one SCHEDULED match in it reached its own kickoff time), a third
    request for the same day must hit the client again."""
    kickoff = datetime(2027, 6, 1, 15, 0, tzinfo=timezone.utc)
    fixture = NormalizedMatch(
        match_id="m1", utc_date=kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"), status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.return_value = [fixture]

        fake_wall_clock = [kickoff - timedelta(hours=1)]  # 1h (3600s) before kickoff
        fake_monotonic = [1_000.0]
        with patch("app.backend.fixture_cache.wall_clock_now", side_effect=lambda: fake_wall_clock[0]), \
             patch("app.backend.fixture_cache.now_monotonic", side_effect=lambda: fake_monotonic[0]):
            with TestClient(app) as client:
                first = client.get(
                    "/api/fixtures", params={"date_from": "2027-06-01", "date_to": "2027-06-01"}
                )
                # Still ~55 minutes before kickoff -- must serve the cached entry.
                fake_monotonic[0] += 3300.0
                second = client.get(
                    "/api/fixtures", params={"date_from": "2027-06-01", "date_to": "2027-06-01"}
                )
                # Past kickoff (the computed TTL) -- must re-fetch.
                fake_monotonic[0] += 400.0
                third = client.get(
                    "/api/fixtures", params={"date_from": "2027-06-01", "date_to": "2027-06-01"}
                )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert mock_client.get_fixtures.call_count == 2
    mock_client.get_fixtures.assert_called_with(date_from="2027-06-01", date_to="2027-06-01")
```

Add `timedelta, timezone` to the existing `from datetime import date` import line at the top of the file (`from datetime import date, timedelta, timezone`).

- [ ] **Step 6: Add a new test for the original motivating bug -- cross-page day sharing**

Append:

```python
def test_fixtures_endpoint_shares_overlapping_days_across_two_different_requested_ranges():
    """The actual bug report this plan exists for: Dashboard requesting
    today..+90 and Match Explorer requesting today-30..+90 must share
    every day they both cover -- not just the lucky case where their
    future-side split points happen to coincide. Simulated here as two
    requests with different outer ranges that overlap on 2026-08-22."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 19)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_client = mock_get_client.return_value
            mock_client.get_fixtures.return_value = [fixture]
            with TestClient(app) as client:
                first = client.get(
                    "/api/fixtures", params={"date_from": "2026-08-22", "date_to": "2026-08-24"}
                )
                mock_client.get_fixtures.reset_mock()
                mock_client.get_fixtures.return_value = []
                second = client.get(
                    "/api/fixtures", params={"date_from": "2026-08-20", "date_to": "2026-08-24"}
                )

    assert first.status_code == 200
    assert second.status_code == 200
    # Every day the second request needs (08-22..08-24) was already warmed
    # by the first -- only the genuinely new days (08-20, 08-21) are missing.
    mock_client.get_fixtures.assert_called_once_with(date_from="2026-08-20", date_to="2026-08-21")
    assert len(second.json()) == 1
    assert second.json()[0]["home_team"] == "Arsenal"
```

- [ ] **Step 7: Run the full file to verify**

Run: `python -m pytest app/backend/tests/test_fixtures_endpoint.py -v 2>&1 | tail -80`
Expected: all tests pass except the same 8 pre-existing, unrelated environment-gap failures (verify this exact set matches what `git stash` showed before this plan started).

- [ ] **Step 8: Run the full backend suite for a broader regression check**

Run: `python -m pytest app/backend/tests -q 2>&1 | tail -20`
Expected: same pass/fail counts as the pre-plan baseline (1912 passed / 8 pre-existing failures / 3 skipped, per the baseline recorded earlier this session), plus the new tests from this plan.

- [ ] **Step 9: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_fixtures_endpoint.py
git commit -m "feat(backend): rewire /api/fixtures onto the per-day fixture_cache"
```

---

### Task 5: `register_fixture_reconciliation_job` in `scheduler_wiring.py`

**Files:**
- Modify: `app/backend/scheduler_wiring.py` (add imports, constants, the new function)
- Test: `app/backend/tests/test_scheduler_wiring.py`

- [ ] **Step 1: Write the failing tests**

Add to `app/backend/tests/test_scheduler_wiring.py`'s imports:

```python
from app.backend import fixture_cache
```

and add `RECONCILIATION_HOURS, RECONCILIATION_MINUTE, register_fixture_reconciliation_job` to the existing `from app.backend.scheduler_wiring import (...)` block.

Append these tests:

```python
@pytest.fixture(autouse=True)
def _clear_fixture_cache_between_reconciliation_tests():
    fixture_cache.clear()
    yield
    fixture_cache.clear()


def test_register_fixture_reconciliation_job_registers_one_job_per_hour(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 1, 0, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = []
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(0, 4),
    )

    job_ids = {job.id for job in scheduler._scheduler.get_jobs()}
    assert "fixture_reconciliation_00" in job_ids
    assert "fixture_reconciliation_04" in job_ids


def test_register_fixture_reconciliation_job_warms_the_fixture_cache(tmp_path: Path) -> None:
    """The merged warming half of W249: a reconciliation run's own live
    fetch must leave fixture_cache warm, so a subsequent get_range() call
    for an overlapping day is a cache hit."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 0, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(0,),
    )
    assert _wait_until(lambda: run_log.has_run("fixture_reconciliation_00", _FUTURE_DAY_STR))

    import asyncio as _asyncio
    cached = _asyncio.run(fixture_cache.get_range(
        "fixtures", MagicMock(side_effect=AssertionError("must not refetch -- already warm")),
        date_from=_FUTURE_DAY_STR, date_to=_FUTURE_DAY_STR,
    ))
    assert len(cached) == 1
    assert cached[0].match_id == "m1"


def test_register_fixture_reconciliation_job_resyncs_t30_for_every_not_yet_kicked_off_fixture(tmp_path: Path) -> None:
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(23,),
    )
    assert _wait_until(lambda: run_log.has_run("fixture_reconciliation_23", _FUTURE_DAY_STR))
    assert run_log.has_run("t30_m1", t30_run_at(fixture).isoformat())


def test_register_fixture_reconciliation_job_does_not_resync_an_already_kicked_off_fixture(tmp_path: Path) -> None:
    past_day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    finished_fixture = NormalizedMatch(
        match_id="m2", utc_date=f"{past_day.isoformat()}T15:00:00Z", status="FINISHED",
        home_team="A", away_team="B", home_goals=1, away_goals=0,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 0, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = []
    fixtures_client.get_results.return_value = [finished_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(0,),
    )
    assert _wait_until(lambda: run_log.has_run("fixture_reconciliation_00", _FUTURE_DAY_STR))
    assert not run_log.has_run("t30_m2", t30_run_at(finished_fixture).isoformat())


def test_register_fixture_reconciliation_job_one_leagues_failure_does_not_block_the_others(tmp_path: Path) -> None:
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 0, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    fixtures_client.get_results.return_value = []
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.side_effect = requests.exceptions.ConnectionError("down")
    sweden_fixtures_client.get_results.side_effect = requests.exceptions.ConnectionError("down")
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch(
        "app.backend.scheduler_wiring.list_display_enabled_competition_ids", return_value=["E0", "SWE"]
    ):
        register_fixture_reconciliation_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
            now_fn=lambda: now, sweden_fixtures_client=sweden_fixtures_client, hours=(0,),
        )
        assert _wait_until(lambda: run_log.has_run("fixture_reconciliation_00", _FUTURE_DAY_STR))

    assert run_log.has_run("t30_m1", t30_run_at(e0_fixture).isoformat())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k reconciliation -v`
Expected: FAIL with `ImportError: cannot import name 'register_fixture_reconciliation_job'`

- [ ] **Step 3: Implement the job in `scheduler_wiring.py`**

Add `timezone` to the existing `from datetime import datetime, timedelta` import (becomes `from datetime import datetime, timedelta, timezone`). Add `has_kicked_off` to the existing `from app.backend.eod_batch import COMPETITION_CODE, LEAGUE_CODE, run_eod_batch` import (becomes `from app.backend.eod_batch import COMPETITION_CODE, LEAGUE_CODE, has_kicked_off, run_eod_batch`). Add a new import: `from app.backend import fixture_cache`.

Add these constants near the other job-ID constants (after `DATA_REFRESH_MINUTE = 0`):

```python
# W249: direct user request -- periodic safety net for a postponed/
# rescheduled fixture (neither the nightly EOD job nor a match's own
# one-shot T-30 job would ever notice a kickoff time changing outside
# their own normal cadence), merged with proactively warming the ±N-day
# window most users actually look at. RecoverableScheduler has no
# interval trigger -- registered at a handful of fixed hours instead,
# the same trick register_data_refresh_job's own single-hour registration
# generalizes trivially.
RECONCILIATION_JOB_ID_PREFIX = "fixture_reconciliation"
RECONCILIATION_HOURS: tuple[int, ...] = (0, 4, 8, 12, 16, 20)
RECONCILIATION_MINUTE = 30
RECONCILIATION_WINDOW_DAYS_BACK = 2
RECONCILIATION_WINDOW_DAYS_FORWARD = 2
```

Add the function itself, after `register_data_refresh_job`:

```python
def _reconciliation_clients_for_league(
    league: str,
    fixtures_client: FootballDataClient,
    sweden_fixtures_client: SwedenFixturesClient | None,
    la_liga_fixtures_client: FootballDataClient | None,
    serie_a_fixtures_client: FootballDataClient | None,
    bundesliga_fixtures_client: FootballDataClient | None,
    ligue1_fixtures_client: FootballDataClient | None,
) -> tuple[Callable, Callable, dict] | None:
    """Returns (get_results, get_fixtures, extra_kwargs) for a league, or
    None if that league isn't configured this run -- mirrors
    _fetch_fixtures_for_league's own per-league branching and
    "not configured" contract exactly."""
    if league == "SWE":
        if sweden_fixtures_client is None:
            return None
        return (sweden_fixtures_client.get_results, sweden_fixtures_client.get_fixtures, {})
    if league == "SP1":
        if la_liga_fixtures_client is None:
            return None
        return (
            la_liga_fixtures_client.get_results, la_liga_fixtures_client.get_fixtures,
            {"competition_code": LA_LIGA_COMPETITION_CODE},
        )
    if league == "I1":
        if serie_a_fixtures_client is None:
            return None
        return (
            serie_a_fixtures_client.get_results, serie_a_fixtures_client.get_fixtures,
            {"competition_code": SERIE_A_COMPETITION_CODE},
        )
    if league == "D1":
        if bundesliga_fixtures_client is None:
            return None
        return (
            bundesliga_fixtures_client.get_results, bundesliga_fixtures_client.get_fixtures,
            {"competition_code": BUNDESLIGA_COMPETITION_CODE},
        )
    if league == "F1":
        if ligue1_fixtures_client is None:
            return None
        return (
            ligue1_fixtures_client.get_results, ligue1_fixtures_client.get_fixtures,
            {"competition_code": LIGUE_1_COMPETITION_CODE},
        )
    return (fixtures_client.get_results, fixtures_client.get_fixtures, {"competition_code": COMPETITION_CODE})


def register_fixture_reconciliation_job(
    scheduler: RecoverableScheduler,
    fixtures_client: FootballDataClient,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    now_fn: Callable[[], datetime] = lambda: sandbox_now(NY_TZ),
    sweden_fixtures_client: SwedenFixturesClient | None = None,
    la_liga_fixtures_client: FootballDataClient | None = None,
    serie_a_fixtures_client: FootballDataClient | None = None,
    bundesliga_fixtures_client: FootballDataClient | None = None,
    ligue1_fixtures_client: FootballDataClient | None = None,
    hours: tuple[int, ...] = RECONCILIATION_HOURS,
    minute: int = RECONCILIATION_MINUTE,
    days_back: int = RECONCILIATION_WINDOW_DAYS_BACK,
    days_forward: int = RECONCILIATION_WINDOW_DAYS_FORWARD,
) -> None:
    """Registers the periodic fixture-reconciliation job (W249) at each of
    `hours` (RecoverableScheduler's own fixed-hour trick for "every N
    hours" -- see this module's RECONCILIATION_HOURS comment).

    Each run, per enabled league: live-fetches (bypassing fixture_cache's
    own TTL on purpose -- the whole point is to catch a change that cache
    might still think is "fresh") results for [today-days_back, today] and
    fixtures for [today, today+days_forward], via
    fixture_cache.force_refresh_range -- this both detects a changed
    kickoff time (the correctness fix) and leaves the cache warm for the
    window real users actually look at (the warming fix, merged into one
    job rather than two). Then re-registers the T-30 job
    (build_schedule_t30) for every fixture in the future side that hasn't
    kicked off yet. schedule_once's own (job_id, run_at)-keyed design
    (scheduler.py) makes this safe to call unconditionally every cycle,
    changed kickoff or not -- an unchanged run_at is a harmless no-op
    re-registration; nothing diffs against a previous snapshot."""

    def _reconcile_job() -> None:
        today = now_fn().date()
        past_from = (today - timedelta(days=days_back)).isoformat()
        future_to = (today + timedelta(days=days_forward)).isoformat()
        today_str = today.isoformat()
        enabled = set(list_display_enabled_competition_ids())
        now = sandbox_now(timezone.utc)

        for league in COMPETITIONS:
            if league not in enabled:
                continue
            clients = _reconciliation_clients_for_league(
                league, fixtures_client, sweden_fixtures_client,
                la_liga_fixtures_client, serie_a_fixtures_client,
                bundesliga_fixtures_client, ligue1_fixtures_client,
            )
            if clients is None:
                continue
            fetch_results, fetch_fixtures, extra_kwargs = clients

            async def _refresh() -> list[NormalizedMatch]:
                _, future_matches = await asyncio.gather(
                    fixture_cache.force_refresh_range(
                        fixture_cache.results_call_type(league), fetch_results,
                        date_from=past_from, date_to=today_str, **extra_kwargs,
                    ),
                    fixture_cache.force_refresh_range(
                        fixture_cache.fixtures_call_type(league), fetch_fixtures,
                        date_from=today_str, date_to=future_to, **extra_kwargs,
                    ),
                )
                return future_matches

            try:
                future_matches = asyncio.run(_refresh())
            except Exception:
                LOGGER.warning(
                    "Fixture reconciliation: live refresh failed for league=%s -- skipping, "
                    "cache keeps whatever it had (not invalidated).", league, exc_info=True,
                )
                continue

            schedule_t30 = build_schedule_t30(scheduler, odds_client, cache, config, today_str, league=league)
            for fixture in future_matches:
                if not has_kicked_off(fixture, now):
                    schedule_t30(fixture)

    for hour in hours:
        scheduler.schedule_daily(f"{RECONCILIATION_JOB_ID_PREFIX}_{hour:02d}", _reconcile_job, hour=hour, minute=minute)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k reconciliation -v`
Expected: 5 passed

- [ ] **Step 5: Run the full scheduler_wiring test file to check for regressions**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v 2>&1 | tail -40`
Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(backend): add fixture-reconciliation job (postponement safety net + cache warming)"
```

---

### Task 6: Wire the job into `main.py`'s lifespan

**Files:**
- Modify: `app/backend/main.py:56-58` (import), `:449-486` (lifespan registration)

- [ ] **Step 1: Add the import**

In `main.py`'s existing import block:

```python
from app.backend.scheduler_wiring import (
    build_odds_client, build_oddspapi_client, build_schedule_t30, register_data_refresh_job, register_eod_job,
    register_fixture_reconciliation_job, register_lessons_job,
)
```

- [ ] **Step 2: Register the job in `lifespan()`**

Add right after the existing `register_data_refresh_job(scheduler)` call (around line 475):

```python
        # W249: periodic postponement/reschedule safety net + ±2-day
        # fixture-cache warming -- see scheduler_wiring.py's
        # register_fixture_reconciliation_job docstring.
        register_fixture_reconciliation_job(
            scheduler,
            fixtures_client=get_fixtures_client(),
            odds_client=build_odds_client(),
            cache=recommendations.get_cache(),
            config=config,
            sweden_fixtures_client=get_sweden_fixtures_client(),
            la_liga_fixtures_client=get_la_liga_fixtures_client(),
            serie_a_fixtures_client=get_serie_a_fixtures_client(),
            bundesliga_fixtures_client=get_bundesliga_fixtures_client(),
            ligue1_fixtures_client=get_ligue1_fixtures_client(),
        )
```

- [ ] **Step 3: Run the full backend suite**

Run: `python -m pytest app/backend/tests -q 2>&1 | tail -20`
Expected: same baseline as Task 4 Step 8, plus no new failures (this task only adds a call inside an `ENABLE_SCHEDULER=1`-gated block that no test suite exercises directly -- see the existing comment right above it in `main.py` explaining why that gating exists).

- [ ] **Step 4: Commit**

```bash
git add app/backend/main.py
git commit -m "feat(backend): wire the fixture-reconciliation job into app startup"
```

---

### Task 7: Documentation (per this repo's CLAUDE.md)

**Files:**
- Modify: `documents/app_techspec.md`
- Modify: `documents/app_user_stories.md`

- [ ] **Step 1: Add a techspec entry**

Append a new dated entry to `documents/app_techspec.md`, in the same section/style as the existing W237/W247 entries (find them via `grep -n "W237\|W247" documents/app_techspec.md` to locate the right section), covering: the per-day cache restructuring, why `date_range_utils.py`/`fixture_cache.py` were extracted (circular-import avoidance), the cross-thread locking rationale, and the reconciliation job's merged warm+resync design -- mirroring the detail level of this session's earlier W248 entry (`FOOTBALL_DATA_API_KEY_2`/`_3` fallback).

- [ ] **Step 2: Append a new user story**

Add a new row (next sequential ID after W248) to `documents/app_user_stories.md`'s most recent phase table, `completed (2026-10-09)`, summarizing: the original bug report (cross-page fixture-request blocking), the design review conversation that led here (per-range vs per-day caching), and the three pieces implemented (per-day cache, event-computed TTL, merged reconciliation+warming job) with file references.

- [ ] **Step 3: Commit**

```bash
git add documents/app_techspec.md documents/app_user_stories.md
git commit -m "docs: record the per-day fixture cache and reconciliation job (W249)"
```

---

## Self-Review

**Spec coverage:**
- "Per-day cache, shared across overlapping ranges" -> Task 3/4 (`get_range`, Task 4 Step 6's cross-range test).
- "Event-computed TTL instead of a flat constant" -> Task 2 (`compute_ttl_seconds`).
- "Periodic poll for postponed/rescheduled matches, ~4h" -> Task 5 (`register_fixture_reconciliation_job`, `RECONCILIATION_HOURS`).
- "Merge reconciliation with proactive warming" -> Task 5 (`force_refresh_range` both re-syncs T-30 and leaves the cache warm; Task 5 Step 1's warming test).
- "Avoid circular imports between scheduler_wiring.py and main.py" -> Task 2/3 (`fixture_cache.py` depends on neither).
- "Cross-thread safety for the new scheduler-thread cache writes" -> Task 3 (`_cache_lock`), called out explicitly in this plan's Context section.

**Placeholder scan:** every step has complete, runnable code; no "TBD"/"add error handling"/"similar to Task N" shortcuts.

**Type consistency:** `fixture_cache.get_range`/`force_refresh_range` signatures match their usage in both `main.py` (Task 4) and `scheduler_wiring.py` (Task 5); `results_call_type`/`fixtures_call_type` (Task 3) are the only place league-suffix strings are defined, used identically in both call sites.
