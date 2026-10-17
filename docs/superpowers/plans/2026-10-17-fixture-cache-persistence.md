# Fixture Cache Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fixture_cache.py`'s per-day cache survive a process restart, so a redeploy no longer forces every league back into a full cold-start re-fetch race against football-data.org's shared rate limit.

**Architecture:** A new SQLite-backed store (`FixtureCacheStore`, stdlib `sqlite3`, `data/fixture_cache.db`) mirrors `results_cache.py`'s already-proven `ResultsCache` pattern exactly. `fixture_cache.py` writes through to it on every cache update and, once per boot (gated behind `ENABLE_SCHEDULER`, same as this app's other boot-time hooks), prunes anything older than a 2-week retention window and loads everything else back into the in-memory cache, each entry marked already-expired so the existing BUG-084/085 fallback logic takes over immediately with zero new runtime branching.

**Tech Stack:** Python, stdlib `sqlite3` (no new dependency), pytest.

**Design doc:** `docs/superpowers/specs/2026-10-17-fixture-cache-persistence-design.md`

---

## Context for the implementing engineer

- `app/backend/fixture_cache.py` is a per-day TTL cache (module-level dict `_cache: dict[tuple[str, str], tuple[float, list[NormalizedMatch]]]`, keyed by `(call_type, day)`) shared by `main.py`'s `GET /api/fixtures` and the fixture-reconciliation scheduled job. It was touched twice earlier today (BUG-084: `_read_days()` now serves stale data instead of filtering by TTL; BUG-085: `get_range()`'s per-span failure handling). This plan builds on top of both unchanged.
- `app/backend/results_cache.py`'s `ResultsCache` is the proven template to mirror: stdlib `sqlite3`, one dedicated file under `data/`, a fresh connection per call (safe across `fixture_cache.py`'s two documented concurrency domains -- the HTTP event loop and the reconciliation job's own background thread via `asyncio.run()`).
- Every other small backend store in this codebase follows the identical pattern: its own file, `DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "<name>.db"`, a `db_path` constructor param tests override with `tmp_path`. Follow it exactly -- don't invent a different convention.
- `app/backend/tests/test_fixture_cache.py` already has an autouse `_clear_cache` fixture resetting `fixture_cache`'s in-memory state before/after every test. No test in this codebase currently sets `ENABLE_SCHEDULER=1` for a real `TestClient(app)` run (confirmed via grep), so gating the new boot-time load behind that flag means it is never exercised by the test suite at all -- the only test-isolation work needed is for `fixture_cache.py`'s own direct unit tests, not the wiring into `main.py`.
- `main.py` already imports `fixture_cache` (`from app.backend import bets, eod_batch, fixture_cache, recommendations, sandbox_clock`) -- no new import needed there.

---

### Task 1: `FixtureCacheStore`

**Files:**
- Create: `app/backend/fixture_cache_store.py`
- Test: `app/backend/tests/test_fixture_cache_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# app/backend/tests/test_fixture_cache_store.py
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.fixture_cache_store import FixtureCacheStore
from app.backend.football_data_client import NormalizedMatch


def _match(match_id: str, utc_date: str, status: str = "SCHEDULED") -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date=utc_date, status=status,
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )


def test_store_then_load_round_trips_matches(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [_match("m1", "2026-08-21T15:00:00Z")])

    loaded = store.load_all()

    assert loaded[("fixtures", "2026-08-21")][0].match_id == "m1"


def test_an_empty_day_round_trips_as_an_empty_list_not_a_missing_key(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [])

    loaded = store.load_all()

    assert ("fixtures", "2026-08-21") in loaded
    assert loaded[("fixtures", "2026-08-21")] == []


def test_a_day_never_stored_has_no_key_at_all(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [])

    loaded = store.load_all()

    assert ("fixtures", "2026-08-22") not in loaded


def test_storing_the_same_day_twice_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [_match("m1", "2026-08-21T15:00:00Z")])
    store.store_day("fixtures", "2026-08-21", [_match("m2", "2026-08-21T18:00:00Z")])

    loaded = store.load_all()

    match_ids = {m.match_id for m in loaded[("fixtures", "2026-08-21")]}
    assert match_ids == {"m2"}


def test_prune_older_than_removes_old_entries_and_keeps_newer_ones(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-10", [_match("old", "2026-08-10T15:00:00Z")])
    store.store_day("fixtures", "2026-08-25", [_match("new", "2026-08-25T15:00:00Z")])

    store.prune_older_than("2026-08-20")
    loaded = store.load_all()

    assert ("fixtures", "2026-08-10") not in loaded
    assert ("fixtures", "2026-08-25") in loaded


def test_different_call_types_on_the_same_day_are_independent(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [_match("m1", "2026-08-21T15:00:00Z")])
    store.store_day("results_sp1", "2026-08-21", [_match("m2", "2026-08-21T15:00:00Z")])

    loaded = store.load_all()

    assert {m.match_id for m in loaded[("fixtures", "2026-08-21")]} == {"m1"}
    assert {m.match_id for m in loaded[("results_sp1", "2026-08-21")]} == {"m2"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest app/backend/tests/test_fixture_cache_store.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'app.backend.fixture_cache_store'`

- [ ] **Step 3: Write `FixtureCacheStore`**

```python
# app/backend/fixture_cache_store.py
"""BUG-086: persistent backing store for fixture_cache.py's per-day cache
-- fixture_cache._cache is a plain in-memory dict, wiped on every process
restart (every Railway redeploy), forcing a full re-fetch storm against
football-data.org's shared rate-limited budget on every single deploy.
Mirrors results_cache.py's ResultsCache exactly -- same dedicated SQLite
file under data/, same stdlib sqlite3 usage (no new dependency), same
fresh-connection-per-call pattern (safe across the two concurrency domains
fixture_cache.py's own module docstring already documents: the HTTP event
loop and the reconciliation job's own thread).

Unlike ResultsCache, which caches only immutable FINISHED results forever,
this store covers fixture_cache's full mix (including SCHEDULED/TIMED/LIVE
days that genuinely do change) -- so entries aren't kept forever:
prune_older_than() deletes anything whose day has passed the given cutoff,
called once at boot (fixture_cache.load_from_disk()) before the persisted
data is loaded into the in-memory cache, so the table never accumulates
more than a bounded retention window's worth of history regardless of how
long the process stays up between redeploys."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from app.backend.football_data_client import NormalizedMatch

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fixture_cache.db"


class FixtureCacheStore:
    """SQLite-backed persistence for fixture_cache.py's (call_type, day) ->
    matches entries. See module docstring for why/how."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_days (
                    call_type TEXT NOT NULL,
                    day TEXT NOT NULL,
                    PRIMARY KEY (call_type, day)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_matches (
                    call_type TEXT NOT NULL,
                    day TEXT NOT NULL,
                    match_id TEXT NOT NULL,
                    utc_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    PRIMARY KEY (call_type, day, match_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cached_matches_lookup ON cached_matches (call_type, day)"
            )

    def store_day(self, call_type: str, day: str, matches: list[NormalizedMatch]) -> None:
        """Persists one (call_type, day)'s matches (possibly empty),
        replacing anything previously stored for that key."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cached_days (call_type, day) VALUES (?, ?)",
                (call_type, day),
            )
            conn.execute(
                "DELETE FROM cached_matches WHERE call_type = ? AND day = ?",
                (call_type, day),
            )
            for match in matches:
                conn.execute(
                    """
                    INSERT INTO cached_matches
                    (call_type, day, match_id, utc_date, status, home_team, away_team, home_goals, away_goals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (call_type, day, match.match_id, match.utc_date, match.status,
                     match.home_team, match.away_team, match.home_goals, match.away_goals),
                )

    def prune_older_than(self, cutoff_day: str) -> None:
        """Deletes every cached (call_type, day) entry strictly before
        cutoff_day (an ISO date string, both tables)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cached_matches WHERE day < ?", (cutoff_day,))
            conn.execute("DELETE FROM cached_days WHERE day < ?", (cutoff_day,))

    def load_all(self) -> dict[tuple[str, str], list[NormalizedMatch]]:
        """Everything currently persisted, keyed by (call_type, day) -- a
        key with zero matches still appears (as an empty list) if it has a
        cached_days row, distinguishing a genuinely empty day from one
        never stored at all."""
        with self._connect() as conn:
            days = conn.execute("SELECT call_type, day FROM cached_days").fetchall()
            rows = conn.execute(
                "SELECT call_type, day, match_id, utc_date, status, home_team, away_team, home_goals, away_goals "
                "FROM cached_matches"
            ).fetchall()
        result: dict[tuple[str, str], list[NormalizedMatch]] = {
            (call_type, day): [] for call_type, day in days
        }
        for call_type, day, match_id, utc_date, status, home_team, away_team, home_goals, away_goals in rows:
            result.setdefault((call_type, day), []).append(
                NormalizedMatch(
                    match_id=match_id, utc_date=utc_date, status=status,
                    home_team=home_team, away_team=away_team,
                    home_goals=home_goals, away_goals=away_goals,
                )
            )
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest app/backend/tests/test_fixture_cache_store.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add app/backend/fixture_cache_store.py app/backend/tests/test_fixture_cache_store.py
git commit -m "feat(backend): add FixtureCacheStore, a SQLite-backed persistence layer

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Wire `FixtureCacheStore` into `fixture_cache.py`

**Files:**
- Modify: `app/backend/fixture_cache.py`
- Modify: `app/backend/tests/test_fixture_cache.py`

- [ ] **Step 1: Extend the test file's isolation fixture and imports first (TDD groundwork)**

Find (near the top of `app/backend/tests/test_fixture_cache.py`):

```python
from app.backend import fixture_cache
from app.backend.fixture_cache import (
    TTL_CEILING_SECONDS, TTL_DEFAULT_SECONDS, TTL_FLOOR_SECONDS, clear, compute_ttl_seconds,
    fixtures_call_type, force_refresh_range, get_range, results_call_type,
)
from app.backend.football_data_client import NormalizedMatch

_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear()
    yield
    clear()
```

Change to:

```python
from app.backend import fixture_cache
from app.backend.fixture_cache import (
    TTL_CEILING_SECONDS, TTL_DEFAULT_SECONDS, TTL_FLOOR_SECONDS, clear, compute_ttl_seconds,
    fixtures_call_type, force_refresh_range, get_range, results_call_type,
)
from app.backend.fixture_cache_store import FixtureCacheStore
from app.backend.football_data_client import NormalizedMatch

_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # BUG-086: fixture_cache now write-through persists to a module-level
    # _store -- point it at a throwaway tmp_path file for every test in
    # this file, fully isolating these tests from the real
    # data/fixture_cache.db (same tmp_path-scoped-db convention already
    # used elsewhere in this codebase, e.g. JobRunLog/RecommendationCache
    # in other test files).
    monkeypatch.setattr(fixture_cache, "_store", FixtureCacheStore(db_path=tmp_path / "fixture_cache_test.db"))
    clear()
    yield
    clear()
```

Also change the top-of-file datetime import (confirmed as of this plan's writing: `date` is not yet imported there):

Find (line 4):

```python
from datetime import datetime, timedelta, timezone
```

Change to:

```python
from datetime import date, datetime, timedelta, timezone
```

- [ ] **Step 2: Write the new failing tests**

Add these to `app/backend/tests/test_fixture_cache.py`, anywhere after the `_clear_cache` fixture (e.g. right before `test_results_call_type_and_fixtures_call_type_naming` at the end of the file):

```python
def test_store_day_persists_to_the_store():
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        fixture_cache.store_day("fixtures", "2026-08-21", [_future_match("2026-08-21")])

    persisted = fixture_cache._store.load_all()
    assert persisted[("fixtures", "2026-08-21")][0].match_id == "m-2026-08-21-15"


def test_load_from_disk_seeds_the_in_memory_cache_already_expired():
    fixture_cache._store.store_day("fixtures", "2026-08-21", [_future_match("2026-08-21")])

    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        fixture_cache.load_from_disk()

    # Already expired -- _missing_days() wants to refetch it on next use...
    assert fixture_cache._missing_days("fixtures", ["2026-08-21"]) == ["2026-08-21"]
    # ...but _read_days() (BUG-084) still serves it immediately regardless.
    assert fixture_cache._read_days("fixtures", ["2026-08-21"])[0].match_id == "m-2026-08-21-15"


def test_load_from_disk_prunes_entries_older_than_retention_before_loading():
    old_day = (date.today() - timedelta(days=fixture_cache.RETENTION_DAYS + 1)).isoformat()
    fixture_cache._store.store_day("fixtures", old_day, [_future_match(old_day)])

    fixture_cache.load_from_disk()

    assert fixture_cache._read_days("fixtures", [old_day]) == []
    assert fixture_cache._store.load_all() == {}


def test_a_real_get_range_call_also_persists_to_the_store():
    """End-to-end: the existing get_range() path (not a direct store_day()
    call) must also reach the persisted store -- proves the write-through
    is wired into the actual fetch flow, not just the low-level function."""
    fetch = MagicMock(return_value=[_future_match("2026-08-21")])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))

    persisted = fixture_cache._store.load_all()
    assert persisted[("fixtures", "2026-08-21")][0].match_id == "m-2026-08-21-15"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest app/backend/tests/test_fixture_cache.py -v -k "store_day_persists or load_from_disk or also_persists"`
Expected: FAIL -- `AttributeError: module 'app.backend.fixture_cache' has no attribute '_store'` (and similarly for `load_from_disk`, `RETENTION_DAYS`)

- [ ] **Step 4: Wire the store into `fixture_cache.py`**

Find (current imports, lines 15-29):

```python
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import threading
import time
from typing import Callable

from starlette.concurrency import run_in_threadpool

from app.backend.date_range_utils import contiguous_date_ranges, date_range
from app.backend.football_data_client import NormalizedMatch
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

TTL_FLOOR_SECONDS = 60.0
TTL_CEILING_SECONDS = 4 * 3600.0
TTL_DEFAULT_SECONDS = 15 * 60.0
```

Change to:

```python
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
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
```

- [ ] **Step 5: Add the `_store` instance and `load_from_disk()`**

Find (the module-level cache state, currently):

```python
# (call_type, day) -> (monotonic expiry, that day's matches). Guarded by
# _cache_lock -- see this module's own docstring for why: the
# reconciliation job (scheduler_wiring.py) writes here from APScheduler's
# background thread via its own asyncio.run() (a *different* event loop
# than the one serving HTTP requests), so plain dict access here is a real
# cross-thread race, unlike every other module-level cache in this
# codebase (all HTTP-route-only, single event loop, no lock needed).
_cache: dict[tuple[str, str], tuple[float, list[NormalizedMatch]]] = {}
_cache_lock = threading.Lock()
```

Change to:

```python
# BUG-086: eager construction is cheap (just CREATE TABLE IF NOT EXISTS,
# no meaningful I/O) -- the expensive/risky parts (load_all(), the
# per-write store_day() calls) are what matter for test isolation, and
# those are handled at their own call sites (see load_from_disk()'s own
# ENABLE_SCHEDULER gate in main.py, and test_fixture_cache.py's
# _clear_cache fixture monkeypatching this attribute per test).
_store = FixtureCacheStore()

# (call_type, day) -> (monotonic expiry, that day's matches). Guarded by
# _cache_lock -- see this module's own docstring for why: the
# reconciliation job (scheduler_wiring.py) writes here from APScheduler's
# background thread via its own asyncio.run() (a *different* event loop
# than the one serving HTTP requests), so plain dict access here is a real
# cross-thread race, unlike every other module-level cache in this
# codebase (all HTTP-route-only, single event loop, no lock needed).
_cache: dict[tuple[str, str], tuple[float, list[NormalizedMatch]]] = {}
_cache_lock = threading.Lock()
```

Find (the `clear()` function):

```python
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
```

Change to:

```python
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
    cutoff = (date.today() - timedelta(days=RETENTION_DAYS)).isoformat()
    _store.prune_older_than(cutoff)
    persisted = _store.load_all()
    now = now_monotonic()
    with _cache_lock:
        for (call_type, day), matches in persisted.items():
            _cache[(call_type, day)] = (now - 1, matches)


def store_day(call_type: str, day: str, matches: list[NormalizedMatch]) -> None:
    ttl = compute_ttl_seconds(matches, wall_clock_now())
    with _cache_lock:
        _cache[(call_type, day)] = (now_monotonic() + ttl, matches)
    _store.store_day(call_type, day, matches)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest app/backend/tests/test_fixture_cache.py -v`
Expected: PASS (every test in the file, including the 4 new ones)

- [ ] **Step 7: Run the broader backend suite**

Run: `source venv/bin/activate && python -m pytest app/backend/tests/ -q`
Expected: same result as the pre-existing baseline (8 known pre-existing, unrelated date-rot failures -- `test_agent_performance_dashboard_endpoint.py` x2, `test_fixtures_endpoint.py` x4, `test_recommendation_outcomes_endpoints.py` x1 -- no NEW failures beyond those).

- [ ] **Step 8: Commit**

```bash
git add app/backend/fixture_cache.py app/backend/tests/test_fixture_cache.py
git commit -m "feat(backend): persist fixture_cache's per-day cache via FixtureCacheStore

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Hydrate at boot

**Files:**
- Modify: `app/backend/main.py`

- [ ] **Step 1: Add the boot-time call**

Find (in `main.py`'s `lifespan` function, inside the `if os.environ.get("ENABLE_SCHEDULER", ...)` block -- search for this exact comment+code, since exact line numbers may have shifted slightly from other work on this branch):

```python
    scheduler: RecoverableScheduler | None = None
    if os.environ.get("ENABLE_SCHEDULER", "").lower() in ("1", "true", "yes"):
        # W29: sandbox mode routes JobRunLog to a scratch path so it never touches real dev data.
        run_log = JobRunLog(db_path=_SANDBOX_JOB_RUNS_DB_PATH) if sandbox_clock.is_sandbox_mode() else None
```

Change to:

```python
    scheduler: RecoverableScheduler | None = None
    if os.environ.get("ENABLE_SCHEDULER", "").lower() in ("1", "true", "yes"):
        # BUG-086: hydrate fixture_cache's in-memory cache from its
        # on-disk store before anything else in this block -- every
        # redeploy otherwise starts the cache fully cold, forcing a full
        # re-fetch storm against football-data.org's shared rate-limited
        # budget (confirmed live, 2026-10-17: several redeploys in a row
        # each caused the same cold-start cascade, competitions missing
        # for minutes at a time). Gated behind this same ENABLE_SCHEDULER
        # check as every other boot-time hook in this block, for the
        # identical reason the pregenerate-recommendations hook below
        # already documents: unconditional here would fire on every test
        # file's `with TestClient(app)` too.
        fixture_cache.load_from_disk()
        # W29: sandbox mode routes JobRunLog to a scratch path so it never touches real dev data.
        run_log = JobRunLog(db_path=_SANDBOX_JOB_RUNS_DB_PATH) if sandbox_clock.is_sandbox_mode() else None
```

`fixture_cache` is already imported at the top of `main.py` (`from app.backend import bets, eod_batch, fixture_cache, recommendations, sandbox_clock`) -- no new import needed.

- [ ] **Step 2: Typecheck/syntax-check**

Run: `source venv/bin/activate && python -c "import app.backend.main"`
Expected: no error (clean import)

- [ ] **Step 3: Run the full backend suite one more time**

Run: `source venv/bin/activate && python -m pytest app/backend/tests/ -q`
Expected: same baseline as Task 2's Step 7 -- confirms this change (which no test exercises directly, per the Context section above) didn't break anything else.

- [ ] **Step 4: Commit**

```bash
git add app/backend/main.py
git commit -m "feat(backend): hydrate fixture_cache from disk at boot

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-review notes (for whoever executes this plan)

- **Spec coverage:** Design doc Section 3 (storage) -> Task 1. Section 4 (wiring into fixture_cache.py) -> Task 2. Section 5 (calling load_from_disk at boot) -> Task 3. Section 6 (testing) -> the test steps across all three tasks.
- **Why `_store` is a plain module-level instance, not passed as a parameter everywhere:** matches `fixture_cache.py`'s own existing style -- `_cache`/`_cache_lock`/`_pending` are already module-level state, not threaded through every function call. Consistent, not a new pattern.
- **Why no new dependency:** `results_cache.py` already proves stdlib `sqlite3` is sufficient for this exact shape of problem in this exact codebase -- reusing it, not introducing DuckDB or anything heavier, per the ladder (already-installed/stdlib before anything new).
