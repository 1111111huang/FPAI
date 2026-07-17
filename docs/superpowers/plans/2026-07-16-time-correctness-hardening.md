# Time-Correctness Test Hardening (W33–W36, W39) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close five independent, previously-untested time-correctness gaps identified in `documents/app_user_stories.md` Phase 8: a scheduler bug where a rescheduled T-30 job can be permanently blocked, a multi-day/DST soak test for the daily scheduler, an injectable clock for `raw_matches` staleness, a credit-counter month-boundary-restart integration test, a rate-limiter multi-step sequence test, and a coexistence check between the app's scheduler and the standalone weekly data-refresh scheduler.

**Architecture:** All five stories are independent (see `app_user_stories.md` Phase 8 dependency graph — none depend on the Phase 7 sandbox work, none depend on each other). Every task follows the codebase's existing convention: injectable `now_fn`/`time_fn` clocks, no `freezegun`, no real sleeping, `tmp_path`-scoped SQLite/JSON stores to simulate process restarts. One task (W33) requires a real source-code fix, not just new tests — see Task 5.

**Tech Stack:** Python, pytest, APScheduler, `unittest.mock`.

---

## File Structure

- Modify: `src/tools/data_tools.py` — add an injectable `now_fn` to `get_data_freshness()` (W34).
- Create: `tests/test_data_tools.py` — first-ever test file for this module (W34).
- Modify: `app/backend/tests/test_football_data_client.py` — append a rate-limiter sequence test (W36, no source change).
- Modify: `app/backend/tests/test_odds_api_client.py` — append a credit-counter restart test (W35, no source change).
- Create: `tests/test_scheduler_coexistence.py` — new cross-cutting test importing both `app.backend.scheduler` and `src.scheduling.data_refresh_scheduler` (W39, no source change).
- Modify: `app/backend/scheduler.py` — fix `schedule_once()`'s run-key bug (W33).
- Modify: `app/backend/tests/test_scheduler.py`, `app/backend/tests/test_scheduler_integration.py`, `app/backend/tests/test_scheduler_wiring.py` — update the 5 existing assertions that hardcode the old `"once"` run-key literal (W33, required by the fix above).
- Create: `app/backend/tests/test_scheduler_soak.py` — multi-day soak, DST, and reschedule tests (W33).

---

## Task 1: W34 — Injectable clock for `get_data_freshness()`

**Files:**
- Modify: `src/tools/data_tools.py:12-44`
- Test: `tests/test_data_tools.py` (new)

**Context:** `get_data_freshness()` calls `pd.Timestamp.now()` directly inline (`src/tools/data_tools.py:33`) with no seam to test the `is_stale` threshold boundary without globally monkeypatching `pandas.Timestamp.now` (risky — pandas is used throughout the codebase). No test file for this module exists today.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_tools.py`:

```python
"""W34: injectable clock for get_data_freshness(), plus 7/8-day staleness
boundary tests. Previously untested -- pd.Timestamp.now() was called
directly inline, with no seam to simulate day-by-day advancement without
globally monkeypatching pandas.Timestamp.now (risky, used throughout the
codebase)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.tools.data_tools import get_data_freshness


def _mock_manager(match_count: int, max_date) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (match_count, max_date)
    manager = MagicMock()
    manager.connection.return_value.__enter__.return_value = conn
    manager.connection.return_value.__exit__.return_value = False
    return manager


def test_exactly_seven_days_old_is_not_stale() -> None:
    max_date = pd.Timestamp("2026-07-01")
    now = pd.Timestamp("2026-07-08")  # exactly 7 days later

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(100, max_date)):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["days_since_update"] == 7
    assert result["is_stale"] is False


def test_exactly_eight_days_old_is_stale() -> None:
    max_date = pd.Timestamp("2026-07-01")
    now = pd.Timestamp("2026-07-09")  # exactly 8 days later

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(100, max_date)):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["days_since_update"] == 8
    assert result["is_stale"] is True


def test_staleness_flips_correctly_as_simulated_days_advance() -> None:
    max_date = pd.Timestamp("2026-07-01")
    manager = _mock_manager(100, max_date)

    with patch("src.tools.data_tools.DuckDBManager", return_value=manager):
        for offset, expected_stale in [(0, False), (6, False), (7, False), (8, True), (30, True)]:
            now = max_date + pd.Timedelta(days=offset)
            result = get_data_freshness(now_fn=lambda now=now: now)
            assert result["is_stale"] is expected_stale, f"offset={offset}"
            assert result["days_since_update"] == offset


def test_real_clock_default_is_unchanged() -> None:
    """Calling with no now_fn must still use the real wall clock -- zero
    behavior change for every existing caller."""
    max_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(50, max_date)):
        result = get_data_freshness()

    assert result["days_since_update"] == 3
    assert result["is_stale"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_tools.py -v`
Expected: FAIL with `TypeError: get_data_freshness() got an unexpected keyword argument 'now_fn'`

- [ ] **Step 3: Add the injectable clock**

Edit `src/tools/data_tools.py`. Add `Callable` to the imports and thread `now_fn` through:

```python
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.utils.db_manager import DuckDBManager


def get_data_freshness(now_fn: Callable[[], pd.Timestamp] = pd.Timestamp.now) -> dict[str, Any]:
    """Return data freshness metadata from the raw_matches table.

    Args:
        now_fn: Returns the current time; injectable for testing the
            staleness boundary without monkeypatching pandas globally.
            Defaults to the real wall clock.

    Returns:
        Dict with keys:
            latest_match_date: ISO date string of the most recent match (or None).
            days_since_update: Number of days since the latest match.
            match_count: Total number of rows in raw_matches.
            is_stale: True if latest_match_date is more than 7 days ago.
    """
    db = DuckDBManager()
    try:
        with db.connection(read_only=True) as conn:
            row = conn.execute("SELECT COUNT(*), MAX(date) FROM raw_matches").fetchone()
    except Exception:
        return {"latest_match_date": None, "days_since_update": None, "match_count": 0, "is_stale": True}

    match_count = int(row[0]) if row else 0
    max_date = row[1] if row else None
    if max_date is not None:
        latest_ts = pd.Timestamp(max_date).tz_localize(None)
        days_since = (now_fn().normalize() - latest_ts.normalize()).days
        latest_str = latest_ts.date().isoformat()
    else:
        days_since = None
        latest_str = None

    return {
        "latest_match_date": latest_str,
        "days_since_update": days_since,
        "match_count": match_count,
        "is_stale": (days_since is None or days_since > 7),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_tools.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ app/backend/tests/ -q`
Expected: same pass count as before this task, plus 4 new passing tests, zero failures. `get_data_freshness()` is called with zero args from `app/backend/main.py:111` and any MCP tool registration — confirm both still work unchanged (they use the default `now_fn`).

- [ ] **Step 6: Commit**

```bash
git add src/tools/data_tools.py tests/test_data_tools.py
git commit -m "test: add injectable clock + staleness boundary tests for get_data_freshness (W34)"
```

---

## Task 2: W36 — Rate-limiter multi-step sequence test

**Files:**
- Test: `app/backend/tests/test_football_data_client.py` (append)

**Context:** `_RateLimiter` (`app/backend/football_data_client.py:43-72`) is already injectable via `sleep_fn`/`time_fn`, but all 3 existing tests use a single frozen instant (`time_fn=lambda: 100.0`), never proving the realistic multi-call sequence: exhaustion → sleep → simulated time advances past the reset instant → next response's headers refresh `_remaining` → a subsequent call proceeds without waiting. This is purely additive test coverage — no source change is required, since the existing implementation is already correct (reading `wait_if_needed()`'s logic at `football_data_client.py:67-72` confirms the fallthrough behavior already works). If either new test unexpectedly fails, that reveals a real bug to fix before proceeding — do not skip investigating a failure here.

- [ ] **Step 1: Write the sequence tests**

Append to `app/backend/tests/test_football_data_client.py` (after the existing 3 `_RateLimiter` tests, i.e. after line 170):

```python
def test_rate_limiter_full_sequence_exhaustion_wait_reset_refreshed_call() -> None:
    """Drives the limiter through a realistic multi-call sequence: exhausted
    -> sleep computed -> simulated time genuinely advances past the reset
    instant -> the next real response's headers refresh _remaining -> a
    subsequent call proceeds without waiting. The 3 pre-existing tests above
    only ever used a single frozen time_fn instant, never proving this."""
    sleep_fn = MagicMock()
    now = [100.0]
    limiter = _RateLimiter(sleep_fn=sleep_fn, time_fn=lambda: now[0])

    # Step 1: a response arrives showing the budget is exhausted, resetting in 45s.
    limiter.update_from_headers({"x-requests-available-minute": "0", "X-RequestCounter-Reset": "45"})
    limiter.wait_if_needed()
    sleep_fn.assert_called_once_with(45)

    # Step 2: simulated time genuinely advances past the reset instant (145.0).
    now[0] = 146.0

    # Step 3: the next real response refreshes _remaining via fresh headers.
    limiter.update_from_headers({"x-requests-available-minute": "10", "X-RequestCounter-Reset": "60"})

    # Step 4: a subsequent call proceeds without waiting -- budget was refreshed.
    sleep_fn.reset_mock()
    limiter.wait_if_needed()
    sleep_fn.assert_not_called()


def test_rate_limiter_degrades_gracefully_when_headers_missing_after_time_has_passed() -> None:
    """A later response with no rate-limit headers at all must not crash or
    permanently wedge the limiter -- once real time has passed the old
    _reset_at, wait_if_needed() falls through to 'proceed' even without
    fresh headers, since update_from_headers no-ops on missing keys. A
    deliberate, tested guarantee, not an unverified side effect."""
    sleep_fn = MagicMock()
    now = [100.0]
    limiter = _RateLimiter(sleep_fn=sleep_fn, time_fn=lambda: now[0])

    limiter.update_from_headers({"x-requests-available-minute": "0", "X-RequestCounter-Reset": "10"})

    now[0] = 111.0  # past the old reset_at (110.0)
    limiter.update_from_headers({})  # no headers this time -- must no-op, not crash

    limiter.wait_if_needed()
    sleep_fn.assert_not_called()
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest app/backend/tests/test_football_data_client.py -v -k rate_limiter`
Expected: 5 passed (3 pre-existing + 2 new). If either new test fails, read `_RateLimiter.wait_if_needed()`/`update_from_headers()` (`app/backend/football_data_client.py:59-72`) and fix the implementation — do not weaken the test to match broken behavior.

- [ ] **Step 3: Commit**

```bash
git add app/backend/tests/test_football_data_client.py
git commit -m "test: add rate-limiter multi-step sequence test (W36)"
```

---

## Task 3: W35 — Credit-counter persistence across a month-boundary restart

**Files:**
- Test: `app/backend/tests/test_odds_api_client.py` (append)

**Context:** `CreditCounter`'s own clock is already injectable and unit-tested in isolation (`test_counter_resets_at_simulated_month_boundary`, `test_odds_api_client.py:219-229`). The untested gap is one level up, at `FileCreditCounterStore`: `_roll_month_if_needed()` is only ever called lazily from `credits_used`/`would_exceed`/`record_usage` — never eagerly from `load()` (`app/backend/odds_api_client.py:135-139`) — so the realistic "save state in July, restart in August, first real call after restart" sequence has never been exercised end-to-end. This is purely additive: tracing through `load()` → `CreditCounter.from_dict()` → the lazy rollover confirms the existing code already handles this correctly. As with Task 2, a failing test here means a real bug — fix the source, don't weaken the test.

- [ ] **Step 1: Write the restart-boundary test**

Append to `app/backend/tests/test_odds_api_client.py` (after `test_file_credit_counter_store_round_trip`, i.e. after line 241):

```python
def test_credit_counter_rolls_over_on_first_check_after_a_month_boundary_restart(tmp_path: Path) -> None:
    """Realistic restart sequence: instance A accumulates usage and saves in
    July; instance B ('process restart') loads that same file in August --
    its very first real usage check must correctly roll over rather than
    carrying the stale July count forward. Same 'two separate process
    instances sharing the same on-disk file' pattern W21 used for the
    scheduler (test_scheduler_integration.py)."""
    counter_path = tmp_path / "odds_credit_usage.json"
    store = FileCreditCounterStore(counter_path)

    # --- instance A: July, accumulates usage, saves before "restarting" ---
    counter_a = store.load(now_fn=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc))
    counter_a.record_usage(480)
    store.save(counter_a)

    # --- instance B: a fresh process, loads the same file in August ---
    counter_b = store.load(now_fn=lambda: datetime(2026, 8, 1, tzinfo=timezone.utc))

    # the very first real usage check after the restart must roll over
    assert counter_b.credits_used == 0
    assert not counter_b.would_exceed(cost=1, limit=500, safety_margin=50)

    counter_b.record_usage(10)
    assert counter_b.credits_used == 10

    # confirm the rollover was persisted, not just held in memory
    store.save(counter_b)
    reloaded = json.loads(counter_path.read_text())
    assert reloaded == {"credits_used": 10, "month_key": "2026-08"}
```

Check the top of the file (`app/backend/tests/test_odds_api_client.py:22-27`) already imports `datetime`, `timezone`, `Path`, and `json` is used elsewhere in the same module (`odds_api_client.py` itself imports `json`) — confirm `import json` is present in the test file's own imports; add `import json` near the top if it isn't already there.

- [ ] **Step 2: Run the test**

Run: `python -m pytest app/backend/tests/test_odds_api_client.py -v -k month_boundary_restart`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add app/backend/tests/test_odds_api_client.py
git commit -m "test: cover credit-counter rollover across a month-boundary restart (W35)"
```

---

## Task 4: W39 — Scheduler coexistence test

**Files:**
- Test: `tests/test_scheduler_coexistence.py` (new)

**Context:** `src/scheduling/data_refresh_scheduler.py` (US#109, weekly Sunday-3am cron) and `app/backend/scheduler.py`'s `RecoverableScheduler` (W08, daily/T-30, timezone-aware via `NY_TZ`) have never been constructed side by side in the same process. Each builds its own independent `BackgroundScheduler()` instance (`data_refresh_scheduler.py:65` has no `timezone=` arg at all, unlike `scheduler.py:107`'s `BackgroundScheduler(timezone=timezone)`), so there is no shared APScheduler jobstore for a job-id collision to occur in — this test confirms that, and that both fire independently without exceptions.

- [ ] **Step 1: Write the coexistence test**

Create `tests/test_scheduler_coexistence.py`:

```python
"""W39: does the still-disconnected weekly data-refresh scheduler (US#109)
coexist safely with W08's RecoverableScheduler if ever run in the same
process? Each constructs its own independent BackgroundScheduler instance
(data_refresh_scheduler.py's has no explicit timezone, unlike
RecoverableScheduler's NY_TZ-aware one) -- there is no shared jobstore for
a job-id collision to occur in; this test confirms both fire independently,
with no exception and no cross-registration, as a forward-looking check in
case they're ever wired together."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("apscheduler")

from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler
from src.scheduling.data_refresh_scheduler import build_weekly_refresh_scheduler


def test_both_schedulers_construct_and_start_in_the_same_process_without_interfering(tmp_path: Path) -> None:
    calls: list[str] = []

    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    recoverable = RecoverableScheduler(
        run_log=run_log, now_fn=lambda: datetime(2026, 7, 12, 10, 0, tzinfo=NY_TZ),
    )
    recoverable.schedule_daily("eod_batch_generation", lambda: calls.append("eod"), hour=23, minute=0)

    weekly = build_weekly_refresh_scheduler(refresh_fn=lambda: calls.append("weekly"))

    recoverable.start()
    weekly.start()
    try:
        recoverable_job_ids = {job.id for job in recoverable._scheduler.get_jobs()}
        weekly_job_ids = {job.id for job in weekly.get_jobs()}

        assert recoverable_job_ids == {"eod_batch_generation"}
        assert weekly_job_ids == {"weekly_data_refresh"}
        assert recoverable_job_ids.isdisjoint(weekly_job_ids)

        # too early for the EOD trigger -- confirms starting didn't
        # spuriously fire anything via APScheduler's own trigger
        assert calls == []

        # each scheduler's job fires independently when invoked directly,
        # with no cross-registration onto the other's scheduler
        weekly.get_job("weekly_data_refresh").func()
        assert calls == ["weekly"]
        assert recoverable._scheduler.get_job("weekly_data_refresh") is None
    finally:
        recoverable.shutdown()
        weekly.shutdown()
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_scheduler_coexistence.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_scheduler_coexistence.py
git commit -m "test: confirm RecoverableScheduler and the weekly refresh scheduler coexist safely (W39)"
```

---

## Task 5: W33 — Scheduler multi-day soak test + DST correctness + reschedule fix

**Files:**
- Modify: `app/backend/scheduler.py:43-46,122-130`
- Modify: `app/backend/tests/test_scheduler.py:70-96`
- Modify: `app/backend/tests/test_scheduler_integration.py:29,92-108`
- Modify: `app/backend/tests/test_scheduler_wiring.py:113`
- Test (new): `app/backend/tests/test_scheduler_soak.py`

**Context:** This is the one story in this plan that requires a real source fix, not just new tests. `schedule_once()`'s catch-up marker is keyed on a **constant** `run_key = ONCE_RUN_KEY = "once"` (`app/backend/scheduler.py:46,124,129-130`) rather than on the job's actual `run_at`. Since a postponed/rescheduled fixture re-registers the *same* `job_id` (`f"t30_{match_id}"`, `app/backend/scheduler_wiring.py:84`) with a *new* `run_at`, the old `"once"` marker from the original firing permanently blocks the new time from ever catching up. The fix: key the run marker on `run_at.isoformat()` instead of a constant, so each distinct kickoff time for a given `job_id` gets its own marker while same-time re-registrations still correctly dedupe.

### Part A: reproduce the bug, then fix it

- [ ] **Step 1: Write the failing reschedule test**

Create `app/backend/tests/test_scheduler_soak.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_scheduler_soak.py -v`
Expected: FAIL — `assert calls == ["original", "rescheduled"]` fails because `calls == ["original"]` (the rescheduled job never fires, blocked by the constant `"once"` marker).

- [ ] **Step 3: Fix `schedule_once()` in `app/backend/scheduler.py`**

Replace the comment and constant at lines 43-46:

```python
# One-off (non-daily) jobs are keyed by (job_id, run_at) rather than a
# constant run_key: job_id alone is not enough, since a postponed/
# rescheduled fixture re-registers the *same* job_id (f"t30_{match_id}")
# with a *new* run_at -- a constant run_key would let the old marker
# permanently block the new time from ever firing (W33).
```

(Remove the `ONCE_RUN_KEY = "once"` line entirely — it's replaced by computing `run_key` from `run_at` directly below.)

Replace `schedule_once()` (lines 122-130):

```python
    def schedule_once(self, job_id: str, fn: Callable[[], None], run_at: datetime) -> None:
        run_key = run_at.isoformat()
        self._scheduler.add_job(
            lambda: self._run_and_mark(job_id, fn, run_key=run_key),
            trigger=DateTrigger(run_date=run_at, timezone=self.timezone),
            id=job_id,
            replace_existing=True,
        )
        if self._now_fn() >= run_at and not self.run_log.has_run(job_id, run_key):
            self._run_and_mark(job_id, fn, run_key)
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `python -m pytest app/backend/tests/test_scheduler_soak.py -v`
Expected: 1 passed.

- [ ] **Step 5: Update the 5 existing assertions that hardcode the old `"once"` literal**

This constant is now gone, so every existing test that referenced the literal `"once"` run-key must be updated to use the real `run_at`-derived key. Confirm the full list first:

Run: `grep -rn 'ONCE_RUN_KEY\|"once"' app/backend/ src/ tests/`
Expected: no remaining references to `ONCE_RUN_KEY`; the following 5 literal-`"once"` call sites are exactly what needs updating (confirm no others turn up).

Edit `app/backend/tests/test_scheduler.py` — replace both `schedule_once` tests (lines 70-96):

```python
def test_schedule_once_catches_up_when_run_at_already_passed(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 22, 15, 5, tzinfo=NY_TZ)
    run_at = now - timedelta(minutes=5)  # scheduled 5 min ago -- missed

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_once(
        "t30_m1", lambda: calls.append("ran"), run_at=run_at
    )

    assert calls == ["ran"]
    assert run_log.has_run("t30_m1", run_at.isoformat())


def test_schedule_once_does_not_rerun_once_marked_ran(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 8, 22, 15, 5, tzinfo=NY_TZ)
    run_at = now - timedelta(minutes=5)
    run_log.mark_ran("t30_m1", run_at.isoformat())
    calls = []

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_once(
        "t30_m1", lambda: calls.append("ran"), run_at=run_at
    )

    assert calls == []
```

Edit `app/backend/tests/test_scheduler_integration.py` — add `t30_run_at` to the import at line 29:

```python
from app.backend.scheduler_wiring import EOD_JOB_ID, build_schedule_t30, register_eod_job, t30_run_at
```

Then in `test_missed_t30_job_catches_up_after_an_outage_spanning_the_kickoff_window` (lines 78-108), add `expected_run_key = t30_run_at(fixture).isoformat()` right after `run_log = JobRunLog(db_path=job_runs_db)` (line 93), and change both assertions:

```python
    assert not run_log.has_run("t30_m2", expected_run_key)
```

and (replacing line 108):

```python
    assert run_log.has_run("t30_m2", expected_run_key)  # caught up immediately on restart
```

Edit `app/backend/tests/test_scheduler_wiring.py` line 113 (`t30_run_at` is already imported at the top of this file):

```python
    assert run_log.has_run("t30_m1", t30_run_at(fixture).isoformat())
```

- [ ] **Step 6: Run the full scheduler test suite to verify the fix and updated assertions all pass**

Run: `python -m pytest app/backend/tests/test_scheduler.py app/backend/tests/test_scheduler_integration.py app/backend/tests/test_scheduler_wiring.py app/backend/tests/test_scheduler_soak.py -v`
Expected: all passed, zero failures.

- [ ] **Step 7: Commit the fix**

```bash
git add app/backend/scheduler.py app/backend/tests/test_scheduler.py app/backend/tests/test_scheduler_integration.py app/backend/tests/test_scheduler_wiring.py app/backend/tests/test_scheduler_soak.py
git commit -m "fix: key schedule_once's run marker on run_at, not a constant, so a rescheduled T-30 job can fire (W33)"
```

### Part B: multi-day soak test

- [ ] **Step 8: Write the multi-day soak test**

Append to `app/backend/tests/test_scheduler_soak.py`:

```python
def test_daily_eod_job_fires_exactly_once_per_calendar_day_across_a_multi_day_run(tmp_path: Path) -> None:
    """Simulates N days of continuous operation: each day, a fresh
    RecoverableScheduler is (re)constructed at a moment past the 23:00
    trigger (as a real long-running process's own CronTrigger fire, or a
    same-day restart, would both do) and registers the same daily job_id.
    Exactly one fire per day -- never zero, never twice, even if the same
    day's registration happens more than once (e.g. two restarts on the
    same day)."""
    db_path = tmp_path / "job_runs.db"
    run_log = JobRunLog(db_path=db_path)
    calls = []
    base = datetime(2026, 7, 1, 23, 30, tzinfo=NY_TZ)

    for day_offset in range(10):
        now = base + timedelta(days=day_offset)
        RecoverableScheduler(run_log=run_log, now_fn=lambda now=now: now).schedule_daily(
            "eod_batch_generation", lambda day=day_offset: calls.append(day), hour=23, minute=0
        )
        # a same-day restart re-registering must not double-fire today
        RecoverableScheduler(run_log=run_log, now_fn=lambda now=now: now).schedule_daily(
            "eod_batch_generation", lambda day=day_offset: calls.append(day), hour=23, minute=0
        )

    assert calls == list(range(10))  # exactly one fire per day, in order
```

- [ ] **Step 9: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_scheduler_soak.py -v -k multi_day`
Expected: 1 passed.

### Part C: DST correctness test

- [ ] **Step 10: Write the DST-transition test**

Append to `app/backend/tests/test_scheduler_soak.py`:

```python
def test_daily_job_fires_correctly_at_local_2300_on_both_sides_of_a_dst_transition(tmp_path: Path) -> None:
    """2026-03-08 is a real US spring-forward DST transition (clocks jump
    02:00 EST -> 03:00 EDT). A naive fixed-UTC-offset trigger comparison
    would drift by an hour across it; NY_TZ is a real zoneinfo timezone, so
    local 23:00 must resolve correctly on both the day before and the day
    of the transition."""
    db_path = tmp_path / "job_runs.db"
    run_log = JobRunLog(db_path=db_path)
    calls = []

    before_transition = datetime(2026, 3, 7, 23, 30, tzinfo=NY_TZ)  # still EST (UTC-5)
    RecoverableScheduler(run_log=run_log, now_fn=lambda: before_transition).schedule_daily(
        "eod_batch_generation", lambda: calls.append("2026-03-07"), hour=23, minute=0
    )
    assert calls == ["2026-03-07"]
    assert before_transition.utcoffset() == timedelta(hours=-5)

    on_transition_day = datetime(2026, 3, 8, 23, 30, tzinfo=NY_TZ)  # now EDT (UTC-4)
    RecoverableScheduler(run_log=run_log, now_fn=lambda: on_transition_day).schedule_daily(
        "eod_batch_generation", lambda: calls.append("2026-03-08"), hour=23, minute=0
    )
    assert calls == ["2026-03-07", "2026-03-08"]
    assert on_transition_day.utcoffset() == timedelta(hours=-4)

    # a same-day re-registration on the transition day itself must not double-fire
    RecoverableScheduler(run_log=run_log, now_fn=lambda: on_transition_day).schedule_daily(
        "eod_batch_generation", lambda: calls.append("2026-03-08"), hour=23, minute=0
    )
    assert calls == ["2026-03-07", "2026-03-08"]
```

- [ ] **Step 11: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_scheduler_soak.py -v`
Expected: 3 passed (reschedule + multi-day + DST).

- [ ] **Step 12: Run the full backend suite to confirm zero regressions**

Run: `python -m pytest app/backend/tests/ -q`
Expected: all passed, same count as before this task plus the new tests in `test_scheduler_soak.py`, zero failures.

- [ ] **Step 13: Commit**

```bash
git add app/backend/tests/test_scheduler_soak.py
git commit -m "test: add multi-day soak test and DST-transition correctness test for the daily scheduler (W33)"
```

---

## Self-Review Notes

- **Spec coverage:** W33 (Task 5, all 3 parts), W34 (Task 1), W35 (Task 3), W36 (Task 2), W39 (Task 4) — every acceptance criterion in the Phase 8 story text is covered by a task above.
- **Placeholder scan:** no `TBD`/`implement later` markers; every step has complete, runnable code.
- **Type consistency:** `now_fn`/`time_fn` signatures match their call sites throughout; `schedule_once`'s new `run_key = run_at.isoformat()` is used identically in the fix (`scheduler.py`) and in every updated test assertion.
- After Task 5, mark **W33, W34, W35, W36, W39** as `completed` in `documents/app_user_stories.md`'s Stories table (per `CLAUDE.md`: "When tasks are complete, mark the user story as completed"), each with a completion note following the existing narrative style (see W01–W26's entries) — file:line references, what was verified, full-suite pass count.

---

## Execution Handoff

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.
