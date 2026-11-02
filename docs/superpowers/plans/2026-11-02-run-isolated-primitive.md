# run_isolated Primitive + Failures Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared `run_isolated`/`run_isolated_async` primitive that catches any exception from a unit of work, records it to a structured, queryable `failures` table, alerts (debounced) on first occurrence, and always lets the caller's loop continue — then wire it into the two clearest, highest-value live gaps found in the 2026-11-02 audit: T-30's completely unguarded `add_secondary_odds`/`cache.record_generation` calls, and the unguarded push-notify steps in the EOD and fixture-reconciliation jobs.

**Architecture:** One new module, `app/backend/run_isolated.py`, holding a `FailuresStore` (stdlib `sqlite3`, same fresh-connection-per-call pattern as `scheduler.py`'s `JobRunLog`) and the two wrapper functions. No changes to any function's own signature or behavior on success — only the failure path changes, from "unguarded" or "ad hoc try/except" to "caught once, recorded once, logged once."

**Tech Stack:** Python 3.13, stdlib `sqlite3`, pytest, existing `app.backend.slack_alerts.post_slack_alert`.

**Scope note:** This plan wires `run_isolated` into the two gaps with no return-value ambiguity (a notify call and a "do work or skip" call, neither of which anything downstream branches on). It deliberately does *not* retrofit `eod_batch.py`'s `has_kicked_off`/`already_fresh`/`validate_and_degrade`/`cache.record_generation` calls (the "broad-umbrella-only" stages from the design spec) — those are boolean/control-flow-bearing calls where swapping in `run_isolated`'s "return `None` on failure" contract needs a per-call-site decision about what `None` should mean for that specific control flow (e.g. does a `has_kicked_off` failure mean "treat as kicked off, skip defensively" or "treat as not kicked off, proceed"?). That retrofit is real but is its own follow-up, not bundled into this plan where the wrong default could silently change production behavior.

---

### Task 1: `FailuresStore` — schema and write path

**Files:**
- Create: `app/backend/run_isolated.py`
- Test: `app/backend/tests/test_run_isolated.py`

- [ ] **Step 1: Write the failing test for schema creation and recording a failure**

```python
"""Tests for run_isolated.py -- the shared per-unit isolation primitive and
its backing FailuresStore. Mirrors scheduler.py's JobRunLog test pattern
(fresh sqlite3 connection per call, tmp_path-backed instances)."""

from __future__ import annotations

from pathlib import Path

from app.backend.run_isolated import FailuresStore


def test_failures_store_records_a_failure(tmp_path: Path) -> None:
    store = FailuresStore(db_path=tmp_path / "failures.db")
    exc = ValueError("bad odds payload")
    store.record(subsystem="t30", unit_id="match_id=558542", exc=exc, alerted=True, league="I1")

    rows = store.list_recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "t30"
    assert row["unit_id"] == "match_id=558542"
    assert row["league"] == "I1"
    assert row["exc_type"] == "ValueError"
    assert row["message"] == "bad odds payload"
    assert "ValueError: bad odds payload" in row["traceback"]
    assert row["retryable"] is None
    assert row["alerted"] == 1


def test_failures_store_league_is_optional(tmp_path: Path) -> None:
    """A shared-builder failure (BUG-101's class) isn't scoped to any one
    league -- league stays NULL rather than guessing at one."""
    store = FailuresStore(db_path=tmp_path / "failures.db")
    store.record(subsystem="pregenerate", unit_id="builder=odds_client", exc=ValueError("x"), alerted=False)

    rows = store.list_recent()
    assert rows[0]["league"] is None


def test_failures_store_filters_by_league(tmp_path: Path) -> None:
    store = FailuresStore(db_path=tmp_path / "failures.db")
    store.record(subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0")
    store.record(subsystem="t30", unit_id="match_id=2", exc=ValueError("b"), alerted=False, league="I1")

    rows = store.list_recent(league="I1")
    assert len(rows) == 1
    assert rows[0]["unit_id"] == "match_id=2"


def test_failures_store_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "failures.db"
    FailuresStore(db_path=db_path).record(
        subsystem="eod_batch", unit_id="league=I1", exc=RuntimeError("boom"), alerted=False,
    )

    reloaded = FailuresStore(db_path=db_path)
    rows = reloaded.list_recent()
    assert len(rows) == 1
    assert rows[0]["subsystem"] == "eod_batch"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_run_isolated.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.run_isolated'`

- [ ] **Step 3: Implement `FailuresStore`**

```python
"""Shared per-unit isolation primitive for the background/async surface.

run_isolated()/run_isolated_async() wrap a single unit of work (one
fixture, one league, one notify call) so a caught exception never
propagates to the caller's loop -- the loop always continues to the next
unit. Every caught failure is written to FailuresStore (sqlite3, same
fresh-connection-per-call pattern as scheduler.py's JobRunLog) instead of
just a log line, and alerted (debounced) via the existing slack_alerts
webhook path on first occurrence within the cooldown window.

2026-11-02 design: docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import traceback as traceback_module
from typing import Any, Awaitable, Callable, TypeVar

from app.backend.slack_alerts import post_slack_alert
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

T = TypeVar("T")

# W163-style convention: repo-root data/, not app/data/ -- matches every
# other per-concern store in this codebase (job_runs.db, recommendation_cache.db).
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "failures.db"

_FAILURES_COLUMNS = ("id", "ts", "subsystem", "unit_id", "league", "exc_type", "message", "traceback", "retryable", "alerted")


class FailuresStore:
    """Persists one row per run_isolated-caught failure. retryable is
    reserved (always NULL today) -- see the design spec's own note on why
    automatic transient/deterministic classification isn't part of this
    primitive."""

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
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    league TEXT,
                    exc_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    traceback TEXT NOT NULL,
                    retryable INTEGER,
                    alerted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_subsystem_ts ON failures (subsystem, ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_league_ts ON failures (league, ts)"
            )

    def record(
        self, subsystem: str, unit_id: str, exc: BaseException, alerted: bool, league: str | None = None,
    ) -> None:
        tb = "".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO failures (ts, subsystem, unit_id, league, exc_type, message, traceback, retryable, alerted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    subsystem,
                    unit_id,
                    league,
                    type(exc).__name__,
                    str(exc),
                    tb,
                    1 if alerted else 0,
                ),
            )

    def list_recent(
        self, since: str | None = None, subsystem: str | None = None, league: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT " + ", ".join(_FAILURES_COLUMNS) + " FROM failures WHERE 1=1"
        params: list[Any] = []
        if since is not None:
            query += " AND ts >= ?"
            params.append(since)
        if subsystem is not None:
            query += " AND subsystem = ?"
            params.append(subsystem)
        if league is not None:
            query += " AND league = ?"
            params.append(league)
        query += " ORDER BY ts DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(zip(_FAILURES_COLUMNS, row)) for row in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_run_isolated.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/run_isolated.py app/backend/tests/test_run_isolated.py
git commit -m "feat: add FailuresStore for structured failure recording"
```

---

### Task 2: `run_isolated` / `run_isolated_async` wrappers

**Files:**
- Modify: `app/backend/run_isolated.py`
- Test: `app/backend/tests/test_run_isolated.py`

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/tests/test_run_isolated.py`:

```python
import asyncio
from unittest.mock import patch

from app.backend.run_isolated import run_isolated, run_isolated_async


def test_run_isolated_returns_the_function_result_on_success(tmp_path: Path) -> None:
    with patch("app.backend.run_isolated.get_failures_store") as mock_get_store:
        result = run_isolated("eod_batch", "match_id=123", lambda: 42)
    assert result == 42
    mock_get_store.assert_not_called()


def test_run_isolated_returns_none_and_records_on_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.backend.run_isolated.DEFAULT_DB_PATH", tmp_path / "failures.db")
    import app.backend.run_isolated as run_isolated_module
    run_isolated_module._singleton_store = None

    def _raise():
        raise ValueError("boom")

    with patch("app.backend.run_isolated.post_slack_alert", return_value=False):
        result = run_isolated("eod_batch", "match_id=123", _raise)

    assert result is None
    rows = run_isolated_module.get_failures_store().list_recent()
    assert len(rows) == 1
    assert rows[0]["exc_type"] == "ValueError"
    assert rows[0]["unit_id"] == "match_id=123"


def test_run_isolated_never_raises_even_if_recording_itself_fails(tmp_path: Path) -> None:
    def _raise():
        raise ValueError("boom")

    with patch("app.backend.run_isolated.get_failures_store", side_effect=RuntimeError("store is down")):
        result = run_isolated("eod_batch", "match_id=123", _raise)

    assert result is None


def test_run_isolated_async_returns_the_function_result_on_success() -> None:
    async def _ok():
        return "fine"

    result = asyncio.run(run_isolated_async("t30", "match_id=558542", _ok))
    assert result == "fine"


def test_run_isolated_async_returns_none_and_records_on_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.backend.run_isolated.DEFAULT_DB_PATH", tmp_path / "failures.db")
    import app.backend.run_isolated as run_isolated_module
    run_isolated_module._singleton_store = None

    async def _raise():
        raise RuntimeError("vendor timeout")

    with patch("app.backend.run_isolated.post_slack_alert", return_value=True):
        result = asyncio.run(run_isolated_async("t30", "match_id=558542", _raise))

    assert result is None
    rows = run_isolated_module.get_failures_store().list_recent()
    assert len(rows) == 1
    assert rows[0]["exc_type"] == "RuntimeError"
    assert rows[0]["alerted"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_run_isolated.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_isolated'`

- [ ] **Step 3: Implement the wrappers**

Append to `app/backend/run_isolated.py`:

```python
_singleton_store: FailuresStore | None = None


def get_failures_store() -> FailuresStore:
    """Module-level singleton, matching bets.get_bet_tracker's own pattern
    (main.py) -- overridden in tests by resetting this module attribute."""
    global _singleton_store
    if _singleton_store is None:
        _singleton_store = FailuresStore()
    return _singleton_store


def _record_failure(subsystem: str, unit_id: str, exc: BaseException, league: str | None) -> None:
    """Never raises -- a failure to record a failure must degrade to a log
    line, not become a second unhandled exception (same contract
    slack_alerts.post_slack_alert already holds itself to)."""
    try:
        alerted = post_slack_alert(
            f":warning: {subsystem} failed for {unit_id}: {type(exc).__name__}: {exc}",
            debounce_key=f"run_isolated:{subsystem}:{type(exc).__name__}",
        )
        get_failures_store().record(subsystem, unit_id, exc, alerted=alerted, league=league)
    except Exception:
        LOGGER.exception(
            "run_isolated: failed to record failure for subsystem=%s unit_id=%s -- logging only.",
            subsystem, unit_id,
        )


def run_isolated(subsystem: str, unit_id: str, fn: Callable[[], T], league: str | None = None) -> T | None:
    """Runs fn(), catching any exception. On failure: logs, alerts
    (debounced, first occurrence of this subsystem+exc_type pair within the
    cooldown window), records a failures row, and returns None instead of
    raising -- the caller's loop always continues to the next unit.
    `league`, when known at the call site, is stored so the /ops dashboard
    can filter by it -- NULL for failures not scoped to one league (e.g. a
    shared-builder failure, BUG-101's class)."""
    try:
        return fn()
    except Exception as exc:
        LOGGER.warning("run_isolated: %s/%s failed: %s", subsystem, unit_id, exc, exc_info=exc)
        _record_failure(subsystem, unit_id, exc, league)
        return None


async def run_isolated_async(
    subsystem: str, unit_id: str, fn: Callable[[], Awaitable[T]], league: str | None = None,
) -> T | None:
    """Same contract as run_isolated, for an async unit of work (an
    asyncio.to_thread-wrapped vendor/LLM call)."""
    try:
        return await fn()
    except Exception as exc:
        LOGGER.warning("run_isolated_async: %s/%s failed: %s", subsystem, unit_id, exc, exc_info=exc)
        _record_failure(subsystem, unit_id, exc, league)
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_run_isolated.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/run_isolated.py app/backend/tests/test_run_isolated.py
git commit -m "feat: add run_isolated/run_isolated_async wrappers"
```

---

### Task 3: Close T-30's unguarded `add_secondary_odds` gap

**Files:**
- Modify: `app/backend/t30_refresh.py:156-180`
- Test: `app/backend/tests/test_t30_refresh.py`

This is the clearest live gap from the audit: `add_secondary_odds` is wrapped in a `try/except` in `eod_batch.py` (BUG-093) but called completely unguarded in `t30_refresh.py` — an exception there propagates straight out of `refresh_match_at_t30`, producing no `T30RefreshResult` at all.

- [ ] **Step 1: Read the current code to confirm exact line numbers**

Run: `sed -n '150,180p' app/backend/t30_refresh.py`

Confirm the `add_secondary_odds(...)` call (no surrounding try/except) and the `cache.record_generation(...)` call further down, matching:

```python
    add_secondary_odds(
        match_info, fresh_odds, odds_client, cache, fixture, date_str,
        agent_config_hash, ODDS_SPORT_KEY_BY_COMPETITION[league], odds_by_teams,
    )

    try:
        agent_result = recommendations.run_agent(match_info=match_info, config=config)
    except Exception as exc:
        LOGGER.warning("T-30 refresh: run_agent failed for match_id=%s: %s", fixture.match_id, exc)
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_error")

    raw, reasoning_trace, forecast_payload = recommendations.unwrap_agent_result(agent_result)
    degraded = validate_and_degrade(raw, fixture.home_team, fixture.away_team)
    cache.record_generation(
        match_id=fixture.match_id, date=date_str, agent_config_hash=agent_config_hash,
        odds=fresh_odds, recommendation=degraded.model_dump(), triggered_by="scheduled",
        reasoning_trace=reasoning_trace, forecast_payload=forecast_payload,
    )
```

- [ ] **Step 2: Write the failing test**

Check the existing fixture/mocking pattern first:

Run: `grep -n "^def test_refresh_match_at_t30\|def _fixture\|MagicMock" app/backend/tests/test_t30_refresh.py | head -20`

Add this test, matching the file's existing style (reuse whatever fixture-building helper already exists in that file for `NormalizedMatch`/`odds_client`/`cache`):

```python
def test_refresh_match_at_t30_isolates_a_failing_add_secondary_odds_call(monkeypatch) -> None:
    """2026-11-02: add_secondary_odds used to be completely unguarded here
    (unlike the identical call in eod_batch.py, wrapped since BUG-093) --
    an exception propagated straight out of refresh_match_at_t30 with no
    T30RefreshResult produced at all. run_isolated now catches it."""
    from unittest.mock import MagicMock, patch
    from app.backend.t30_refresh import refresh_match_at_t30

    fixture = _make_fixture()  # reuse this file's existing fixture-building helper
    odds_client = MagicMock()
    odds_client.get_odds.return_value = _make_odds_events()  # reuse this file's existing odds-events helper
    cache = MagicMock()
    config = _make_config()  # reuse this file's existing config-building helper

    with patch("app.backend.t30_refresh.add_secondary_odds", side_effect=RuntimeError("vendor exploded")), \
         patch("app.backend.t30_refresh.recommendations.run_agent") as mock_run_agent:
        result = refresh_match_at_t30(
            fixture=fixture, odds_client=odds_client, cache=cache, config=config, date_str="2026-11-02",
        )

    assert result.outcome == "skipped_error"
    mock_run_agent.assert_not_called()
    cache.record_generation.assert_not_called()
```

(Use the file's actual existing helper names in place of `_make_fixture`/`_make_odds_events`/`_make_config` — check `grep -n "^def _make\|^def test_refresh_match_at_t30_refreshes" app/backend/tests/test_t30_refresh.py` first and match them exactly.)

- [ ] **Step 2b: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_t30_refresh.py -k isolates_a_failing_add_secondary_odds -v`
Expected: FAIL — either an uncaught `RuntimeError` propagates out of the test, or `mock_run_agent.assert_not_called()` fails because today's code has no guard stopping it from being reached... actually confirm which: today's code has no try/except, so the `RuntimeError` propagates straight out of `refresh_match_at_t30` and the test fails with that exception, not a clean assertion failure. That itself is the proof of the gap.

- [ ] **Step 3: Wrap `add_secondary_odds` in `run_isolated`**

In `app/backend/t30_refresh.py`, add the import:

```python
from app.backend.run_isolated import run_isolated
```

Replace:

```python
    add_secondary_odds(
        match_info, fresh_odds, odds_client, cache, fixture, date_str,
        agent_config_hash, ODDS_SPORT_KEY_BY_COMPETITION[league], odds_by_teams,
    )
```

with:

```python
    # 2026-11-02: add_secondary_odds makes real network calls to two odds
    # vendors, same as eod_batch.py's own call -- that one's been wrapped
    # since BUG-093, this one never was. An exception here used to
    # propagate straight out of refresh_match_at_t30 with no
    # T30RefreshResult produced at all.
    secondary_odds_ok = run_isolated(
        "t30", f"match_id={fixture.match_id}",
        lambda: add_secondary_odds(
            match_info, fresh_odds, odds_client, cache, fixture, date_str,
            agent_config_hash, ODDS_SPORT_KEY_BY_COMPETITION[league], odds_by_teams,
        ) or True,
        league=league,
    )
    if secondary_odds_ok is None:
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_error")
```

(`add_secondary_odds` returns `None` on its own normal success path -- the `or True` makes `run_isolated`'s success/failure distinction unambiguous: `True` on success, `None` only on a caught exception.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest app/backend/tests/test_t30_refresh.py -k isolates_a_failing_add_secondary_odds -v`
Expected: PASS

- [ ] **Step 5: Run the full T-30 test file to confirm no regressions**

Run: `python -m pytest app/backend/tests/test_t30_refresh.py -v`
Expected: all tests PASS, including every pre-existing one

- [ ] **Step 6: Commit**

```bash
git add app/backend/t30_refresh.py app/backend/tests/test_t30_refresh.py
git commit -m "fix: isolate T-30's add_secondary_odds call with run_isolated"
```

---

### Task 4: Close T-30's unguarded `cache.record_generation` gap

**Files:**
- Modify: `app/backend/t30_refresh.py` (same area, just below Task 3's change)
- Test: `app/backend/tests/test_t30_refresh.py`

- [ ] **Step 1: Write the failing test**

```python
def test_refresh_match_at_t30_isolates_a_failing_cache_write(monkeypatch) -> None:
    """2026-11-02: cache.record_generation was also unguarded -- a DB write
    failure here (disk full, lock contention) propagated out the same way
    the add_secondary_odds gap did."""
    from unittest.mock import MagicMock, patch
    from app.backend.t30_refresh import refresh_match_at_t30

    fixture = _make_fixture()
    odds_client = MagicMock()
    odds_client.get_odds.return_value = _make_odds_events()
    cache = MagicMock()
    cache.record_generation.side_effect = RuntimeError("disk full")
    config = _make_config()

    fake_agent_result = {"recommendation": {
        "match": {"home": fixture.home_team, "away": fixture.away_team, "date": "2026-11-02", "league": "E0"},
        "overall": "no_bet", "candidates": [], "recommendation_pick": None,
        "explanation": [], "confidence": "low", "limitations": [], "prediction_basis": "unknown",
    }}

    with patch("app.backend.t30_refresh.add_secondary_odds"), \
         patch("app.backend.t30_refresh.recommendations.run_agent", return_value=fake_agent_result), \
         patch("app.backend.t30_refresh.recommendations.unwrap_agent_result", return_value=(fake_agent_result["recommendation"], None, None)):
        result = refresh_match_at_t30(
            fixture=fixture, odds_client=odds_client, cache=cache, config=config, date_str="2026-11-02",
        )

    assert result.outcome == "skipped_error"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_t30_refresh.py -k isolates_a_failing_cache_write -v`
Expected: FAIL — the `RuntimeError("disk full")` propagates straight out of `refresh_match_at_t30`

- [ ] **Step 3: Wrap `cache.record_generation` in `run_isolated`**

Replace:

```python
    raw, reasoning_trace, forecast_payload = recommendations.unwrap_agent_result(agent_result)
    degraded = validate_and_degrade(raw, fixture.home_team, fixture.away_team)
    cache.record_generation(
        match_id=fixture.match_id, date=date_str, agent_config_hash=agent_config_hash,
        odds=fresh_odds, recommendation=degraded.model_dump(), triggered_by="scheduled",
        reasoning_trace=reasoning_trace, forecast_payload=forecast_payload,
    )
```

with:

```python
    raw, reasoning_trace, forecast_payload = recommendations.unwrap_agent_result(agent_result)
    degraded = validate_and_degrade(raw, fixture.home_team, fixture.away_team)
    cache_write_ok = run_isolated(
        "t30", f"match_id={fixture.match_id}",
        lambda: cache.record_generation(
            match_id=fixture.match_id, date=date_str, agent_config_hash=agent_config_hash,
            odds=fresh_odds, recommendation=degraded.model_dump(), triggered_by="scheduled",
            reasoning_trace=reasoning_trace, forecast_payload=forecast_payload,
        ) or True,
        league=league,
    )
    if cache_write_ok is None:
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_error")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest app/backend/tests/test_t30_refresh.py -k isolates_a_failing_cache_write -v`
Expected: PASS

- [ ] **Step 5: Run the full T-30 test file to confirm no regressions**

Run: `python -m pytest app/backend/tests/test_t30_refresh.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/backend/t30_refresh.py app/backend/tests/test_t30_refresh.py
git commit -m "fix: isolate T-30's cache.record_generation call with run_isolated"
```

---

### Task 5: Close the EOD job's unguarded push-notify gap

**Files:**
- Modify: `app/backend/scheduler_wiring.py:783-790` (confirm exact current line numbers before editing — line numbers drift)
- Test: `app/backend/tests/test_scheduler_wiring.py`

Confirmed exact current code (read via `sed -n '783,790p' app/backend/scheduler_wiring.py`):

```python
        if push_store is not None:
            send_to_all_subscribers(
                store=push_store, title="New recommendations ready",
                body=f"Tomorrow's picks ({date_str}) are in.", url="/",
            )
            for cluster in cluster_fixtures_by_kickoff(all_fixtures):
                _schedule_t30_batch_notify(scheduler, push_store, cluster, date_str)
```

This runs *after* the `for league in COMPETITIONS:` loop, spanning every competition's fixtures together (`all_fixtures`) — neither call is scoped to one league, so both are recorded with `league=None` (same convention as a shared-builder failure).

- [ ] **Step 1: Write the failing test**

Check the existing test pattern first:

Run: `grep -n "^def test.*_eod_job\|register_eod_job\|send_to_all_subscribers" app/backend/tests/test_scheduler_wiring.py | head -20`

Add a test matching this file's existing `_eod_job`-testing style (reuse its existing fixture/mock-building helpers for `fixtures_client`/`odds_client`/`cache`/`config`/`push_store`):

```python
def test_eod_job_push_notify_failure_does_not_crash_the_job(monkeypatch) -> None:
    """2026-11-02: send_to_all_subscribers (and the T-30 batch-notify loop
    right after it) ran after the whole per-competition loop with no
    try/except of its own -- a failure there (bad subscription, network
    error) used to fail the whole _eod_job, which only the scheduler's own
    outer catch (scheduler.py's _run_and_mark) caught, marking a run that
    may have fully succeeded as a failed one."""
    with patch("app.backend.scheduler_wiring.send_to_all_subscribers", side_effect=RuntimeError("push service down")):
        # invoke the same _eod_job-building helper this file's other tests
        # use, with a non-None push_store so the push-notify branch runs
        ...  # assert the job completes without raising, and that generation
             # for every competition still happened (same assertions this
             # file's other _eod_job tests already make on `cache`/`result`)
```

(This file already has an established pattern for constructing and invoking `_eod_job` in its other tests — write the test using that exact pattern, with a `push_store` that isn't `None` so the branch under test actually runs.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k push_notify_failure -v`
Expected: FAIL — the `RuntimeError` propagates out of `_eod_job`

- [ ] **Step 3: Wrap both calls in `run_isolated`**

Add the import to `app/backend/scheduler_wiring.py`:

```python
from app.backend.run_isolated import run_isolated
```

Replace the block from above with:

```python
        if push_store is not None:
            run_isolated(
                "eod_job", "push_notify:new_recommendations",
                lambda: send_to_all_subscribers(
                    store=push_store, title="New recommendations ready",
                    body=f"Tomorrow's picks ({date_str}) are in.", url="/",
                ) or True,
            )
            for cluster in cluster_fixtures_by_kickoff(all_fixtures):
                run_isolated(
                    "eod_job", "push_notify:t30_batch",
                    lambda c=cluster: _schedule_t30_batch_notify(scheduler, push_store, c, date_str) or True,
                )
```

(`cluster=cluster` default-argument capture avoids the classic late-binding closure bug — without it, every lambda in the loop would close over the same `cluster` variable and all fire with its final value.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k push_notify_failure -v`
Expected: PASS

- [ ] **Step 5: Run the full scheduler_wiring test file to confirm no regressions**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "fix: isolate EOD job's push-notify calls with run_isolated"
```

---

### Task 6: Close the fixture-reconciliation job's `_schedule_t30_batch_notify` gap

**Files:**
- Modify: `app/backend/scheduler_wiring.py:1030-1032` (confirm exact current line numbers before editing)
- Test: `app/backend/tests/test_scheduler_wiring.py`

A related but *not identical* gap to Task 5's: this one lives inside the reconciliation job's own `for league in COMPETITIONS:` / per-date loop, so — unlike the EOD job's cross-league notify block — `league` genuinely is in scope here.

Confirmed exact current code (read via `sed -n '1025,1033p' app/backend/scheduler_wiring.py`):

```python
                if push_store is not None:
                    for cluster in cluster_fixtures_by_kickoff(date_fixtures):
                        _schedule_t30_batch_notify(scheduler, push_store, cluster, date_str)
```

- [ ] **Step 1: Write the failing test**

Find this file's existing `_reconcile_job`/`register_fixture_reconciliation_job` test-construction helper first:

Run: `grep -n "^def test.*reconcil\|register_fixture_reconciliation_job" app/backend/tests/test_scheduler_wiring.py | head -20`

```python
def test_reconcile_job_batch_notify_failure_does_not_crash_the_job(monkeypatch) -> None:
    """2026-11-02: _schedule_t30_batch_notify had no try/except of its own
    inside the reconciliation job's per-league/per-date loop -- a failure
    here (bad subscription, network error) used to abort reconciliation for
    every date/league processed after the failing cluster, not just that
    one cluster's own notify."""
    with patch("app.backend.scheduler_wiring._schedule_t30_batch_notify", side_effect=RuntimeError("push service down")):
        # invoke the same _reconcile_job-building helper this file's other
        # tests use, with a non-None push_store so the branch under test runs
        ...  # assert the job completes without raising, and that T-30
             # scheduling for fixtures in other clusters/dates still happened
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k reconcile_job_batch_notify_failure -v`
Expected: FAIL

- [ ] **Step 3: Wrap the call in `run_isolated`**

Replace the block from above with:

```python
                if push_store is not None:
                    for cluster in cluster_fixtures_by_kickoff(date_fixtures):
                        run_isolated(
                            "fixture_reconciliation", f"push_notify:date={date_str}",
                            lambda c=cluster: _schedule_t30_batch_notify(scheduler, push_store, c, date_str) or True,
                            league=league,
                        )
```

(Same default-argument `c=cluster` capture as Task 5, same late-binding reason.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k reconcile_job_batch_notify_failure -v`
Expected: PASS

- [ ] **Step 5: Run the full scheduler_wiring test file to confirm no regressions**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "fix: isolate fixture-reconciliation job's T-30 batch-notify call with run_isolated"
```

---

### Task 7: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full backend test suite**

Run: `python -m pytest app/backend/tests -q`
Expected: every previously-passing test still passes (same 8 pre-existing, unrelated failures documented in `documents/bugs.md`'s BUG-092 notes are acceptable; no new failures)

- [ ] **Step 2: Run the run_isolated and t30/scheduler_wiring files specifically, for a clean summary**

Run: `python -m pytest app/backend/tests/test_run_isolated.py app/backend/tests/test_t30_refresh.py app/backend/tests/test_scheduler_wiring.py -v`
Expected: all PASS

---

## Self-review notes

- **Spec coverage**: implements §4 and §6 of the design spec (the `run_isolated` primitive and the failures table) in full, and wires it into the two call sites the spec's own table marks as "unguarded" with no return-value ambiguity (T-30's `add_secondary_odds`/`cache.record_generation`, both EOD-job-class push-notify steps). Deliberately does not implement the `eod_batch.py` broad-umbrella-stage retrofit, the circuit breaker (§5), the failures table API/dashboard (§7), or `build_odds_client()`'s defense-in-depth wrap — each is its own follow-up plan or task, scoped out explicitly at the top of this document.
- **Type consistency**: `run_isolated`/`run_isolated_async` signatures (`subsystem: str, unit_id: str, fn: Callable[[], T]`) are used identically across every call site in Tasks 3-6.
