# Sandbox Testing Environment (W27–W31, W37, W38) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an app-wide, overridable "as-of now" (sandbox mode) so whole user journeys can be tested against a chosen historical date instead of the real, off-season wall clock — fixtures, real historical odds, a real agent recommendation, bet logging, and settlement, all isolated from real dev data, with a repeatable runbook proving it end-to-end.

**Architecture:** `SANDBOX_MODE=1`/`SANDBOX_DATE=YYYY-MM-DD` are two env vars, both absent by default so normal operation is byte-for-byte unchanged. A single `app/backend/sandbox_clock.py` module is the one source of truth for "is sandbox mode on" and "what date"; every other piece (a historical odds client sourced from `raw_matches`, sandbox-scoped cache/bet-tracker/job-run-log db paths, the frontend's date-window logic, the agent's snapshot record/replay) reads from it rather than rolling its own env-var checks. Dependency order (per `app_user_stories.md` Phase 7): **W27 + W28 (independent) → W29 → W30 → W37/W38 → W31 (exercises everything together)**.

**Tech Stack:** Python/FastAPI backend (`app/backend`), Next.js/React/TypeScript frontend (`app/frontend`), DuckDB (`raw_matches`), SQLite (cache/bet-tracker/job-run-log stores), pytest, vitest.

---

## File Structure

- Create: `app/backend/sandbox_clock.py` — `is_sandbox_mode()`, `sandbox_date()`, `sandbox_now()`, `sandbox_status()` (W27).
- Test: `app/backend/tests/test_sandbox_clock.py` (W27).
- Modify: `app/backend/main.py` — new `GET /api/sandbox/status` endpoint (W27); sandbox-scoped `JobRunLog` in `lifespan` (W29).
- Test: `app/backend/tests/test_sandbox_status_endpoint.py` (W27).
- Modify: `app/backend/scheduler_wiring.py` — `next_day_date_str`/`register_eod_job` default clock routed through `sandbox_now` (W27); `build_odds_client()` returns `HistoricalOddsClient` in sandbox mode (W29).
- Create: `app/backend/historical_odds_client.py` — `HistoricalOddsClient`, same `get_odds()` shape as `OddsAPIClient` (W28).
- Test: `app/backend/tests/test_historical_odds_client.py` (W28).
- Modify: `app/backend/recommendations.py` — sandbox-scoped cache path (W29); `run_agent()` wrapper routing through `SnapshotStore` record/replay in sandbox mode (W37).
- Test: `app/backend/tests/test_sandbox_agent_snapshot.py` (W37).
- Modify: `app/backend/bets.py` — sandbox-scoped bet-tracker path (W29).
- Test: `app/backend/tests/test_sandbox_wiring.py` (W29).
- Modify: `src/agent/tools.py` — `configure_snapshot_store()` gains a `base_dir` param (W37).
- Modify: `src/agent/snapshot_store.py` — public `DEFAULT_BASE_DIR` alias + `base_dir` property (W37).
- Create: `app/frontend/lib/useSandboxAsOf.ts` — shared hook resolving "today" from the sandbox clock (W30).
- Test: `app/frontend/lib/useSandboxAsOf.test.ts` (W30).
- Modify: `app/frontend/lib/api.ts`, `app/frontend/lib/types.ts` — `getSandboxStatus()`, `SandboxStatus` type (W30).
- Modify: `app/frontend/components/MatchUI.tsx`, `app/frontend/components/BetTracker.tsx` — replace `new Date()` date-window calls with `useSandboxAsOf()` (W30).
- Test: `app/frontend/components/MatchUI.dateboundary.test.tsx` (W38).
- Create: `scripts/sandbox_runbook.py`, `scripts/__init__.py` — the repeatable, runnable scenario script (W31).
- Test: `scripts/test_sandbox_runbook.py` (W31).
- Create: `documents/sandbox_testing_runbook.md` — recorded results of a real run (W31).

---

## Task 1: W27 — Sandbox clock

**Files:**
- Create: `app/backend/sandbox_clock.py`
- Test: `app/backend/tests/test_sandbox_clock.py`
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_sandbox_status_endpoint.py`
- Modify: `app/backend/scheduler_wiring.py`
- Modify: `app/backend/tests/test_scheduler_wiring.py`

**Context:** No sandbox code exists anywhere in the repo yet (confirmed via grep). `SANDBOX_MODE`/`SANDBOX_DATE` must both be absent by default so every existing test/behavior is byte-for-byte unchanged. `app/backend/scheduler_wiring.py:62`'s `next_day_date_str()` is the story's own named example of a "today"-dependent call site to wire through this.

### Part A: the module itself

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_sandbox_clock.py`:

```python
"""W27: sandbox clock -- an app-wide, overridable "as-of now", driven by two
env vars (SANDBOX_MODE=1, SANDBOX_DATE=YYYY-MM-DD), both absent by default
so normal operation is completely unaffected."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.sandbox_clock import is_sandbox_mode, sandbox_date, sandbox_now, sandbox_status


def test_sandbox_mode_defaults_to_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)
    assert is_sandbox_mode() is False
    assert sandbox_date() is None


def test_sandbox_mode_off_ignores_a_stray_sandbox_date(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert sandbox_date() is None  # gated on SANDBOX_MODE, not just SANDBOX_DATE's presence


def test_sandbox_mode_on_with_date_returns_the_override_date(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert is_sandbox_mode() is True
    assert sandbox_date().isoformat() == "2026-03-01"


def test_sandbox_now_returns_real_clock_when_sandbox_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    before = datetime.now(timezone.utc)
    result = sandbox_now(timezone.utc)
    after = datetime.now(timezone.utc)
    assert before <= result <= after


def test_sandbox_now_returns_the_override_date_at_midnight_when_active(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    result = sandbox_now(timezone.utc)
    assert result == datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_sandbox_status_reflects_active_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert sandbox_status() == {"sandbox_mode": True, "as_of": "2026-03-01"}


def test_sandbox_status_reflects_inactive_sandbox(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)
    assert sandbox_status() == {"sandbox_mode": False, "as_of": None}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_sandbox_clock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.sandbox_clock'`

- [ ] **Step 3: Implement the module**

Create `app/backend/sandbox_clock.py`:

```python
"""W27: sandbox clock -- an app-wide, overridable "as-of now". Driven by two
env vars, SANDBOX_MODE=1 and SANDBOX_DATE=YYYY-MM-DD, both absent by default
so normal operation is completely unaffected (purely additive, not a new
mode every code path must branch on defensively). Not a literal container
(Docker etc.) -- "sandbox" here means an isolated *configuration and data*
mode within the existing app.

Every backend call site that currently computes "today"/"now" directly for
date-window purposes should route through sandbox_now()/is_sandbox_mode()
instead of a bare datetime.now()."""

from __future__ import annotations

from datetime import date, datetime, tzinfo
import os


def is_sandbox_mode() -> bool:
    return os.environ.get("SANDBOX_MODE") == "1"


def sandbox_date() -> date | None:
    """The active override date, or None if sandbox mode is off or no
    SANDBOX_DATE is set."""
    if not is_sandbox_mode():
        return None
    raw = os.environ.get("SANDBOX_DATE")
    if not raw:
        return None
    return date.fromisoformat(raw)


def sandbox_now(tz: tzinfo | None = None) -> datetime:
    """Real wall-clock 'now' (in the given tz, if any) unless sandbox mode
    is active with a SANDBOX_DATE set, in which case it returns that date
    at midnight in the given tz -- a stand-in "as-of" instant for
    date-window computations, not a literal simulated clock-tick."""
    override = sandbox_date()
    if override is not None:
        return datetime(override.year, override.month, override.day, tzinfo=tz)
    return datetime.now(tz)


def sandbox_status() -> dict:
    override = sandbox_date()
    return {"sandbox_mode": is_sandbox_mode(), "as_of": override.isoformat() if override else None}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_sandbox_clock.py -v`
Expected: 7 passed.

### Part B: `GET /api/sandbox/status`

- [ ] **Step 5: Write the failing endpoint test**

Create `app/backend/tests/test_sandbox_status_endpoint.py`:

```python
"""W27: GET /api/sandbox/status, so the frontend (W30) and test scripts
(W31) can introspect the active sandbox date rather than each needing their
own access to the env vars."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.main import app


def test_sandbox_status_reports_inactive_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/sandbox/status")
    assert response.status_code == 200
    assert response.json() == {"sandbox_mode": False, "as_of": None}


def test_sandbox_status_reports_the_active_override_date(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with TestClient(app) as client:
        response = client.get("/api/sandbox/status")
    assert response.status_code == 200
    assert response.json() == {"sandbox_mode": True, "as_of": "2026-03-01"}
```

- [ ] **Step 6: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_sandbox_status_endpoint.py -v`
Expected: FAIL — 404 (`/api/sandbox/status` doesn't exist yet).

- [ ] **Step 7: Wire the endpoint into `app/backend/main.py`**

Add the import after line 19 (`from app.backend import bets, recommendations`):

```python
from app.backend import bets, recommendations, sandbox_clock
```

Add the endpoint after `health()` (after line 103):

```python
@app.get("/api/sandbox/status")
def get_sandbox_status() -> dict:
    """W27: lets the frontend and test scripts introspect the active
    sandbox date instead of each needing their own access to the env vars."""
    return sandbox_clock.sandbox_status()
```

- [ ] **Step 8: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_sandbox_status_endpoint.py -v`
Expected: 2 passed.

### Part C: wire one real "today"-dependent site

- [ ] **Step 9: Write the failing test for `next_day_date_str`**

Append to `app/backend/tests/test_scheduler_wiring.py` (after the last test in the file):

```python
def test_next_day_date_str_respects_sandbox_override(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert next_day_date_str() == "2026-03-02"


def test_next_day_date_str_uses_real_clock_when_sandbox_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    real_tomorrow = (datetime.now(NY_TZ) + timedelta(days=1)).date().isoformat()
    assert next_day_date_str() == real_tomorrow
```

- [ ] **Step 10: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v -k sandbox`
Expected: FAIL — `test_next_day_date_str_respects_sandbox_override` computes tomorrow from the real clock, not `2026-03-02`.

- [ ] **Step 11: Route `next_day_date_str`/`register_eod_job` through `sandbox_now`**

Edit `app/backend/scheduler_wiring.py`. Add the import after line 19 (`from app.backend.scheduler import NY_TZ, RecoverableScheduler`):

```python
from app.backend.sandbox_clock import sandbox_now
```

Change line 62:

```python
def next_day_date_str(now_fn: Callable[[], datetime] = lambda: sandbox_now(NY_TZ)) -> str:
```

Change line 95 (in `register_eod_job`'s signature):

```python
def register_eod_job(
    scheduler: RecoverableScheduler,
    fixtures_client: FootballDataClient,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    now_fn: Callable[[], datetime] = lambda: sandbox_now(NY_TZ),
) -> None:
```

- [ ] **Step 12: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: all passed, including the 2 new tests.

- [ ] **Step 13: Run the full backend suite to confirm zero regressions**

Run: `python -m pytest app/backend/tests/ -q`
Expected: same pass count as before this task plus the new tests, zero failures — every existing test runs with `SANDBOX_MODE` unset, so `sandbox_now(NY_TZ)` behaves identically to the old `datetime.now(NY_TZ)`.

- [ ] **Step 14: Commit**

```bash
git add app/backend/sandbox_clock.py app/backend/tests/test_sandbox_clock.py app/backend/main.py app/backend/tests/test_sandbox_status_endpoint.py app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat: add the sandbox clock, GET /api/sandbox/status, and wire next_day_date_str through it (W27)"
```

---

## Task 2: W28 — Historical odds source

**Files:**
- Create: `app/backend/historical_odds_client.py`
- Test: `app/backend/tests/test_historical_odds_client.py`

**Context:** The Odds API (W07) is live-current-odds-only with no historical replay. `raw_matches` already carries real `odds_h`/`odds_d`/`odds_a` from football-data.co.uk (3,800 rows, 2016-08-13 through 2026-05-24, confirmed live). `HistoricalOddsClient` needs the exact same `get_odds() -> list[NormalizedOdds] | None` shape `OddsAPIClient` does, so `eod_batch.py`/`t30_refresh.py` need zero changes to consume it, and team names must flow through in `raw_matches`'s own canonical form so BUG-015's existing `match_odds()`/`odds_lookup()` resolves them unmodified.

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_historical_odds_client.py`:

```python
"""W28: sandbox odds source -- real historical 1X2 odds from raw_matches
(football-data.co.uk), not synthetic. Seeds a real temp DuckDB with the
same raw_matches columns production uses, so this exercises a real DuckDB
query end-to-end rather than a mocked connection. Deliberately excludes
odds movement (a single closing-line snapshot, not a time series) -- see
W32."""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import yaml

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.historical_odds_client import HistoricalOddsClient


def _seed_db(tmp_path: Path):
    from src.utils.db_manager import DuckDBManager

    db_path = tmp_path / "sandbox.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}))

    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE raw_matches (
            match_id TEXT, league TEXT, date TIMESTAMP,
            home_team TEXT, away_team TEXT,
            odds_h FLOAT, odds_d FLOAT, odds_a FLOAT
        )
        """
    )
    conn.execute(
        "INSERT INTO raw_matches VALUES "
        "('1', 'E0', '2026-03-01', 'Arsenal', 'Everton', 1.80, 3.60, 4.20), "
        "('2', 'E0', '2026-03-01', 'Chelsea', 'Fulham', 1.50, 4.00, 6.50), "
        "('3', 'E0', '2026-03-02', 'Liverpool', 'Burnley', 1.30, 5.50, 9.00)"  # different date -- must be excluded
    )
    conn.close()
    return DuckDBManager(config_path=str(config_path))


def test_get_odds_returns_real_odds_for_the_sandbox_date(tmp_path: Path) -> None:
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds()

    assert result is not None
    assert len(result) == 2
    by_home = {odds.home_team: odds for odds in result}
    assert by_home["Arsenal"].home_odds == 1.80
    assert by_home["Arsenal"].draw_odds == 3.60
    assert by_home["Arsenal"].away_odds == 4.20
    assert by_home["Chelsea"].away_team == "Fulham"


def test_get_odds_excludes_fixtures_on_other_dates(tmp_path: Path) -> None:
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)

    result = client.get_odds()

    assert all(odds.home_team != "Liverpool" for odds in result)


def test_get_odds_returns_none_when_no_fixtures_that_date(tmp_path: Path) -> None:
    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2099-01-01", db_manager=manager)

    assert client.get_odds() is None


def test_get_odds_matches_the_normalizedodds_shape_odds_api_client_uses() -> None:
    import dataclasses

    from app.backend.odds_api_client import NormalizedOdds

    assert {f.name for f in dataclasses.fields(NormalizedOdds)} == {
        "home_team", "away_team", "commence_time", "home_odds", "draw_odds", "away_odds",
    }


def test_odds_lookup_and_match_odds_resolve_historical_client_output_unmodified(tmp_path: Path) -> None:
    """W28's acceptance: a real football-data.org fixture for the sandbox
    date must successfully match to HistoricalOddsClient's output via the
    existing, unmodified odds_lookup()/match_odds() (BUG-015's team-name
    resolution) -- zero changes needed to either function."""
    from app.backend.eod_batch import match_odds, odds_lookup
    from app.backend.football_data_client import NormalizedMatch

    manager = _seed_db(tmp_path)
    client = HistoricalOddsClient(sandbox_date="2026-03-01", db_manager=manager)
    odds_events = client.get_odds()
    odds_by_teams = odds_lookup(odds_events)

    fixture = NormalizedMatch(
        match_id="1", utc_date="2026-03-01T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )

    matched = match_odds(fixture, odds_by_teams)

    assert matched == {"home": 1.80, "draw": 3.60, "away": 4.20}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_historical_odds_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.historical_odds_client'`

- [ ] **Step 3: Implement `HistoricalOddsClient`**

Create `app/backend/historical_odds_client.py`:

```python
"""W28: sandbox odds source -- real historical 1X2 odds, not synthetic. The
Odds API (W07) is a live-current-odds-only service with no historical
replay -- a past sandbox date needs a different real source for odds.
raw_matches already carries real odds_h/odds_d/odds_a (football-data.co.uk),
covering 2016-08-13 through the table's last refresh.

Implements the exact same get_odds() -> list[NormalizedOdds] | None shape
OddsAPIClient does, so eod_batch.py/t30_refresh.py need zero changes to
consume it -- same interface, different backing data. Team names come
through in raw_matches's own canonical form, so BUG-015's existing
TeamNameMapper-based matching in odds_lookup()/match_odds() resolves them
unmodified. Deliberately excludes odds movement -- a single closing-line
snapshot per match, not a time series (see W32)."""

from __future__ import annotations

from app.backend.odds_api_client import NormalizedOdds
from src.utils.db_manager import DuckDBManager


class HistoricalOddsClient:
    def __init__(self, sandbox_date: str, db_manager: DuckDBManager | None = None) -> None:
        self._sandbox_date = sandbox_date
        self._db_manager = db_manager or DuckDBManager()

    def get_odds(self, sport_key: str = "soccer_epl") -> list[NormalizedOdds] | None:
        with self._db_manager.connection(read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT home_team, away_team, date, odds_h, odds_d, odds_a
                FROM raw_matches
                WHERE league = 'E0' AND CAST(date AS DATE) = CAST(? AS DATE)
                """,
                (self._sandbox_date,),
            ).fetchall()

        if not rows:
            return None

        return [
            NormalizedOdds(
                home_team=home_team,
                away_team=away_team,
                commence_time=match_date.isoformat() if hasattr(match_date, "isoformat") else str(match_date),
                home_odds=odds_h,
                draw_odds=odds_d,
                away_odds=odds_a,
            )
            for home_team, away_team, match_date, odds_h, odds_d, odds_a in rows
        ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_historical_odds_client.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the full backend suite to confirm no regressions**

Run: `python -m pytest app/backend/tests/ -q`
Expected: same pass count plus 5 new, zero failures.

- [ ] **Step 6: Commit**

```bash
git add app/backend/historical_odds_client.py app/backend/tests/test_historical_odds_client.py
git commit -m "feat: add HistoricalOddsClient, a real-historical-odds source for sandbox mode (W28)"
```

---

## Task 3: W29 — Wire sandbox mode into the backend

**Files:**
- Modify: `app/backend/scheduler_wiring.py` (`build_odds_client`)
- Modify: `app/backend/recommendations.py` (`get_cache`)
- Modify: `app/backend/bets.py` (`get_bet_tracker`)
- Modify: `app/backend/main.py` (`lifespan`'s scheduler construction)
- Test: `app/backend/tests/test_sandbox_wiring.py`

**Context:** `get_fixtures_client()` already needs no change — it keeps using the real `FootballDataClient`; only the date range it's queried with changes, and that's already sourced from W27's `sandbox_now()` wherever `next_day_date_str()`'s default is used. What's left: (1) `build_odds_client()` must return W28's `HistoricalOddsClient` in sandbox mode; (2) the `RecommendationCache`/`BetTracker`/`JobRunLog` singletons must point at `app/data/sandbox/*.db` in sandbox mode so sandbox runs never touch real dev data. `app/data/` is already fully gitignored (`.gitignore:48`), so the new `app/data/sandbox/` subdirectory needs no gitignore changes.

- [ ] **Step 1: Write the failing wiring tests**

Create `app/backend/tests/test_sandbox_wiring.py`:

```python
"""W29: wiring sandbox mode into fixtures/odds/cache/bet-tracker
construction. Each seam is tested in isolation via monkeypatched env vars
and singleton resets -- the full real end-to-end proof (real agent call,
real fixtures/odds, zero effect on real dbs) is W31's runbook, not
re-proven here with a real LLM call."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import bets, recommendations
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.scheduler_wiring import build_odds_client


def _reset_singletons() -> None:
    recommendations._cache_singleton = None
    bets._bet_tracker_singleton = None


def test_get_cache_uses_sandbox_scoped_db_path_when_sandbox_active(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    cache = recommendations.get_cache()

    assert "sandbox" in str(cache._db_path)
    _reset_singletons()


def test_get_cache_uses_real_db_path_when_sandbox_inactive(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.delenv("SANDBOX_MODE", raising=False)

    cache = recommendations.get_cache()

    assert "sandbox" not in str(cache._db_path)
    _reset_singletons()


def test_get_bet_tracker_uses_sandbox_scoped_db_path_when_sandbox_active(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    tracker = bets.get_bet_tracker()

    assert "sandbox" in str(tracker._db_path)
    _reset_singletons()


def test_get_bet_tracker_uses_real_db_path_when_sandbox_inactive(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.delenv("SANDBOX_MODE", raising=False)

    tracker = bets.get_bet_tracker()

    assert "sandbox" not in str(tracker._db_path)
    _reset_singletons()


def test_build_odds_client_returns_historical_client_when_sandbox_active(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    client = build_odds_client()

    assert isinstance(client, HistoricalOddsClient)


def test_build_odds_client_returns_live_client_when_sandbox_inactive(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "fake-key")

    client = build_odds_client()

    assert not isinstance(client, HistoricalOddsClient)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_sandbox_wiring.py -v`
Expected: FAIL — `get_cache()`/`get_bet_tracker()` always return real-path instances today; `build_odds_client()` never returns a `HistoricalOddsClient`.

- [ ] **Step 3: Wire `build_odds_client()`**

Edit `app/backend/scheduler_wiring.py`. Add imports after the `sandbox_now` import added in Task 1 Step 11:

```python
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.sandbox_clock import sandbox_date, sandbox_now
```

Replace `build_odds_client()` (originally lines 51-59):

```python
def build_odds_client() -> OddsAPIClient | HistoricalOddsClient | None:
    """Returns W28's HistoricalOddsClient when sandbox mode is active with a
    SANDBOX_DATE set (a real historical odds source, since The Odds API is
    live-current-odds-only); otherwise the real, live OddsAPIClient -- None
    if no ODDS_API_KEY is configured."""
    override_date = sandbox_date()
    if override_date is not None:
        return HistoricalOddsClient(sandbox_date=override_date.isoformat())

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return None
    store = FileCreditCounterStore(CREDIT_COUNTER_PATH)
    counter = store.load()
    return PersistingOddsClient(client=OddsAPIClient(api_key=api_key, credit_counter=counter), counter=counter, store=store)
```

- [ ] **Step 4: Wire `get_cache()` in `app/backend/recommendations.py`**

Add imports at the top (after `from __future__ import annotations`):

```python
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.backend.match_info import gate_league
from app.backend.recommendation_cache import RecommendationCache
from app.backend.sandbox_clock import is_sandbox_mode
from src.agent.graph import run_agent
```

Add the path constant and update `get_cache()`:

```python
_cache_singleton: RecommendationCache | None = None
_SANDBOX_CACHE_DB_PATH = Path(__file__).parent.parent / "data" / "sandbox" / "recommendation_cache.db"


def get_cache() -> RecommendationCache:
    """FastAPI dependency -- overridden in tests via app.dependency_overrides.
    Sandbox mode (W29) points this at a scratch db path so sandbox runs
    never touch real dev data."""
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = (
            RecommendationCache(db_path=_SANDBOX_CACHE_DB_PATH) if is_sandbox_mode() else RecommendationCache()
        )
    return _cache_singleton
```

- [ ] **Step 5: Wire `get_bet_tracker()` in `app/backend/bets.py`**

Add imports at the top (after `from __future__ import annotations`):

```python
from pathlib import Path

from pydantic import BaseModel, Field

from app.backend.bet_tracker import Bet, BetTracker
from app.backend.sandbox_clock import is_sandbox_mode
```

Add the path constant and update `get_bet_tracker()`:

```python
_bet_tracker_singleton: BetTracker | None = None
_SANDBOX_BET_TRACKER_DB_PATH = Path(__file__).parent.parent / "data" / "sandbox" / "user_bets.db"


def get_bet_tracker() -> BetTracker:
    """FastAPI dependency -- overridden in tests via app.dependency_overrides.
    Sandbox mode (W29) points this at a scratch db path so sandbox runs
    never touch real dev data."""
    global _bet_tracker_singleton
    if _bet_tracker_singleton is None:
        _bet_tracker_singleton = (
            BetTracker(db_path=_SANDBOX_BET_TRACKER_DB_PATH) if is_sandbox_mode() else BetTracker()
        )
    return _bet_tracker_singleton
```

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_sandbox_wiring.py -v`
Expected: 6 passed.

### Part B: sandbox-scoped `JobRunLog` in the real scheduler wiring

- [ ] **Step 7: Update `app/backend/main.py`'s `lifespan`**

Add `Path` and `JobRunLog` imports. Change line 9-15 area (add `from pathlib import Path` near the top, after `from contextlib import asynccontextmanager`):

```python
from contextlib import asynccontextmanager
from pathlib import Path
import os
```

Change line 28 (`from app.backend.scheduler import RecoverableScheduler`):

```python
from app.backend.scheduler import JobRunLog, RecoverableScheduler
```

Add a path constant near the top-level constants (after `LOGGER = get_logger(__name__)`):

```python
_SANDBOX_JOB_RUNS_DB_PATH = Path(__file__).parent / "data" / "sandbox" / "job_runs.db"
```

Change the scheduler construction inside `lifespan` (originally line 70):

```python
    scheduler: RecoverableScheduler | None = None
    if os.environ.get("ENABLE_SCHEDULER", "").lower() in ("1", "true", "yes"):
        run_log = JobRunLog(db_path=_SANDBOX_JOB_RUNS_DB_PATH) if sandbox_clock.is_sandbox_mode() else None
        scheduler = RecoverableScheduler(run_log=run_log)
        register_eod_job(
            scheduler,
            fixtures_client=get_fixtures_client(),
            odds_client=build_odds_client(),
            cache=recommendations.get_cache(),
            config=config,
        )
        scheduler.start()
        LOGGER.info("W08/W09/W10 scheduler started (ENABLE_SCHEDULER=1).")
```

- [ ] **Step 8: Run the full backend suite to confirm no regressions**

Run: `python -m pytest app/backend/tests/ -q`
Expected: same pass count as before this task plus the 6 new tests, zero failures. Confirm in particular that `test_status_endpoint.py`, `test_recommendation_cache_endpoints.py`, and `test_bets_endpoints.py` still pass unchanged (they construct `RecommendationCache`/`BetTracker` directly or via `app.dependency_overrides`, bypassing the singletons — unaffected by this change).

- [ ] **Step 9: Manual live verification (real dependencies, not automated)**

This step requires real `FOOTBALL_DATA_API_KEY`/LLM credentials and is not part of the automated suite — run it once, note the result in the commit message or a scratch note, and rely on W31's runbook (Task 7) for the permanent, repeatable record:

```bash
SANDBOX_MODE=1 SANDBOX_DATE=2026-05-24 ENABLE_SCHEDULER=0 uvicorn app.backend.main:app --reload &
curl -s http://localhost:8000/api/sandbox/status
curl -s -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Brighton", "away_team": "Manchester United", "date": "2026-05-24"}'
ls -la app/data/sandbox/       # sandbox-scoped db files now exist
ls -la app/data/*.db           # real db files' mtimes unchanged
```

Expected: `/api/sandbox/status` returns `{"sandbox_mode": true, "as_of": "2026-05-24"}`; the recommendation call succeeds using real historical odds for that date; `app/data/sandbox/recommendation_cache.db` is created; the real `app/data/recommendation_cache.db`'s modification time is untouched.

- [ ] **Step 10: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/recommendations.py app/backend/bets.py app/backend/main.py app/backend/tests/test_sandbox_wiring.py
git commit -m "feat: wire sandbox mode into odds-client selection and cache/bet-tracker/job-run-log db paths (W29)"
```

---

## Task 4: W30 — Frontend sandbox-clock awareness

**Files:**
- Create: `app/frontend/lib/useSandboxAsOf.ts`
- Test: `app/frontend/lib/useSandboxAsOf.test.ts`
- Modify: `app/frontend/lib/api.ts`, `app/frontend/lib/types.ts`
- Modify: `app/frontend/components/MatchUI.tsx`, `app/frontend/components/BetTracker.tsx`

**Context:** `new Date()` is called directly at 4 known sites for date-window purposes: `MatchUI.tsx:143` (`formatDay`, relative-day labels — not a fixture query, out of scope here), `MatchUI.tsx:548` (Dashboard's fixture query), `MatchUI.tsx:605-611` (Match Explorer's 90-day window), and `BetTracker.tsx:31-34` (`ManualBetForm`'s fixture search, same 90-day-window pattern as Explorer). `StatusFooter.tsx` does **not** call `new Date()` itself — its staleness display is entirely server-computed, so it needs no frontend change (only the backend `get_data_freshness()` path, out of scope for this story). Each site must fetch W27's `GET /api/sandbox/status` and use its `as_of` date instead of the real `new Date()` when `sandbox_mode` is true, with zero behavior change when it's false.

### Part A: the shared hook

- [ ] **Step 1: Write the failing hook tests**

Create `app/frontend/lib/useSandboxAsOf.test.ts`:

```ts
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useSandboxAsOf } from "./useSandboxAsOf";
import { getSandboxStatus } from "./api";

vi.mock("./api", () => ({ getSandboxStatus: vi.fn() }));

describe("useSandboxAsOf", () => {
  beforeEach(() => {
    vi.mocked(getSandboxStatus).mockReset();
  });

  it("stays on the real clock when sandbox mode is inactive", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    const before = new Date();

    const { result } = renderHook(() => useSandboxAsOf());

    await waitFor(() => expect(getSandboxStatus).toHaveBeenCalled());
    const after = new Date();
    expect(result.current.getTime()).toBeGreaterThanOrEqual(before.getTime());
    expect(result.current.getTime()).toBeLessThanOrEqual(after.getTime());
  });

  it("switches to the sandbox as_of date once fetched", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    const { result } = renderHook(() => useSandboxAsOf());

    await waitFor(() => expect(result.current.toISOString().slice(0, 10)).toBe("2026-03-01"));
  });

  it("falls back to the real clock if the sandbox status call fails", async () => {
    vi.mocked(getSandboxStatus).mockRejectedValue(new Error("network error"));

    const { result } = renderHook(() => useSandboxAsOf());

    await waitFor(() => expect(getSandboxStatus).toHaveBeenCalled());
    expect(result.current).toBeInstanceOf(Date);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd app/frontend && npx vitest run lib/useSandboxAsOf.test.ts`
Expected: FAIL — `useSandboxAsOf.ts` doesn't exist yet.

- [ ] **Step 3: Add `getSandboxStatus()` and the `SandboxStatus` type**

Edit `app/frontend/lib/types.ts` — add near `StatusResponse`:

```ts
// W27: reflects the backend's sandbox_clock module -- as_of is null when
// sandbox_mode is false.
export type SandboxStatus = {
  sandbox_mode: boolean;
  as_of: string | null;
};
```

Edit `app/frontend/lib/api.ts` — update the type import at the top:

```ts
import type { Bet, BetStats, Fixture, MatchRecommendationOut, SandboxStatus, StatusResponse } from "./types";
```

Add near `getStatus()`:

```ts
/** W27: introspects whether sandbox mode is active and, if so, the as-of date. */
export async function getSandboxStatus(): Promise<SandboxStatus> {
  const response = await fetch(`${API_BASE}/api/sandbox/status`);
  if (!response.ok) {
    throw new ApiError(`Failed to load sandbox status (${response.status})`, response.status);
  }
  return response.json();
}
```

- [ ] **Step 4: Implement the hook**

Create `app/frontend/lib/useSandboxAsOf.ts`:

```ts
import { useEffect, useState } from "react";
import { getSandboxStatus } from "./api";

/**
 * W30: resolves "today" for date-window purposes -- the sandbox as_of date
 * when sandbox mode is active, the real browser Date() otherwise. Fetches
 * W27's /api/sandbox/status once per mount. Works for both a human clicking
 * through the real UI and Playwright-driven automated checks, since both
 * just read this hook's returned Date the same way.
 */
export function useSandboxAsOf(): Date {
  const [asOf, setAsOf] = useState<Date>(() => new Date());

  useEffect(() => {
    let cancelled = false;
    getSandboxStatus()
      .then((status) => {
        if (!cancelled && status.sandbox_mode && status.as_of) {
          setAsOf(new Date(`${status.as_of}T00:00:00`));
        }
      })
      .catch(() => {
        // sandbox status endpoint unreachable/erroring -- fall back to the
        // real browser clock rather than blocking the page.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return asOf;
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd app/frontend && npx vitest run lib/useSandboxAsOf.test.ts`
Expected: 3 passed.

### Part B: wire the hook into Dashboard, Match Explorer, and the manual bet form

- [ ] **Step 6: Read the enclosing effects before editing**

Read `app/frontend/components/MatchUI.tsx` in full (or at minimum lines 520-620) to find `DashboardPage`'s and `MatchExplorerPage`'s `load()` functions and the `useEffect` (or equivalent) that calls them, so the new `asOf` value can be added to the correct dependency array. Read `app/frontend/components/BetTracker.tsx` in full (or at minimum lines 1-60) for the same reason around `ManualBetForm`'s fixture search.

- [ ] **Step 7: Wire `DashboardPage`**

In `app/frontend/components/MatchUI.tsx`, add the hook call at the top of `DashboardPage`'s component body:

```tsx
const asOf = useSandboxAsOf();
```

Replace the fixture-query line (originally `MatchUI.tsx:548`):

```tsx
const today = asOf.toISOString().slice(0, 10);
const fixtures = await getFixtures(today, today);
```

Add `asOf` to the dependency array of whichever `useEffect`/callback invokes `load()`, so a later async resolution of the sandbox date (after the initial real-`Date()` render) re-triggers the fixture query with the corrected date. Add the import at the top of the file:

```tsx
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
```

- [ ] **Step 8: Wire `MatchExplorerPage`**

In the same file, add the hook call at the top of `MatchExplorerPage`'s component body:

```tsx
const asOf = useSandboxAsOf();
```

Replace the 90-day-window block (originally `MatchUI.tsx:605-611`):

```tsx
const from = new Date(asOf);
const to = new Date(asOf);
to.setDate(to.getDate() + 90);
const fixtures = await getFixtures(from.toISOString().slice(0, 10), to.toISOString().slice(0, 10));
```

Add `asOf` to the dependency array of the effect/callback invoking `load()`, same as Step 7.

- [ ] **Step 9: Wire `ManualBetForm`'s fixture search**

In `app/frontend/components/BetTracker.tsx`, add the hook call at the top of `ManualBetForm`'s component body and replace the 90-day-window block (originally `BetTracker.tsx:31-34`):

```tsx
const asOf = useSandboxAsOf();
...
const from = new Date(asOf);
const to = new Date(asOf);
to.setDate(to.getDate() + 90);
getFixtures(from.toISOString().slice(0, 10), to.toISOString().slice(0, 10))...
```

Add the import at the top of the file:

```tsx
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
```

- [ ] **Step 10: Run the full frontend test suite**

Run: `cd app/frontend && npx vitest run`
Expected: all passed, zero regressions. `formatDay`'s own `new Date()` at `MatchUI.tsx:143` is deliberately left untouched (relative-day labels are a display nuance, not a fixture-query date window — not in this story's scope).

- [ ] **Step 11: Manual browser verification**

Start both processes and confirm visually:

```bash
SANDBOX_MODE=1 SANDBOX_DATE=2026-05-24 uvicorn app.backend.main:app --reload &
cd app/frontend && npm run dev &
```

Open `http://localhost:3000`, confirm the Dashboard shows 2026-05-24's real fixtures instead of today's real, likely off-season-empty list.

- [ ] **Step 12: Commit**

```bash
git add app/frontend/lib/useSandboxAsOf.ts app/frontend/lib/useSandboxAsOf.test.ts app/frontend/lib/api.ts app/frontend/lib/types.ts app/frontend/components/MatchUI.tsx app/frontend/components/BetTracker.tsx
git commit -m "feat: add useSandboxAsOf and wire Dashboard/Match Explorer/manual bet search through it (W30)"
```

---

## Task 5: W37 — Wire `SnapshotStore` into the sandbox agent-invocation path

**Files:**
- Modify: `src/agent/snapshot_store.py`
- Modify: `src/agent/tools.py`
- Modify: `app/backend/recommendations.py`
- Test: `app/backend/tests/test_sandbox_agent_snapshot.py`

**Context:** The app's real agent-invocation path never calls `configure_snapshot_store()`, so `SnapshotStore.mode` stays at its default, `"live"` — the one mode `web_search`'s `before:<match_date>` leakage filter does **not** cover. Once sandbox mode makes the agent reason about a chosen past date, an unfiltered `web_search` could return the sandboxed match's own real final score. Fix: sandbox mode routes through `configure_snapshot_store("record", ...)` on a sandboxed match's first run and `"replay"` on every subsequent run of the same match, in a namespace separate from the real evaluation corpus (`data/agent_snapshots/sandbox/` vs. the default `data/agent_snapshots/`).

- [ ] **Step 1: Read `src/agent/snapshot_store.py` in full**

Confirm the exact name of the private default-base-dir constant (expected `_DEFAULT_BASE_DIR`, per `src/agent/snapshot_store.py:41`) and the internal attribute `__init__` stores `base_dir` into (expected `self._base_dir`, consistent with this codebase's `self._x` convention used by every other store class — `JobRunLog._db_path`, `CreditCounter._now_fn`, etc.). If either differs from these expectations, use the real names in the steps below instead.

- [ ] **Step 2: Add a public `base_dir` accessor to `SnapshotStore`**

Edit `src/agent/snapshot_store.py`. Directly below the private constant definition (`_DEFAULT_BASE_DIR = Path("data/agent_snapshots")` or equivalent), add a public alias:

```python
DEFAULT_BASE_DIR = _DEFAULT_BASE_DIR
```

Inside the `SnapshotStore` class, add a `base_dir` property (near the existing `mode`/`match_id`/`match_date` properties):

```python
    @property
    def base_dir(self) -> Path:
        return self._base_dir
```

- [ ] **Step 3: Write the failing wrapper tests**

Create `app/backend/tests/test_sandbox_agent_snapshot.py`:

```python
"""W37: wires SnapshotStore record/replay into the sandbox agent-invocation
path. When sandbox mode is active, recommendations.run_agent() records the
first live run of a sandboxed match (date-filtered web_search, per A10) and
replays every subsequent run of the same match (zero live calls) --
otherwise it passes straight through to the real, live run_agent,
unchanged from before this story."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import recommendations


_MATCH_INFO = {"home_team": "Arsenal", "away_team": "Everton", "date": "2026-03-01"}
_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-03-01", "league": "E0"},
    "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
    "limitations": [], "prediction_basis": "market_odds_only",
}


def _reset_sandbox_recording_state() -> None:
    recommendations._sandbox_recorded_matches.clear()


def test_passes_through_to_the_real_run_agent_when_sandbox_mode_is_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION) as mock_run, \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        result = recommendations.run_agent(_MATCH_INFO)

    assert result == _RECOMMENDATION
    mock_run.assert_called_once_with(_MATCH_INFO, config=None)
    mock_configure.assert_not_called()


def test_first_sandboxed_run_of_a_match_uses_record_mode(monkeypatch) -> None:
    _reset_sandbox_recording_state()
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["record", "live"]
    _reset_sandbox_recording_state()


def test_second_sandboxed_run_of_the_same_match_uses_replay_mode(monkeypatch) -> None:
    _reset_sandbox_recording_state()
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)  # first run: records
        mock_configure.reset_mock()
        recommendations.run_agent(_MATCH_INFO)  # second run: must replay

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live"]
    _reset_sandbox_recording_state()


def test_sandboxed_run_uses_the_sandbox_snapshot_namespace(monkeypatch) -> None:
    _reset_sandbox_recording_state()
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    record_call = mock_configure.call_args_list[0]
    assert record_call.kwargs["base_dir"] == recommendations._SANDBOX_SNAPSHOT_BASE_DIR
    _reset_sandbox_recording_state()
```

- [ ] **Step 4: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_sandbox_agent_snapshot.py -v`
Expected: FAIL — `recommendations.run_agent` is currently just the direct `src.agent.graph.run_agent` reference, has no sandbox branching, and `_real_run_agent`/`_sandbox_recorded_matches`/`agent_tools` don't exist in the module yet.

- [ ] **Step 5: Add `base_dir` to `configure_snapshot_store()` in `src/agent/tools.py`**

Edit `src/agent/tools.py`. Add the `Path` import if not already present, and replace `configure_snapshot_store()`:

```python
from pathlib import Path

from src.agent.snapshot_store import DEFAULT_BASE_DIR, SnapshotMode, SnapshotStore

_snapshot_store = SnapshotStore()


def configure_snapshot_store(
    mode: SnapshotMode,
    match_id: str | None = None,
    match_date: str | None = None,
    base_dir: str | Path | None = None,
) -> None:
    """Configure the module-level SnapshotStore shared by all tool functions.
    Call this before run_agent() to switch between live/record/replay. In
    record and replay mode, match_id is required (raises ValueError
    otherwise, from SnapshotStore._path). match_date, if given, is appended
    to web_search queries as 'before:<match_date>' to reduce post-match
    result leakage (A10). base_dir (W37) lets a caller -- the app's sandbox
    agent-invocation path -- point recordings at a separate namespace (e.g.
    data/agent_snapshots/sandbox/) instead of the default corpus; omit it
    to use (or return to) the default."""
    global _snapshot_store
    effective_base_dir = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    if effective_base_dir != _snapshot_store.base_dir:
        _snapshot_store = SnapshotStore(base_dir=effective_base_dir)
    _snapshot_store.set_mode(mode)
    if match_id is not None:
        _snapshot_store.set_match(match_id, match_date)
```

- [ ] **Step 6: Add the sandbox-aware `run_agent` wrapper in `app/backend/recommendations.py`**

Change the import at the top (originally `from src.agent.graph import run_agent`):

```python
from src.agent import tools as agent_tools
from src.agent.graph import run_agent as _real_run_agent
```

Add near the top-level constants (after `_SANDBOX_CACHE_DB_PATH`, added in Task 3):

```python
_SANDBOX_SNAPSHOT_BASE_DIR = Path(__file__).parent.parent.parent / "data" / "agent_snapshots" / "sandbox"
_sandbox_recorded_matches: set[str] = set()


def run_agent(match_info: dict, config=None):
    """W37: routes through SnapshotStore record/replay when sandbox mode is
    active, so a sandboxed match's real web_search calls are date-filtered
    (record) and every subsequent run of the same match makes zero live
    calls at all (replay) -- otherwise passes straight through to the real,
    live run_agent, unchanged from before this story."""
    if not is_sandbox_mode():
        return _real_run_agent(match_info, config=config)

    match_key = f"{match_info.get('home_team')}__{match_info.get('away_team')}__{match_info.get('date')}"
    mode = "replay" if match_key in _sandbox_recorded_matches else "record"
    agent_tools.configure_snapshot_store(
        mode, match_id=match_key, match_date=match_info.get("date"), base_dir=_SANDBOX_SNAPSHOT_BASE_DIR,
    )
    try:
        result = _real_run_agent(match_info, config=config)
    finally:
        agent_tools.configure_snapshot_store("live")
    if mode == "record":
        _sandbox_recorded_matches.add(match_key)
    return result
```

This intentionally leaves `main.py:136` (`recommendations.run_agent`), `eod_batch.py:100`, and `t30_refresh.py:84` completely unchanged — all 3 call `recommendations.run_agent(...)`, a module attribute looked up at call time, so redefining it here transparently gives every caller the sandbox behavior with zero edits to those 3 files.

- [ ] **Step 7: Run to verify it passes**

Run: `python -m pytest app/backend/tests/test_sandbox_agent_snapshot.py -v`
Expected: 4 passed.

- [ ] **Step 8: Run the full backend + agent test suites to confirm no regressions**

Run: `python -m pytest app/backend/tests/ tests/ -q`
Expected: same pass count plus 4 new, zero failures. In particular confirm `src/agent/backtest.py`'s existing `configure_snapshot_store("replay", match_id=...)` calls (no `base_dir` given) still resolve to the default corpus directory — `effective_base_dir` falls back to `DEFAULT_BASE_DIR` when `base_dir` is omitted, matching current behavior exactly.

- [ ] **Step 9: Commit**

```bash
git add src/agent/snapshot_store.py src/agent/tools.py app/backend/recommendations.py app/backend/tests/test_sandbox_agent_snapshot.py
git commit -m "feat: route sandboxed agent runs through SnapshotStore record/replay, isolated from the real corpus (W37)"
```

---

## Task 6: W38 — Frontend date-boundary correctness

**Files:**
- Test: `app/frontend/components/MatchUI.dateboundary.test.tsx`

**Context:** With W30's `useSandboxAsOf()` in place, Dashboard's fixture query and Match Explorer's 90-day window can now be driven to any instant on demand instead of waiting for real wall-clock boundaries. This story is pure test coverage — no new source changes, since W30 already wired the components correctly; these tests prove the boundary behavior via the mocked `getSandboxStatus()`/`getFixtures()` call args.

- [ ] **Step 1: Write the date-boundary tests**

Create `app/frontend/components/MatchUI.dateboundary.test.tsx`:

```tsx
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { DashboardPage, MatchExplorerPage } from "./MatchUI";
import { getFixtures, getSandboxStatus } from "@/lib/api";

vi.mock("@/lib/api");

describe("date-boundary correctness via the sandbox clock (W38)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockResolvedValue([]);
  });

  it("Dashboard queries fixtures for the sandbox as_of date, not the real browser date", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-03-01"));
  });

  it("Dashboard's fixture query shifts to the next simulated day at midnight", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValueOnce({ sandbox_mode: true, as_of: "2026-03-01" });
    const { rerender } = render(<DashboardPage />);
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-03-01"));

    vi.mocked(getSandboxStatus).mockResolvedValueOnce({ sandbox_mode: true, as_of: "2026-03-02" });
    rerender(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-02", "2026-03-02"));
  });

  it("Match Explorer's 90-day window is anchored to the sandbox as_of date, not the real browser date", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    render(<MatchExplorerPage />);

    // 2026-03-01 + 90 days = 2026-05-30
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-05-30"));
  });
});
```

- [ ] **Step 2: Run and verify the outcome**

Run: `cd app/frontend && npx vitest run components/MatchUI.dateboundary.test.tsx`
Expected: 3 passed, since W30 already wired both components correctly. If any test fails, that reveals a real gap in W30's wiring (e.g. `asOf` missing from an effect's dependency array) — fix `MatchUI.tsx` before proceeding, don't weaken the test.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/MatchUI.dateboundary.test.tsx
git commit -m "test: cover Dashboard/Match Explorer date-boundary correctness using the sandbox clock (W38)"
```

---

## Task 7: W31 — Sandbox scenario runbook

**Files:**
- Create: `scripts/__init__.py`, `scripts/sandbox_runbook.py`
- Test: `scripts/test_sandbox_runbook.py`
- Create: `documents/sandbox_testing_runbook.md`

**Context:** Generalizes W23's one-off smoke-test checklist (`documents/prelaunch_smoke_test_checklist.md`) into something re-runnable for any historical date, exercising W27–W30 and W37 together: Dashboard fixtures, a real recommendation generation, logging one bet from that recommendation and one manually, and settling both against the real historical result. Written as a runnable script, not just a manual checklist. No `scripts/` directory exists yet.

- [ ] **Step 1: Create the `scripts` package**

Run: `mkdir -p scripts && touch scripts/__init__.py`

- [ ] **Step 2: Write the failing guard test**

Create `scripts/test_sandbox_runbook.py`:

```python
"""W31: sandbox scenario runbook. Only the argument-guard is unit-tested
here (no real network/LLM calls needed) -- the full live run against a real
historical date is Step 5/6 below, recorded in
documents/sandbox_testing_runbook.md, same as W23's original checklist."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.sandbox_runbook import main


def test_main_exits_with_a_clear_message_when_sandbox_mode_not_set(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "SANDBOX_MODE=1" in capsys.readouterr().out
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest scripts/test_sandbox_runbook.py -v`
Expected: FAIL — `scripts.sandbox_runbook` doesn't exist yet.

- [ ] **Step 4: Implement the runbook script**

Create `scripts/sandbox_runbook.py`:

```python
#!/usr/bin/env python3
"""W31: sandbox scenario runbook -- repeatable for any historical date.
Boots the app in sandbox mode for a given date and drives the full real
user journey against it: Dashboard fixtures, a real recommendation
generation (real agent/LLM call, real ML model prediction from real
point-in-time features), logging one bet from that recommendation and one
manually, then settling both against the real historical result.

Non-determinism in the agent's own predicted values is expected and
accepted -- this validates the app's plumbing works end-to-end for an
arbitrary day, not that the agent's predicted values are "correct" (a
distinct, separate concern this script does not test).

Usage:
    SANDBOX_MODE=1 SANDBOX_DATE=2026-05-24 FOOTBALL_DATA_API_KEY=... python scripts/sandbox_runbook.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.backend import bets, recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.football_data_client import FootballDataClient
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.recommendations import RecommendationRequest, validate_and_degrade
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_date
from app.backend.settlement import settle_open_bets
from src.agent.agent_config import AgentConfig


def main() -> None:
    if not is_sandbox_mode() or sandbox_date() is None:
        print("SANDBOX_MODE=1 and SANDBOX_DATE=<real historical date> must both be set.")
        sys.exit(1)

    date_str = sandbox_date().isoformat()
    print(f"=== Sandbox runbook for {date_str} ===")

    # 1. Dashboard: real fixtures for the sandbox date
    fixtures_client = FootballDataClient(api_key=os.environ["FOOTBALL_DATA_API_KEY"])
    fixtures = fixtures_client.get_results(competition_code="PL", date_from=date_str, date_to=date_str)
    if not fixtures:
        print(f"No completed E0 fixtures found for {date_str} -- pick a different date.")
        sys.exit(1)
    fixture = fixtures[0]
    print(f"Fixture: {fixture.home_team} vs {fixture.away_team} ({fixture.home_goals}-{fixture.away_goals})")

    # 2. Real historical odds for that fixture
    odds_client = HistoricalOddsClient(sandbox_date=date_str)
    odds_events = odds_client.get_odds()
    assert odds_events is not None, "HistoricalOddsClient returned no odds for this date"
    print(f"Odds events found: {len(odds_events)}")

    # 3. Generate a real recommendation (real agent/LLM call)
    config = AgentConfig.default()
    request = RecommendationRequest(home_team=fixture.home_team, away_team=fixture.away_team, date=date_str)
    raw = recommendations.run_agent(request.to_match_info())
    result = validate_and_degrade(raw)
    print(f"Recommendation overall={result.overall} markets={len(result.markets)}")

    cache = recommendations.get_cache()
    cache.record_generation(
        match_id=request.effective_match_id(), date=date_str,
        agent_config_hash=compute_agent_config_hash(config),
        odds={}, recommendation=result.model_dump(), triggered_by="sandbox_runbook",
    )
    print(f"Recorded to sandbox cache: {cache._db_path}")

    # 4. Log one bet from the recommendation (if any market was generated)
    tracker = bets.get_bet_tracker()
    if result.markets:
        market = result.markets[0]
        bet_from_rec = tracker.create_bet(
            match_id=request.effective_match_id(), date=date_str,
            home_team=fixture.home_team, away_team=fixture.away_team,
            market=market.market, selection=market.selection,
            odds=market.current_odds or 2.0, stake=10.0,
            source="from_recommendation", recommendation_snapshot=result.model_dump(),
        )
        print(f"Logged bet from recommendation: id={bet_from_rec.id}")
    else:
        print("No markets in the recommendation -- skipping the from-recommendation bet.")

    # 5. Log one bet manually
    bet_manual = tracker.create_bet(
        match_id=request.effective_match_id(), date=date_str,
        home_team=fixture.home_team, away_team=fixture.away_team,
        market="match_odds", selection="home", odds=2.0, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    print(f"Logged manual bet: id={bet_manual.id}")

    # 6. Settle both against the real historical result
    settled = settle_open_bets(tracker, fixtures_client)
    print(f"Settled {len(settled)} bet(s):")
    for bet in settled:
        print(f"  id={bet.id} outcome={bet.outcome} profit_loss={bet.profit_loss}")

    print("\nSandbox-scoped resources used -- real app/data/*.db untouched:")
    print(f"  cache: {cache._db_path}")
    print(f"  bets:  {tracker._db_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the guard test to verify it passes**

Run: `python -m pytest scripts/test_sandbox_runbook.py -v`
Expected: 1 passed.

- [ ] **Step 6: Run the runbook for real, against a real historical date**

This is the story's actual acceptance criterion — it requires real credentials and real network access, and cannot be automated in CI:

```bash
SANDBOX_MODE=1 SANDBOX_DATE=2026-05-24 python scripts/sandbox_runbook.py
```

Pick a date with completed E0 fixtures (e.g. `2026-05-24`, `raw_matches`'s last-refreshed date, confirmed to have real fixtures/odds — see W28's research). If that date's fixture happens to yield an `insufficient_data`/no-markets recommendation, try another completed-fixture date from `raw_matches` before concluding something is broken.

- [ ] **Step 7: Record the results in a new runbook document**

Create `documents/sandbox_testing_runbook.md`, following `documents/prelaunch_smoke_test_checklist.md`'s structure (intro explaining the scratch-db approach, numbered checklist items, a "Last run" results block per item with real observed values, a closing summary table). Populate it with the actual output captured from Step 6 — exact fixture names/date, exact `overall`/market count, exact bet ids, exact settlement outcomes/profit-loss, and confirmation that `app/data/*.db`'s modification times were untouched by the run (`ls -la app/data/*.db` before and after).

- [ ] **Step 8: Commit**

```bash
git add scripts/__init__.py scripts/sandbox_runbook.py scripts/test_sandbox_runbook.py documents/sandbox_testing_runbook.md
git commit -m "feat: add the repeatable sandbox scenario runbook and record a real run (W31)"
```

---

## Self-Review Notes

- **Spec coverage:** W27 (Task 1), W28 (Task 2), W29 (Task 3), W30 (Task 4), W37 (Task 5), W38 (Task 6), W31 (Task 7) — every acceptance criterion in the Phase 7 story text is covered.
- **Placeholder scan:** no `TBD`/vague steps; the two steps that require real credentials/network (Task 3 Step 9, Task 7 Step 6) are explicitly called out as manual verification rather than disguised as automated tests, consistent with how W01/W02/W25 recorded "Verified live" separately from their TDD suites.
- **Type consistency:** `HistoricalOddsClient.get_odds()` matches `OddsAPIClient.get_odds()`'s exact signature and `NormalizedOdds` return type throughout (Task 2, reused unchanged in Task 3's `build_odds_client()`); `run_agent(match_info, config=None)`'s signature (Task 5) matches both call conventions used at the 3 real call sites (`main.py`, `eod_batch.py`, `t30_refresh.py`); `useSandboxAsOf()`'s `Date` return type is used identically across `MatchUI.tsx` and `BetTracker.tsx` (Task 4) and in the W38 tests (Task 6).
- **Cross-task ordering:** Task 3's edits to `scheduler_wiring.py`/`main.py` build on imports added in Task 1; Task 5's `recommendations.py` edits build on the `_SANDBOX_CACHE_DB_PATH`/import block added in Task 3 — execute in the numbered order above, not in parallel, if using subagent-driven execution with file-level isolation.
- After Task 7, mark **W27, W28, W29, W30, W31, W37, W38** as `completed` in `documents/app_user_stories.md`'s Stories table (per `CLAUDE.md`: "When tasks are complete, mark the user story as completed"), each with a completion note following the existing narrative style (see W01–W26's entries) — file:line references, what was verified, full-suite pass counts, and the real runbook run's date/results.

---

## Execution Handoff

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Given Tasks 1→3→5 share edits to overlapping files (`scheduler_wiring.py`, `recommendations.py`), execute tasks strictly in order (1, 2, 3, 4, 5, 6, 7) rather than in parallel.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.
