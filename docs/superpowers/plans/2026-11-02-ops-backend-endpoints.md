# Ops Backend Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three new `/api/admin/ops/*` endpoints backing three of the four `/ops` dashboard tabs — failures & circuit breakers, odds availability (with cause), and feature health — each supporting date-range and league filtering. The fourth tab (recommendation performance) needs no new backend work: `/api/recommendations/stats` and `/performance-dashboard` already exist (Task 5 just confirms they already support the same filtering shape).

**Architecture:** Thin FastAPI routes over data that already exists: `run_isolated.py`'s `FailuresStore` and `scheduler_wiring.py`'s `get_circuit_breaker_status()` (both from the two prior plans), and `RecommendationCache.list_latest_per_match()` (already exists) scanned and filtered in Python — at today's scale (an admin diagnostics surface, not a hot path) this mirrors `list_latest_per_match()`'s own documented "fine for now, revisit if this grows past 100K rows" stance. No new database tables beyond the `failures` table Plan 1 already created.

**Tech Stack:** FastAPI, existing `RequireAppTokenMiddleware` gate (no new auth).

---

### Task 1: `missing_markets_from_cache_odds` + cause classification

**Files:**
- Modify: `app/backend/missing_odds_alert.py`
- Test: `app/backend/tests/test_missing_odds_alert.py` (check first whether this file exists: `ls app/backend/tests/test_missing_odds_alert.py`; if not, create it)

`missing_live_odds()` (already in this file) expects `match_info`'s nested `*_odds` key shape (`match_info["total_goals_odds"]`, `match_info["asian_handicap_odds"]`). `RecommendationCache`'s stored `CacheEntry.odds` is a *different*, flat shape — `add_secondary_odds()` (`eod_batch.py`) mutates the same dict used for the 1X2 match (`home`/`draw`/`away` keys) in place, adding `total_goals`/`btts`/`asian_handicap`/`corners`/`home_goals`/`away_goals` as sibling keys on that same flat dict, and *that* dict is what gets persisted via `cache.record_generation(odds=odds, ...)`. Reusing `missing_live_odds()` directly against a `CacheEntry.odds` dict would silently check the wrong keys — this task writes the correct adapter instead of forcing a mismatched reuse.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for missing_markets_from_cache_odds() -- the /ops odds-availability
tab's adapter from RecommendationCache's flat odds-dict storage shape to
the same three-market check missing_live_odds() already does for
match_info's nested shape. 2026-11-02 design:
docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md"""

from __future__ import annotations

from app.backend.missing_odds_alert import missing_markets_from_cache_odds


def test_missing_markets_from_cache_odds_reports_nothing_when_everything_present() -> None:
    odds = {
        "home": 2.1, "draw": 3.4, "away": 3.6,
        "total_goals": {"over_2.5": 1.9, "under_2.5": 1.9},
        "asian_handicap": {"line": -0.5, "home": 1.9, "away": 1.9},
        "corners": {"over_9.5": 1.9, "under_9.5": 1.9},
    }
    assert missing_markets_from_cache_odds(odds) == []


def test_missing_markets_from_cache_odds_reports_missing_1x2() -> None:
    odds = {"home": None, "draw": None, "away": None}
    assert "1X2" in missing_markets_from_cache_odds(odds)


def test_missing_markets_from_cache_odds_reports_missing_over_under() -> None:
    odds = {"home": 2.1, "draw": 3.4, "away": 3.6, "total_goals": None}
    assert "over/under 2.5" in missing_markets_from_cache_odds(odds)


def test_missing_markets_from_cache_odds_reports_missing_asian_handicap() -> None:
    odds = {"home": 2.1, "draw": 3.4, "away": 3.6, "asian_handicap": {}}
    assert "Asian handicap" in missing_markets_from_cache_odds(odds)


def test_missing_markets_from_cache_odds_reports_missing_corners() -> None:
    """corners is the one market missing_live_odds() (match_info-shaped)
    never checked -- OddsPapi's total_corners wasn't part of BUG-096's own
    scope. The /ops tab adds it since it's the fourth market this app
    actually prices."""
    odds = {
        "home": 2.1, "draw": 3.4, "away": 3.6,
        "total_goals": {"over_2.5": 1.9}, "asian_handicap": {"line": -0.5, "home": 1.9, "away": 1.9},
        "corners": None,
    }
    assert missing_markets_from_cache_odds(odds) == ["total_corners"]


def test_missing_markets_from_cache_odds_treats_empty_corners_dict_as_missing() -> None:
    """{} means "OddsPapi answered, no line posted yet" (BUG-097) -- still
    missing from the dashboard's point of view, just not a vendor failure."""
    odds = {
        "home": 2.1, "draw": 3.4, "away": 3.6,
        "total_goals": {"over_2.5": 1.9}, "asian_handicap": {"line": -0.5, "home": 1.9, "away": 1.9},
        "corners": {},
    }
    assert missing_markets_from_cache_odds(odds) == ["total_corners"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_missing_odds_alert.py -v`
Expected: FAIL with `ImportError: cannot import name 'missing_markets_from_cache_odds'`

- [ ] **Step 3: Implement `missing_markets_from_cache_odds`**

Append to `app/backend/missing_odds_alert.py`:

```python
def missing_markets_from_cache_odds(odds: dict) -> list[str]:
    """Same three-market check as missing_live_odds() above, adapted to
    RecommendationCache's own flat odds-dict storage shape (home/draw/away
    plus total_goals/asian_handicap/corners as sibling keys, set in place by
    add_secondary_odds() -- see eod_batch.py) rather than match_info's
    nested *_odds key shape. Also checks corners (total_corners), which
    missing_live_odds() never covered -- OddsPapi wasn't in BUG-096's scope,
    but it's the fourth market this app actually prices, and the /ops
    odds-availability tab should show all four. Used to query already-cached
    rows (2026-11-02 design) instead of only firing once at generation time."""
    missing: list[str] = []
    if not all(odds.get(k) for k in ("home", "draw", "away")):
        missing.append("1X2")
    if not (odds.get("total_goals") or {}).get("over_2.5"):
        missing.append("over/under 2.5")
    ah = odds.get("asian_handicap") or {}
    if ah.get("line") is None or not ah.get("home") or not ah.get("away"):
        missing.append("Asian handicap")
    if not odds.get("corners"):
        missing.append("total_corners")
    return missing
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_missing_odds_alert.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/missing_odds_alert.py app/backend/tests/test_missing_odds_alert.py
git commit -m "feat: add missing_markets_from_cache_odds for the ops odds-availability tab"
```

---

### Task 2: `GET /api/admin/ops/failures`

**Files:**
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_main_ops_endpoints.py` (new file)

- [ ] **Step 1: Write the failing test**

Check the existing admin-endpoint test pattern first:

Run: `grep -n "RequireAppTokenMiddleware\|APP_ACCESS_TOKEN\|TestClient(app)" app/backend/tests/test_fixtures_endpoint.py | head -10`

```python
"""Tests for the /api/admin/ops/* endpoints -- the four-tab /ops dashboard's
backend. 2026-11-02 design:
docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.run_isolated import FailuresStore


def test_get_ops_failures_returns_recorded_failures_and_breaker_status(tmp_path: Path, monkeypatch) -> None:
    import app.backend.run_isolated as run_isolated_module
    monkeypatch.setattr(run_isolated_module, "DEFAULT_DB_PATH", tmp_path / "failures.db")
    run_isolated_module._singleton_store = None
    run_isolated_module.get_failures_store().record(
        subsystem="t30", unit_id="match_id=558542", exc=ValueError("vendor exploded"),
        alerted=True, league="I1",
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures")

    assert response.status_code == 200
    body = response.json()
    assert len(body["failures"]) == 1
    assert body["failures"][0]["subsystem"] == "t30"
    assert body["failures"][0]["league"] == "I1"
    assert set(body["circuit_breakers"].keys()) == {"the_odds_api", "oddspapi"}


def test_get_ops_failures_filters_by_league(tmp_path: Path, monkeypatch) -> None:
    import app.backend.run_isolated as run_isolated_module
    monkeypatch.setattr(run_isolated_module, "DEFAULT_DB_PATH", tmp_path / "failures.db")
    run_isolated_module._singleton_store = None
    store = run_isolated_module.get_failures_store()
    store.record(subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0")
    store.record(subsystem="t30", unit_id="match_id=2", exc=ValueError("b"), alerted=False, league="I1")

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures", params={"league": "I1"})

    body = response.json()
    assert len(body["failures"]) == 1
    assert body["failures"][0]["unit_id"] == "match_id=2"


def test_get_ops_failures_filters_by_since(tmp_path: Path, monkeypatch) -> None:
    import app.backend.run_isolated as run_isolated_module
    monkeypatch.setattr(run_isolated_module, "DEFAULT_DB_PATH", tmp_path / "failures.db")
    run_isolated_module._singleton_store = None
    run_isolated_module.get_failures_store().record(
        subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0",
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures", params={"since": "2099-01-01T00:00:00+00:00"})

    assert response.json()["failures"] == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_main_ops_endpoints.py -v`
Expected: FAIL with 404 (route doesn't exist yet)

- [ ] **Step 3: Implement the endpoint**

Add to `app/backend/main.py`, near the other `/api/admin/*` routes:

```python
from app.backend.run_isolated import get_failures_store
from app.backend.scheduler_wiring import get_circuit_breaker_status


@app.get("/api/admin/ops/failures")
def get_ops_failures(
    since: str | None = None,
    league: str | None = None,
    subsystem: str | None = None,
) -> dict:
    """2026-11-02: /ops dashboard's failures & circuit breakers tab.
    Protected by RequireAppTokenMiddleware like every other /api/admin/*
    route."""
    failures = get_failures_store().list_recent(since=since, subsystem=subsystem, league=league)
    return {"failures": failures, "circuit_breakers": get_circuit_breaker_status()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_main_ops_endpoints.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_main_ops_endpoints.py
git commit -m "feat: add GET /api/admin/ops/failures endpoint"
```

---

### Task 3: `GET /api/admin/ops/odds-availability`

**Files:**
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_main_ops_endpoints.py`

Cause classification, per fixture/market: if no cache row exists at all for a `(match_id, date)` pair the caller asked about, that's `"never_attempted"`; if a row exists and a market is missing, the cause is `"circuit_open"` when that market's vendor breaker is currently `open`, otherwise `"not_posted"` (the common, benign case — a book simply hasn't priced it yet). This endpoint only reports on fixtures *with* a cache row (there's no "fixture the pipeline hasn't reached yet" source of truth independent of the cache itself) — the `/ops` frontend cross-references `/api/fixtures` for the `never_attempted` case, same as Plan 4's own Task covers.

- [ ] **Step 1: Write the failing test**

```python
def test_get_ops_odds_availability_reports_missing_markets_with_cause(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock, patch
    from app.backend.recommendation_cache import CacheEntry

    entry = CacheEntry(
        match_id="558542", date="2026-11-02", agent_config_hash="abc",
        odds={"home": 2.1, "draw": 3.4, "away": 3.6, "total_goals": None, "asian_handicap": None, "corners": None},
        recommendation={"match": {"home": "Udinese", "away": "Roma", "date": "2026-11-02", "league": "I1"}},
        generated_at="2026-11-02T10:00:00+00:00", triggered_by="scheduled",
    )
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = [entry]

    with patch("app.backend.scheduler_wiring.get_circuit_breaker_status", return_value={
        "the_odds_api": {"state": "open"}, "oddspapi": {"state": "closed"},
    }):
        with TestClient(app) as client:
            app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
            try:
                response = client.get("/api/admin/ops/odds-availability", params={"date_from": "2026-11-02", "date_to": "2026-11-02"})
            finally:
                app.dependency_overrides.pop(recommendations.get_cache, None)

    body = response.json()
    assert len(body["fixtures"]) == 1
    row = body["fixtures"][0]
    assert row["match_id"] == "558542"
    assert row["league"] == "I1"
    assert "over/under 2.5" in row["missing_markets"]
    assert row["likely_cause"] == "circuit_open"
```

(Match this file's actual existing import style for `recommendations` and `TestClient`/`app` — check `app/backend/tests/test_fixtures_endpoint.py` for the established `app.dependency_overrides` pattern this codebase already uses for swapping in a fake cache.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_main_ops_endpoints.py -k odds_availability -v`
Expected: FAIL with 404

- [ ] **Step 3: Implement the endpoint**

Add to `app/backend/main.py`:

```python
from app.backend.missing_odds_alert import missing_markets_from_cache_odds


@app.get("/api/admin/ops/odds-availability")
def get_ops_odds_availability(
    date_from: str,
    date_to: str,
    league: str | None = None,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> dict:
    """2026-11-02: /ops dashboard's odds-availability tab. Scans every
    cached match's latest generation (list_latest_per_match already exists
    for this exact "every match, not just ones I already know the key for"
    shape -- recommendation_outcomes.py's own resolution job uses it the
    same way) and reports which markets are missing, with a best-effort
    cause: circuit_open when that market's vendor breaker is currently
    open, not_posted otherwise (the common, benign case)."""
    breaker_status = get_circuit_breaker_status()
    odds_api_open = breaker_status["the_odds_api"]["state"] == "open"
    oddspapi_open = breaker_status["oddspapi"]["state"] == "open"

    fixtures: list[dict] = []
    for entry in cache.list_latest_per_match():
        if not (date_from <= entry.date <= date_to):
            continue
        match = entry.recommendation.get("match") or {}
        entry_league = match.get("league")
        if league is not None and entry_league != league:
            continue
        missing = missing_markets_from_cache_odds(entry.odds)
        if not missing:
            continue
        likely_cause = "not_posted"
        if "total_corners" in missing and oddspapi_open:
            likely_cause = "circuit_open"
        elif any(m in missing for m in ("1X2", "over/under 2.5", "Asian handicap")) and odds_api_open:
            likely_cause = "circuit_open"
        fixtures.append({
            "match_id": entry.match_id, "date": entry.date, "league": entry_league,
            "home_team": match.get("home"), "away_team": match.get("away"),
            "missing_markets": missing, "likely_cause": likely_cause,
        })
    return {"fixtures": fixtures}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_main_ops_endpoints.py -k odds_availability -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_main_ops_endpoints.py
git commit -m "feat: add GET /api/admin/ops/odds-availability endpoint"
```

---

### Task 4: `GET /api/admin/ops/feature-health`

**Files:**
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_main_ops_endpoints.py`

- [ ] **Step 1: Write the failing test**

```python
def test_get_ops_feature_health_aggregates_by_league(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    from app.backend.recommendation_cache import CacheEntry

    entries = [
        CacheEntry(
            match_id="1", date="2026-11-02", agent_config_hash="a",
            odds={}, recommendation={"match": {"league": "I1"}},
            generated_at="2026-11-02T10:00:00+00:00", triggered_by="scheduled",
            forecast_payload={"diagnostics": {"cold_start_risk": True, "feature_completeness": 0.4}, "data_quality": {"unknown_team": False}},
        ),
        CacheEntry(
            match_id="2", date="2026-11-02", agent_config_hash="a",
            odds={}, recommendation={"match": {"league": "I1"}},
            generated_at="2026-11-02T11:00:00+00:00", triggered_by="scheduled",
            forecast_payload={"diagnostics": {"cold_start_risk": False, "feature_completeness": 0.9}, "data_quality": {"unknown_team": False}},
        ),
    ]
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = entries

    with TestClient(app) as client:
        app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
        try:
            response = client.get("/api/admin/ops/feature-health", params={"date_from": "2026-11-02", "date_to": "2026-11-02"})
        finally:
            app.dependency_overrides.pop(recommendations.get_cache, None)

    body = response.json()
    assert body["by_league"]["I1"]["sample_size"] == 2
    assert body["by_league"]["I1"]["cold_start_risk_rate"] == 0.5
    assert body["by_league"]["I1"]["avg_feature_completeness"] == 0.65
    assert body["by_league"]["I1"]["unknown_team_rate"] == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_main_ops_endpoints.py -k feature_health -v`
Expected: FAIL with 404

- [ ] **Step 3: Implement the endpoint**

Add to `app/backend/main.py`:

```python
@app.get("/api/admin/ops/feature-health")
def get_ops_feature_health(
    date_from: str,
    date_to: str,
    league: str | None = None,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> dict:
    """2026-11-02: /ops dashboard's feature-health tab. feature_completeness/
    cold_start_risk/unknown_team already live on every cached recommendation's
    forecast_payload (A31 diagnostics, src/agent/graph.py's
    _extract_forecast_diagnostics) -- nothing new to compute, only to
    aggregate by league/date, which nothing does today."""
    by_league: dict[str, dict] = {}
    for entry in cache.list_latest_per_match():
        if not (date_from <= entry.date <= date_to):
            continue
        match = entry.recommendation.get("match") or {}
        entry_league = match.get("league")
        if league is not None and entry_league != league:
            continue
        if entry_league is None or entry.forecast_payload is None:
            continue
        diagnostics = entry.forecast_payload.get("diagnostics") or {}
        data_quality = entry.forecast_payload.get("data_quality") or {}
        bucket = by_league.setdefault(entry_league, {
            "sample_size": 0, "cold_start_count": 0, "unknown_team_count": 0, "completeness_sum": 0.0,
        })
        bucket["sample_size"] += 1
        if diagnostics.get("cold_start_risk"):
            bucket["cold_start_count"] += 1
        if data_quality.get("unknown_team"):
            bucket["unknown_team_count"] += 1
        completeness = diagnostics.get("feature_completeness")
        if completeness is not None:
            bucket["completeness_sum"] += completeness

    result: dict[str, dict] = {}
    for league_code, bucket in by_league.items():
        n = bucket["sample_size"]
        result[league_code] = {
            "sample_size": n,
            "cold_start_risk_rate": round(bucket["cold_start_count"] / n, 6) if n else 0.0,
            "unknown_team_rate": round(bucket["unknown_team_count"] / n, 6) if n else 0.0,
            "avg_feature_completeness": round(bucket["completeness_sum"] / n, 6) if n else None,
        }
    return {"by_league": result}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_main_ops_endpoints.py -k feature_health -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_main_ops_endpoints.py
git commit -m "feat: add GET /api/admin/ops/feature-health endpoint"
```

---

### Task 5: Confirm the existing performance endpoints already support league filtering

**Files:**
- Read: `app/backend/recommendation_stats.py`, `app/backend/agent_performance_dashboard.py`
- No code changes expected — this task is verification, producing a follow-up task only if a real gap is found.

- [ ] **Step 1: Check `/api/recommendations/stats`'s existing league support**

Run: `grep -n "by_league\|competition" app/backend/recommendation_stats.py`

Confirm `by_league` (or `by_league_metrics`) breakdowns already exist (seen during the 2026-11-02 design conversation: `_segment_kelly_report(outcomes, lambda o: o.competition, starting_bankroll)`). If a `league` query parameter to *filter* the whole response (not just break it down) doesn't exist on the endpoint itself, note this as a one-line gap for Plan 4 to close client-side (the frontend tab reads `by_league[selectedLeague]` from the existing full response rather than needing the backend to filter) — do **not** add a new backend parameter here; the existing breakdown shape already contains everything a league filter needs.

- [ ] **Step 2: Check date filtering**

Run: `grep -n "days: int" app/backend/main.py | grep -A 2 "recommendations/stats\|performance-dashboard"`

Confirm the existing `days` query parameter (seen in both endpoints during the audit) covers the dashboard's date-range filter requirement — a "last N days" window, not an explicit `date_from`/`date_to` pair like the two new endpoints above. Note this shape difference for Plan 4: the shared filter bar's date-range picker needs to translate into a `days` value for this one tab (e.g. `days = (today - date_from).days`), not pass `date_from`/`date_to` straight through like the other three tabs do.

- [ ] **Step 3: No commit** — this task produces notes for Plan 4, not code changes.

---

### Task 6: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full backend test suite**

Run: `python -m pytest app/backend/tests -q`
Expected: no new failures beyond the same 8 pre-existing, unrelated ones documented in `documents/bugs.md`'s BUG-092 notes

- [ ] **Step 2: Run the new ops-endpoint files for a clean summary**

Run: `python -m pytest app/backend/tests/test_missing_odds_alert.py app/backend/tests/test_main_ops_endpoints.py -v`
Expected: all PASS

---

## Self-review notes

- **Spec coverage**: implements the backend half of §7's four tabs — failures (Task 2), odds availability (Task 3), feature health (Task 4) — plus confirms (Task 5) the fourth tab needs no new backend work, matching the spec's explicit "link, don't rebuild" call.
- **Dependency on prior plans**: Task 2 requires `run_isolated.py`'s `FailuresStore` (Plan 1, Task 1) and `scheduler_wiring.py`'s `get_circuit_breaker_status()` (Plan 2, Task 4) to already exist — this plan must run after both.
- **Type consistency**: `missing_markets_from_cache_odds(odds: dict) -> list[str]` matches the shape `CacheEntry.odds` actually has (confirmed by reading `eod_batch.py`'s `add_secondary_odds` mutation site, not assumed from `missing_live_odds()`'s different, match_info-shaped signature).
