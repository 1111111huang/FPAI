# Ops Dashboard Trend Visualizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `/ops` dashboard's Odds Availability, Feature Health, and
Performance tabs' single-window snapshots with Thursday-anchored weekly trend
views, and make the shared filter bar (now a Window selector, not raw dates)
actually drive all three — including Performance, which today ignores it
entirely.

**Architecture:** Three new backend endpoints aggregate existing data
(cached recommendations' `forecast_payload`, resolved outcomes, cached
`candidates`) into weekly buckets — no new instrumentation, no changes to how
recommendations are generated/staked/resolved. Frontend: a Window chip
selector replaces the date-range pickers; two existing tab components swap
their data source for the new trend endpoints; a new `PerformanceTab`
component (not `AgentPerformanceContent`, which stays untouched for the
standalone `/agent-performance` route) adds metric + betting-strategy toggles
driven by the shared filter bar.

**Tech Stack:** FastAPI (Python) backend, Next.js/React (TypeScript, Vitest)
frontend — same stack as the rest of this codebase, no new dependencies.

**Design doc:** `docs/superpowers/specs/2026-11-03-ops-dashboard-trends-design.md`

---

## Before you start (read once, applies to every task below)

- This plan is executed directly on `main` (not a worktree) unless whoever
  executes it decides otherwise with the user — check current branch/worktree
  state before starting Task 1.
- Backend tests: `cd` to the repo root, `python -m pytest app/backend/tests -q`
  (or the specific test file). A pre-existing baseline of unrelated failures
  exists in this repo (`test_agent_performance_dashboard_endpoint.py`,
  `test_fixtures_endpoint.py`, `test_recommendation_outcomes_endpoints.py` —
  roughly 8-9 tests, a documented, long-standing, unrelated gap). Confirm your
  change introduces nothing *beyond* that known set — don't chase or "fix"
  those unrelated failures as part of this plan.
- Frontend tests: **Vitest, not Jest** (`cd app/frontend && npx vitest run`).
  This codebase's real mock convention is an explicit factory:
  `vi.mock("@/lib/api", () => ({ someFn: vi.fn() }))`, then
  `vi.mocked(someFn).mockResolvedValue(...)` inside each test — see
  `app/frontend/components/ops/FailuresTab.test.tsx` for the canonical,
  already-approved example of this whole pattern (fetch-in-`useEffect` with a
  `cancelled` guard, `ErrorState`+`retryTick` for errors, `"Loading…"` with the
  real ellipsis character, not three periods).
- Every new frontend data-fetching component in this plan must follow that
  exact established pattern (`FailuresTab.tsx`/`OddsAvailabilityTab.tsx`/
  `FeatureHealthTab.tsx` are the live reference implementations) — read at
  least one of them before writing a new component in this plan.
- `npx tsc --noEmit` must stay clean after every frontend task.

---

### Task 1: Thursday-anchored week-bucketing utility (backend)

**Files:**
- Create: `app/backend/ops_trends.py`
- Test: `app/backend/tests/test_ops_trends.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for app/backend/ops_trends.py's week-bucketing helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.ops_trends import recent_week_starts, thursday_week_start


def test_thursday_maps_to_itself():
    assert thursday_week_start("2026-10-08") == "2026-10-08"  # a Thursday


def test_friday_maps_to_the_preceding_thursday():
    assert thursday_week_start("2026-10-09") == "2026-10-08"


def test_wednesday_maps_to_the_thursday_six_days_earlier():
    assert thursday_week_start("2026-10-14") == "2026-10-08"  # the Wednesday ending that week


def test_recent_week_starts_returns_oldest_to_newest():
    # 2026-11-03 is a Tuesday; its week started Thursday 2026-10-29.
    starts = recent_week_starts(date(2026, 11, 3), weeks=4)
    assert starts == ["2026-10-08", "2026-10-15", "2026-10-22", "2026-10-29"]


def test_recent_week_starts_length_matches_weeks_param():
    starts = recent_week_starts(date(2026, 11, 3), weeks=8)
    assert len(starts) == 8
    assert starts[-1] == "2026-10-29"  # newest bucket is always "today"'s own week
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_ops_trends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backend.ops_trends'`

- [ ] **Step 3: Implement `ops_trends.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_ops_trends.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/ops_trends.py app/backend/tests/test_ops_trends.py
git commit -m "feat: add Thursday-anchored week-bucketing for ops dashboard trends"
```

---

### Task 2: Odds Availability trend endpoint

**Files:**
- Modify: `app/backend/main.py`
- Modify: `app/backend/tests/test_main_ops_endpoints.py`

Context: the existing `GET /api/admin/ops/odds-availability` (main.py, already
shipped) iterates `cache.list_latest_per_match()`, filters by
`date_from <= entry.date <= date_to`, and reports missing markets per fixture.
This task adds a sibling endpoint that instead buckets by week and aggregates
per-market counts, reusing the exact same `missing_markets_from_cache_odds`
call and `likely_cause` logic already in that function — read it in full in
`main.py` before writing this (search `def get_ops_odds_availability`).

- [ ] **Step 1: Write the failing test**

Add to `app/backend/tests/test_main_ops_endpoints.py` (check its existing
imports/fixtures first — it already has a `TestClient(app)` + cache-seeding
pattern for the sibling snapshot endpoint's own tests; follow that exact
setup, don't invent a new one):

```python
def test_odds_availability_trend_buckets_by_week_and_ranks_markets(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    cache = RecommendationCache(db_path=":memory:")  # match this file's existing in-memory cache pattern
    app.dependency_overrides[recommendations.get_cache] = lambda: cache

    # Week of 2026-10-08 (Thu-Wed): 2 fixtures, one missing total_corners.
    cache.record_generation(
        match_id="m1", date="2026-10-09", agent_config_hash="h",
        odds={"result_3way": {"home": 2.0}}, triggered_by="scheduled",
        recommendation={"match": {"league": "E0", "home": "A", "away": "B"}, "overall": "no_bet"},
    )
    cache.record_generation(
        match_id="m2", date="2026-10-10", agent_config_hash="h",
        odds={"result_3way": {"home": 2.0}, "total_corners": {"over_9.5": 1.9}}, triggered_by="scheduled",
        recommendation={"match": {"league": "E0", "home": "C", "away": "D"}, "overall": "no_bet"},
    )
    # Week of 2026-10-15: 1 fixture, missing total_corners too.
    cache.record_generation(
        match_id="m3", date="2026-10-16", agent_config_hash="h",
        odds={"result_3way": {"home": 2.0}}, triggered_by="scheduled",
        recommendation={"match": {"league": "E0", "home": "E", "away": "F"}, "overall": "no_bet"},
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/odds-availability-trend", params={"weeks": 2, "today": "2026-10-15"})

    app.dependency_overrides.pop(recommendations.get_cache, None)
    assert response.status_code == 200
    body = response.json()
    assert [w["week_start"] for w in body["weeks"]] == ["2026-10-08", "2026-10-15"]
    week1 = body["weeks"][0]
    assert week1["total_fixtures"] == 2
    assert week1["by_market"]["total_corners"]["missing"] == 1
    week2 = body["weeks"][1]
    assert week2["total_fixtures"] == 1
    assert week2["by_market"]["total_corners"]["missing"] == 1
```

Check `missing_markets_from_cache_odds`'s real missing-market label strings
(e.g. is it `"total_corners"` or `"over/under total corners"`? — read
`app/backend/missing_odds_alert.py`'s `missing_markets_from_cache_odds`
function before writing this test's assertions, and correct the labels above
to match reality, not this sketch, if they differ) and adjust the test's
expected keys to match exactly what that function actually returns.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_main_ops_endpoints.py -k odds_availability_trend -v`
Expected: FAIL — 404 (route doesn't exist yet)

- [ ] **Step 3: Implement the endpoint**

Add to `app/backend/main.py`, directly below the existing
`get_ops_odds_availability` function (import `recent_week_starts` from
`app.backend.ops_trends` at the top of the file alongside this file's other
`app.backend.*` imports):

```python
@app.get("/api/admin/ops/odds-availability-trend")
def get_ops_odds_availability_trend(
    weeks: int = Query(8, ge=1, le=52),
    league: str | None = None,
    today: str | None = None,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> dict:
    """2026-11-03: /ops dashboard's Odds Availability tab, trend view --
    buckets the same missing-market detection get_ops_odds_availability
    already does by Thursday-anchored week instead of a flat date range, so
    the frontend can rank "which market is constantly missing" across the
    whole window rather than one snapshot. `today` is test-only (overrides
    the real clock so week boundaries are deterministic in tests); omit it in
    production, where it defaults to the real current date."""
    breaker_status = get_circuit_breaker_status()
    odds_api_open = breaker_status["the_odds_api"]["state"] == "open"
    oddspapi_open = breaker_status["oddspapi"]["state"] == "open"
    today_date = date.fromisoformat(today) if today else _current_real_date()
    week_starts = recent_week_starts(today_date, weeks)
    week_index = {start: i for i, start in enumerate(week_starts)}

    buckets: list[dict] = [{"week_start": start, "total_fixtures": 0, "by_market": {}} for start in week_starts]

    for entry in cache.list_latest_per_match():
        week_start = thursday_week_start(entry.date)
        idx = week_index.get(week_start)
        if idx is None:
            continue  # outside the requested window
        match = entry.recommendation.get("match") or {}
        entry_league = match.get("league")
        if league is not None and entry_league != league:
            continue
        bucket = buckets[idx]
        bucket["total_fixtures"] += 1
        missing = missing_markets_from_cache_odds(entry.odds)
        if not missing:
            continue
        for market in missing:
            cause_key = "circuit_open" if (
                (market == "total_corners" and oddspapi_open)
                or (market != "total_corners" and odds_api_open)
            ) else "not_posted"
            market_bucket = bucket["by_market"].setdefault(market, {"missing": 0, "circuit_open": 0, "not_posted": 0})
            market_bucket["missing"] += 1
            market_bucket[cause_key] += 1

    return {"weeks": buckets}
```

Double-check the `likely_cause` branching above against the real logic in
`get_ops_odds_availability` (read it again right before writing this) — this
sketch mirrors it but simplify/correct it to match the real per-market cause
rule exactly (the existing function's real branching may be more specific
than "total_corners vs everything else"; don't invent a different rule here).

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_main_ops_endpoints.py -k odds_availability_trend -v`
Expected: PASS

- [ ] **Step 5: Run the full ops endpoint test file + full backend suite**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_main_ops_endpoints.py -v && python -m pytest app/backend -q`
Expected: all pass (full suite: no new failures beyond the documented pre-existing baseline)

- [ ] **Step 6: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_main_ops_endpoints.py
git commit -m "feat: add GET /api/admin/ops/odds-availability-trend"
```

---

### Task 3: Feature Health trend endpoint

**Files:**
- Modify: `app/backend/main.py`
- Modify: `app/backend/tests/test_main_ops_endpoints.py`

Context: `forecast_payload.explainability.top_features` is a list of
**up to 8** items per cached recommendation (the most important features for
*that* match's forecast — not a fixed/complete feature set; see
`src/forecast/forecast_service.py`'s `_top_features`, `limit: int = 8`). Each
item's real shape is `{"name": str, "value": float | None, "importance": float}`
— **`"name"`, not `"feature"`**; there is no `shap_value` key here (that's a
different, per-candidate field elsewhere in this codebase — don't confuse the
two). Read `forecast_service.py`'s `_top_features` function yourself to
confirm this shape before writing code against it.

Because a feature only appears in a match's `top_features` when it was
actually among that match's top 8 by importance, **a feature's null-rate and
drift must be computed only over the matches where it actually appeared** —
not over every recommendation in the week (a rarely-important feature that
simply doesn't surface often is not the same thing as a feature going null).

- [ ] **Step 1: Write the failing tests**

Add to `app/backend/tests/test_ops_trends.py`:

```python
from app.backend.ops_trends import FeatureWeekStats, aggregate_feature_weeks


def test_aggregate_feature_weeks_computes_mean_and_null_rate_per_feature_per_week():
    # Two matches in the same week: XOC_HOME appears both times (values 2.0, 4.0);
    # DEF_ANCHOR_AWAY appears once, null that time.
    entries = [
        {"week_start": "2026-10-08", "top_features": [
            {"name": "XOC_HOME", "value": 2.0, "importance": 0.5},
            {"name": "DEF_ANCHOR_AWAY", "value": None, "importance": 0.3},
        ]},
        {"week_start": "2026-10-08", "top_features": [
            {"name": "XOC_HOME", "value": 4.0, "importance": 0.4},
        ]},
    ]
    result = aggregate_feature_weeks(entries, week_starts=["2026-10-08"])
    assert result["XOC_HOME"]["2026-10-08"] == FeatureWeekStats(appearances=2, mean=3.0, null_rate=0.0)
    assert result["DEF_ANCHOR_AWAY"]["2026-10-08"] == FeatureWeekStats(appearances=1, mean=None, null_rate=1.0)


def test_flag_drifting_features_requires_minimum_sample_size_in_both_halves():
    from app.backend.ops_trends import flag_drifting_features

    # XOC_HOME: baseline mean 2.0 (2 weeks), recent mean 8.0 (2 weeks) -- a real 4x jump, enough samples.
    weekly = {
        "XOC_HOME": {
            "2026-10-08": FeatureWeekStats(appearances=5, mean=2.0, null_rate=0.0),
            "2026-10-15": FeatureWeekStats(appearances=5, mean=2.0, null_rate=0.0),
            "2026-10-22": FeatureWeekStats(appearances=5, mean=8.0, null_rate=0.0),
            "2026-10-29": FeatureWeekStats(appearances=5, mean=8.0, null_rate=0.0),
        },
        # LOW_SAMPLE: real-looking drift ratio but only 1 appearance total -- must NOT be flagged.
        "LOW_SAMPLE": {
            "2026-10-08": FeatureWeekStats(appearances=1, mean=1.0, null_rate=0.0),
            "2026-10-15": FeatureWeekStats(appearances=0, mean=None, null_rate=0.0),
            "2026-10-22": FeatureWeekStats(appearances=0, mean=None, null_rate=0.0),
            "2026-10-29": FeatureWeekStats(appearances=1, mean=9.0, null_rate=0.0),
        },
    }
    flagged = flag_drifting_features(weekly, week_starts=["2026-10-08", "2026-10-15", "2026-10-22", "2026-10-29"])
    names = {f["name"] for f in flagged}
    assert "XOC_HOME" in names
    assert "LOW_SAMPLE" not in names
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_ops_trends.py -k "aggregate_feature_weeks or flag_drifting" -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the aggregation + flagging helpers**

Append to `app/backend/ops_trends.py`:

```python
from collections import defaultdict
from dataclasses import dataclass

# Tunable, first-pass thresholds -- see design doc section 6. A feature needs
# at least this many appearances in BOTH the baseline and recent half to be
# eligible for flagging at all (guards against a 1-2-sample feature looking
# like it "drifted" when it's just noise from rarely surfacing).
MIN_APPEARANCES_PER_HALF = 3
# Flag drift when the recent-half pooled mean is at least this many times
# the baseline pooled mean (or its reciprocal, for a drop) -- a magnitude-
# ratio heuristic, not a statistical test; revisit once this has real
# production weeks to tune against.
DRIFT_RATIO_THRESHOLD = 2.0
# Flag an elevated null rate when the recent-half null rate exceeds the
# baseline by at least this many percentage points.
NULL_RATE_DELTA_THRESHOLD = 0.05


@dataclass(frozen=True)
class FeatureWeekStats:
    appearances: int
    mean: float | None
    null_rate: float


def aggregate_feature_weeks(
    entries: list[dict], week_starts: list[str]
) -> dict[str, dict[str, FeatureWeekStats]]:
    """`entries`: one dict per cached recommendation with `top_features`
    (already `forecast_payload["explainability"]["top_features"]`) already
    extracted, plus the entry's own pre-computed `week_start` (callers bucket
    via thursday_week_start() before calling this -- kept as a plain
    precondition rather than re-bucketing here, since main.py's endpoint
    already needs the raw date for other filtering too). Returns
    {feature_name: {week_start: FeatureWeekStats}}, only for weeks actually
    present in `week_starts` -- a feature with zero appearances in a given
    week simply has no key for that week (not a zero-appearance stats
    object), so callers can tell "never surfaced that week" from "surfaced
    and was null" by key presence."""
    # name -> week_start -> running totals
    totals: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {
        "appearances": 0, "null_count": 0, "value_sum": 0.0, "value_count": 0,
    }))
    week_set = set(week_starts)
    for entry in entries:
        week_start = entry["week_start"]
        if week_start not in week_set:
            continue
        for feature in entry.get("top_features") or []:
            name = feature.get("name")
            if not name:
                continue
            bucket = totals[name][week_start]
            bucket["appearances"] += 1
            value = feature.get("value")
            if value is None:
                bucket["null_count"] += 1
            else:
                bucket["value_sum"] += float(value)
                bucket["value_count"] += 1

    result: dict[str, dict[str, FeatureWeekStats]] = {}
    for name, by_week in totals.items():
        result[name] = {}
        for week_start, bucket in by_week.items():
            mean = bucket["value_sum"] / bucket["value_count"] if bucket["value_count"] else None
            null_rate = bucket["null_count"] / bucket["appearances"] if bucket["appearances"] else 0.0
            result[name][week_start] = FeatureWeekStats(
                appearances=bucket["appearances"], mean=mean, null_rate=round(null_rate, 6),
            )
    return result


def _pooled(stats: list[FeatureWeekStats]) -> tuple[int, float | None, float]:
    """Pools several weeks' FeatureWeekStats into one (appearances, mean,
    null_rate) triple -- sums counts rather than averaging per-week means, so
    weeks with more samples correctly carry more weight."""
    total_appearances = sum(s.appearances for s in stats)
    total_null = sum(round(s.null_rate * s.appearances) for s in stats)
    value_count = sum(s.appearances - round(s.null_rate * s.appearances) for s in stats)
    value_sum = sum((s.mean or 0.0) * (s.appearances - round(s.null_rate * s.appearances)) for s in stats)
    mean = value_sum / value_count if value_count else None
    null_rate = total_null / total_appearances if total_appearances else 0.0
    return total_appearances, mean, null_rate


def flag_drifting_features(
    weekly: dict[str, dict[str, FeatureWeekStats]], week_starts: list[str]
) -> list[dict]:
    """Splits `week_starts` into an earlier baseline half and a later recent
    half (floor split -- e.g. 8 weeks -> 4/4, 5 weeks -> 2/3), pools each
    feature's stats within each half, and flags features whose recent pooled
    mean/null_rate deviates from baseline beyond the module's threshold
    constants. Only features with >= MIN_APPEARANCES_PER_HALF total
    appearances in BOTH halves are eligible -- see that constant's own
    comment for why. Returns a list sorted by severity (largest deviation
    first), each entry: {"name", "category", "flags": [...], "baseline_mean",
    "recent_mean", "baseline_null_rate", "recent_null_rate", "magnitude",
    "weekly": [{"week_start", "mean", "null_rate"}, ...]}."""
    split = len(week_starts) // 2
    baseline_weeks, recent_weeks = week_starts[:split], week_starts[split:]
    flagged: list[dict] = []
    for name, by_week in weekly.items():
        baseline_stats = [by_week[w] for w in baseline_weeks if w in by_week]
        recent_stats = [by_week[w] for w in recent_weeks if w in by_week]
        baseline_n, baseline_mean, baseline_null = _pooled(baseline_stats)
        recent_n, recent_mean, recent_null = _pooled(recent_stats)
        if baseline_n < MIN_APPEARANCES_PER_HALF or recent_n < MIN_APPEARANCES_PER_HALF:
            continue

        flags: list[str] = []
        magnitude = 0.0
        if baseline_mean is not None and recent_mean is not None:
            if baseline_mean == 0:
                if abs(recent_mean) > 0.01:
                    flags.append("drift")
                    magnitude = max(magnitude, abs(recent_mean))
            else:
                ratio = abs(recent_mean / baseline_mean)
                if ratio >= DRIFT_RATIO_THRESHOLD or ratio <= (1.0 / DRIFT_RATIO_THRESHOLD):
                    flags.append("drift")
                    magnitude = max(magnitude, max(ratio, 1.0 / ratio) if ratio else 0.0)
        if (recent_null - baseline_null) >= NULL_RATE_DELTA_THRESHOLD:
            flags.append("nullrate")
            magnitude = max(magnitude, (recent_null - baseline_null) * 10)  # scaled to be roughly comparable to a drift ratio

        if not flags:
            continue
        flagged.append({
            "name": name,
            "category": _feature_category(name),
            "flags": flags,
            "baseline_mean": round(baseline_mean, 6) if baseline_mean is not None else None,
            "recent_mean": round(recent_mean, 6) if recent_mean is not None else None,
            "baseline_null_rate": round(baseline_null, 6),
            "recent_null_rate": round(recent_null, 6),
            "magnitude": round(magnitude, 3),
            "weekly": [
                {"week_start": w, "mean": by_week[w].mean, "null_rate": by_week[w].null_rate}
                for w in week_starts if w in by_week
            ],
        })
    flagged.sort(key=lambda f: f["magnitude"], reverse=True)
    return flagged


_CATEGORY_PREFIXES = [
    "OFF_", "DEF_", "MKT_", "CTX_", "DIS_", "SQUAD_", "H2H_", "LUCK_", "EFFICIENCY_",
    "OPP_ADJ_", "DC_", "XOC", "DEF_ANCHOR",
]


def _feature_category(name: str) -> str:
    """Best-effort category tag from the feature name's own existing
    namespace-prefix convention (already visible across this codebase's real
    feature names, e.g. OFF_AWAY_SHOT_ACCURACY_R5, MKT_POISSON_BTTS_PROB) --
    a simple prefix lookup, not a new taxonomy to maintain. Falls back to
    "other" for a name matching none of these."""
    for prefix in _CATEGORY_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_").lower()
    return "other"
```

Before committing to this exact drift/null-rate math, double-check
`_pooled`'s null-count reconstruction (`round(s.null_rate * s.appearances)`)
against `aggregate_feature_weeks`'s own rounding of `null_rate` — if this
round-trip loses precision in a way that breaks a test, store raw
`null_count`/`value_count`/`value_sum` in `FeatureWeekStats` instead of a
derived `null_rate`, and compute `null_rate` as a property/separately where
needed, rather than reconstructing counts from a rounded rate. Prefer
correctness over matching this sketch exactly if the two conflict.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_ops_trends.py -v`
Expected: all PASS

- [ ] **Step 5: Wire the HTTP endpoint**

Add to `app/backend/main.py`, below `get_ops_feature_health`:

```python
@app.get("/api/admin/ops/feature-health-trend")
def get_ops_feature_health_trend(
    weeks: int = Query(8, ge=2, le=52),
    league: str | None = None,
    today: str | None = None,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> dict:
    """2026-11-03: /ops dashboard's Feature Health tab, trend view -- flags
    only features whose values drifted from their own baseline or whose
    null-rate climbed, across the window's earlier vs. later half. See
    ops_trends.py's aggregate_feature_weeks/flag_drifting_features for the
    actual math; this endpoint is just the fetch + filter + call."""
    today_date = date.fromisoformat(today) if today else _current_real_date()
    week_starts = recent_week_starts(today_date, weeks)
    week_set = set(week_starts)

    entries_for_aggregation: list[dict] = []
    for entry in cache.list_latest_per_match():
        week_start = thursday_week_start(entry.date)
        if week_start not in week_set:
            continue
        match = entry.recommendation.get("match") or {}
        if league is not None and match.get("league") != league:
            continue
        if entry.forecast_payload is None:
            continue
        top_features = (entry.forecast_payload.get("explainability") or {}).get("top_features") or []
        entries_for_aggregation.append({"week_start": week_start, "top_features": top_features})

    weekly = aggregate_feature_weeks(entries_for_aggregation, week_starts)
    flagged = flag_drifting_features(weekly, week_starts)
    return {
        "week_starts": week_starts,
        "tracked_feature_count": len(weekly),
        "flagged_features": flagged,
    }
```

Import `aggregate_feature_weeks`, `flag_drifting_features`,
`recent_week_starts`, `thursday_week_start` from `app.backend.ops_trends` at
the top of `main.py`, and `date` from `datetime` if not already imported
there (check first — this file likely already imports `date`/`datetime` for
other endpoints; don't add a duplicate import).

- [ ] **Step 6: Write an HTTP-level test, run it, confirm it passes**

Add to `app/backend/tests/test_main_ops_endpoints.py`, following the same
in-memory-cache setup pattern as Task 2's test — seed 2+ cached recommendations
across 2+ weeks with `forecast_payload={"explainability": {"top_features": [...]}}`
such that one feature clearly drifts (e.g. mean 2.0 in the baseline week(s),
mean 20.0 in the recent week(s), with >= `MIN_APPEARANCES_PER_HALF` samples in
each half — reuse multiple `record_generation` calls per week if one call
alone isn't enough to clear the minimum-sample threshold), and assert the
response's `flagged_features` contains that feature's name with `"drift"` in
its `flags` list. Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_main_ops_endpoints.py -k feature_health_trend -v`
Expected: PASS.

- [ ] **Step 7: Full suite**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend -q`
Expected: no new failures beyond the documented pre-existing baseline.

- [ ] **Step 8: Commit**

```bash
git add app/backend/ops_trends.py app/backend/tests/test_ops_trends.py app/backend/main.py app/backend/tests/test_main_ops_endpoints.py
git commit -m "feat: add GET /api/admin/ops/feature-health-trend with drift/null-rate flagging"
```

---

### Task 4: Performance trend endpoint — candidate grading + weekly records

**Files:**
- Modify: `app/backend/ops_trends.py`
- Modify: `app/backend/tests/test_ops_trends.py`

This is the most involved piece — read `src/agent/market_resolution.py`
(`market_correct`, `build_actual_outcome`), `src/agent/staking.py`
(`simulate_kelly_stake`, `BetOutcome`), `src/agent/evaluation.py`
(`build_evaluation_report`), and `app/backend/recommendation_outcomes.py`
(`RecommendationOutcome` — note it already stores `home_goals`/`away_goals`
per resolved match, post-W175) in full before writing this task's code, if
you haven't already — this task reuses all of them, not reimplements.

- [ ] **Step 1: Write the failing tests**

Add to `app/backend/tests/test_ops_trends.py`:

```python
from app.backend.ops_trends import build_week_records
from app.backend.recommendation_outcomes import RecommendationOutcome


def _outcome(match_id, date_, market, selection, rec_type, odds, edge, correct, home_goals=2, away_goals=1):
    return RecommendationOutcome(
        id=0, match_id=match_id, date=date_, competition="E0", market=market, selection=selection,
        recommendation_type=rec_type, confidence="medium", odds=odds, value_edge=edge, correct=correct,
        generated_at=f"{date_}T12:00:00Z", resolved_at=f"{date_}T13:00:00Z",
        home_goals=home_goals, away_goals=away_goals,
    )


def test_one_per_match_uses_the_single_stored_outcome_directly():
    outcomes = [_outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True)]
    records = build_week_records(outcomes, {}, pick_scope="one_per_match", include_conditional=False)
    assert len(records) == 1
    assert records[0].market_results == [
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1, "correct": True},
    ]


def test_all_direct_grades_every_direct_bet_candidate_from_the_cached_recommendation():
    outcome = _outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True, home_goals=1, away_goals=0)
    cached_candidates_by_key = {
        ("m1", "2026-10-09"): [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1},
            {"market": "btts", "selection": "no", "recommendation_type": "direct_bet", "current_odds": 1.8, "value_edge": 0.05},
            {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 2.1, "value_edge": -0.1},
        ],
    }
    records = build_week_records([outcome], cached_candidates_by_key, pick_scope="all_direct", include_conditional=False)
    assert len(records) == 1
    results = {(m["market"], m["selection"]): m for m in records[0].market_results}
    assert len(results) == 2  # only the two direct_bet candidates, not the no_bet one
    assert results[("result_3way", "home")]["correct"] is True  # home won 1-0
    assert results[("btts", "no")]["correct"] is True  # 1-0 is indeed btts=no


def test_include_conditional_adds_conditional_candidates_staked_at_current_odds():
    outcome = _outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True, home_goals=1, away_goals=0)
    cached_candidates_by_key = {
        ("m1", "2026-10-09"): [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1},
            {"market": "total_goals", "selection": "under_2.5", "recommendation_type": "conditional", "current_odds": 1.9, "value_edge": 0.03},
        ],
    }
    without = build_week_records([outcome], cached_candidates_by_key, pick_scope="all_direct", include_conditional=False)
    with_cond = build_week_records([outcome], cached_candidates_by_key, pick_scope="all_direct", include_conditional=True)
    assert len(without[0].market_results) == 1
    assert len(with_cond[0].market_results) == 2
    conditional_result = next(m for m in with_cond[0].market_results if m["market"] == "total_goals")
    assert conditional_result["recommendation_type"] == "direct_bet"  # staked AS IF direct, per the user's chosen semantics
    assert conditional_result["correct"] is True  # 1-0 total=1, under_2.5 is correct


def test_all_direct_skips_matches_with_no_known_final_score():
    outcome = _outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True)
    outcome = dataclasses.replace(outcome, home_goals=None, away_goals=None)  # pre-W175 row, no stored score
    records = build_week_records([outcome], {("m1", "2026-10-09"): [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1}]}, pick_scope="all_direct", include_conditional=False)
    assert records == []
```

Add `import dataclasses` to this test file's imports if not already present.
Note `test_include_conditional_adds_conditional_candidates_staked_at_current_odds`
re-tags a staked conditional's `recommendation_type` as `"direct_bet"` in the
built record (not `"conditional"`) specifically so it flows through
`simulate_kelly_stake`'s existing `recommendation_type != "direct_bet": continue`
filter unmodified (see Task 4 Step 3's docstring for why) — confirm this is
really what you want to build before implementing; if it feels like the wrong
mechanism once you're looking at `staking.py` directly, say so and propose the
alternative (e.g. a kwarg on a new staking function) rather than silently
building something else.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_ops_trends.py -k "one_per_match or all_direct or include_conditional" -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `build_week_records`**

Append to `app/backend/ops_trends.py`:

```python
from dataclasses import dataclass as _dataclass  # avoid shadowing the stdlib import already used above

from src.agent.backtest import BacktestRecord
from src.agent.market_resolution import build_actual_outcome, market_correct


def build_week_records(
    outcomes: list["RecommendationOutcome"],
    cached_candidates_by_key: dict[tuple[str, str], list[dict]],
    pick_scope: str,
    include_conditional: bool,
) -> list[BacktestRecord]:
    """Builds one BacktestRecord per match, ready for
    src.agent.staking.simulate_kelly_stake, under one of two strategies:

    - pick_scope="one_per_match": the single already-resolved outcome row
      *is* the bet -- same population compute_recommendation_stats already
      uses today. `cached_candidates_by_key` is unused in this mode (pass
      {} if you have nothing else handy).
    - pick_scope="all_direct": every `direct_bet`-flagged candidate in that
      match's cached recommendation (looked up via
      `cached_candidates_by_key[(match_id, date)]`) is graded independently
      against the match's own known final score (from the outcome row's
      stored home_goals/away_goals) via market_correct() -- each becomes its
      own stakeable unit, not just the one historically-chosen pick. A match
      with no cached_candidates_by_key entry, or no stored final score, is
      skipped entirely (nothing to grade).

    include_conditional=True additionally includes `conditional`-flagged
    candidates in the same population (either pick_scope), staked as if they
    were direct_bet -- a deliberate, explicit user choice (design doc
    section 2), not a claim this reflects real conditional-bet economics.
    Re-tagged to recommendation_type="direct_bet" in the returned
    market_results specifically so simulate_kelly_stake's own
    `recommendation_type != "direct_bet": continue` filter picks them up
    unmodified -- this function is the one place that distinction is made,
    staking.py itself needs no change."""
    records: list[BacktestRecord] = []

    if pick_scope == "one_per_match":
        for outcome in outcomes:
            records.append(BacktestRecord(
                match_id=outcome.match_id, home_team="", away_team="", date=outcome.date,
                league=outcome.competition or "", recommendation={"confidence": outcome.confidence}, actual={},
                market_results=[{
                    "market": outcome.market, "selection": outcome.selection,
                    "recommendation_type": outcome.recommendation_type,
                    "current_odds": outcome.odds, "value_edge": outcome.value_edge, "correct": outcome.correct,
                }],
            ))
        return records

    # pick_scope == "all_direct"
    allowed_types = {"direct_bet", "conditional"} if include_conditional else {"direct_bet"}
    for outcome in outcomes:
        if outcome.home_goals is None or outcome.away_goals is None:
            continue
        candidates = cached_candidates_by_key.get((outcome.match_id, outcome.date))
        if not candidates:
            continue
        actual = build_actual_outcome(outcome.home_goals, outcome.away_goals)
        market_results: list[dict] = []
        for candidate in candidates:
            if candidate.get("recommendation_type") not in allowed_types:
                continue
            correct = market_correct(candidate, actual)
            if correct is None:
                continue
            market_results.append({
                "market": candidate.get("market"), "selection": candidate.get("selection"),
                "recommendation_type": "direct_bet",  # see docstring -- staking.py's own filter expects this literal value
                "current_odds": candidate.get("current_odds"), "value_edge": candidate.get("value_edge"),
                "correct": correct,
            })
        if not market_results:
            continue
        records.append(BacktestRecord(
            match_id=outcome.match_id, home_team="", away_team="", date=outcome.date,
            league=outcome.competition or "", recommendation={"confidence": outcome.confidence}, actual={},
            market_results=market_results,
        ))
    return records
```

Check `BacktestRecord`'s real field names/required args in
`src/agent/backtest.py` before finalizing this — the call sites above mirror
`recommendation_stats.py`'s own `_to_backtest_records` usage, but confirm
exactly (field names, required vs. optional) rather than trusting this
sketch blindly.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_ops_trends.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/backend/ops_trends.py app/backend/tests/test_ops_trends.py
git commit -m "feat: add build_week_records for performance-trend's pick-scope/conditional strategies"
```

---

### Task 5: Performance trend endpoint — HTTP wiring

**Files:**
- Modify: `app/backend/main.py`
- Modify: `app/backend/tests/test_main_ops_endpoints.py`

- [ ] **Step 1: Write the failing test**

Add to `app/backend/tests/test_main_ops_endpoints.py`, following this file's
existing dependency-override pattern for both `recommendations.get_cache` and
`get_recommendation_outcome_store` (check the existing feature-health/odds-
availability tests in this same file for how they override `get_cache`; check
`app/backend/main.py`'s own `get_recommendation_outcome_store` dependency
name for the outcome-store override):

```python
def test_performance_trend_defaults_to_one_pick_per_match(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    store = RecommendationOutcomeStore(db_path=":memory:")
    store.insert(
        match_id="m1", date="2026-10-09", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-10-09T12:00:00Z", home_goals=1, away_goals=0,
    )
    app.dependency_overrides[get_recommendation_outcome_store] = lambda: store

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/performance-trend", params={"weeks": 1, "today": "2026-10-15"})

    app.dependency_overrides.pop(get_recommendation_outcome_store, None)
    assert response.status_code == 200
    weeks = response.json()["weeks"]
    assert len(weeks) == 1
    assert weeks[0]["week_start"] == "2026-10-08"
    assert weeks[0]["bets_placed"] == 1
    assert weeks[0]["hit_rate"] == 1.0


def test_performance_trend_all_direct_pick_scope_needs_the_cache_too(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    store = RecommendationOutcomeStore(db_path=":memory:")
    store.insert(
        match_id="m1", date="2026-10-09", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-10-09T12:00:00Z", home_goals=1, away_goals=0,
    )
    cache = RecommendationCache(db_path=":memory:")
    cache.record_generation(
        match_id="m1", date="2026-10-09", agent_config_hash="h", odds={}, triggered_by="scheduled",
        recommendation={
            "match": {"league": "E0"}, "overall": "direct_bet",
            "candidates": [
                {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1},
                {"market": "btts", "selection": "no", "recommendation_type": "direct_bet", "current_odds": 1.8, "value_edge": 0.05},
            ],
        },
    )
    app.dependency_overrides[get_recommendation_outcome_store] = lambda: store
    app.dependency_overrides[recommendations.get_cache] = lambda: cache

    with TestClient(app) as client:
        response = client.get(
            "/api/admin/ops/performance-trend",
            params={"weeks": 1, "today": "2026-10-15", "pick_scope": "all_direct"},
        )

    app.dependency_overrides.pop(get_recommendation_outcome_store, None)
    app.dependency_overrides.pop(recommendations.get_cache, None)
    assert response.status_code == 200
    assert response.json()["weeks"][0]["bets_placed"] == 2  # both direct_bet candidates, not just the stored pick
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_main_ops_endpoints.py -k performance_trend -v`
Expected: FAIL — 404

- [ ] **Step 3: Implement the endpoint**

Add to `app/backend/main.py` (import `build_week_records` from
`app.backend.ops_trends`, `simulate_kelly_stake` from `src.agent.staking`,
`build_evaluation_report` from `src.agent.evaluation`, and
`DEFAULT_STARTING_BANKROLL` from `app.backend.recommendation_stats` — check
that constant's real name/location before importing it, don't duplicate it
if it already exists importable):

```python
@app.get("/api/admin/ops/performance-trend")
def get_ops_performance_trend(
    weeks: int = Query(8, ge=1, le=52),
    league: str | None = None,
    today: str | None = None,
    pick_scope: str = Query("one_per_match", pattern="^(one_per_match|all_direct)$"),
    include_conditional: bool = False,
    store: RecommendationOutcomeStore = Depends(get_recommendation_outcome_store),
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> dict:
    """2026-11-03: /ops dashboard's Performance tab, trend view -- weekly
    hit-rate/ROI/profit, under a chosen betting strategy (pick_scope x
    include_conditional, see ops_trends.build_week_records). Each week is
    simulated independently from the same fixed starting_bankroll (not a
    continuous cross-week equity curve) so weeks are directly comparable to
    each other, not polluted by a prior week's drawdown."""
    today_date = date.fromisoformat(today) if today else _current_real_date()
    week_starts = recent_week_starts(today_date, weeks)

    all_outcomes = store.list_all()
    if league is not None:
        all_outcomes = [o for o in all_outcomes if o.competition == league]
    outcomes_by_week: dict[str, list] = {w: [] for w in week_starts}
    for outcome in all_outcomes:
        week_start = thursday_week_start(outcome.date)
        if week_start in outcomes_by_week:
            outcomes_by_week[week_start].append(outcome)

    cached_candidates_by_key: dict[tuple[str, str], list[dict]] = {}
    if pick_scope == "all_direct":
        keys = [(o.match_id, o.date) for week_outcomes in outcomes_by_week.values() for o in week_outcomes]
        entries = cache.list_latest_for_keys(keys)
        for (match_id, entry_date), entry in entries.items():
            cached_candidates_by_key[(match_id, entry_date)] = entry.recommendation.get("candidates") or []

    weeks_out = []
    for week_start in week_starts:
        records = build_week_records(
            outcomes_by_week[week_start], cached_candidates_by_key, pick_scope, include_conditional,
        )
        bankroll_result = simulate_kelly_stake(records, starting_bankroll=DEFAULT_STARTING_BANKROLL)
        report = build_evaluation_report(records, bankroll_result)
        weeks_out.append({
            "week_start": week_start,
            "hit_rate": report["hit_rate"],
            "roi": report["roi"],
            "bets_placed": report["bets_placed"],
            "total_profit": report["total_profit"],
            "ending_bankroll": report["ending_bankroll"],
        })
    return {"weeks": weeks_out, "pick_scope": pick_scope, "include_conditional": include_conditional}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_main_ops_endpoints.py -k performance_trend -v`
Expected: PASS

- [ ] **Step 5: Full backend suite**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend -q`
Expected: no new failures beyond the documented pre-existing baseline.

- [ ] **Step 6: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_main_ops_endpoints.py
git commit -m "feat: add GET /api/admin/ops/performance-trend"
```

---

### Task 6: Frontend `lib/api.ts` + `lib/types.ts` additions

**Files:**
- Modify: `app/frontend/lib/api.ts`
- Modify: `app/frontend/lib/types.ts`

No direct unit test for these functions — matches this codebase's real,
already-established convention (confirmed repeatedly earlier in this same
dashboard's build: `lib/api.ts` fetch wrappers are only ever tested
indirectly, via component tests that `vi.mock("@/lib/api", ...)` wholesale).

- [ ] **Step 1: Add the types**

Append to `app/frontend/lib/types.ts`, matching this file's existing `type`
(not `interface`) convention:

```typescript
export type OddsAvailabilityTrendWeek = {
  week_start: string;
  total_fixtures: number;
  by_market: Record<string, { missing: number; circuit_open: number; not_posted: number }>;
};

export type OpsOddsAvailabilityTrendResponse = {
  weeks: OddsAvailabilityTrendWeek[];
};

export type FeatureHealthFlag = "drift" | "nullrate";

export type FlaggedFeature = {
  name: string;
  category: string;
  flags: FeatureHealthFlag[];
  baseline_mean: number | null;
  recent_mean: number | null;
  baseline_null_rate: number;
  recent_null_rate: number;
  magnitude: number;
  weekly: { week_start: string; mean: number | null; null_rate: number }[];
};

export type OpsFeatureHealthTrendResponse = {
  week_starts: string[];
  tracked_feature_count: number;
  flagged_features: FlaggedFeature[];
};

export type PerformanceTrendWeek = {
  week_start: string;
  hit_rate: number;
  roi: number;
  bets_placed: number;
  total_profit: number;
  ending_bankroll: number;
};

export type PickScope = "one_per_match" | "all_direct";

export type OpsPerformanceTrendResponse = {
  weeks: PerformanceTrendWeek[];
  pick_scope: PickScope;
  include_conditional: boolean;
};
```

- [ ] **Step 2: Add the fetch functions**

Read `app/frontend/lib/api.ts`'s existing `getOpsOddsAvailability`/
`getOpsFeatureHealth` functions (already in the file) for the exact
`apiFetch`/`ApiError`/query-string-building style to match, then append:

```typescript
export async function getOpsOddsAvailabilityTrend(opts: { weeks: number; league?: string }): Promise<OpsOddsAvailabilityTrendResponse> {
  const params = new URLSearchParams({ weeks: String(opts.weeks) });
  if (opts.league) params.set("league", opts.league);
  const response = await apiFetch(`/api/admin/ops/odds-availability-trend?${params.toString()}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load odds availability trend (${response.status})`, response.status);
  }
  return response.json();
}

export async function getOpsFeatureHealthTrend(opts: { weeks: number; league?: string }): Promise<OpsFeatureHealthTrendResponse> {
  const params = new URLSearchParams({ weeks: String(opts.weeks) });
  if (opts.league) params.set("league", opts.league);
  const response = await apiFetch(`/api/admin/ops/feature-health-trend?${params.toString()}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load feature health trend (${response.status})`, response.status);
  }
  return response.json();
}

export async function getOpsPerformanceTrend(opts: {
  weeks: number; league?: string; pickScope?: PickScope; includeConditional?: boolean;
}): Promise<OpsPerformanceTrendResponse> {
  const params = new URLSearchParams({ weeks: String(opts.weeks) });
  if (opts.league) params.set("league", opts.league);
  if (opts.pickScope) params.set("pick_scope", opts.pickScope);
  if (opts.includeConditional) params.set("include_conditional", "true");
  const response = await apiFetch(`/api/admin/ops/performance-trend?${params.toString()}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load performance trend (${response.status})`, response.status);
  }
  return response.json();
}
```

Merge the new type imports into `api.ts`'s existing `./types` import line
rather than adding a second one, matching the file's established style.

- [ ] **Step 3: Type-check**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: clean

- [ ] **Step 4: Confirm no regressions**

Run: `cd app/frontend && npx vitest run`
Expected: all pass, same count as before this task

- [ ] **Step 5: Commit**

```bash
git add app/frontend/lib/api.ts app/frontend/lib/types.ts
git commit -m "feat: add lib/api.ts functions and types for the three ops trend endpoints"
```

---

### Task 7: `OpsDashboard.tsx` — Window chip selector replaces date pickers

**Files:**
- Modify: `app/frontend/components/OpsDashboard.tsx`
- Modify: `app/frontend/components/__tests__/OpsDashboard.test.tsx`

- [ ] **Step 1: Update the failing/changed tests first**

Read the current `OpsDashboard.test.tsx` in full. Update its league-filter
test and any date-input-specific assertions to match the new Window chip UI
instead (e.g. a test asserting a default-selected "8w" chip exists via
`screen.getByRole("button", { name: "8w" })` with some selected-state
attribute, rather than asserting on `<input type="date">` values). Keep the
existing tab-rendering/tab-switching tests as-is — those aren't affected by
this change. Run `cd app/frontend && npx vitest run components/__tests__/OpsDashboard.test.tsx` first, to see what's currently there and confirm which assertions need updating before you touch the component.

- [ ] **Step 2: Implement the Window chip selector**

In `OpsDashboard.tsx`, replace the `dateFromOverride`/`dateToOverride` state
and the From/To `<input type="date">` pair with:

```tsx
const WEEK_OPTIONS = [4, 8, 12, 26];

export function OpsDashboard() {
  const [tab, setTab] = useState<TabKey>("failures");
  const [league, setLeague] = useState<string>("");
  const [weeks, setWeeks] = useState<number>(8);

  return (
    <AppShell active="ops">
      <div className="mx-auto max-w-5xl p-4">
        <h1 className="text-xl font-semibold text-ink">Ops Dashboard</h1>

        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface p-3 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-muted">Window</span>
            <div className="flex gap-1">
              {WEEK_OPTIONS.map((w) => (
                <button
                  key={w}
                  type="button"
                  aria-pressed={weeks === w}
                  onClick={() => setWeeks(w)}
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    weeks === w ? "bg-accent text-white" : "border border-border text-muted hover:text-ink"
                  }`}
                >
                  {w}w
                </button>
              ))}
            </div>
          </div>
          <label className="flex items-center gap-2">
            <span className="text-muted">League</span>
            <select
              value={league}
              onChange={(e) => setLeague(e.target.value)}
              className="rounded border border-border bg-bg px-2 py-1 text-ink"
            >
              <option value="">All</option>
              {Object.keys(LEAGUE_LABEL).map((code) => (
                <option key={code} value={code}>
                  {LEAGUE_LABEL[code]}
                </option>
              ))}
            </select>
          </label>
        </div>

        {/* tabs unchanged */}

        <div className="mt-4">
          {tab === "failures" && <FailuresTab league={league || undefined} />}
          {tab === "odds" && <OddsAvailabilityTab weeks={weeks} league={league || undefined} />}
          {tab === "features" && <FeatureHealthTab weeks={weeks} league={league || undefined} />}
          {tab === "performance" && <PerformanceTab weeks={weeks} league={league || undefined} />}
        </div>
      </div>
    </AppShell>
  );
}
```

Remove the now-unused `useSandboxAsOf`/`dateString`/`addDays`/
`AgentPerformanceContent` imports from this file if nothing else in it still
uses them (check before removing — `AgentPerformanceContent` specifically is
no longer rendered here once Task 10 lands `PerformanceTab`, but if you're
doing this task before Task 10 in execution order, `AgentPerformanceContent`
is still referenced for the Performance tab until Task 10 swaps it — leave
that one import/usage in place for now if so, and only remove it once
`PerformanceTab` actually replaces it). Update the header comment at the top
of the file (currently describing the date-range + league filter bar) to
describe the Window selector instead — don't leave it stale.

- [ ] **Step 3: Run the test file, iterate until green**

Run: `cd app/frontend && npx vitest run components/__tests__/OpsDashboard.test.tsx`
Expected: PASS (this will likely fail initially because `OddsAvailabilityTab`/
`FeatureHealthTab`/`PerformanceTab` don't yet accept a `weeks` prop or
`PerformanceTab` doesn't exist yet — that's expected and fine at this point in
the plan; Tasks 8-10 close those gaps. If this task is executed before those,
the test should fail specifically on those missing pieces, not on anything
related to the Window selector itself — confirm that's the actual failure
reason before moving on, same discipline as every other task in this
dashboard's build.)

- [ ] **Step 4: Commit**

```bash
git add app/frontend/components/OpsDashboard.tsx app/frontend/components/__tests__/OpsDashboard.test.tsx
git commit -m "feat: replace OpsDashboard's date-range pickers with a Window (weeks) selector"
```

(If Step 3's test can't fully pass yet because of the dependent components,
that's fine — commit this task's own change now; the full green state lands
once Task 10 is also done, same deferred-commit pattern this dashboard's
original build already used once for `OpsDashboard.tsx` itself.)

---

### Task 8: `OddsAvailabilityTab` redesign — ranked trend chart

**Files:**
- Modify: `app/frontend/components/ops/OddsAvailabilityTab.tsx`
- Modify: `app/frontend/components/ops/__tests__/OddsAvailabilityTab.test.tsx`

Read the current `OddsAvailabilityTab.tsx` in full first — this task changes
its prop signature from `{ dateFrom, dateTo, league }` to `{ weeks, league }`
and its data source from `getOpsOddsAvailability` to
`getOpsOddsAvailabilityTrend`, but keeps the exact same
`useEffect`+`cancelled`-guard / `ErrorState`+`retryTick` / `"Loading…"`
pattern already in this file — a swap of what's fetched and rendered, not a
rewrite of the data-fetching skeleton.

- [ ] **Step 1: Write the failing test**

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { OddsAvailabilityTab } from "../OddsAvailabilityTab";
import { getOpsOddsAvailabilityTrend } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  getOpsOddsAvailabilityTrend: vi.fn(),
}));

describe("OddsAvailabilityTab", () => {
  it("ranks markets by % of fixtures missing them across the window", async () => {
    vi.mocked(getOpsOddsAvailabilityTrend).mockResolvedValue({
      weeks: [
        { week_start: "2026-10-08", total_fixtures: 50, by_market: { total_corners: { missing: 40, circuit_open: 30, not_posted: 10 } } },
        { week_start: "2026-10-15", total_fixtures: 50, by_market: { total_corners: { missing: 36, circuit_open: 20, not_posted: 16 }, asian_handicap: { missing: 5, circuit_open: 0, not_posted: 5 } } },
      ],
    });

    render(<OddsAvailabilityTab weeks={8} />);

    expect(await screen.findByText("total_corners")).toBeInTheDocument();
    // (40+36) missing / (50+50) total = 76%
    expect(screen.getByText("76%")).toBeInTheDocument();
  });

  it("renders an empty state with no missing markets anywhere in the window", async () => {
    vi.mocked(getOpsOddsAvailabilityTrend).mockResolvedValue({ weeks: [{ week_start: "2026-10-08", total_fixtures: 10, by_market: {} }] });
    render(<OddsAvailabilityTab weeks={8} />);
    expect(await screen.findByText(/no missing markets/i)).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsOddsAvailabilityTrend).mockRejectedValue(new Error("boom"));
    render(<OddsAvailabilityTab weeks={8} />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app/frontend && npx vitest run components/ops/__tests__/OddsAvailabilityTab.test.tsx`
Expected: FAIL (prop/function mismatch against the current implementation)

- [ ] **Step 3: Implement the redesigned component**

```tsx
"use client";

import { useEffect, useState } from "react";

import { getOpsOddsAvailabilityTrend } from "@/lib/api";
import type { OpsOddsAvailabilityTrendResponse } from "@/lib/types";
import { ErrorState } from "../MatchUI";

function aggregateByMarket(data: OpsOddsAvailabilityTrendResponse): { market: string; pct: number; missing: number; total: number }[] {
  const totals: Record<string, { missing: number; total: number }> = {};
  let grandTotal = 0;
  for (const week of data.weeks) {
    grandTotal += week.total_fixtures;
    for (const [market, stats] of Object.entries(week.by_market)) {
      const bucket = totals[market] ?? { missing: 0, total: 0 };
      bucket.missing += stats.missing;
      totals[market] = bucket;
    }
  }
  return Object.entries(totals)
    .map(([market, { missing }]) => ({ market, missing, total: grandTotal, pct: grandTotal > 0 ? missing / grandTotal : 0 }))
    .sort((a, b) => b.pct - a.pct);
}

export function OddsAvailabilityTab({ weeks, league }: { weeks: number; league?: string }) {
  const [data, setData] = useState<OpsOddsAvailabilityTrendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getOpsOddsAvailabilityTrend({ weeks, league })
      .then((result) => { if (!cancelled) setData(result); })
      .catch((err) => { if (!cancelled) setError(err.message ?? "Failed to load odds availability trend"); })
    return () => { cancelled = true; };
  }, [weeks, league, retryTick]);

  if (error) return <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />;
  if (!data) return <div className="text-sm text-muted">Loading…</div>;

  const ranked = aggregateByMarket(data);
  if (ranked.length === 0) {
    return <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">No missing markets in this window.</div>;
  }

  const maxPct = Math.max(...ranked.map((r) => r.pct), 0.01);
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4">
      {ranked.map((r) => (
        <div key={r.market} className="flex items-center gap-3">
          <div className="w-40 shrink-0 text-xs font-medium text-ink">{r.market}</div>
          <div className="h-4 flex-1 rounded bg-surface-2">
            <div
              className={`h-full rounded ${r.pct >= 0.5 ? "bg-serious" : r.pct >= 0.2 ? "bg-warning" : "bg-muted/50"}`}
              style={{ width: `${(r.pct / maxPct) * 100}%` }}
            />
          </div>
          <div className="w-12 shrink-0 text-right text-xs font-mono">{Math.round(r.pct * 100)}%</div>
        </div>
      ))}
    </div>
  );
}
```

Confirm the exact Tailwind color-token class names (`bg-serious`/`bg-warning`/
`text-muted`/etc.) against `FailuresTab.tsx`'s real, already-approved usage
before finalizing — match its real token names exactly, don't guess new ones.
Confirm `ErrorState`'s real import path/prop names the same way.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app/frontend && npx vitest run components/ops/__tests__/OddsAvailabilityTab.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/ops/OddsAvailabilityTab.tsx app/frontend/components/ops/__tests__/OddsAvailabilityTab.test.tsx
git commit -m "feat: redesign OddsAvailabilityTab as a ranked constantly-missing-markets chart"
```

---

### Task 9: `FeatureHealthTab` redesign — severity-sorted flagged list

**Files:**
- Modify: `app/frontend/components/ops/FeatureHealthTab.tsx`
- Modify: `app/frontend/components/ops/__tests__/FeatureHealthTab.test.tsx`

Prop signature changes from `{ dateFrom, dateTo, league }` to
`{ weeks, league }`; data source changes to `getOpsFeatureHealthTrend`. Same
fetch/error/loading skeleton as the rest of this dashboard.

- [ ] **Step 1: Write the failing test**

```typescript
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FeatureHealthTab } from "../FeatureHealthTab";
import { getOpsFeatureHealthTrend } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  getOpsFeatureHealthTrend: vi.fn(),
}));

function flaggedFeature(overrides: Partial<Parameters<typeof vi.fn>[0]> = {}) {
  return {
    name: "XOC_HOME", category: "offense", flags: ["drift"],
    baseline_mean: 1.8, recent_mean: 12.7, baseline_null_rate: 0.0, recent_null_rate: 0.0,
    magnitude: 7.1, weekly: [{ week_start: "2026-10-08", mean: 1.8, null_rate: 0.0 }],
    ...overrides,
  };
}

describe("FeatureHealthTab", () => {
  it("renders the scorecard and the flagged-feature list, sorted by the API's own order", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08"], tracked_feature_count: 52,
      flagged_features: [flaggedFeature()],
    });
    render(<FeatureHealthTab weeks={8} />);
    expect(await screen.findByText("XOC_HOME")).toBeInTheDocument();
    expect(screen.getByText("52")).toBeInTheDocument(); // tracked_feature_count tile
  });

  it("caps the list and reveals the rest on click", async () => {
    const many = Array.from({ length: 12 }, (_, i) => flaggedFeature({ name: `FEAT_${i}` } as never));
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({ week_starts: ["2026-10-08"], tracked_feature_count: 52, flagged_features: many });
    render(<FeatureHealthTab weeks={8} />);
    await screen.findByText("FEAT_0");
    expect(screen.queryByText("FEAT_11")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/show .* more/i));
    expect(screen.getByText("FEAT_11")).toBeInTheDocument();
  });

  it("filters by name via the search box", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08"], tracked_feature_count: 2,
      flagged_features: [flaggedFeature(), flaggedFeature({ name: "DEF_ANCHOR_AWAY" } as never)],
    });
    render(<FeatureHealthTab weeks={8} />);
    await screen.findByText("XOC_HOME");
    fireEvent.change(screen.getByPlaceholderText(/filter by name/i), { target: { value: "DEF" } });
    expect(screen.queryByText("XOC_HOME")).not.toBeInTheDocument();
    expect(screen.getByText("DEF_ANCHOR_AWAY")).toBeInTheDocument();
  });

  it("renders a clean empty state when nothing is flagged", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({ week_starts: ["2026-10-08"], tracked_feature_count: 52, flagged_features: [] });
    render(<FeatureHealthTab weeks={8} />);
    expect(await screen.findByText(/no.*flagged/i)).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockRejectedValue(new Error("boom"));
    render(<FeatureHealthTab weeks={8} />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app/frontend && npx vitest run components/ops/__tests__/FeatureHealthTab.test.tsx`
Expected: FAIL

- [ ] **Step 3: Implement the redesigned component**

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";

import { getOpsFeatureHealthTrend } from "@/lib/api";
import type { FlaggedFeature, OpsFeatureHealthTrendResponse } from "@/lib/types";
import { ErrorState } from "../MatchUI";

const INITIAL_VISIBLE = 8;

function FeatureRow({ feature }: { feature: FlaggedFeature }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2">
      <div>
        <span className="font-mono text-xs font-semibold">{feature.name}</span>
        <span className="ml-2 text-[10px] uppercase text-muted">{feature.category}</span>
      </div>
      <div className="flex items-center gap-2 text-[10px]">
        {feature.flags.includes("drift") && <span className="rounded-full bg-serious/15 px-2 py-0.5 font-semibold text-serious">DRIFT</span>}
        {feature.flags.includes("nullrate") && <span className="rounded-full bg-warning/15 px-2 py-0.5 font-semibold text-warning">NULL RATE ↑</span>}
      </div>
      <div className="text-right text-xs text-muted">
        {feature.flags.includes("drift") && feature.baseline_mean !== null && feature.recent_mean !== null
          ? `${feature.baseline_mean.toFixed(2)} → ${feature.recent_mean.toFixed(2)}`
          : `${(feature.baseline_null_rate * 100).toFixed(0)}% → ${(feature.recent_null_rate * 100).toFixed(0)}% null`}
      </div>
    </div>
  );
}

export function FeatureHealthTab({ weeks, league }: { weeks: number; league?: string }) {
  const [data, setData] = useState<OpsFeatureHealthTrendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);
  const [search, setSearch] = useState("");
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    setVisibleCount(INITIAL_VISIBLE);
    getOpsFeatureHealthTrend({ weeks, league })
      .then((result) => { if (!cancelled) setData(result); })
      .catch((err) => { if (!cancelled) setError(err.message ?? "Failed to load feature health trend"); });
    return () => { cancelled = true; };
  }, [weeks, league, retryTick]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = search.trim().toLowerCase();
    return q ? data.flagged_features.filter((f) => f.name.toLowerCase().includes(q)) : data.flagged_features;
  }, [data, search]);

  if (error) return <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />;
  if (!data) return <div className="text-sm text-muted">Loading…</div>;

  const drifting = data.flagged_features.filter((f) => f.flags.includes("drift")).length;
  const nullRate = data.flagged_features.filter((f) => f.flags.includes("nullrate")).length;
  const both = data.flagged_features.filter((f) => f.flags.length > 1).length;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-4 divide-x divide-border rounded-lg border border-border bg-surface text-center">
        <div className="p-3"><div className="font-mono text-xl font-bold">{data.tracked_feature_count}</div><div className="text-[10px] uppercase text-muted">Tracked</div></div>
        <div className="p-3"><div className="font-mono text-xl font-bold text-serious">{drifting}</div><div className="text-[10px] uppercase text-muted">Drifting</div></div>
        <div className="p-3"><div className="font-mono text-xl font-bold text-warning">{nullRate}</div><div className="text-[10px] uppercase text-muted">Elevated Null Rate</div></div>
        <div className="p-3"><div className="font-mono text-xl font-bold">{both}</div><div className="text-[10px] uppercase text-muted">Both</div></div>
      </div>

      {data.flagged_features.length === 0 ? (
        <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">No features flagged in this window.</div>
      ) : (
        <div className="rounded-lg border border-border bg-surface p-4">
          <input
            placeholder="Filter by name, e.g. XOC"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-3 w-56 rounded border border-border bg-bg px-2 py-1 font-mono text-xs"
          />
          <div className="flex flex-col gap-2">
            {filtered.slice(0, visibleCount).map((f) => <FeatureRow key={f.name} feature={f} />)}
          </div>
          {filtered.length > visibleCount && (
            <button
              type="button"
              onClick={() => setVisibleCount(filtered.length)}
              className="mt-3 w-full border-t border-dashed border-border pt-2 text-xs font-semibold text-accent"
            >
              Show {filtered.length - visibleCount} more flagged features ↓
            </button>
          )}
        </div>
      )}
    </div>
  );
}
```

Same note as Task 8: confirm real color-token class names against
`FailuresTab.tsx` before finalizing.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app/frontend && npx vitest run components/ops/__tests__/FeatureHealthTab.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/ops/FeatureHealthTab.tsx app/frontend/components/ops/__tests__/FeatureHealthTab.test.tsx
git commit -m "feat: redesign FeatureHealthTab as a severity-sorted, searchable flagged-feature list"
```

---

### Task 10: New `PerformanceTab` — weekly chart + metric/strategy toggles

**Files:**
- Create: `app/frontend/components/ops/PerformanceTab.tsx`
- Test: `app/frontend/components/ops/__tests__/PerformanceTab.test.tsx`
- Modify: `app/frontend/components/OpsDashboard.tsx` (swap `<AgentPerformanceContent />` for `<PerformanceTab weeks={weeks} league={league || undefined} />` in the Performance tab's panel; remove the now-fully-unused `AgentPerformanceContent` import if nothing else in the file uses it)
- Modify: `app/frontend/components/__tests__/OpsDashboard.test.tsx` (update its mock for `@/lib/api` to include `getOpsPerformanceTrend` instead of `getAgentPerformanceDashboard`/`getStatus`/`getSandboxStatus`, since the Performance tab no longer renders `AgentPerformanceContent`)

This is a new component, not a redesign of an existing one — but it still
follows the exact same established fetch/loading/error pattern as every other
tab in this dashboard.

- [ ] **Step 1: Write the failing test**

```typescript
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PerformanceTab } from "../PerformanceTab";
import { getOpsPerformanceTrend } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  getOpsPerformanceTrend: vi.fn(),
}));

describe("PerformanceTab", () => {
  it("renders a bar per week and a week-over-week delta", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [
        { week_start: "2026-10-29", hit_rate: 0.58, roi: 0.1, bets_placed: 10, total_profit: 50, ending_bankroll: 1050 },
        { week_start: "2026-11-05", hit_rate: 0.71, roi: 0.2, bets_placed: 8, total_profit: 80, ending_bankroll: 1080 },
      ],
      pick_scope: "one_per_match", include_conditional: false,
    });
    render(<PerformanceTab weeks={8} />);
    expect(await screen.findByText(/71%/)).toBeInTheDocument();
    expect(screen.getByText(/\+13pp/)).toBeInTheDocument();
  });

  it("re-fetches with the roi metric mode when selected", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-11-05", hit_rate: 0.5, roi: 0.25, bets_placed: 4, total_profit: 20, ending_bankroll: 1020 }],
      pick_scope: "one_per_match", include_conditional: false,
    });
    render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    fireEvent.click(screen.getByRole("button", { name: "ROI" }));
    expect(await screen.findByText(/25%/)).toBeInTheDocument();
  });

  it("re-fetches with the chosen strategy toggles", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-11-05", hit_rate: 0.5, roi: 0.1, bets_placed: 4, total_profit: 20, ending_bankroll: 1020 }],
      pick_scope: "all_direct", include_conditional: true,
    });
    render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    fireEvent.click(screen.getByRole("button", { name: /all direct/i }));
    fireEvent.click(screen.getByRole("button", { name: /include conditional/i }));
    expect(getOpsPerformanceTrend).toHaveBeenLastCalledWith(
      expect.objectContaining({ pickScope: "all_direct", includeConditional: true }),
    );
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsPerformanceTrend).mockRejectedValue(new Error("boom"));
    render(<PerformanceTab weeks={8} />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app/frontend && npx vitest run components/ops/__tests__/PerformanceTab.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `PerformanceTab.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";

import { getOpsPerformanceTrend } from "@/lib/api";
import type { OpsPerformanceTrendResponse, PerformanceTrendWeek, PickScope } from "@/lib/types";
import { ErrorState } from "../MatchUI";

type Metric = "hit_rate" | "roi" | "bets_placed" | "total_profit";

const METRIC_LABELS: { key: Metric; label: string }[] = [
  { key: "hit_rate", label: "Hit Rate" },
  { key: "roi", label: "ROI" },
  { key: "bets_placed", label: "Bet Frequency" },
  { key: "total_profit", label: "Total Profit" },
];

function formatMetric(metric: Metric, week: PerformanceTrendWeek): string {
  const value = week[metric];
  if (metric === "hit_rate" || metric === "roi") return `${Math.round(value * 100)}%`;
  if (metric === "bets_placed") return String(value);
  return value.toFixed(1);
}

export function PerformanceTab({ weeks, league }: { weeks: number; league?: string }) {
  const [data, setData] = useState<OpsPerformanceTrendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);
  const [metric, setMetric] = useState<Metric>("hit_rate");
  const [pickScope, setPickScope] = useState<PickScope>("one_per_match");
  const [includeConditional, setIncludeConditional] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getOpsPerformanceTrend({ weeks, league, pickScope, includeConditional })
      .then((result) => { if (!cancelled) setData(result); })
      .catch((err) => { if (!cancelled) setError(err.message ?? "Failed to load performance trend"); });
    return () => { cancelled = true; };
  }, [weeks, league, pickScope, includeConditional, retryTick]);

  if (error) return <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />;
  if (!data) return <div className="text-sm text-muted">Loading…</div>;

  const displayWeeks = data.weeks;
  const last = displayWeeks[displayWeeks.length - 1];
  const prev = displayWeeks[displayWeeks.length - 2];
  const delta = last && prev && (metric === "hit_rate" || metric === "roi")
    ? Math.round((last[metric] - prev[metric]) * 100)
    : null;

  const maxValue = Math.max(...displayWeeks.map((w) => w[metric]), 0.0001);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        {METRIC_LABELS.map((m) => (
          <button
            key={m.key}
            type="button"
            onClick={() => setMetric(m.key)}
            className={`rounded-full px-3 py-1 text-xs font-semibold ${metric === m.key ? "bg-accent text-white" : "border border-border text-muted"}`}
          >
            {m.label}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        <button
          type="button"
          onClick={() => setPickScope(pickScope === "one_per_match" ? "all_direct" : "one_per_match")}
          className="rounded border border-border px-2 py-1"
        >
          {pickScope === "one_per_match" ? "One pick per match" : "All direct bets"}
        </button>
        <button
          type="button"
          onClick={() => setIncludeConditional((v) => !v)}
          aria-pressed={includeConditional}
          className="rounded border border-border px-2 py-1"
        >
          {includeConditional ? "Including conditional" : "Include conditional"}
        </button>
      </div>

      <div className="flex items-end gap-2 rounded-lg border border-border bg-surface p-4" style={{ height: 160 }}>
        {displayWeeks.map((w) => (
          <div key={w.week_start} className="flex flex-1 flex-col items-center justify-end gap-1" style={{ height: "100%" }}>
            <span className="font-mono text-[10px]">{formatMetric(metric, w)}</span>
            <div
              className={`w-full rounded-t ${w.week_start === last?.week_start ? "bg-good" : "bg-accent"}`}
              style={{ height: `${Math.max((w[metric] / maxValue) * 100, 2)}%` }}
            />
            <span className="text-[9px] text-muted">{w.week_start}</span>
          </div>
        ))}
      </div>

      {delta !== null && (
        <span className={`inline-block w-fit rounded px-2 py-1 text-xs font-bold ${delta >= 0 ? "bg-good/15 text-good" : "bg-serious/15 text-serious"}`}>
          {delta >= 0 ? "+" : ""}{delta}pp vs previous week
        </span>
      )}
    </div>
  );
}
```

Replace the hardcoded bar-chart height/axis approach above with real
labeled-axis tick marks if a plain CSS flex-height bar chart (no numeric axis
drawn) feels like a regression against the already-approved mockup's explicit
y-axis requirement — re-read this plan's design doc section 7 and the
approved mockup before finalizing: the mockup used inline SVG with real
gridlines/tick labels for the weekly bar chart, which is the actual bar
(no pun intended) to match, not a plainer CSS-only chart. Adapt the SVG
pattern from the mockup (`.superpowers/brainstorm/.../ops-dashboard-full.html`,
if still on disk, or reconstruct the same shape: `viewBox`, gridlines at
0/25/50/75/100% or an auto-scaled equivalent for non-percentage metrics,
`<rect>` bars, `<text>` labels) rather than the simplified div-height
approach sketched here, which was a shortcut for this plan document's own
brevity, not the final bar.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app/frontend && npx vitest run components/ops/__tests__/PerformanceTab.test.tsx`
Expected: PASS

- [ ] **Step 5: Wire it into `OpsDashboard.tsx`**

Replace `{tab === "performance" && <AgentPerformanceContent />}` with
`{tab === "performance" && <PerformanceTab weeks={weeks} league={league || undefined} />}`.
Remove the `AgentPerformanceContent` import if it's now unused in this file.
Update `OpsDashboard.test.tsx`'s `vi.mock("@/lib/api", ...)` factory to add
`getOpsPerformanceTrend: vi.fn()` (and you may remove
`getAgentPerformanceDashboard`/`getStatus`/`getSandboxStatus` from that
mock if this file's tests no longer exercise the Performance tab's old
content — check first whether any existing test in this file actually
renders the Performance tab and asserts on `AgentPerformanceContent`'s own
content; if so, update that test's expectations to the new `PerformanceTab`
instead of just deleting coverage).

- [ ] **Step 6: Run the full ops test suite**

Run: `cd app/frontend && npx vitest run components/ops components/__tests__/OpsDashboard.test.tsx`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add app/frontend/components/ops/PerformanceTab.tsx app/frontend/components/ops/__tests__/PerformanceTab.test.tsx app/frontend/components/OpsDashboard.tsx app/frontend/components/__tests__/OpsDashboard.test.tsx
git commit -m "feat: add PerformanceTab with weekly trend chart + metric/strategy toggles"
```

---

### Task 11: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Full backend suite**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend -q`
Expected: no new failures beyond the documented pre-existing baseline.

- [ ] **Step 2: Full frontend suite + typecheck + build**

Run:
```bash
cd app/frontend
npx vitest run
npx tsc --noEmit
npm run build
```
Expected: all clean. The production build in particular will catch any
remaining integration issue a unit test wouldn't (static generation, "use
client" boundaries, etc.) — same extra signal this dashboard's original build
used at its own final task.

- [ ] **Step 3: Manual/best-effort smoke check**

Follow the same adapted approach used for this dashboard's original Task 8
(no browser automation tool available in this environment): confirm the
production build succeeds, and name exactly which automated tests cover
"changing the Window selector actually changes the Performance tab's
numbers" (this plan's whole reason for existing) — point to the specific
test(s) in `PerformanceTab.test.tsx`/`OpsDashboard.test.tsx` that prove this,
rather than claiming it was manually verified in a browser if it wasn't.

- [ ] **Step 4: Report**

Summarize: test counts before/after this plan, confirmation of the documented
pre-existing-failure baseline (unchanged), any judgment calls made while
implementing (e.g. the exact drift/null-rate threshold constants actually
shipped, if tuned differently than this plan's starting suggestions), and any
follow-ups worth flagging (e.g. `max_drawdown`/`insufficient_data_rate` modes
explicitly deferred per the design doc's non-goals).

---

## Self-review notes

- **Spec coverage**: every section of `docs/superpowers/specs/2026-11-03-ops-dashboard-trends-design.md` has a corresponding task — §4 (filter bar) → Task 7; §5 (odds availability) → Tasks 2, 8; §6 (feature health) → Tasks 1, 3, 9; §7 (performance) → Tasks 4, 5, 10.
- **Placeholder scan**: every code step has real, complete code; the two
  places code is explicitly flagged as "verify/correct against the real
  function before trusting this sketch" (Task 2's `likely_cause` branching,
  Task 4's `BacktestRecord` field names, Task 10's bar-chart axis) are
  deliberate "don't blindly copy, the real signature/requirement wins"
  instructions — the same category of guidance used successfully throughout
  this dashboard's earlier build (e.g. Task 2's `FailureRow.alerted` type
  correction), not missing requirements.
- **Type consistency**: frontend types (`OpsOddsAvailabilityTrendResponse`,
  `OpsFeatureHealthTrendResponse`, `OpsPerformanceTrendResponse`) match the
  backend endpoints' literal response shapes field-for-field as specified in
  each backend task.
- **Scope**: 11 tasks, each independently testable and committable — larger
  than this dashboard's original 8-task plan, reflecting the genuinely larger
  scope here (new aggregation logic + new staking-strategy reuse, not just
  new UI on existing snapshot data), but each task is still bite-sized and
  self-contained, matching this plan document's own granularity rules.
