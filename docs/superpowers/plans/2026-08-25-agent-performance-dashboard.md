# Agent Performance Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local-only dashboard page showing the agent's live performance — main P&L metrics, three segmentation breakdown tables, three distribution charts, and top-5 winning/losing bet examples — built entirely on data already tracked (`recommendation_outcomes`, W167/W168).

**Architecture:** Extend the existing Kelly-simulation pipeline (`src/agent/evaluation.py`, `app/backend/recommendation_stats.py`) with per-segment reports and a raw staked-bets list; add one new backend module for match/team-name enrichment of the top/bottom examples (the only piece needing DB I/O); wire one new endpoint; build one new, unlinked frontend page reusing the app's existing dark-theme design tokens and dataviz-validated categorical palette.

**Tech Stack:** Python (FastAPI, pytest), TypeScript/React (Vitest) — no new dependencies, no new charting library (hand-rolled SVG-free CSS bars/histograms matching the codebase's existing zero-dependency style).

**Spec:** `docs/superpowers/specs/2026-08-25-agent-performance-dashboard-design.md`
**Stories:** `documents/agent_user_stories.md` A83 (new), `documents/app_user_stories.md` W170–W174 (new)

---

## File Structure

**Backend (modify/create):**
- Modify `src/agent/evaluation.py` — `build_evaluation_report` gains `total_staked`/`total_profit` in its return dict.
- Modify `app/backend/recommendation_stats.py` — new `_segment_kelly_report`, three new segment-metrics keys, `staked_bets` key.
- Create `app/backend/agent_performance_dashboard.py` — `compute_agent_performance_dashboard()` (top/bottom sorting + team-name enrichment via `RecommendationCache`).
- Modify `app/backend/main.py` — `GET /api/recommendations/performance-dashboard`.

**Frontend (modify/create):**
- Modify `app/frontend/lib/types.ts` — response types.
- Modify `app/frontend/lib/api.ts` — `getAgentPerformanceDashboard()`.
- Modify `app/frontend/components/MatchUI.tsx` — export `marketLabel` (currently private, needed by the new page).
- Modify `app/frontend/components/AppShell.tsx` — widen `active` prop's type to accept `"agent-performance"` (same "valid active value with no nav entry" pattern `"bets"` already uses).
- Create `app/frontend/components/AgentPerformanceDashboard.tsx` — the page component.
- Create `app/frontend/app/agent-performance/page.tsx` — thin route wrapper (not added to `AppShell`'s `NAV_ITEMS`).

**Tests (modify/create):**
- Modify `tests/test_agent_evaluation.py`, `app/backend/tests/test_recommendation_stats.py`.
- Create `app/backend/tests/test_agent_performance_dashboard.py`, `app/backend/tests/test_agent_performance_dashboard_endpoint.py`.
- Create `app/frontend/components/__tests__/AgentPerformanceDashboard.test.tsx`.

---

### Task 1: Extend `build_evaluation_report` with total_staked/total_profit (A83)

**Files:**
- Modify: `src/agent/evaluation.py`
- Test: `tests/test_agent_evaluation.py`

- [ ] **Step 1: Write the failing test assertions**

In `tests/test_agent_evaluation.py`, find `test_build_evaluation_report_computes_roi_and_hit_rate` and add two assertions at the end of it:

```python
    assert report["roi"] == 1.0  # 10 profit / 10 staked
    assert report["bet_frequency"] == 0.5  # 1 bet / 2 matches
    assert report["insufficient_data_rate"] == 0.5
    assert report["matches_evaluated"] == 2
    assert report["total_staked"] == 10.0
    assert report["total_profit"] == 10.0
```

(This replaces the existing 4-line assertion block ending in `assert report["matches_evaluated"] == 2` — the two new lines are appended after it, same function.)

Also add a new test right after `test_build_evaluation_report_handles_zero_bets`:

```python
def test_build_evaluation_report_total_staked_and_profit_zero_when_no_bets():
    bankroll = BankrollResult(starting_bankroll=1000.0, ending_bankroll=1000.0, equity_curve=[1000.0], bets=[])

    class _Rec:
        recommendation = {"overall": "no_bet"}

    report = build_evaluation_report([_Rec()], bankroll)
    assert report["total_staked"] == 0.0
    assert report["total_profit"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_evaluation.py -v`
Expected: FAIL — `KeyError: 'total_staked'` on the modified test, and the new test errors the same way.

- [ ] **Step 3: Add the two fields**

In `src/agent/evaluation.py`, find the `return` statement inside `build_evaluation_report`:

```python
    return {
        "matches_evaluated": len(records),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "roi": round(roi, 6),
        "hit_rate": round(hit_rate, 6),
        "bet_frequency": round(bet_frequency, 6),
        "max_drawdown": round(compute_max_drawdown(bankroll_result.equity_curve), 6),
        "insufficient_data_rate": round(insufficient_data_rate, 6),
        "starting_bankroll": bankroll_result.starting_bankroll,
        "ending_bankroll": round(bankroll_result.ending_bankroll, 2),
    }
```

Replace with:

```python
    return {
        "matches_evaluated": len(records),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "roi": round(roi, 6),
        "hit_rate": round(hit_rate, 6),
        "bet_frequency": round(bet_frequency, 6),
        "max_drawdown": round(compute_max_drawdown(bankroll_result.equity_curve), 6),
        "insufficient_data_rate": round(insufficient_data_rate, 6),
        "starting_bankroll": bankroll_result.starting_bankroll,
        "ending_bankroll": round(bankroll_result.ending_bankroll, 2),
        # A83: already computed above (total_staked/total_profit locals) --
        # just never returned. Needed by the agent performance dashboard's
        # Main Metrics row (Total Stake, Money Won). Purely additive: every
        # existing caller (main.py's agent-backtest/agent-train reporting,
        # src/agent/comparison.py, recommendation_stats.py) reads specific
        # keys or dumps the dict generically (print_report/save_report both
        # iterate report.items()) -- nothing breaks from two new keys.
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_evaluation.py -v`
Expected: PASS, all tests including the modified and new one.

Then run the wider suite to confirm nothing that reads this dict's keys generically broke:

Run: `pytest tests/test_comparison.py app/backend/tests/test_recommendation_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/evaluation.py tests/test_agent_evaluation.py
git commit -m "feat(agent): add total_staked/total_profit to build_evaluation_report (A83)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 1 of a 7-task plan for the Agent Performance Dashboard. `build_evaluation_report` (`src/agent/evaluation.py`) already computes `total_staked`/`total_profit` as local variables but never returns them — this is a purely additive extension, no signature change, no behavior change to any existing field.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 2: Extend recommendation_stats.py with segment breakdowns + staked_bets (W170)

**Files:**
- Modify: `app/backend/recommendation_stats.py`
- Modify: `app/backend/tests/test_recommendation_stats.py`

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/tests/test_recommendation_stats.py`:

```python
def test_segment_kelly_report_groups_by_key_fn():
    outcomes = [
        _outcome(match_id="m1", market="result_3way", odds=2.0, value_edge=0.1, correct=True),
        _outcome(match_id="m2", market="btts", odds=1.8, value_edge=0.1, correct=False),
    ]
    result = _segment_kelly_report(outcomes, lambda o: o.market)
    assert set(result.keys()) == {"result_3way", "btts"}
    assert result["result_3way"]["bets_placed"] == 1
    assert result["result_3way"]["bets_won"] == 1
    assert result["btts"]["bets_placed"] == 1
    assert result["btts"]["bets_won"] == 0


def test_segment_kelly_report_groups_none_key_as_unknown():
    outcomes = [_outcome(match_id="m1", competition=None, odds=2.0, value_edge=0.1, correct=True)]
    result = _segment_kelly_report(outcomes, lambda o: o.competition)
    assert set(result.keys()) == {"unknown"}


def test_by_market_metrics_present_on_compute_recommendation_stats():
    outcomes = [_outcome(match_id="m1", market="result_3way", odds=2.0, value_edge=0.1, correct=True)]
    stats = compute_recommendation_stats(outcomes)
    assert stats["by_market_metrics"]["result_3way"]["bets_placed"] == 1
    assert "total_staked" in stats["by_market_metrics"]["result_3way"]


def test_by_market_selection_metrics_uses_composite_key():
    outcomes = [_outcome(match_id="m1", market="result_3way", selection="home", odds=2.0, value_edge=0.1, correct=True)]
    stats = compute_recommendation_stats(outcomes)
    assert "result_3way:home" in stats["by_market_selection_metrics"]


def test_by_league_metrics_uses_competition_field():
    outcomes = [_outcome(match_id="m1", competition="E0", odds=2.0, value_edge=0.1, correct=True)]
    stats = compute_recommendation_stats(outcomes)
    assert "E0" in stats["by_league_metrics"]


def test_staked_bets_is_a_list_of_plain_dicts():
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True, recommendation_type="direct_bet")]
    stats = compute_recommendation_stats(outcomes)
    assert isinstance(stats["staked_bets"], list)
    assert len(stats["staked_bets"]) == 1
    bet = stats["staked_bets"][0]
    assert isinstance(bet, dict)
    assert bet["match_id"] == "m1"
    assert bet["won"] is True
    assert bet["payout"] > 0


def test_staked_bets_excludes_conditional_picks():
    # simulate_kelly_stake only ever stakes direct_bet -- conditional picks
    # never appear in staked_bets even though they're resolved outcomes.
    outcomes = [_outcome(match_id="m1", recommendation_type="conditional", odds=2.0, value_edge=0.1, correct=True)]
    stats = compute_recommendation_stats(outcomes)
    assert stats["staked_bets"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_recommendation_stats.py -v`
Expected: FAIL — `ImportError`/`NameError` on `_segment_kelly_report`, `KeyError` on the new dict keys.

- [ ] **Step 3: Implement the extension**

In `app/backend/recommendation_stats.py`, add `import dataclasses` to the imports (after `from collections import defaultdict`):

```python
from collections import defaultdict
import dataclasses
from typing import Any
```

Add `_segment_kelly_report` right after `_to_backtest_records` (before `compute_recommendation_stats`):

```python
def _segment_kelly_report(
    outcomes: list[RecommendationOutcome],
    key_fn,
    starting_bankroll: float = DEFAULT_STARTING_BANKROLL,
) -> dict[str, dict[str, Any]]:
    """Same hit-rate-plus-Kelly-ROI report compute_recommendation_stats's
    own kelly_roi_simulation produces, run once per group instead of once
    overall -- powers the dashboard's Market / Market+Direction / League
    breakdown tables (W170). None-valued keys (competition can be null)
    group under "unknown", matching _breakdown_by's own convention."""
    groups: dict[str, list[RecommendationOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[key_fn(outcome) or "unknown"].append(outcome)
    result: dict[str, dict[str, Any]] = {}
    for group_key, group_outcomes in groups.items():
        records = _to_backtest_records(group_outcomes)
        bankroll_result = simulate_kelly_stake(records, starting_bankroll=starting_bankroll)
        result[group_key] = build_evaluation_report(records, bankroll_result)
    return result
```

Replace `compute_recommendation_stats`'s body:

```python
def compute_recommendation_stats(
    outcomes: list[RecommendationOutcome], starting_bankroll: float = DEFAULT_STARTING_BANKROLL
) -> dict[str, Any]:
    records = _to_backtest_records(outcomes)
    bankroll_result = simulate_kelly_stake(records, starting_bankroll=starting_bankroll)

    return {
        "overall": _hit_rate(outcomes),
        "by_market": _breakdown_by(outcomes, "market"),
        "by_competition": _breakdown_by(outcomes, "competition"),
        "by_confidence": _breakdown_by(outcomes, "confidence"),
        "kelly_roi_simulation": build_evaluation_report(records, bankroll_result),
    }
```

Replace with:

```python
def compute_recommendation_stats(
    outcomes: list[RecommendationOutcome], starting_bankroll: float = DEFAULT_STARTING_BANKROLL
) -> dict[str, Any]:
    records = _to_backtest_records(outcomes)
    bankroll_result = simulate_kelly_stake(records, starting_bankroll=starting_bankroll)

    return {
        "overall": _hit_rate(outcomes),
        "by_market": _breakdown_by(outcomes, "market"),
        "by_competition": _breakdown_by(outcomes, "competition"),
        "by_confidence": _breakdown_by(outcomes, "confidence"),
        "kelly_roi_simulation": build_evaluation_report(records, bankroll_result),
        # W170: full (hit-rate + Kelly ROI) metrics per segment, for the
        # dashboard's three breakdown tables -- distinct from by_market/
        # by_competition above, which only report hit-rate across every
        # resolved pick (conditional included). These three are scoped to
        # the same staked (direct_bet-only) population as kelly_roi_simulation.
        "by_market_metrics": _segment_kelly_report(outcomes, lambda o: o.market, starting_bankroll),
        "by_market_selection_metrics": _segment_kelly_report(
            outcomes, lambda o: f"{o.market}:{o.selection}", starting_bankroll
        ),
        "by_league_metrics": _segment_kelly_report(outcomes, lambda o: o.competition, starting_bankroll),
        # Raw per-bet list (plain dicts, not BetOutcome instances -- always
        # JSON-safe without relying on FastAPI's implicit dataclass
        # handling) -- feeds the dashboard's odds/stake histograms (bucketed
        # client-side) and is the source list compute_agent_performance_dashboard
        # sorts for its top/bottom examples.
        "staked_bets": [dataclasses.asdict(bet) for bet in bankroll_result.bets],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendation_stats.py -v`
Expected: PASS, 12 passed (5 pre-existing + 7 new).

Then run the wider stats/endpoint suite to confirm the existing `/stats` endpoint (which calls `compute_recommendation_stats` directly) still works:

Run: `pytest app/backend/tests/test_recommendation_outcomes_endpoints.py -v`
Expected: PASS, all still passing (that endpoint returns this dict verbatim -- more keys, no removed/changed ones).

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendation_stats.py app/backend/tests/test_recommendation_stats.py
git commit -m "feat(app): add per-segment Kelly reports + staked_bets to recommendation_stats (W170)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 2 of a 7-task plan. Depends on Task 1 (already done, committed): `build_evaluation_report` now returns `total_staked`/`total_profit` too, which `_segment_kelly_report`'s per-group calls automatically pick up (no code change needed there beyond what Task 1 already did). `app/backend/recommendation_stats.py` already exists with `_hit_rate`, `_breakdown_by`, `_to_backtest_records`, `compute_recommendation_stats`, `RecommendationOutcome`, `BacktestRecord`, `simulate_kelly_stake` all imported/defined — you're extending this existing file, not creating a new one. The existing `GET /api/recommendations/stats` endpoint (`main.py`) already calls `compute_recommendation_stats` and returns its result directly (`-> dict`) — adding new keys to that dict is purely additive from that endpoint's perspective, nothing to change there.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 3: Create agent_performance_dashboard.py (W171)

**Files:**
- Create: `app/backend/agent_performance_dashboard.py`
- Create: `app/backend/tests/test_agent_performance_dashboard.py`

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_agent_performance_dashboard.py`:

```python
"""W171: top/bottom staked-bet examples enriched with match/team context
for the agent performance dashboard. The only piece of this feature that
needs RecommendationCache (DB I/O) -- recommendation_stats.py stays pure
aggregation on purpose."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.agent_performance_dashboard import compute_agent_performance_dashboard
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcome


def _outcome(match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
             recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True) -> RecommendationOutcome:
    return RecommendationOutcome(
        id=1, match_id=match_id, date=date, competition=competition, market=market, selection=selection,
        recommendation_type=recommendation_type, confidence=confidence, odds=odds, value_edge=value_edge,
        correct=correct, generated_at="2026-08-22T10:00:00+00:00", resolved_at="2026-08-23T00:00:00+00:00",
    )


_REC = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1}],
    "confidence": "medium", "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
}


def test_top_winners_enriched_with_team_names(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _REC, "scheduled")
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]

    result = compute_agent_performance_dashboard(outcomes, cache)

    assert len(result["top_winners"]) == 1
    winner = result["top_winners"][0]
    assert winner["home_team"] == "Arsenal"
    assert winner["away_team"] == "Everton"
    assert winner["payout"] > 0
    assert winner["date"] == "2026-08-22"
    assert winner["competition"] == "E0"


def test_top_losers_enriched_and_sorted_most_negative_first(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _REC, "scheduled")
    cache.record_generation("m2", "2026-08-23", "hash1", {}, {**_REC, "match": {**_REC["match"], "home": "Chelsea", "away": "Brighton"}}, "scheduled")
    outcomes = [
        _outcome(match_id="m1", date="2026-08-22", odds=3.0, value_edge=0.2, correct=False),
        _outcome(match_id="m2", date="2026-08-23", odds=2.0, value_edge=0.1, correct=False),
    ]

    result = compute_agent_performance_dashboard(outcomes, cache)

    assert len(result["top_losers"]) == 2
    # Most negative payout first (bigger stake loses more since value_edge is higher).
    assert result["top_losers"][0]["payout"] < result["top_losers"][1]["payout"]
    assert result["top_losers"][0]["home_team"] == "Arsenal"


def test_cache_miss_degrades_team_names_to_none_not_a_crash(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")  # nothing recorded
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]

    result = compute_agent_performance_dashboard(outcomes, cache)

    assert len(result["top_winners"]) == 1
    assert result["top_winners"][0]["home_team"] is None
    assert result["top_winners"][0]["away_team"] is None
    assert result["top_winners"][0]["match_id"] == "m1"


def test_respects_top_n(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    outcomes = [
        _outcome(match_id=f"m{i}", date="2026-08-22", odds=2.0, value_edge=0.1 + i * 0.01, correct=True)
        for i in range(3)
    ]

    result = compute_agent_performance_dashboard(outcomes, cache, top_n=2)

    assert len(result["top_winners"]) == 2


def test_empty_outcomes_returns_empty_top_lists(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    result = compute_agent_performance_dashboard([], cache)
    assert result["top_winners"] == []
    assert result["top_losers"] == []


def test_no_losers_when_every_staked_bet_won(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]
    result = compute_agent_performance_dashboard(outcomes, cache)
    assert result["top_losers"] == []
    assert len(result["top_winners"]) == 1


def test_result_still_includes_everything_compute_recommendation_stats_returns(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]
    result = compute_agent_performance_dashboard(outcomes, cache)
    assert "overall" in result
    assert "by_market_metrics" in result
    assert "kelly_roi_simulation" in result
    assert "staked_bets" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_agent_performance_dashboard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backend.agent_performance_dashboard'`

- [ ] **Step 3: Implement agent_performance_dashboard.py**

Create `app/backend/agent_performance_dashboard.py`:

```python
"""W171: top/bottom staked-bet examples for the agent performance
dashboard, enriched with match date/competition/team names. The only piece
of this feature needing RecommendationCache (DB I/O) -- recommendation_stats.py
stays pure aggregation on purpose, mirroring bet_stats.py's own separation
from bet_tracker.py."""

from __future__ import annotations

from typing import Any

from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcome
from app.backend.recommendation_stats import compute_recommendation_stats
from src.agent.schema import reported_teams


def _enrich_bet(
    bet: dict[str, Any],
    outcomes_by_match: dict[str, RecommendationOutcome],
    cache: RecommendationCache,
) -> dict[str, Any]:
    """Attaches date/competition (from the resolved outcome itself, always
    available) and home_team/away_team (from the cached recommendation's
    own self-reported match field, via reported_teams() -- the same
    helper already shared for this exact home/home_team key-spelling
    ambiguity elsewhere in this codebase, BUG-023/024). A cache miss (the
    recommendation was purged, or a genuine race) degrades team names to
    None rather than failing the whole dashboard -- same "never let one
    bad row break the page" discipline validate_and_degrade already uses."""
    outcome = outcomes_by_match.get(bet["match_id"])
    date = outcome.date if outcome else None
    competition = outcome.competition if outcome else None
    home_team: str | None = None
    away_team: str | None = None
    if date is not None:
        entry = cache.get_latest_any_config(bet["match_id"], date)
        if entry is not None:
            teams = reported_teams(entry.recommendation.get("match") or {})
            if teams is not None:
                home_team, away_team = teams
    return {**bet, "date": date, "competition": competition, "home_team": home_team, "away_team": away_team}


def compute_agent_performance_dashboard(
    outcomes: list[RecommendationOutcome], cache: RecommendationCache, top_n: int = 5
) -> dict[str, Any]:
    """Everything compute_recommendation_stats already returns, plus
    top_winners/top_losers: the top_n highest- and lowest-payout entries
    from staked_bets, enriched with match context. Only these ≤2×top_n
    rows get the (relatively) expensive cache lookup -- not every staked
    bet, however many there are."""
    stats = compute_recommendation_stats(outcomes)
    staked_bets = stats["staked_bets"]
    outcomes_by_match = {o.match_id: o for o in outcomes}

    winners = sorted((b for b in staked_bets if b["payout"] > 0), key=lambda b: b["payout"], reverse=True)[:top_n]
    losers = sorted((b for b in staked_bets if b["payout"] < 0), key=lambda b: b["payout"])[:top_n]

    return {
        **stats,
        "top_winners": [_enrich_bet(b, outcomes_by_match, cache) for b in winners],
        "top_losers": [_enrich_bet(b, outcomes_by_match, cache) for b in losers],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_agent_performance_dashboard.py -v`
Expected: PASS, 7 passed.

- [ ] **Step 5: Commit**

```bash
git add app/backend/agent_performance_dashboard.py app/backend/tests/test_agent_performance_dashboard.py
git commit -m "feat(app): agent_performance_dashboard -- top/bottom bet enrichment (W171)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 3 of a 7-task plan. Depends on Task 2 (already done, committed): `compute_recommendation_stats` now returns `staked_bets` (a list of plain dicts with `match_id, market, selection, odds, stake, won, payout`). `RecommendationCache.get_latest_any_config(match_id, date) -> CacheEntry | None` already exists (`app/backend/recommendation_cache.py`); `CacheEntry.recommendation` is the raw recommendation dict, whose `"match"` key holds `{"home"/"home_team": ..., "away"/"away_team": ...}`. `reported_teams(match_field: dict) -> tuple[str, str] | None` already exists in `src/agent/schema.py` and handles exactly that key-spelling ambiguity -- import and use it, don't reimplement it.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 4: Wire GET /api/recommendations/performance-dashboard (W172)

**Files:**
- Modify: `app/backend/main.py`
- Create: `app/backend/tests/test_agent_performance_dashboard_endpoint.py`

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_agent_performance_dashboard_endpoint.py`:

```python
"""W172: GET /api/recommendations/performance-dashboard."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.main import app
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore, get_recommendation_outcome_store


def _override(tmp_path: Path) -> tuple[RecommendationCache, RecommendationOutcomeStore]:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    app.dependency_overrides[recommendations.get_cache] = lambda: cache
    app.dependency_overrides[get_recommendation_outcome_store] = lambda: store
    return cache, store


def test_dashboard_endpoint_returns_empty_state_with_no_data(tmp_path: Path):
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard")
        assert response.status_code == 200
        body = response.json()
        assert body["kelly_roi_simulation"]["bets_placed"] == 0
        assert body["top_winners"] == []
        assert body["top_losers"] == []
        assert body["staked_bets"] == []
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_reflects_a_resolved_outcome(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard")
        assert response.status_code == 200
        body = response.json()
        assert body["kelly_roi_simulation"]["bets_placed"] == 1
        assert len(body["top_winners"]) == 1
        assert "by_market_metrics" in body
        assert "result_3way" in body["by_market_metrics"]
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_respects_top_n_param(tmp_path: Path):
    _, store = _override(tmp_path)
    for i in range(3):
        store.insert(
            match_id=f"m{i}", date="2026-08-22", competition="E0", market="result_3way", selection="home",
            recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1 + i * 0.01, correct=True,
            generated_at="2026-08-22T10:00:00+00:00",
        )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard?top_n=2")
        assert len(response.json()["top_winners"]) == 2
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_rejects_out_of_range_top_n(tmp_path: Path):
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard?top_n=0")
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_days_param_filters_by_date(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="old", date="2020-01-01", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2020-01-01T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard?days=30")
        assert response.json()["kelly_roi_simulation"]["bets_placed"] == 0
    finally:
        app.dependency_overrides.clear()


def test_dashboard_route_registered_before_match_id_route(tmp_path: Path):
    """Same route-ordering hazard already caught once for /stats (W168) --
    {match_id} is a single-segment pattern that would otherwise swallow the
    literal "performance-dashboard" path segment."""
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard")
        # A 422 here (missing required `date` query param) would mean this
        # request got routed to get_cached_recommendation(match_id=...)
        # instead of the dashboard endpoint.
        assert response.status_code != 422
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_agent_performance_dashboard_endpoint.py -v`
Expected: FAIL with 404s (route doesn't exist yet).

- [ ] **Step 3: Add the endpoint**

In `app/backend/main.py`, add the import near the other `app.backend` imports. Find:

```python
from app.backend.recommendation_stats import compute_recommendation_stats
```

Replace with:

```python
from app.backend.agent_performance_dashboard import compute_agent_performance_dashboard
from app.backend.recommendation_stats import compute_recommendation_stats
```

Now add the new endpoint, right after `get_recommendation_stats` and before `GET /api/recommendations/{match_id}` (same ordering requirement as `/stats` itself -- see that endpoint's own docstring for why). Find:

```python
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    outcomes = store.list_all(since=cutoff)
    return compute_recommendation_stats(outcomes)


@app.get("/api/recommendations/{match_id}")
```

Replace with:

```python
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    outcomes = store.list_all(since=cutoff)
    return compute_recommendation_stats(outcomes)


@app.get("/api/recommendations/performance-dashboard")
async def get_agent_performance_dashboard(
    days: int = Query(30, ge=0, le=3650),
    top_n: int = Query(5, ge=1, le=50),
    cache: RecommendationCache = Depends(recommendations.get_cache),
    store: RecommendationOutcomeStore = Depends(get_recommendation_outcome_store),
) -> dict:
    """W171/W172: local-only diagnostics dashboard -- main metrics, segment
    breakdowns, distributions, and top/bottom staked-bet examples, all in
    one response. Not called by the deployed frontend's nav-linked pages;
    reachable only via app/agent-performance/page.tsx, which is itself
    unlinked from AppShell's nav (W174).

    Registered ahead of GET /api/recommendations/{match_id} below, same
    reason /stats already had to be: {match_id} is a single-path-segment
    pattern that would otherwise swallow "performance-dashboard" as its
    own match_id value."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    outcomes = store.list_all(since=cutoff)
    return compute_agent_performance_dashboard(outcomes, cache, top_n=top_n)


@app.get("/api/recommendations/{match_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_agent_performance_dashboard_endpoint.py -v`
Expected: PASS, 6 passed.

Then run the full backend suite:

Run: `pytest tests/ app/backend/tests/ -v`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_agent_performance_dashboard_endpoint.py
git commit -m "feat(app): wire GET /api/recommendations/performance-dashboard (W172)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 4 of a 7-task plan, the last backend task. Depends on Task 3 (already done, committed): `compute_agent_performance_dashboard(outcomes, cache, top_n=5)` exists in `app/backend/agent_performance_dashboard.py`. `app/backend/main.py` already imports `Query`, `Depends`, `RecommendationCache`, `RecommendationOutcomeStore`, `get_recommendation_outcome_store`, `recommendations` (for `recommendations.get_cache`), `datetime`, `timezone`, `timedelta` -- all reused as-is, no new imports besides the one function shown above.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 5: Frontend types + API client function (W173)

**Files:**
- Modify: `app/frontend/lib/types.ts`
- Modify: `app/frontend/lib/api.ts`

- [ ] **Step 1: Add the response types**

Append to `app/frontend/lib/types.ts`:

```typescript
// W170/W171: app/backend/agent_performance_dashboard.py's return shape.
export type HitRateSummary = {
  sample_size: number;
  correct: number;
  hit_rate: number;
};

// Mirrors src/agent/evaluation.py's build_evaluation_report() return shape,
// denominated in UB (an abstract Unit Bet, not dollars -- same convention
// as the rest of this feature).
export type SegmentMetrics = {
  matches_evaluated: number;
  bets_placed: number;
  bets_won: number;
  roi: number;
  hit_rate: number;
  bet_frequency: number;
  max_drawdown: number;
  insufficient_data_rate: number;
  starting_bankroll: number;
  ending_bankroll: number;
  total_staked: number;
  total_profit: number;
};

export type StakedBet = {
  match_id: string;
  market: string;
  selection: string;
  odds: number;
  stake: number;
  won: boolean;
  payout: number;
};

// StakedBet plus match context, only populated for top_winners/top_losers
// (W171) -- home_team/away_team are null on a cache miss, degraded, not absent.
export type TopBet = StakedBet & {
  date: string | null;
  competition: string | null;
  home_team: string | null;
  away_team: string | null;
};

export type AgentPerformanceDashboard = {
  overall: HitRateSummary;
  by_market: Record<string, HitRateSummary>;
  by_competition: Record<string, HitRateSummary>;
  by_confidence: Record<string, HitRateSummary>;
  kelly_roi_simulation: SegmentMetrics;
  by_market_metrics: Record<string, SegmentMetrics>;
  by_market_selection_metrics: Record<string, SegmentMetrics>;
  by_league_metrics: Record<string, SegmentMetrics>;
  staked_bets: StakedBet[];
  top_winners: TopBet[];
  top_losers: TopBet[];
};
```

- [ ] **Step 2: Add the API client function**

In `app/frontend/lib/api.ts`, add `AgentPerformanceDashboard` to the existing type import. Find:

```typescript
import type {
  Bet,
  BetStats,
  Fixture,
  MatchRecommendationOut,
  SandboxStatus,
  StatusResponse,
} from "./types";
```

Replace with:

```typescript
import type {
  AgentPerformanceDashboard,
  Bet,
  BetStats,
  Fixture,
  MatchRecommendationOut,
  SandboxStatus,
  StatusResponse,
} from "./types";
```

Append this function at the end of the file:

```typescript
/** W172: local-only diagnostics dashboard -- not called from any nav-linked
 * page, only app/agent-performance/page.tsx (W174, itself unlinked). */
export async function getAgentPerformanceDashboard(
  days?: number,
  topN?: number
): Promise<AgentPerformanceDashboard> {
  const params = new URLSearchParams();
  if (days !== undefined) params.set("days", String(days));
  if (topN !== undefined) params.set("top_n", String(topN));
  const query = params.toString();
  const response = await apiFetch(`/api/recommendations/performance-dashboard${query ? `?${query}` : ""}`);
  if (!response.ok) {
    throw new ApiError(`Failed to load agent performance dashboard (${response.status})`, response.status);
  }
  return response.json();
}
```

- [ ] **Step 3: Verify it typechecks**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: clean (these are pure additions -- no existing code references them yet, so nothing can be broken by them, but the new code itself must be valid TypeScript).

- [ ] **Step 4: Commit**

```bash
git add app/frontend/lib/types.ts app/frontend/lib/api.ts
git commit -m "feat(app): frontend types + API client for the performance dashboard (W173)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 5 of a 7-task plan. Depends on Task 4 (already done, committed): the backend endpoint's exact response shape is now fixed (`overall`, `by_market`, `by_competition`, `by_confidence`, `kelly_roi_simulation`, `by_market_metrics`, `by_market_selection_metrics`, `by_league_metrics`, `staked_bets`, `top_winners`, `top_losers`). This task is pure type/client-function plumbing, no UI yet -- Task 7 consumes what this task adds. `app/frontend/lib/api.ts` already has an `apiFetch()` helper and an `ApiError` class used by every other function in the file (e.g. `getFixtures`, `getBetStats`) -- follow that exact existing pattern, don't invent a new one.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, verify it typechecks, commit, self-review, report back. (No runtime tests for this task -- it's pure types + a thin fetch wrapper matching an established pattern; Task 7's component tests exercise it end-to-end.)

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, typecheck result, files changed, self-review findings, commit SHA.

---

### Task 6: Small prerequisite exports (MatchUI.tsx, AppShell.tsx)

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx`
- Modify: `app/frontend/components/AppShell.tsx`

- [ ] **Step 1: Export `marketLabel` from MatchUI.tsx**

Find:

```typescript
function marketLabel(market: string): { label: string; subtitle: string | null } {
```

Replace with:

```typescript
// W174: exported so AgentPerformanceDashboard.tsx can reuse the same
// human-readable market names ("3-Way Result" instead of "result_3way")
// instead of duplicating MARKET_LABEL.
export function marketLabel(market: string): { label: string; subtitle: string | null } {
```

- [ ] **Step 2: Widen AppShell's `active` prop type**

In `app/frontend/components/AppShell.tsx`, find:

```typescript
export function AppShell({
  active,
  railTrigger,
  children,
}: {
  active: "dashboard" | "matches" | "bets";
```

Replace with:

```typescript
export function AppShell({
  active,
  railTrigger,
  children,
}: {
  // W174: "agent-performance" has no NAV_ITEMS entry (the page itself is
  // unlinked -- direct-URL-only, matching "bets"'s own existing precedent:
  // NAV_ITEMS omits it too, but "bets" stays a valid `active` value here
  // for BetTrackerPage's own unaffected use). Adding a value here never
  // requires a matching NAV_ITEMS entry -- the two are independent.
  active: "dashboard" | "matches" | "bets" | "agent-performance";
```

- [ ] **Step 3: Verify nothing broke**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: clean.

Run: `cd app/frontend && npx vitest run components/__tests__/MatchUI.test.tsx components/__tests__/AppShell.test.tsx`
Expected: PASS, all tests (widening a union type and adding `export` to an existing function are both non-breaking).

- [ ] **Step 4: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/AppShell.tsx
git commit -m "refactor(app): export marketLabel, widen AppShell active union for agent-performance (W174 prereq)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 6 of a 7-task plan -- two tiny, independent prerequisite changes for Task 7 (the actual dashboard page). Both are non-breaking widenings (adding `export`, adding a union member) with no behavior change to any existing consumer.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, verify, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, verification output, files changed, self-review findings, commit SHA.

---

### Task 7: Frontend dashboard page (W174)

**Files:**
- Create: `app/frontend/components/AgentPerformanceDashboard.tsx`
- Create: `app/frontend/app/agent-performance/page.tsx`
- Create: `app/frontend/components/__tests__/AgentPerformanceDashboard.test.tsx`

- [ ] **Step 1: Write the failing tests**

Create `app/frontend/components/__tests__/AgentPerformanceDashboard.test.tsx`:

```typescript
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { AgentPerformancePage } from "../AgentPerformanceDashboard";
import { getAgentPerformanceDashboard, getStatus, getSandboxStatus } from "@/lib/api";
import type { AgentPerformanceDashboard as DashboardData, SegmentMetrics } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getAgentPerformanceDashboard: vi.fn(),
  getStatus: vi.fn(),
  getSandboxStatus: vi.fn(),
  ApiError: class ApiError extends Error {
    status?: number;
  },
}));

function segmentMetrics(overrides: Partial<SegmentMetrics> = {}): SegmentMetrics {
  return {
    matches_evaluated: 10, bets_placed: 5, bets_won: 2, roi: 0.1, hit_rate: 0.4,
    bet_frequency: 0.5, max_drawdown: 0.1, insufficient_data_rate: 0.0,
    starting_bankroll: 1000, ending_bankroll: 1100, total_staked: 50, total_profit: 5,
    ...overrides,
  };
}

function dashboardData(overrides: Partial<DashboardData> = {}): DashboardData {
  return {
    overall: { sample_size: 10, correct: 5, hit_rate: 0.5 },
    by_market: {},
    by_competition: {},
    by_confidence: {},
    // Deliberately distinct bets_placed from the segment tables below (each
    // default to 5 via segmentMetrics()'s own default) -- keeps
    // getByText("7") in the KPI-row test unambiguous instead of colliding
    // with every segment table's own "5" in its Bets column.
    kelly_roi_simulation: segmentMetrics({ bets_placed: 7 }),
    by_market_metrics: { result_3way: segmentMetrics() },
    by_market_selection_metrics: { "result_3way:home": segmentMetrics() },
    by_league_metrics: { E0: segmentMetrics({ bets_placed: 5 }) },
    staked_bets: [
      { match_id: "m1", market: "result_3way", selection: "home", odds: 2.1, stake: 3.0, won: true, payout: 3.3 },
    ],
    top_winners: [
      { match_id: "m1", market: "result_3way", selection: "home", odds: 2.1, stake: 3.0, won: true, payout: 3.3, date: "2026-08-22", competition: "E0", home_team: "Arsenal", away_team: "Everton" },
    ],
    top_losers: [],
    ...overrides,
  };
}

describe("AgentPerformancePage", () => {
  beforeEach(() => {
    vi.mocked(getAgentPerformanceDashboard).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockResolvedValue({} as never);
  });

  it("renders the Main Metrics KPI row from kelly_roi_simulation", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    render(<AgentPerformancePage />);

    expect(await screen.findByText("ROI")).toBeInTheDocument();
    expect(screen.getByText("Total Stake")).toBeInTheDocument();
    expect(screen.getByText("Money Won")).toBeInTheDocument();
    expect(screen.getByText("Bets Placed")).toBeInTheDocument();
    expect(screen.getByText("Hit %")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument(); // kelly_roi_simulation.bets_placed
  });

  it("renders the three breakdown tables", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    render(<AgentPerformancePage />);

    expect(await screen.findByText("By Market")).toBeInTheDocument();
    expect(screen.getByText("By Market + Direction")).toBeInTheDocument();
    expect(screen.getByText("By League")).toBeInTheDocument();
    expect(screen.getByText("3-Way Result")).toBeInTheDocument(); // marketLabel("result_3way")
  });

  it("renders top winners and top losers tables, including an empty-losers state", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    render(<AgentPerformancePage />);

    expect(await screen.findByText("Arsenal v Everton")).toBeInTheDocument();
    expect(screen.getByText("Top 5 Winners")).toBeInTheDocument();
    expect(screen.getByText("Top 5 Losers")).toBeInTheDocument();
    expect(screen.getByText("None yet.")).toBeInTheDocument();
  });

  it("falls back to match_id when a top bet has no team names (cache miss)", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(
      dashboardData({
        top_winners: [
          { match_id: "m404", market: "btts", selection: "yes", odds: 1.9, stake: 2.0, won: true, payout: 1.8, date: "2026-08-22", competition: "E0", home_team: null, away_team: null },
        ],
      })
    );
    render(<AgentPerformancePage />);

    expect(await screen.findByText("m404")).toBeInTheDocument();
  });

  it("re-fetches with a new days value when a time-range pill is clicked", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    const user = userEvent.setup();
    render(<AgentPerformancePage />);

    await screen.findByText("ROI");
    await user.click(screen.getByText("Last 30 days"));

    await waitFor(() => expect(getAgentPerformanceDashboard).toHaveBeenLastCalledWith(30));
  });

  it("shows an error state when the fetch fails", async () => {
    const { ApiError } = await import("@/lib/api");
    vi.mocked(getAgentPerformanceDashboard).mockRejectedValue(new ApiError("boom", 500));
    render(<AgentPerformancePage />);

    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app/frontend && npx vitest run components/__tests__/AgentPerformanceDashboard.test.tsx`
Expected: FAIL — module not found (`../AgentPerformanceDashboard` doesn't exist yet).

- [ ] **Step 3: Implement the component and route**

Create `app/frontend/components/AgentPerformanceDashboard.tsx`:

```typescript
"use client";

/** W174: local-only agent performance dashboard -- not linked from
 * AppShell's nav (see app/agent-performance/page.tsx), reachable only by
 * direct URL. Same "unlinked, not removed" precedent as BetTracker's own
 * /bets route (W106/W115). */

import { useEffect, useState } from "react";

import { ApiError, getAgentPerformanceDashboard } from "@/lib/api";
import { LEAGUE_LABEL } from "@/lib/dashboardMetrics";
import type { AgentPerformanceDashboard, SegmentMetrics, TopBet } from "@/lib/types";
import { AppShell } from "./AppShell";
import { ErrorState, marketLabel } from "./MatchUI";

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}
function formatUB(v: number): string {
  return `${v.toFixed(1)} UB`;
}
function pnlColor(v: number): string {
  return v > 0 ? "text-good" : v < 0 ? "text-serious" : "text-ink";
}
function formatSelection(selection: string): string {
  return selection.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function StatTile({ label, value, colorClass }: { label: string; value: string; colorClass?: string }) {
  return (
    <div className="min-w-[140px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="text-[10px] uppercase tracking-wide text-muted">{label}</div>
      <div className={`mt-1 font-mono text-2xl font-bold ${colorClass ?? "text-ink"}`}>{value}</div>
    </div>
  );
}

function BreakdownTable({ title, rows }: { title: string; rows: { label: string; metrics: SegmentMetrics }[] }) {
  return (
    <div className="min-w-[260px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 text-xs font-semibold text-ink">{title}</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted">
            <th className="pb-1.5 text-left text-[10px] font-normal uppercase">Segment</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">ROI</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Stake</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Won</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Bets</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Hit%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-hairline">
              <td className="py-1.5 text-ink">{r.label}</td>
              <td className={`py-1.5 text-right font-mono ${pnlColor(r.metrics.roi)}`}>{formatPct(r.metrics.roi)}</td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{formatUB(r.metrics.total_staked)}</td>
              <td className={`py-1.5 text-right font-mono ${pnlColor(r.metrics.total_profit)}`}>
                {r.metrics.total_profit >= 0 ? "+" : ""}
                {r.metrics.total_profit.toFixed(1)}
              </td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{r.metrics.bets_placed}</td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{formatPct(r.metrics.hit_rate)}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={6} className="py-3 text-center text-ink-secondary">
                No staked bets in this window.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function bucketize(values: number[], edges: number[], labels: string[]): { label: string; count: number }[] {
  const counts = new Array(labels.length).fill(0);
  for (const v of values) {
    let idx = edges.length - 2;
    for (let i = 0; i < edges.length - 1; i++) {
      if (v >= edges[i] && v < edges[i + 1]) {
        idx = i;
        break;
      }
    }
    counts[idx] += 1;
  }
  return labels.map((label, i) => ({ label, count: counts[i] }));
}

const ODDS_EDGES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, Infinity];
const ODDS_LABELS = ["1.0", "1.5", "2.0", "2.5", "3.0", "4.0", "5.0+"];
const STAKE_EDGES = [0, 2, 4, 6, 8, 10, Infinity];
const STAKE_LABELS = ["0", "2", "4", "6", "8", "10+"];

function Histogram({ title, buckets }: { title: string; buckets: { label: string; count: number }[] }) {
  const max = Math.max(1, ...buckets.map((b) => b.count));
  return (
    <div className="min-w-[200px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 text-xs font-semibold text-ink">{title}</div>
      <div className="flex items-end gap-1" style={{ height: 80 }}>
        {buckets.map((b) => (
          <div key={b.label} className="flex flex-1 flex-col items-center justify-end" title={`${b.label}: ${b.count}`}>
            <div
              className="w-full rounded-t bg-accent"
              style={{ height: `${(b.count / max) * 100}%`, minHeight: b.count > 0 ? 2 : 0 }}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted">
        {buckets.map((b) => (
          <span key={b.label}>{b.label}</span>
        ))}
      </div>
    </div>
  );
}

// Dataviz skill's reference palette, dark-mode categorical steps -- slot 1
// (accent) already used by the rest of this app; slots 2-8 introduced here
// for the first time, same fixed order the palette validates.
const CATEGORICAL_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"];

function LeagueBarChart({ entries }: { entries: { league: string; count: number }[] }) {
  const max = Math.max(1, ...entries.map((e) => e.count));
  return (
    <div className="min-w-[200px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 text-xs font-semibold text-ink">Bets by League</div>
      <div className="flex flex-col gap-2">
        {entries.map((e, i) => (
          <div key={e.league} className="flex items-center gap-2 text-xs">
            <span className="w-10 shrink-0 text-ink-secondary">{LEAGUE_LABEL[e.league] ?? e.league}</span>
            <div className="h-3.5 flex-1 rounded bg-white/[0.04]">
              <div
                className="h-full rounded"
                style={{ width: `${(e.count / max) * 100}%`, background: CATEGORICAL_COLORS[i % CATEGORICAL_COLORS.length] }}
              />
            </div>
            <span className="w-6 shrink-0 text-right text-muted">{e.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BetsExampleTable({ title, bets, colorClass }: { title: string; bets: TopBet[]; colorClass: string }) {
  return (
    <div className="min-w-[280px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className={`mb-2 text-xs font-semibold ${colorClass}`}>{title}</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted">
            <th className="pb-1.5 text-left text-[10px] font-normal uppercase">Match</th>
            <th className="pb-1.5 text-left text-[10px] font-normal uppercase">Pick</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Odds</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Stake</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Payout</th>
          </tr>
        </thead>
        <tbody>
          {bets.map((b, i) => (
            <tr key={i} className="border-t border-hairline">
              <td className="py-1.5 text-ink">{b.home_team && b.away_team ? `${b.home_team} v ${b.away_team}` : b.match_id}</td>
              <td className="py-1.5 text-ink-secondary">
                {marketLabel(b.market).label} · {formatSelection(b.selection)}
              </td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{b.odds.toFixed(2)}</td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{formatUB(b.stake)}</td>
              <td className={`py-1.5 text-right font-mono ${pnlColor(b.payout)}`}>
                {b.payout >= 0 ? "+" : ""}
                {b.payout.toFixed(1)}
              </td>
            </tr>
          ))}
          {bets.length === 0 && (
            <tr>
              <td colSpan={5} className="py-3 text-center text-ink-secondary">
                None yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

const DAY_OPTIONS: { label: string; days: number }[] = [
  { label: "All time", days: 3650 },
  { label: "Last 90 days", days: 90 },
  { label: "Last 30 days", days: 30 },
];

export function AgentPerformancePage() {
  const [days, setDays] = useState(3650);
  const [data, setData] = useState<AgentPerformanceDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getAgentPerformanceDashboard(days)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load performance data.");
      });
    return () => {
      cancelled = true;
    };
  }, [days]);

  const kelly = data?.kelly_roi_simulation;

  return (
    <AppShell active="agent-performance">
      <div className="pt-8">
        <h1 className="text-xl font-semibold tracking-tight text-ink">Agent Performance</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          All resolved live recommendations, Kelly-sized simulated stakes (UB)
        </p>

        <div className="mt-4 flex gap-2">
          {DAY_OPTIONS.map((opt) => (
            <button
              key={opt.label}
              type="button"
              onClick={() => setDays(opt.days)}
              className={`rounded-full border px-3 py-1 text-xs ${
                days === opt.days ? "border-accent bg-accent text-white" : "border-border-strong text-ink-secondary"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="mt-6">
            <ErrorState message={error} onRetry={() => setDays((d) => d)} />
          </div>
        )}

        {!error && !data && <div className="mt-6 text-sm text-ink-secondary">Loading…</div>}

        {data && kelly && (
          <>
            <div className="mt-6">
              <div className="mb-2 text-sm font-semibold text-ink">Main Metrics</div>
              <div className="flex flex-wrap gap-3">
                <StatTile label="ROI" value={formatPct(kelly.roi)} colorClass={pnlColor(kelly.roi)} />
                <StatTile label="Total Stake" value={formatUB(kelly.total_staked)} />
                <StatTile
                  label="Money Won"
                  value={`${kelly.total_profit >= 0 ? "+" : ""}${kelly.total_profit.toFixed(1)} UB`}
                  colorClass={pnlColor(kelly.total_profit)}
                />
                <StatTile label="Bets Placed" value={String(kelly.bets_placed)} />
                <StatTile label="Hit %" value={formatPct(kelly.hit_rate)} />
              </div>
            </div>

            <div className="mt-8">
              <div className="mb-1 text-sm font-semibold text-ink">Segmentation</div>
              <p className="mb-3 text-xs text-ink-secondary">
                Same metrics, sliced by Market / Market+Direction / League
              </p>
              <div className="flex flex-wrap gap-4">
                <BreakdownTable
                  title="By Market"
                  rows={Object.entries(data.by_market_metrics).map(([k, v]) => ({ label: marketLabel(k).label, metrics: v }))}
                />
                <BreakdownTable
                  title="By Market + Direction"
                  rows={Object.entries(data.by_market_selection_metrics).map(([k, v]) => {
                    const [market, selection] = k.split(":");
                    return { label: `${marketLabel(market).label} · ${formatSelection(selection)}`, metrics: v };
                  })}
                />
                <BreakdownTable
                  title="By League"
                  rows={Object.entries(data.by_league_metrics).map(([k, v]) => ({ label: LEAGUE_LABEL[k] ?? k, metrics: v }))}
                />
              </div>
            </div>

            <div className="mt-8">
              <div className="mb-3 text-sm font-semibold text-ink">Distributions</div>
              <div className="flex flex-wrap gap-4">
                <Histogram
                  title="Odds Distribution"
                  buckets={bucketize(data.staked_bets.map((b) => b.odds), ODDS_EDGES, ODDS_LABELS)}
                />
                <LeagueBarChart
                  entries={Object.entries(data.by_league_metrics)
                    .map(([league, m]) => ({ league, count: m.bets_placed }))
                    .sort((a, b) => b.count - a.count)}
                />
                <Histogram
                  title="Stake Distribution"
                  buckets={bucketize(data.staked_bets.map((b) => b.stake), STAKE_EDGES, STAKE_LABELS)}
                />
              </div>
            </div>

            <div className="mb-8 mt-8">
              <div className="mb-3 text-sm font-semibold text-ink">Top Winning &amp; Losing Bets</div>
              <div className="flex flex-wrap gap-4">
                <BetsExampleTable title="Top 5 Winners" bets={data.top_winners} colorClass="text-good" />
                <BetsExampleTable title="Top 5 Losers" bets={data.top_losers} colorClass="text-serious" />
              </div>
            </div>
          </>
        )}
      </div>
    </AppShell>
  );
}
```

Create `app/frontend/app/agent-performance/page.tsx`:

```typescript
import { AgentPerformancePage } from "@/components/AgentPerformanceDashboard";

export default function Page() {
  return <AgentPerformancePage />;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app/frontend && npx vitest run components/__tests__/AgentPerformanceDashboard.test.tsx`
Expected: PASS, 6 passed.

Then typecheck and build:

Run: `cd app/frontend && npx tsc --noEmit && npm run build`
Expected: clean. Confirm the build output lists a new `/agent-performance` route.

Then run the full frontend suite:

Run: `cd app/frontend && npx vitest run`
Expected: PASS (aside from `MatchUI.dateboundary.test.tsx`'s own unrelated pre-existing flake under parallel execution, if it recurs -- rerun standalone to confirm, per this codebase's established practice).

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/AgentPerformanceDashboard.tsx app/frontend/app/agent-performance/page.tsx app/frontend/components/__tests__/AgentPerformanceDashboard.test.tsx
git commit -m "feat(app): agent performance dashboard page, unlinked from nav (W174)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 7 of a 7-task plan, the final implementation task. Depends on Task 5 (types + API client) and Task 6 (exported `marketLabel`, widened `AppShell` active union), both already done and committed. `AppShell` wraps page content and expects `active`/`children` props (see `app/frontend/components/BetTracker.tsx`'s `BetTrackerPage` for the exact same wrapping pattern this task mirrors). `LEAGUE_LABEL` is already exported from `app/frontend/lib/dashboardMetrics.ts`. `ErrorState` is already exported from `app/frontend/components/MatchUI.tsx` (props: `{ message: string; onRetry?: () => void }`). This page is deliberately **not** added to `AppShell.tsx`'s `NAV_ITEMS` array -- don't add it there, that's the whole point of "local only" (see the design spec).

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), typecheck/build results, files changed, self-review findings, commit SHA.

---

### Task 8: Full verification + mark stories completed

**Files:**
- Modify: `documents/agent_user_stories.md`
- Modify: `documents/app_user_stories.md`

- [ ] **Step 1: Run the full backend test suite**

Run: `pytest tests/ app/backend/tests/ -v`
Expected: PASS, 0 failures. Note the total pass count.

- [ ] **Step 2: Run the full frontend test suite + typecheck + build**

Run: `cd app/frontend && npx vitest run && npx tsc --noEmit && npm run build`
Expected: PASS, 0 failures (aside from the documented pre-existing `MatchUI.dateboundary.test.tsx` parallel-run flake, if it recurs -- confirm via a standalone rerun), clean typecheck, clean build with `/agent-performance` listed as a route.

- [ ] **Step 3: Manual sanity check of the new endpoint**

With the backend running locally (`uvicorn app.backend.main:app --reload` from repo root):

```bash
curl http://localhost:8000/api/recommendations/performance-dashboard
```

Should return `200` with valid JSON (an empty-state response is fine if nothing is resolved locally yet).

With the frontend running locally (`npm run dev` from `app/frontend`), visit `http://localhost:3000/agent-performance` directly and confirm the page renders (empty-state is fine) without a console error, and confirm it does **not** appear in the left nav.

- [ ] **Step 4: Mark A83 completed in agent_user_stories.md**

Add A83 as a new row (status `active`, to be flipped to `completed` here) in a new `## PHASE 26: Agent Performance Dashboard` section, mirroring the existing phase-header + table format this file already uses throughout (see PHASE 25 immediately above it for the exact template). Description: "Add `total_staked`/`total_profit` to `build_evaluation_report`'s return dict (`src/agent/evaluation.py`) -- both already computed locally, never previously returned. Needed by the agent performance dashboard's Main Metrics row (W170-W174, `app_user_stories.md` PHASE 27)." Then append a `**Completion notes (<today's date>):**` sentence summarizing what shipped and the real test-suite pass count from Step 1, following the style of the row immediately above it as the template.

- [ ] **Step 5: Mark W170-W174 completed in app_user_stories.md**

Add W170-W174 as new rows (status `active`, to be flipped to `completed` here) in a new `## PHASE 27: Agent Performance Dashboard` section, mirroring the existing phase-header + table format (see PHASE 37 immediately above it for the exact template). One row per story:
- W170: per-segment Kelly reports + `staked_bets` in `recommendation_stats.py`.
- W171: `agent_performance_dashboard.py` (top/bottom enrichment).
- W172: `GET /api/recommendations/performance-dashboard`.
- W173: frontend types + API client function.
- W174: the dashboard page itself, unlinked from nav.

Then append completion notes to each, following W169's own style immediately above as the template, including the real frontend test/build results from Step 2 and confirming the direct-URL/no-nav-link check from Step 3.

- [ ] **Step 6: Commit**

```bash
git add documents/agent_user_stories.md documents/app_user_stories.md
git commit -m "docs: mark A83/W170-W174 completed with verification results

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
