# Live Recommendation Outcome Tracking & Kelly Stake Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Durably resolve every live-generated recommendation's actual pick against real match results (independent of whether the user ever bet on it), expose a diagnostics endpoint the user can query for hit-rate/ROI, and show a Kelly-derived suggested-stake multiplier (in an abstract "Unit Bet" — not a dollar figure) on each recommendation card.

**Architecture:** Agent-side (`src/agent/`) gets three small, reusable building blocks — a standalone Kelly-fraction formula, a ported "which market did we actually recommend" picker, and a deterministic enrichment pass that attaches a stake multiplier to every recommendation. App-side (`app/backend/`) gets a new SQLite-backed outcome store resolved on demand (mirroring the existing bet-settlement job), a pure aggregation module for the diagnostics stats, and two new endpoints. The frontend surfaces the multiplier per card plus one static explainer line.

**Tech Stack:** Python (FastAPI, sqlite3, pytest), TypeScript/React (Vitest, Testing Library) — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-live-recommendation-tracking-design.md`
**Stories:** `documents/agent_user_stories.md` A80–A82 (PHASE 25), `documents/app_user_stories.md` W167–W169 (PHASE 37)

---

## File Structure

**Agent-side (create/modify):**
- Modify `src/agent/staking.py` — extract `kelly_fraction()`, reuse in `simulate_kelly_stake`.
- Modify `src/agent/market_resolution.py` — add `pick_recommended_market()`.
- Modify `src/agent/schema.py` — add `_attach_unit_bet_multiplier()` deterministic pass + `unit_bet_multiplier` field.

**App-side (create/modify):**
- Modify `app/backend/recommendation_cache.py` — add `list_latest_per_match()`.
- Modify `app/backend/recommendations.py` — add `unit_bet_multiplier` to `MatchRecommendationOut`.
- Create `app/backend/recommendation_outcomes.py` — `RecommendationOutcome` dataclass, `RecommendationOutcomeStore`, `resolve_pending_recommendations()`.
- Create `app/backend/recommendation_stats.py` — `compute_recommendation_stats()` (hit-rate breakdown + Kelly ROI simulation).
- Modify `app/backend/main.py` — `POST /api/recommendations/settle-open`, `GET /api/recommendations/stats`.

**Frontend (modify):**
- Modify `app/frontend/lib/types.ts` — add `unit_bet_multiplier` to `MatchRecommendationOut`.
- Modify `app/frontend/components/MatchUI.tsx` — add `unitBetMultiplier` to `Match`, thread through `applyRecommendation`, render per-card suggestion + Daily Edges explainer line.

**Tests (create/modify):**
- Modify `tests/test_staking.py`, `tests/test_market_resolution.py`.
- Create `tests/test_agent_unit_bet_multiplier.py`.
- Modify `app/backend/tests/test_recommendation_cache.py`, `app/backend/tests/test_recommendations_schema.py`.
- Create `app/backend/tests/test_recommendation_outcomes.py`, `app/backend/tests/test_recommendation_stats.py`, `app/backend/tests/test_recommendation_outcomes_endpoints.py`.
- Modify `app/frontend/components/__tests__/MatchUI.test.tsx`, `app/frontend/components/__tests__/MatchUI.precompute.test.tsx`.

---

### Task 1: Extract `kelly_fraction()` in staking.py (A80)

**Files:**
- Modify: `src/agent/staking.py`
- Test: `tests/test_staking.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_staking.py`:

```python
def test_kelly_fraction_positive_edge():
    # 0.10 / (3.0 - 1) = 0.05
    assert kelly_fraction(0.10, 3.0) == 0.05


def test_kelly_fraction_caps_at_max_fraction():
    assert kelly_fraction(0.9, 1.5, max_fraction=0.1) == 0.1


def test_kelly_fraction_returns_zero_for_non_positive_edge():
    assert kelly_fraction(-0.05, 2.0) == 0.0
    assert kelly_fraction(0.0, 2.0) == 0.0


def test_kelly_fraction_returns_zero_for_odds_at_or_below_one():
    assert kelly_fraction(0.1, 1.0) == 0.0
    assert kelly_fraction(0.1, 0.5) == 0.0
```

Add `kelly_fraction` to the existing import at the top of the file:

```python
from src.agent.staking import kelly_fraction, simulate_flat_stake, simulate_kelly_stake
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_staking.py -k kelly_fraction -v`
Expected: FAIL with `ImportError: cannot import name 'kelly_fraction'`

- [ ] **Step 3: Implement `kelly_fraction` and reuse it in `simulate_kelly_stake`**

In `src/agent/staking.py`, add this function above `simulate_kelly_stake`:

```python
def kelly_fraction(value_edge: float, odds: float, max_fraction: float = 0.10) -> float:
    """Kelly stake as a fraction of bankroll: value_edge / (odds - 1), capped
    at max_fraction. Returns 0.0 for non-positive edge or odds <= 1.0 -- no
    Kelly fraction is defined/worth taking there; callers must treat 0.0 as
    "no stake", not a computation error.

    Extracted (A80) from simulate_kelly_stake's own inline formula so
    schema.py's unit_bet_multiplier enrichment (A82) and the app's
    outcome-based ROI simulation (app/backend/recommendation_stats.py, W168)
    reuse the exact same math backtesting already relies on, instead of a
    second, potentially-drifting copy."""
    if odds <= 1.0 or value_edge <= 0:
        return 0.0
    return min(value_edge / (odds - 1.0), max_fraction)
```

Then replace the inline computation inside `simulate_kelly_stake`'s loop. Find this block:

```python
            if odds <= 1.0 or value_edge <= 0:
                continue
            fraction = min(value_edge / (odds - 1.0), max_fraction)
            stake = bankroll * fraction
```

Replace it with:

```python
            fraction = kelly_fraction(value_edge, odds, max_fraction)
            if fraction <= 0:
                continue
            stake = bankroll * fraction
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_staking.py -v`
Expected: PASS, all tests including the pre-existing `test_kelly_stake_*` ones (behavior-preserving refactor).

- [ ] **Step 5: Commit**

```bash
git add src/agent/staking.py tests/test_staking.py
git commit -m "feat(agent): extract kelly_fraction as a standalone pure function (A80)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Add `pick_recommended_market()` to market_resolution.py (A81)

**Files:**
- Modify: `src/agent/market_resolution.py`
- Test: `tests/test_market_resolution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_market_resolution.py`:

```python
def test_pick_recommended_market_returns_none_for_empty_list():
    assert pick_recommended_market([]) is None


def test_pick_recommended_market_prefers_non_no_bet():
    markets = [
        {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "value_edge": 0.20},
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "value_edge": 0.05},
    ]
    picked = pick_recommended_market(markets)
    assert picked["market"] == "result_3way"


def test_pick_recommended_market_breaks_ties_by_value_edge():
    markets = [
        {"market": "btts", "selection": "yes", "recommendation_type": "direct_bet", "value_edge": 0.05},
        {"market": "result_3way", "selection": "home", "recommendation_type": "conditional", "value_edge": 0.12},
    ]
    picked = pick_recommended_market(markets)
    assert picked["market"] == "result_3way"


def test_pick_recommended_market_falls_back_to_no_bet_when_nothing_actionable():
    # Mirrors MatchUI.tsx's bestMarket(): when every market is no_bet, still
    # return the highest-value_edge one rather than nothing at all.
    markets = [
        {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "value_edge": -0.02},
        {"market": "result_3way", "selection": "home", "recommendation_type": "no_bet", "value_edge": 0.01},
    ]
    picked = pick_recommended_market(markets)
    assert picked["market"] == "result_3way"
```

Update the import at the top of `tests/test_market_resolution.py`:

```python
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct, pick_recommended_market
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_market_resolution.py -k pick_recommended_market -v`
Expected: FAIL with `ImportError: cannot import name 'pick_recommended_market'`

- [ ] **Step 3: Implement `pick_recommended_market`**

Append to `src/agent/market_resolution.py`:

```python
def pick_recommended_market(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Which single market a recommendation actually picked -- ports
    MatchUI.tsx's bestMarket() into Python (A81) so the app's outcome
    resolver (app/backend/recommendation_outcomes.py, W167) can determine
    server-side the same market a completed card's Hit/Not-Hit badge
    already reflects client-side. Prefers a non-'no_bet' market; falls back
    to ranking among all markets (including no_bet) only when nothing is
    actionable at all. Ties broken by value_edge, highest first -- Python's
    max() returns the first maximal element on ties, matching a stable
    descending sort's own tie-break order."""
    if not markets:
        return None
    actionable = [m for m in markets if m.get("recommendation_type") != "no_bet"]
    pool = actionable if actionable else markets
    return max(pool, key=lambda m: m.get("value_edge") or 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_market_resolution.py -v`
Expected: PASS, 13 passed (9 pre-existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/agent/market_resolution.py tests/test_market_resolution.py
git commit -m "feat(agent): add pick_recommended_market, porting bestMarket() from TS (A81)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Add `unit_bet_multiplier` enrichment pass to schema.py (A82)

**Files:**
- Modify: `src/agent/schema.py`
- Test: `tests/test_agent_unit_bet_multiplier.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_unit_bet_multiplier.py`:

```python
"""Regression tests for A82: attach a deterministic unit_bet_multiplier to
every extracted recommendation -- a Kelly-derived stake-sizing suggestion
for the recommendation's actual pick (A81's pick_recommended_market),
expressed as a multiple of an abstract "Unit Bet" (UB), not a dollar
figure. See documents/agent_user_stories.md A82."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_VALID_MARKET = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 3.0,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.33,
    "value_edge": 0.10,
}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "markets": [_VALID_MARKET],
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_direct_bet_gets_a_positive_multiplier():
    # kelly_fraction(0.10, 3.0) = 0.10 / 2.0 = 0.05 -> 0.05 / 0.01 = 5.0
    rec = extract_recommendation(_wrap_json(_VALID))
    assert rec["unit_bet_multiplier"] == 5.0


def test_multiplier_capped_at_ten():
    market = {**_VALID_MARKET, "current_odds": 1.5, "value_edge": 0.9}
    data = {**_VALID, "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] == 10.0


def test_no_bet_overall_gets_no_multiplier():
    market = {**_VALID_MARKET, "recommendation_type": "no_bet"}
    data = {**_VALID, "overall": "no_bet", "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] is None


def test_missing_odds_gets_no_multiplier():
    # A67/BUG-013 already forbid direct_bet with null odds, but a
    # conditional market can legitimately have current_odds=None.
    market = {
        **_VALID_MARKET, "market": "btts", "selection": "yes",
        "recommendation_type": "conditional", "current_odds": None,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] is None


def test_conditional_below_value_floor_gets_zero_not_null():
    # A real price exists but doesn't clear the value bar yet -- "wait, 0
    # UB for now" is meaningfully different from "no price at all".
    market = {
        **_VALID_MARKET, "market": "btts", "selection": "yes",
        "recommendation_type": "conditional", "current_odds": 1.6, "value_edge": -0.02,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_unit_bet_multiplier.py -v`
Expected: FAIL with `KeyError: 'unit_bet_multiplier'`

- [ ] **Step 3: Add the field to the TypedDict and implement the enrichment pass**

In `src/agent/schema.py`, add these two imports to the existing import block at the top of the file (after the `from src.ingestion.common.team_mapping import TeamNameMapper` line):

```python
from src.agent.market_resolution import pick_recommended_market
from src.agent.staking import kelly_fraction
```

Add the new field to `MatchRecommendation` (insert after the `prediction_basis: str` line):

```python
    prediction_basis: str
    # A82: Kelly-derived stake-sizing suggestion for the recommendation's
    # actual pick (pick_recommended_market), as a multiple of an abstract
    # "Unit Bet" -- not a dollar figure. Computed here (like target_odds/
    # A52), never by the LLM. None when there's no priced pick (no_bet/
    # insufficient_data, or missing odds); 0.0 is a real, distinct value --
    # a priced 'conditional' market whose edge doesn't clear the bar yet.
    unit_bet_multiplier: float | None
```

Add the enrichment function right after `_reconcile_overall_with_markets` (before `def reported_teams`):

```python
def _attach_unit_bet_multiplier(data: dict) -> dict:
    """A82: deterministic stake-sizing suggestion for the recommendation's
    actual pick, expressed as a multiple of a standard "Unit Bet" (UB) --
    an abstract betting unit, not a dollar figure (bet 2 UB at odds 3.0,
    get 6 UB back). Reuses the exact Kelly-fraction math staking.py already
    computes for backtest sizing (A80's kelly_fraction) against the same
    1%-of-bankroll baseline simulate_flat_stake calls "1x", so
    unit_bet_multiplier=1.0 means "size this like staking.py's own flat
    baseline" -- not an arbitrary new scale. kelly_fraction's own
    max_fraction=0.10 default caps the result at 10.0 automatically, no
    separate clamping needed here.

    Run last, after every downgrade pass and _reconcile_overall_with_markets:
    needs each market's FINAL recommendation_type to pick the right one
    (A81's pick_recommended_market)."""
    picked = pick_recommended_market(data.get("markets") or [])
    if picked is None or picked.get("current_odds") is None:
        data["unit_bet_multiplier"] = None
    else:
        fraction = kelly_fraction(picked.get("value_edge") or 0.0, picked["current_odds"])
        data["unit_bet_multiplier"] = fraction / 0.01
    return data
```

Wire it into `extract_recommendation`'s pipeline. Find:

```python
        data = _reconcile_overall_with_markets(data)
        return data  # type: ignore[return-value]
```

Replace with:

```python
        data = _reconcile_overall_with_markets(data)
        data = _attach_unit_bet_multiplier(data)
        return data  # type: ignore[return-value]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_unit_bet_multiplier.py -v`
Expected: PASS, 5 passed.

Then run the full schema test suite to confirm nothing else broke:

Run: `pytest tests/test_agent_schema.py tests/test_agent_schema_validation.py tests/test_agent_target_odds.py tests/test_agent_conditional_odds_floor.py tests/test_agent_overall_reconciliation.py -v`
Expected: PASS, all pre-existing tests unaffected (none assert on the full dict shape excluding new keys).

- [ ] **Step 5: Commit**

```bash
git add src/agent/schema.py tests/test_agent_unit_bet_multiplier.py
git commit -m "feat(agent): attach unit_bet_multiplier to every extracted recommendation (A82)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Add `RecommendationCache.list_latest_per_match()`

**Files:**
- Modify: `app/backend/recommendation_cache.py`
- Test: `app/backend/tests/test_recommendation_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `app/backend/tests/test_recommendation_cache.py`:

```python
def test_list_latest_per_match_returns_one_row_per_match_regardless_of_hash(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, {"overall": "direct_bet"}, "scheduled")
    cache.record_generation("m1", "2026-08-22", "hash2", {}, {"overall": "conditional"}, "manual_regenerate")
    cache.record_generation("m2", "2026-08-23", "hash1", {}, {"overall": "no_bet"}, "scheduled")

    entries = cache.list_latest_per_match()

    by_match = {e.match_id: e for e in entries}
    assert len(entries) == 2
    # m1's latest generation is the second one (hash2), not the first.
    assert by_match["m1"].recommendation["overall"] == "conditional"
    assert by_match["m1"].agent_config_hash == "hash2"
    assert by_match["m2"].recommendation["overall"] == "no_bet"


def test_list_latest_per_match_returns_empty_list_when_nothing_cached(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    assert cache.list_latest_per_match() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/backend/tests/test_recommendation_cache.py -k list_latest_per_match -v`
Expected: FAIL with `AttributeError: 'RecommendationCache' object has no attribute 'list_latest_per_match'`

- [ ] **Step 3: Implement `list_latest_per_match`**

In `app/backend/recommendation_cache.py`, add this method to `RecommendationCache`, right after `get_history`:

```python
    def list_latest_per_match(self) -> list[CacheEntry]:
        """One row per distinct (match_id, date) -- the single most recent
        generation across any agent_config_hash, mirroring
        get_latest_any_config()'s own "ignore the hash" fallback semantics
        but for every cached match at once. Used by
        recommendation_outcomes.py's resolution job (W167) to find every
        match with a live-generated recommendation, not just ones a caller
        already knows the key for."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by
                FROM recommendation_generations rg
                WHERE id = (
                    SELECT MAX(id) FROM recommendation_generations rg2
                    WHERE rg2.match_id = rg.match_id AND rg2.date = rg.date
                )
                """
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendation_cache.py -v`
Expected: PASS, all tests including the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendation_cache.py app/backend/tests/test_recommendation_cache.py
git commit -m "feat(app): add RecommendationCache.list_latest_per_match (W167)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Add `unit_bet_multiplier` to `MatchRecommendationOut` (recommendations.py)

**Files:**
- Modify: `app/backend/recommendations.py`
- Test: `app/backend/tests/test_recommendations_schema.py`

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/tests/test_recommendations_schema.py`:

```python
def test_unit_bet_multiplier_passes_through_unchanged():
    raw = {**_VALID_RAW, "unit_bet_multiplier": 3.5}
    result = validate_and_degrade(raw, "Arsenal", "Everton")
    assert result.unit_bet_multiplier == 3.5


def test_missing_unit_bet_multiplier_defaults_to_none_for_pre_a82_cached_data():
    result = validate_and_degrade(_VALID_RAW, "Arsenal", "Everton")
    assert result.unit_bet_multiplier is None
```

(These reuse the existing `_VALID_RAW` fixture already defined near the top of this test file, same as the pre-existing `target_odds` tests.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_recommendations_schema.py -k unit_bet_multiplier -v`
Expected: FAIL with `AttributeError: 'MatchRecommendationOut' object has no attribute 'unit_bet_multiplier'`

- [ ] **Step 3: Add the field**

In `app/backend/recommendations.py`, add to `MatchRecommendationOut` (insert after `unknown_team: bool = False`):

```python
    unknown_team: bool = False
    # A82 (agent_user_stories.md): Kelly-derived stake-sizing suggestion for
    # this recommendation's actual pick, as a multiple of an abstract "Unit
    # Bet" -- not a dollar figure. None when there's no priced pick, or
    # absent entirely on a pre-A82 cached row, same convention as
    # target_odds/feature_completeness above.
    unit_bet_multiplier: float | None = None
```

Then in `validate_and_degrade`, add the field to the final return statement. Find:

```python
        cold_start_risk=bool(raw.get("cold_start_risk", False)),
        feature_completeness=raw.get("feature_completeness"),
        unknown_team=bool(raw.get("unknown_team", False)),
    )
```

Replace with:

```python
        cold_start_risk=bool(raw.get("cold_start_risk", False)),
        feature_completeness=raw.get("feature_completeness"),
        unknown_team=bool(raw.get("unknown_team", False)),
        unit_bet_multiplier=raw.get("unit_bet_multiplier"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendations_schema.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendations.py app/backend/tests/test_recommendations_schema.py
git commit -m "feat(app): surface unit_bet_multiplier on MatchRecommendationOut (A82/W169)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Create `recommendation_outcomes.py` (store + resolution job) (W167)

**Files:**
- Create: `app/backend/recommendation_outcomes.py`
- Test: `app/backend/tests/test_recommendation_outcomes.py`

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_recommendation_outcomes.py`:

```python
"""W167: recommendation_outcomes storage + resolve_pending_recommendations,
mirroring test_settlement.py's own structure and cases."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore, resolve_pending_recommendations


def _match(match_id: str, home_goals: int | None, away_goals: int | None) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date="2026-08-22T15:00:00Z", status="FINISHED",
        home_team="Arsenal", away_team="Everton", home_goals=home_goals, away_goals=away_goals,
    )


def _rec(overall: str, market: str, selection: str, recommendation_type: str, current_odds, value_edge=0.1, league="E0", confidence="medium") -> dict:
    return {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": league},
        "overall": overall,
        "markets": [{
            "market": market, "selection": selection, "recommendation_type": recommendation_type,
            "current_odds": current_odds, "value_edge": value_edge,
        }],
        "confidence": confidence,
        "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
    }


def test_resolves_a_won_direct_bet_pick(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
    assert resolved[0].market == "result_3way"
    assert store.list_all()[0].match_id == "m1"


def test_resolves_a_lost_pick(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "away", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved[0].correct is False


def test_skips_no_bet_recommendations(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("no_bet", "result_3way", "home", "no_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved == []
    client.get_results.assert_not_called()


def test_skips_unresolvable_markets(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "home_corners", "over_4.5", "direct_bet", 1.9), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved == []
    client.get_results.assert_not_called()


def test_skips_not_yet_finished_matches(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = []  # not finished yet

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved == []


def test_idempotent_rerun_does_not_duplicate(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    first = resolve_pending_recommendations(cache, store, client)
    second = resolve_pending_recommendations(cache, store, client)

    assert len(first) == 1
    assert second == []
    assert len(store.list_all()) == 1


def test_resolves_a_non_epl_league_via_the_correct_competition_code(tmp_path: Path) -> None:
    """Deliberately does not trust the recommendation's self-reported
    match.league for routing -- merges results across every football-data.org
    competition code instead (same reasoning settlement.py already uses to
    merge EPL + Sweden results)."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0, league="SP1"), "scheduled")
    client = MagicMock()
    # Only the "PD" (La Liga) call returns the match; every other competition
    # code call returns nothing.
    client.get_results.side_effect = lambda competition_code, date_from, date_to: (
        [_match("m1", 2, 1)] if competition_code == "PD" else []
    )

    resolved = resolve_pending_recommendations(cache, store, client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
    assert resolved[0].competition == "SP1"


def test_uses_sweden_client_when_provided(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("sw1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0, league="SWE"), "scheduled")
    client = MagicMock()
    client.get_results.return_value = []
    sweden_client = MagicMock()
    sweden_client.get_results.return_value = [
        NormalizedMatch(match_id="sw1", utc_date="2026-08-22T15:00:00Z", status="FINISHED", home_team="Malmo FF", away_team="AIK", home_goals=2, away_goals=1)
    ]

    resolved = resolve_pending_recommendations(cache, store, client, sweden_client=sweden_client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_recommendation_outcomes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.recommendation_outcomes'`

- [ ] **Step 3: Implement `recommendation_outcomes.py`**

Create `app/backend/recommendation_outcomes.py`:

```python
"""W167: durable outcome tracking for live-generated recommendations,
independent of whether the user ever placed a bet on them -- unlike
bet_tracker.py (which only records what the user chose to log), this
resolves every actionable recommendation the agent produced against real
results, for the user's own diagnostics (GET /api/recommendations/stats,
recommendation_stats.py). One row per (match_id, date): the agent's actual
pick (A81's pick_recommended_market) from that match's latest cached
recommendation, resolved won/lost via src.agent.market_resolution.

Own db file (data/recommendation_outcomes.db), not recommendation_cache.db --
matches this codebase's established one-concern-one-db-file convention
(recommendation_cache.db, user_bets.db, job_runs.db all already do this)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from app.backend.football_data_client import FootballDataClient
from app.backend.football_data_competition_codes import FOOTBALL_DATA_CODE_BY_LEAGUE
from app.backend.recommendation_cache import RecommendationCache
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_scoped_path
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct, pick_recommended_market

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "recommendation_outcomes.db"


@dataclass(frozen=True)
class RecommendationOutcome:
    id: int
    match_id: str
    date: str
    competition: str | None
    market: str
    selection: str
    recommendation_type: str
    confidence: str | None
    odds: float | None
    value_edge: float | None
    correct: bool
    generated_at: str
    resolved_at: str


class RecommendationOutcomeStore:
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
                CREATE TABLE IF NOT EXISTS recommendation_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    competition TEXT,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    recommendation_type TEXT NOT NULL,
                    confidence TEXT,
                    odds REAL,
                    value_edge REAL,
                    correct INTEGER NOT NULL,
                    generated_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    UNIQUE(match_id, date)
                )
                """
            )

    def resolved_keys(self) -> set[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT match_id, date FROM recommendation_outcomes").fetchall()
        return {(row[0], row[1]) for row in rows}

    def insert(
        self,
        match_id: str,
        date: str,
        competition: str | None,
        market: str,
        selection: str,
        recommendation_type: str,
        confidence: str | None,
        odds: float | None,
        value_edge: float | None,
        correct: bool,
        generated_at: str,
    ) -> RecommendationOutcome:
        resolved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recommendation_outcomes
                (match_id, date, competition, market, selection, recommendation_type,
                 confidence, odds, value_edge, correct, generated_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (match_id, date, competition, market, selection, recommendation_type,
                 confidence, odds, value_edge, int(correct), generated_at, resolved_at),
            )
            row_id = cursor.lastrowid
        return RecommendationOutcome(
            id=row_id, match_id=match_id, date=date, competition=competition, market=market,
            selection=selection, recommendation_type=recommendation_type, confidence=confidence,
            odds=odds, value_edge=value_edge, correct=correct, generated_at=generated_at, resolved_at=resolved_at,
        )

    def list_all(self, since: str | None = None) -> list[RecommendationOutcome]:
        query = (
            "SELECT id, match_id, date, competition, market, selection, recommendation_type, "
            "confidence, odds, value_edge, correct, generated_at, resolved_at FROM recommendation_outcomes"
        )
        params: tuple = ()
        if since is not None:
            query += " WHERE date >= ?"
            params = (since,)
        query += " ORDER BY date ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_outcome(row) for row in rows]

    @staticmethod
    def _row_to_outcome(row: tuple) -> RecommendationOutcome:
        return RecommendationOutcome(
            id=row[0], match_id=row[1], date=row[2], competition=row[3], market=row[4], selection=row[5],
            recommendation_type=row[6], confidence=row[7], odds=row[8], value_edge=row[9],
            correct=bool(row[10]), generated_at=row[11], resolved_at=row[12],
        )


_store_singleton: RecommendationOutcomeStore | None = None
_SANDBOX_STORE_DB_PATH = sandbox_scoped_path("recommendation_outcomes.db")


def get_recommendation_outcome_store() -> RecommendationOutcomeStore:
    """FastAPI dependency -- overridden in tests via app.dependency_overrides.
    Sandbox mode (W29) points this at a scratch db path, same convention as
    recommendations.get_cache()/bets.get_bet_tracker()."""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = (
            RecommendationOutcomeStore(db_path=_SANDBOX_STORE_DB_PATH) if is_sandbox_mode() else RecommendationOutcomeStore()
        )
    return _store_singleton


def resolve_pending_recommendations(
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    sweden_client: object | None = None,
) -> list[RecommendationOutcome]:
    """W167: resolves every match's latest cached recommendation's actual
    pick against real results, mirroring settlement.py's settle_open_bets()
    structure (per-date result batching to respect the ~10 req/min
    football-data.org budget) but generalized across every domestic league
    the agent covers (FOOTBALL_DATA_CODE_BY_LEAGUE), not just PL.

    Deliberately does NOT trust the recommendation's own self-reported
    match.league for routing -- that field is LLM-authored and unverified,
    the same trust level as home_team/away_team. Instead merges results
    from every football-data.org competition code plus sweden_client for
    each date, keyed by match_id -- the same "disjoint id space" reasoning
    settlement.py already relies on for its own EPL+Sweden merge, just
    extended from one competition code to all of them."""
    already_resolved = store.resolved_keys()
    candidates: list[tuple] = []
    for entry in cache.list_latest_per_match():
        if (entry.match_id, entry.date) in already_resolved:
            continue
        rec = entry.recommendation
        if rec.get("overall") not in ("direct_bet", "conditional"):
            continue
        picked = pick_recommended_market(rec.get("markets") or [])
        if picked is None or picked.get("market") not in RESOLVABLE_MARKETS:
            continue
        candidates.append((entry, rec, picked))

    by_date: dict[str, list[tuple]] = {}
    for candidate in candidates:
        by_date.setdefault(candidate[0].date, []).append(candidate)

    newly_resolved: list[RecommendationOutcome] = []
    for date, group in by_date.items():
        results_by_id = {}
        for competition_code in FOOTBALL_DATA_CODE_BY_LEAGUE.values():
            for match in client.get_results(competition_code=competition_code, date_from=date, date_to=date):
                results_by_id[match.match_id] = match
        if sweden_client is not None:
            for match in sweden_client.get_results(date_from=date, date_to=date):
                results_by_id[match.match_id] = match

        for entry, rec, picked in group:
            match = results_by_id.get(entry.match_id)
            if match is None or match.home_goals is None or match.away_goals is None:
                continue
            actual = build_actual_outcome(match.home_goals, match.away_goals)
            correct = market_correct(picked, actual)
            if correct is None:
                continue
            outcome = store.insert(
                match_id=entry.match_id,
                date=entry.date,
                competition=(rec.get("match") or {}).get("league"),
                market=picked["market"],
                selection=picked["selection"],
                recommendation_type=picked["recommendation_type"],
                confidence=rec.get("confidence"),
                odds=picked.get("current_odds"),
                value_edge=picked.get("value_edge"),
                correct=correct,
                generated_at=entry.generated_at,
            )
            newly_resolved.append(outcome)
    return newly_resolved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendation_outcomes.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendation_outcomes.py app/backend/tests/test_recommendation_outcomes.py
git commit -m "feat(app): recommendation_outcomes store + resolution job (W167)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Create `recommendation_stats.py` (W168)

**Files:**
- Create: `app/backend/recommendation_stats.py`
- Test: `app/backend/tests/test_recommendation_stats.py`

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_recommendation_stats.py`:

```python
"""W168: hit-rate breakdown + Kelly ROI simulation over resolved
recommendation_outcomes."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.recommendation_outcomes import RecommendationOutcome
from app.backend.recommendation_stats import compute_recommendation_stats


def _outcome(
    match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
    recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
) -> RecommendationOutcome:
    return RecommendationOutcome(
        id=1, match_id=match_id, date=date, competition=competition, market=market, selection=selection,
        recommendation_type=recommendation_type, confidence=confidence, odds=odds, value_edge=value_edge,
        correct=correct, generated_at="2026-08-22T10:00:00+00:00", resolved_at="2026-08-23T00:00:00+00:00",
    )


def test_empty_outcomes_returns_zeroed_stats():
    stats = compute_recommendation_stats([])
    assert stats["overall"]["sample_size"] == 0
    assert stats["overall"]["hit_rate"] == 0.0
    assert stats["kelly_roi_simulation"]["bets_placed"] == 0


def test_overall_hit_rate_across_mixed_outcomes():
    outcomes = [_outcome(correct=True), _outcome(match_id="m2", correct=False)]
    stats = compute_recommendation_stats(outcomes)
    assert stats["overall"]["sample_size"] == 2
    assert stats["overall"]["correct"] == 1
    assert stats["overall"]["hit_rate"] == 0.5


def test_breakdown_by_market_and_competition_and_confidence():
    outcomes = [
        _outcome(market="result_3way", competition="E0", confidence="high", correct=True),
        _outcome(match_id="m2", market="btts", competition="SP1", confidence="low", correct=False),
    ]
    stats = compute_recommendation_stats(outcomes)
    assert stats["by_market"]["result_3way"]["sample_size"] == 1
    assert stats["by_market"]["btts"]["sample_size"] == 1
    assert stats["by_competition"]["E0"]["hit_rate"] == 1.0
    assert stats["by_competition"]["SP1"]["hit_rate"] == 0.0
    assert stats["by_confidence"]["high"]["correct"] == 1
    assert stats["by_confidence"]["low"]["correct"] == 0


def test_kelly_roi_simulation_only_includes_direct_bet_picks():
    # A conditional pick was never actually staked -- same convention
    # src/agent/staking.py's own simulators already use.
    outcomes = [
        _outcome(recommendation_type="direct_bet", odds=3.0, value_edge=0.10, correct=True),
        _outcome(match_id="m2", recommendation_type="conditional", odds=1.6, value_edge=-0.02, correct=False),
    ]
    stats = compute_recommendation_stats(outcomes)
    assert stats["kelly_roi_simulation"]["bets_placed"] == 1


def test_kelly_roi_simulation_skips_null_odds():
    outcomes = [_outcome(recommendation_type="direct_bet", odds=None, correct=True)]
    stats = compute_recommendation_stats(outcomes)
    assert stats["kelly_roi_simulation"]["bets_placed"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_recommendation_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.recommendation_stats'`

- [ ] **Step 3: Implement `recommendation_stats.py`**

Create `app/backend/recommendation_stats.py`:

```python
"""W168: diagnostics aggregation over resolved recommendation_outcomes --
hit-rate breakdown by market/competition/confidence, plus a Kelly-sized ROI
simulation reusing src/agent/staking.py's own simulate_kelly_stake (A80's
kelly_fraction) via a thin BacktestRecord adapter. Mirrors bet_stats.py's
own separation from its storage class (bet_tracker.py) -- this file is pure
aggregation, no DB I/O of its own.

Denominated in UB (an abstract Unit Bet), not dollars -- starting_bankroll
is just a plain number, same as src/agent/staking.py's own bankroll
parameter always was; see docs/superpowers/specs/2026-08-25-live-recommendation-tracking-design.md."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.backend.recommendation_outcomes import RecommendationOutcome
from src.agent.backtest import BacktestRecord
from src.agent.evaluation import build_evaluation_report
from src.agent.staking import simulate_kelly_stake

DEFAULT_STARTING_BANKROLL = 1000.0


def _hit_rate(outcomes: list[RecommendationOutcome]) -> dict[str, Any]:
    correct = sum(1 for o in outcomes if o.correct)
    return {
        "sample_size": len(outcomes),
        "correct": correct,
        "hit_rate": round(correct / len(outcomes), 6) if outcomes else 0.0,
    }


def _breakdown_by(outcomes: list[RecommendationOutcome], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[RecommendationOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[getattr(outcome, key) or "unknown"].append(outcome)
    return {group_key: _hit_rate(group) for group_key, group in groups.items()}


def _to_backtest_records(outcomes: list[RecommendationOutcome]) -> list[BacktestRecord]:
    return [
        BacktestRecord(
            match_id=o.match_id, home_team="", away_team="", date=o.date, league=o.competition or "",
            recommendation={}, actual={},
            market_results=[{
                "market": o.market, "selection": o.selection, "recommendation_type": o.recommendation_type,
                "current_odds": o.odds, "value_edge": o.value_edge, "correct": o.correct,
            }],
        )
        for o in outcomes
    ]


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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendation_stats.py -v`
Expected: PASS, 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendation_stats.py app/backend/tests/test_recommendation_stats.py
git commit -m "feat(app): recommendation_stats aggregation with Kelly ROI simulation (W168)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Wire the two new endpoints into main.py (W167/W168)

**Files:**
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_recommendation_outcomes_endpoints.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_recommendation_outcomes_endpoints.py`:

```python
"""W167/W168: POST /api/recommendations/settle-open, GET /api/recommendations/stats.

Mirrors test_settlement_endpoint.py's own patching convention exactly:
settle_open_recommendations() (like settle_open()) calls get_fixtures_client()/
get_sweden_fixtures_client() as plain module-level function calls inside the
endpoint body, not via Depends() -- so tests patch
"app.backend.main.get_fixtures_client" directly rather than using
app.dependency_overrides for those two."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore, get_recommendation_outcome_store


def _override(tmp_path: Path) -> tuple[RecommendationCache, RecommendationOutcomeStore]:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    app.dependency_overrides[recommendations.get_cache] = lambda: cache
    app.dependency_overrides[get_recommendation_outcome_store] = lambda: store
    return cache, store


@pytest.fixture(autouse=True)
def sweden_client_mock():
    """W57 precedent (test_settlement_endpoint.py): defaulted to empty so
    every test here that doesn't care about Sweden keeps working unchanged --
    resolve_pending_recommendations always consults sweden_client when the
    endpoint supplies one, regardless of whether any candidate is Swedish."""
    with patch("app.backend.main.get_sweden_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_results.return_value = []
        yield mock_get_client.return_value


_REC = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1}],
    "confidence": "medium", "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
}


def test_settle_open_endpoint_resolves_and_returns_outcomes(tmp_path: Path):
    cache, store = _override(tmp_path)
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _REC, "scheduled")
    try:
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = [
                NormalizedMatch(match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED", home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1)
            ]
            with TestClient(app) as client:
                response = client.post("/api/recommendations/settle-open")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["correct"] is True
        assert len(store.list_all()) == 1
    finally:
        app.dependency_overrides.clear()


def test_settle_open_endpoint_returns_empty_list_when_nothing_resolves(tmp_path: Path):
    _override(tmp_path)
    try:
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.post("/api/recommendations/settle-open")
        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_returns_zeroed_stats_with_nothing_resolved(tmp_path: Path):
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/stats")
        assert response.status_code == 200
        assert response.json()["overall"]["sample_size"] == 0
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_reflects_a_resolved_outcome(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/stats")
        assert response.status_code == 200
        assert response.json()["overall"]["sample_size"] == 1
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_days_param_filters_by_date(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="old", date="2020-01-01", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2020-01-01T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/stats?days=30")
        assert response.json()["overall"]["sample_size"] == 0
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_recommendation_outcomes_endpoints.py -v`
Expected: FAIL with 404s (routes don't exist yet).

- [ ] **Step 3: Add the endpoints to main.py**

Add the `timedelta` import. Find:

```python
from datetime import date, datetime, timezone
```

Replace with:

```python
from datetime import date, datetime, timedelta, timezone
```

Add the new imports near the other `app.backend` imports. Find:

```python
from app.backend.recommendation_cache import DEFAULT_DB_PATH as RECOMMENDATION_CACHE_DB_PATH, RecommendationCache
from app.backend.bet_stats import compute_bet_stats
from app.backend.recommendations import MatchRecommendationOut, RecommendationRequest, validate_and_degrade
```

Replace with:

```python
from app.backend.recommendation_cache import DEFAULT_DB_PATH as RECOMMENDATION_CACHE_DB_PATH, RecommendationCache
from app.backend.recommendation_outcomes import (
    RecommendationOutcomeStore,
    get_recommendation_outcome_store,
    resolve_pending_recommendations,
)
from app.backend.recommendation_stats import compute_recommendation_stats
from app.backend.bet_stats import compute_bet_stats
from app.backend.recommendations import MatchRecommendationOut, RecommendationRequest, validate_and_degrade
```

Now add the two endpoints, right after the existing `GET /api/recommendations/{match_id}` block. Find:

```python
    agent_config_hash = compute_agent_config_hash(AgentConfig.default())
    entry = cache.get_latest(match_id, date, agent_config_hash) or cache.get_latest_any_config(match_id, date)
    if entry is None:
        raise HTTPException(status_code=404, detail="No cached recommendation for this match/date yet.")
    return validate_and_degrade(entry.recommendation)

```

Replace with (keeping the existing code, adding new code after it):

```python
    agent_config_hash = compute_agent_config_hash(AgentConfig.default())
    entry = cache.get_latest(match_id, date, agent_config_hash) or cache.get_latest_any_config(match_id, date)
    if entry is None:
        raise HTTPException(status_code=404, detail="No cached recommendation for this match/date yet.")
    return validate_and_degrade(entry.recommendation)


class RecommendationOutcomeOut(BaseModel):
    id: int
    match_id: str
    date: str
    competition: str | None
    market: str
    selection: str
    recommendation_type: str
    confidence: str | None
    odds: float | None
    value_edge: float | None
    correct: bool
    generated_at: str
    resolved_at: str


@app.post("/api/recommendations/settle-open")
async def settle_open_recommendations(
    cache: RecommendationCache = Depends(recommendations.get_cache),
    store: RecommendationOutcomeStore = Depends(get_recommendation_outcome_store),
) -> list[RecommendationOutcomeOut]:
    """W167: on-demand resolution trigger, same trigger story as
    /api/bets/settle-open -- not scheduler-tied. Diagnostics only, for the
    user's own querying; the frontend never calls this. Calls
    get_fixtures_client()/get_sweden_fixtures_client() as plain function
    calls (not Depends()), exactly mirroring settle_open()'s own existing
    shape immediately below -- both reuse the same fixtures/results client
    and its shared rate-limit budget."""
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    resolved = await run_in_threadpool(resolve_pending_recommendations, cache, store, client, sweden_client)
    return [RecommendationOutcomeOut(**dataclasses.asdict(r)) for r in resolved]


@app.get("/api/recommendations/stats")
async def get_recommendation_stats(
    days: int = 30,
    store: RecommendationOutcomeStore = Depends(get_recommendation_outcome_store),
) -> dict:
    """W168: hit-rate breakdown + Kelly ROI simulation over resolved
    recommendation_outcomes, denominated in UB. Diagnostics only -- queried
    directly by the user, no frontend surface."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    outcomes = store.list_all(since=cutoff)
    return compute_recommendation_stats(outcomes)

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendation_outcomes_endpoints.py -v`
Expected: PASS, 5 passed.

Then run the full backend suite to confirm nothing else broke:

Run: `pytest app/backend/tests/ -v`
Expected: PASS, all tests, no regressions.

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_recommendation_outcomes_endpoints.py
git commit -m "feat(app): wire POST /api/recommendations/settle-open, GET /api/recommendations/stats (W167/W168)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: Frontend — thread `unit_bet_multiplier` through and display it on the card (W169)

**Files:**
- Modify: `app/frontend/lib/types.ts`
- Modify: `app/frontend/components/MatchUI.tsx`
- Test: `app/frontend/components/__tests__/MatchUI.test.tsx`

- [ ] **Step 1: Write the failing tests**

Append to `app/frontend/components/__tests__/MatchUI.test.tsx` (inside the same file, alongside the existing `target_odds` tests around line 260-280 -- place these new tests directly after `test("shows the plain Odds box when target_odds is absent (pre-A52 cached data)")` in the same `describe` block):

```typescript
  it("shows the suggested UB stake for an actionable upcoming card", () => {
    const match = baseMatch({
      overall: "direct_bet",
      unitBetMultiplier: 2.3,
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "direct_bet",
          currentOdds: 2.0, minOdds: 0, mlProbability: 0.55, impliedProbability: 0.5, valueEdge: 0.1,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Suggested: 2.3 UB")).toBeInTheDocument();
  });

  it("shows nothing when unitBetMultiplier is null", () => {
    const match = baseMatch({
      overall: "no_bet",
      unitBetMultiplier: null,
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "no_bet",
          currentOdds: 2.0, minOdds: 0, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: -0.01,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText(/Suggested:/)).not.toBeInTheDocument();
  });

  it("does not show the suggested stake on a completed card", () => {
    const match = baseMatch({
      status: "completed",
      overall: "direct_bet",
      unitBetMultiplier: 2.3,
      result: { home: 2, away: 1 },
      markets: [
        {
          market: "result_3way", selection: "home", recommendationType: "direct_bet",
          currentOdds: 2.0, minOdds: 0, mlProbability: 0.55, impliedProbability: 0.5, valueEdge: 0.1,
        },
      ],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText(/Suggested:/)).not.toBeInTheDocument();
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app/frontend && npx vitest run components/__tests__/MatchUI.test.tsx -t "unitBetMultiplier\|suggested UB\|Suggested"`
Expected: FAIL — TypeScript error (`unitBetMultiplier` doesn't exist on `Match`), or the text isn't found.

- [ ] **Step 3: Add the field to types.ts, Match, applyRecommendation, and render it**

In `app/frontend/lib/types.ts`, add to `MatchRecommendationOut` (insert after `unknown_team: boolean;`):

```typescript
  unknown_team: boolean;
  // A82 (agent_user_stories.md): Kelly-derived stake-sizing suggestion for
  // this recommendation's actual pick, as a multiple of an abstract "Unit
  // Bet" -- not a dollar figure. null when there's no priced pick, or
  // absent entirely on a pre-A82 cached row.
  unit_bet_multiplier?: number | null;
};
```

In `app/frontend/components/MatchUI.tsx`, add to the `Match` type (insert after `invalidMarketCount: number;`):

```typescript
  invalidMarketCount: number;
  // A82/W169: Kelly-derived suggested stake for this recommendation's
  // actual pick, as a multiple of an abstract Unit Bet -- not a dollar
  // figure. null/undefined when there's no priced pick to suggest for.
  unitBetMultiplier?: number | null;
};
```

In `applyRecommendation`, add the field to the returned object (insert after `invalidMarketCount: rec.invalid_market_count,`):

```typescript
    invalidMarketCount: rec.invalid_market_count,
    unitBetMultiplier: rec.unit_bet_multiplier ?? null,
```

Now render it in the Pick column. Find this block (the hit echo, right under the Pick's selection line):

```typescript
              {hit !== null && (
                <div className={`flex items-center gap-1 text-xs font-medium ${hit ? "text-good" : "text-serious"}`}>
                  {hit ? <CheckCircle weight="fill" size={11} /> : <XCircle weight="fill" size={11} />}
                  {hit ? "Hit" : "Not Hit"}
                </div>
              )}
            </div>
```

Replace with (adding the new block right after it, still inside the Pick column's `<div>`):

```typescript
              {hit !== null && (
                <div className={`flex items-center gap-1 text-xs font-medium ${hit ? "text-good" : "text-serious"}`}>
                  {hit ? <CheckCircle weight="fill" size={11} /> : <XCircle weight="fill" size={11} />}
                  {hit ? "Hit" : "Not Hit"}
                </div>
              )}
              {/* A82/W169: Kelly-derived suggested stake for the actual
                  pick, in UB (an abstract unit -- see the Daily Edges
                  header explainer, not a dollar figure). Only meaningful
                  pre-match -- a completed match has nothing left to size a
                  stake for. */}
              {!isCompleted && match.unitBetMultiplier != null && (
                <div className="text-xs font-medium text-ink-secondary">
                  Suggested: {match.unitBetMultiplier.toFixed(1)} UB
                </div>
              )}
            </div>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app/frontend && npx vitest run components/__tests__/MatchUI.test.tsx`
Expected: PASS, all tests in the file including the 3 new ones.

Then typecheck:

Run: `cd app/frontend && npx tsc --noEmit`
Expected: clean, no errors.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/lib/types.ts app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.test.tsx
git commit -m "feat(app): show Kelly-derived UB stake suggestion on recommendation cards (W169)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: Frontend — Daily Edges UB explainer line (W169)

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx`
- Test: `app/frontend/components/__tests__/MatchUI.precompute.test.tsx`

- [ ] **Step 1: Write the failing test**

Append to `app/frontend/components/__tests__/MatchUI.precompute.test.tsx`, as a new `describe` block at the end of the file:

```typescript
describe("Daily Edges UB explainer line (W169)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
    vi.mocked(getFixtures).mockResolvedValue([]);
  });

  it("shows a static explanation of the UB (Unit Bet) convention under the header", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText(/UB = Unit Bet, your standard betting unit — bet 2 UB at odds 3\.0, get 6 UB back\./)
    ).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/MatchUI.precompute.test.tsx -t "UB explainer"`
Expected: FAIL — text not found.

- [ ] **Step 3: Add the explainer line**

In `app/frontend/components/MatchUI.tsx`, find the Daily Edges header block:

```typescript
              <div>
                <h1 className="text-xl font-semibold tracking-tight text-ink">Daily Edges</h1>
                {/* Mockup point 3: a live stat summary, not the old static
                    subtitle W119 removed -- only once matches have actually
                    loaded (nothing to summarize before then). */}
                {shownMatches.length > 0 && (
                  <p className="mt-0.5 text-sm text-ink-secondary">
                    {shownMatches.length} match{shownMatches.length === 1 ? "" : "es"} · {positiveEdgeCount} with
                    positive edge
                  </p>
                )}
              </div>
```

Replace with:

```typescript
              <div>
                <h1 className="text-xl font-semibold tracking-tight text-ink">Daily Edges</h1>
                {/* Mockup point 3: a live stat summary, not the old static
                    subtitle W119 removed -- only once matches have actually
                    loaded (nothing to summarize before then). */}
                {shownMatches.length > 0 && (
                  <p className="mt-0.5 text-sm text-ink-secondary">
                    {shownMatches.length} match{shownMatches.length === 1 ? "" : "es"} · {positiveEdgeCount} with
                    positive edge
                  </p>
                )}
                {/* W169: static, no API call -- UB is an abstract betting
                    unit (A82), not a dollar figure, so there's nothing to
                    fetch here, just an explanation of the convention. */}
                <p className="mt-0.5 text-xs text-ink-secondary">
                  UB = Unit Bet, your standard betting unit — bet 2 UB at odds 3.0, get 6 UB back.
                </p>
              </div>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app/frontend && npx vitest run components/__tests__/MatchUI.precompute.test.tsx`
Expected: PASS, all tests in the file including the new one.

Then run the full frontend suite:

Run: `cd app/frontend && npx vitest run`
Expected: PASS, no regressions (aside from the file's own pre-existing, documented flake noted elsewhere in this codebase's story docs -- rerun standalone if anything in `MatchUI.dateboundary.test.tsx` fails under parallel execution).

Then typecheck and build:

Run: `cd app/frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.precompute.test.tsx
git commit -m "feat(app): add UB explainer line under Daily Edges header (W169)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 11: Full verification + mark stories completed

**Files:**
- Modify: `documents/agent_user_stories.md`
- Modify: `documents/app_user_stories.md`

- [ ] **Step 1: Run the full backend test suite**

Run: `pytest tests/ app/backend/tests/ -v`
Expected: PASS, 0 failures. Note the total pass count for the completion notes below.

- [ ] **Step 2: Run the full frontend test suite + typecheck + build**

Run: `cd app/frontend && npx vitest run && npx tsc --noEmit && npm run build`
Expected: PASS, 0 failures, clean typecheck, clean build.

- [ ] **Step 3: Manual sanity check of the new endpoints**

With the backend running locally (`uvicorn app.backend.main:app --reload` from repo root), confirm:

```bash
curl -X POST http://localhost:8000/api/recommendations/settle-open
curl http://localhost:8000/api/recommendations/stats
```

Both should return `200` with valid JSON (an empty list / zeroed stats is fine if nothing is cached locally yet — this just confirms the routes are live and don't 500).

- [ ] **Step 4: Mark A80–A82 completed in agent_user_stories.md**

In `documents/agent_user_stories.md`, change each of A80, A81, A82's `active` to `completed`, and append a `**Completion notes (<today's date>):**` sentence to each row's Comments cell summarizing what shipped and the real test-suite pass count from Step 1 — follow the exact style of the existing completed rows immediately above them in the same file (e.g. A79's own completion notes) as the template.

- [ ] **Step 5: Mark W167–W169 completed in app_user_stories.md**

In `documents/app_user_stories.md`, change each of W167, W168, W169's `active` to `completed`, and append completion notes the same way, following W166's own style immediately above as the template. Include the real frontend test/build results from Step 2.

- [ ] **Step 6: Commit**

```bash
git add documents/agent_user_stories.md documents/app_user_stories.md
git commit -m "docs: mark A80-A82/W167-W169 completed with verification results

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
