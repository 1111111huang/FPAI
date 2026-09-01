# Backtest/Train Recommendation-Pick Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `src/agent/backtest.py`'s `process_match_row()` so it builds `market_results` from the agent's resolved `recommendation_pick` instead of the dead `recommendation.get("markets", [])` — closing out sub-project #3 of the single-market-recommendation redesign.

**Architecture:** One function, one call site. `process_match_row()` already imports `market_correct`/`build_actual_outcome` from `src.agent.market_resolution`; add `resolve_recommendation_pick` to that same import and use it to find the one candidate the agent actually picked, then wrap it (or nothing) into the same `market_results: list[dict]` shape every downstream consumer (`staking.py`, `lessons.py`) already expects.

**Tech Stack:** Python, pytest, `unittest.mock.patch` (existing test file's own style — `run_agent` is mocked, not the real LLM).

---

### Task 1: Fix `process_match_row()` + rework `tests/test_backtest.py`

**Files:**
- Modify: `src/agent/backtest.py:18` (import), `src/agent/backtest.py:203-207` (`market_results` construction)
- Modify: `tests/test_backtest.py`

Full design: `docs/superpowers/specs/2026-09-01-backtest-train-recommendation-pick-design.md`.

**Background for the person/agent implementing this:** `run_agent()` (via `src/agent/graph.py` → `src/agent/schema.py`'s `extract_recommendation()`) has returned a `recommendation` dict shaped `{"match": ..., "overall": ..., "candidates": [...], "recommendation_pick": {"market": ..., "selection": ...} | None, ...}` since an earlier phase of this redesign (Phase 29, already merged to `main`). `process_match_row()` in `backtest.py` still reads the OLD shape (`recommendation.get("markets", [])`, a flat array where every market carried its own `recommendation_type`/`current_odds`/`value_edge`/etc.) — since that key no longer exists on any real `run_agent()` output, this line has been silently returning `[]` on every real backtest/train run since Phase 29 merged. `tests/test_backtest.py` doesn't catch this because it mocks `run_agent` directly with an old-shape dict, bypassing the real schema entirely.

`resolve_recommendation_pick(candidates: list[dict], pick: dict | None) -> dict | None` already exists in `src/agent/market_resolution.py` (used by every other caller across this redesign — `app/backend/recommendations.py`, `recommendation_outcomes.py`, `bets.py`, the frontend's `resolveRecommendation()`). It does a plain equality lookup: finds the entry in `candidates` whose `market`/`selection` match `pick`'s, tolerating malformed/missing data by returning `None` rather than raising. This task reuses it, not reimplements it.

- [ ] **Step 1: Write the failing test proving one-bet-per-match**

Add to `tests/test_backtest.py`, right after `test_process_match_row_scores_markets_correctly` (which this step's Step 3 will also rewrite):

```python
def test_process_match_row_scores_only_the_recommendation_pick_not_every_candidate():
    """A92: market_results must reflect only the one candidate the agent
    actually picked, not every candidate that independently passed
    guardrails -- matches what live settlement/recommendation_stats.py
    already do (see the design spec's LIVE_SOURCE_NOTE discussion)."""
    recommendation = {
        "match": {}, "overall": "direct_bet",
        "candidates": [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 1.9, "min_odds": 1.9, "ml_probability": 0.6, "implied_probability": 0.52, "value_edge": 0.08, "composite_score": 0.6, "reason": "x"},
            {"market": "btts", "selection": "yes", "recommendation_type": "direct_bet", "current_odds": 2.1, "min_odds": 1.8, "ml_probability": 0.55, "implied_probability": 0.48, "value_edge": 0.14, "composite_score": 0.9, "reason": "y"},
        ],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation), \
         patch("src.agent.tools.configure_snapshot_store"):
        record = process_match_row(_row(fthg=2, ftag=1), _make_config())

    # btts/yes has the higher value_edge (0.14 vs 0.08) and is individually
    # direct_bet-eligible, but it was NOT the recommendation_pick -- it must
    # not appear in market_results at all.
    assert [m["market"] for m in record.market_results] == ["result_3way"]
    assert record.market_results[0]["selection"] == "home"
    assert record.market_results[0]["correct"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest.py::test_process_match_row_scores_only_the_recommendation_pick_not_every_candidate -v`
Expected: FAIL — with the current code, `recommendation.get("markets", [])` returns `[]` regardless of `candidates`/`recommendation_pick`, so `record.market_results == []`, not `["result_3way"]`. (The assertion `== ["result_3way"]` fails against `== []`.)

- [ ] **Step 3: Implement the fix**

In `src/agent/backtest.py`, change the import on line 18:

```python
from src.agent.market_resolution import build_actual_outcome, market_correct as _market_correct
```

to:

```python
from src.agent.market_resolution import build_actual_outcome, market_correct as _market_correct, resolve_recommendation_pick
```

Then replace the `market_results` construction (currently):

```python
    actual = load_outcome(row)
    market_results = [
        {**m, "correct": _market_correct(m, actual)}
        for m in recommendation.get("markets", [])
    ]
```

with:

```python
    actual = load_outcome(row)
    picked = resolve_recommendation_pick(
        recommendation.get("candidates") or [], recommendation.get("recommendation_pick")
    )
    market_results = [{**picked, "correct": _market_correct(picked, actual)}] if picked else []
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `python -m pytest tests/test_backtest.py::test_process_match_row_scores_only_the_recommendation_pick_not_every_candidate -v`
Expected: PASS

- [ ] **Step 5: Rework this file's own old-shape fixtures**

`tests/test_backtest.py` has 7 more places building an old-shape `recommendation` dict (grep the file for `"markets"` to find all of them — there should be exactly 7 remaining after Step 1's new test, all of them `recommendation = {...}` literals). Two categories:

**(a) The detailed fixture in `test_process_match_row_scores_markets_correctly`** (around line 102-126) exercises real market-scoring behavior end-to-end (a resolvable `result_3way` pick that's correct, an unresolvable `home_corners` candidate). Rework it to the new shape, keeping the same scoring assertions but now against a single resolved pick:

```python
def test_process_match_row_scores_markets_correctly():
    recommendation = {
        "match": {}, "overall": "direct_bet",
        "candidates": [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 1.9, "min_odds": 1.9, "ml_probability": 0.6, "implied_probability": 0.52, "value_edge": 0.08, "composite_score": 0.7, "reason": "x"},
            {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 1.8, "min_odds": 2.0, "ml_probability": 0.5, "implied_probability": 0.55, "value_edge": -0.05, "composite_score": 0.1, "reason": "y"},
            {"market": "home_corners", "selection": "over_4.5", "recommendation_type": "no_bet", "current_odds": 1.9, "min_odds": 1.9, "ml_probability": 0.5, "implied_probability": 0.52, "value_edge": -0.02, "composite_score": 0.1, "reason": "z"},
        ],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure:
        record = process_match_row(_row(fthg=2, ftag=1), _make_config())

    assert isinstance(record, BacktestRecord)
    assert record.actual["result"] == "home"
    assert len(record.market_results) == 1
    assert record.market_results[0]["market"] == "result_3way"
    assert record.market_results[0]["correct"] is True

    # configure_snapshot_store called with replay then live (record_calls captures the mode transitions)
    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live"]
    mock_run.assert_called_once()
```

(Note: `btts`/`home_corners` are now `no_bet` rather than one of them being an eligible-but-unpicked `direct_bet` — that scenario is what Step 1's new test already covers dedicatedly; no need to duplicate it here. This test's job is proving the resolvable/unresolvable scoring path, which only needs one resolved pick to exercise.)

**(b) The six `"markets": []` placeholder fixtures** in `test_process_match_row_uses_league_scoped_base_dir`, `test_process_match_row_threads_allow_lessons_in_replay_to_configure_snapshot_store`, `test_process_match_row_passes_leakage_guard_instructions_to_run_agent`, `test_process_match_row_passes_leakage_guard_instructions_with_capture_state`, `test_process_match_row_captures_full_state_when_requested`, `test_process_match_row_full_state_none_by_default` — none of these assert anything about `market_results`, they're testing unrelated behavior (base_dir threading, `allow_lessons_in_replay`, leakage-guard instructions, `full_state` capture). Replace each `"markets": []` with `"candidates": [], "recommendation_pick": None` (same no-op result either way, just matching the real shape instead of a dead key):

```python
recommendation = {
    "match": {}, "overall": "no_bet", "candidates": [], "recommendation_pick": None,
    "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
}
```

- [ ] **Step 6: Run the full file, then the full suite**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: PASS, every test in the file (26 tests: the original 25 plus Step 1's new one).

Run: `python -m pytest tests/ app/backend/tests/ -q`
Expected: PASS, matching the already-tracked pre-existing environmental baseline exactly (missing local `data/fpai_core.db`: `test_fixtures_endpoint.py` + `test_prepare_training_data_league_scoping.py`) and nothing else red. If anything outside that baseline is red, stop and fix it before committing — `staking.py`/`lessons.py` are shape-agnostic per the design spec's own investigation, so a regression there would mean that investigation missed something real; don't paper over it.

- [ ] **Step 7: Commit**

```bash
git add src/agent/backtest.py tests/test_backtest.py
git commit -m "fix(agent): A92 -- process_match_row() reads recommendation_pick, not dead markets key

recommendation.get(\"markets\", []) has silently returned [] on every real
agent-backtest/agent-train run since Phase 29 merged (run_agent() has
produced candidates/recommendation_pick since then, never markets) --
every backtest staking simulation and lesson-stats block has recorded
zero bets. Fixed by resolving recommendation_pick against candidates via
the same resolve_recommendation_pick() every other caller in this
redesign already uses, narrowing backtest to one bet per match (matching
what live settlement/recommendation_stats.py already do).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Full regression + user-story completion note

**Files:**
- Modify: `documents/agent_user_stories.md` (A92's `future` status and Comments cell)

- [ ] **Step 1: Run the full suite one more time on the final state**

Run: `python -m pytest tests/ app/backend/tests/ -q 2>&1 | tail -30`
Expected: clean except the already-documented pre-existing environmental failures. Record the exact pass/fail/skip counts for the completion note.

- [ ] **Step 2: Update A92's row in `documents/agent_user_stories.md`**

Change `future` to `completed`, and append a completion note to the `Comments` cell (after the existing `Size ... · Depends on: ...` text) summarizing what actually shipped, naming the real commit SHA, the real test file/count, and the real full-suite numbers from Step 1 — matching this document's own completion-note convention (see A88-A91 in Phase 29, or W193-W197 in `documents/app_user_stories.md`'s Phase 47, for the established format).

- [ ] **Step 3: Commit**

```bash
git add documents/agent_user_stories.md
git commit -m "docs(agent): A92 completion notes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
