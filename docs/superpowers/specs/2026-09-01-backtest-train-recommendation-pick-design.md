# Backtest/train recommendation-pick migration — design

**Status:** approved
**Related:** sub-project #3 of the single-market-recommendation redesign. Sub-project #1 (agent schema/prompt/guardrails) and sub-project #2 (live-serving wiring) are merged to `main`.

## Problem

`run_agent()` has produced `candidates`/`recommendation_pick` instead of a `markets` array since sub-project #1 merged. Every real consumer of the recommendation shape was migrated in sub-project #2 — except `src/agent/backtest.py`, whose `process_match_row()` still builds `market_results` from `recommendation.get("markets", [])`.

This is not just stale code: it is a live, silent bug on `main`. `recommendation.get("markets", [])` now always returns `[]`, so every real `agent-backtest`/`agent-train` run since sub-project #1 merged has produced empty `market_results` for every match — `staking.py`'s simulations record zero bets, and `lessons.py`'s stats/reflection functions report "no markets recommended" for every match. `tests/test_backtest.py` doesn't catch this because it mocks `run_agent` directly with old-shape fixtures instead of exercising the real schema pipeline.

## Scope

Investigated every consumer of the old `markets` shape and of `BacktestRecord.market_results`:

- **`src/agent/evaluation.py`** — only reads `recommendation.get("overall")` (unaffected by the redesign) and `config.markets` (a different concept: `AgentConfig`'s configured market list). No changes needed.
- **`src/agent/staking.py`** — consumes `record.market_results` as a plain list of dicts (`market`/`selection`/`recommendation_type`/`current_odds`/`value_edge`/`correct`), with no assumption about how many entries it has or where they came from. No changes needed.
- **`src/agent/lessons.py`** — same: duck-types on `record.market_results`/`.recommendation`/`.actual`/`.league`, shape-agnostic. No changes needed.
- **`app/backend/recommendation_stats.py`, `app/backend/live_lessons.py`** — already build their own `BacktestRecord`-shaped objects from `RecommendationOutcome` rows (one settled outcome per match), one-entry `market_results` lists. Already correct, already what this design brings `backtest.py` in line with. No changes needed.
- **`main.py`** — no CLI output references `markets`/`candidates` shape directly. No changes needed.

So the fix is contained to one function.

## Design decision: one bet per match

Under the old shape, every market was scored independently, so a single match could contribute multiple `direct_bet` entries to `market_results`, and `staking.py` could stake on more than one market per match. The live side never had this: `resolve_pending_recommendations()` (W194) always resolves exactly one outcome per match — the recommendation_pick — because that's the only bet the agent actually committed to. `live_lessons.py` already carries an explicit code comment (`LIVE_SOURCE_NOTE`) flagging this as a known asymmetry between backtest and live-sourced lesson batches.

Decision: `backtest.py` moves to the same one-bet-per-match model, resolving `market_results` from the recommendation's own `recommendation_pick` (via the same `resolve_recommendation_pick()` every other caller already uses), not from every individually-eligible candidate. This closes the asymmetry rather than papering over it, and makes backtest ROI/hit-rate numbers directly comparable to live numbers for the first time. Historical backtest numbers will shift as a result — multi-bet matches no longer compound — but the new numbers reflect what the agent would actually have done.

## The fix

`src/agent/backtest.py`, `process_match_row()` (currently around line 203-207):

```python
actual = load_outcome(row)
picked = resolve_recommendation_pick(
    recommendation.get("candidates") or [], recommendation.get("recommendation_pick")
)
market_results = [{**picked, "correct": _market_correct(picked, actual)}] if picked else []
```

replacing:

```python
actual = load_outcome(row)
market_results = [
    {**m, "correct": _market_correct(m, actual)}
    for m in recommendation.get("markets", [])
]
```

`resolve_recommendation_pick` is imported from `src.agent.market_resolution` (already imported in this file alongside `build_actual_outcome`/`market_correct`). `market_results` stays a list — 0 or 1 entries — so every downstream consumer (`staking.py`, `lessons.py`) needs no changes: they already treat it as "iterate whatever's there."

No backward-compat handling is needed for old snapshot replays: `process_match_row()` always calls the *current* `run_agent()` fresh (replay re-runs the live agent against recorded research/forecast evidence; the recommendation itself is generated at replay time, not read from an old cached value), and `run_agent()` has unconditionally produced the new shape since sub-project #1 merged.

## Testing

`tests/test_backtest.py` mocks `run_agent` with old-shape (`"markets": [...]`) fixtures — every one of these needs reworking to the `candidates`/`recommendation_pick` shape, following the same substitution pattern used throughout sub-projects #1 and #2. Add a new test proving the one-bet-per-match behavior: a mocked `run_agent` return with multiple `direct_bet`-eligible candidates in `candidates`, only one named by `recommendation_pick` — `market_results` must contain exactly that one entry, not all of them.

`tests/test_staking.py`, `tests/test_agent_lessons.py`, `tests/test_main_agent_train.py` construct `BacktestRecord` objects directly (not through `process_match_row()`) and don't need fixture changes — confirmed by reading each file; they're already shape-agnostic. (One cosmetic note: `test_staking.py`'s fixture builder sets `recommendation={"overall": ..., "markets": markets}` — a vestigial key `staking.py` never reads. Leaving as-is; not in scope, and touching it risks scope creep for zero behavior change.)

Full regression: `python -m pytest tests/ app/backend/tests/ -q` after the fix, expecting only the already-tracked pre-existing environmental failures (`test_fixtures_endpoint.py`, `test_prepare_training_data_league_scoping.py` in a worktree missing a local `data/fpai_core.db`).
