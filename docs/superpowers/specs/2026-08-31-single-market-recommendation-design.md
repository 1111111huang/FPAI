# Single-Recommendation Decision Mechanism — Design

**Date:** 2026-08-31
**Status:** approved, pending implementation plan
**Origin:** direct user request, following a conversation about how `pick_recommended_market()`/`bestMarket()` reduce an agent-produced `markets` array down to "the one shown on the web app." The user's own framing: "I want the agent to look at all markets together, with the news, odds to make one bet recommendation, preferably balance the edge and the hit probability."

## The problem this replaces

Today, `extract_recommendation()` (`src/agent/schema.py`) asks the LLM to independently evaluate every market (`result_3way`, `btts`, `total_goals`, `home_corners`, `away_corners`) in one shot, each getting its own `recommendation_type`/`ml_probability`/`value_edge`. Nothing about that per-market judgment considers the other markets — a `result_3way`/draw pick and a `btts`/no pick can both independently clear `direct_bet` on the same match with no awareness of each other.

"Which one is shown" is then a pure `max(value_edge)` reduction, applied three separate times by three independently-maintained implementations: `pick_recommended_market()` (Python, generation-time sizing and settlement grading), and `bestMarket()` (TypeScript, every frontend render). The reduction rewards raw edge only — a market with a smaller edge but much higher hit probability always loses to a noisier, larger one. It also means a genuinely comparative judgment (should a smaller, safer edge beat a larger, riskier one? are two markets correlated enough that recommending both is double-counting the same signal?) never actually happens anywhere in the system.

## What changes

The agent produces **one** recommendation per match, chosen by weighing all evaluated markets against each other — not five independent verdicts reduced by a `max()` afterward. This spec covers the decision mechanism only: the schema, the prompt, and the guardrail pipeline. It defines the contract two follow-up sub-projects will consume:

- **Live serving** (not this spec): `pipeline.py`/`graph.py` wiring, settlement (`recommendation_outcomes.py`), the frontend (`MatchUI.tsx`), and cache-compatibility for rows written under the old schema.
- **Backtest/train harness** (not this spec): `BacktestRecord`, `evaluation.py`'s ROI/Kelly simulation, `staking.py`, `live_lessons.py`'s historical adapter, the `agent-backtest`/`agent-train` CLI.

Both were deliberately scoped out after the project was found too large for a single spec — this document defines what they'll each build on top of.

## Call structure: still one LLM call

Two calls (independent per-market scoring, then a synthesis call) was considered and rejected. This system is already tight on LLM/tool-call budget — `max_tool_calls` was cut 10→3 in production seven days ago (A87) after Tavily quota exhaustion traced back to `research_node`'s guaranteed baseline searches running on every `run_agent()` call, including every T-30 refresh. Doubling the LLM cost of every single generation (pregenerate, EOD, T-30, manual regenerate) for this feature would be a real, recurring expense, not a one-time engineering cost. The restructured prompt asks for the same one-shot reasoning it already does today, just organized differently.

## Schema

Replaces `markets: list[MarketRecommendationModel]` with two fields:

```python
class MarketCandidateModel(BaseModel):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float | None
    min_odds: float = 0.0
    ml_probability: float
    implied_probability: float
    value_edge: float
    target_odds: float | None = None
    composite_score: float   # NEW -- the LLM's own edge/hit-probability balance, self-reported
    reason: str              # NEW -- one line: why this candidate won or lost

class RecommendationPick(BaseModel):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]

class MatchRecommendationModel(BaseModel):
    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    candidates: list[MarketCandidateModel]            # every market with real matched odds
    recommendation_pick: RecommendationPick | None     # which candidate is the pick
    explanation: list[str]
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str
```

`recommendation_pick` is deliberately a pointer — `market`+`selection` only, not a duplicate copy of the candidate's numeric fields. Code resolves the real `recommendation` object by matching it against `candidates`. This makes it structurally impossible for "the pick" and "its own listed numbers" to quietly disagree, since there is only ever one copy of the data. If the LLM points `recommendation_pick` at a market/selection absent from its own `candidates` list, that is a clean, new validation failure (degrades to `insufficient_data`), not a silent bug to discover later.

`overall` stays, for naming continuity, but is now just `resolved_recommendation.recommendation_type` (or `no_bet`/`insufficient_data` when there's no valid pick) — there is nothing left to reconcile it against, so `_reconcile_overall_with_markets` (A65) is deleted entirely rather than adapted.

The snippet above only shows fields that change shape. `unit_bet_multiplier`, `cold_start_risk`, `feature_completeness`, and `unknown_team` (the `MatchRecommendation` TypedDict's own post-processing fields, `src/agent/schema.py`) are unaffected and stay exactly as they are today — see the Guardrails section below for how the two that derive from "the picked market" (`unit_bet_multiplier`, `target_odds`) adapt.

## Prompt changes

`config/prompts/agent_v1.txt` and its three posture siblings (conservative/balanced/aggressive) gain three things:

1. **Every guardrail threshold, templated in as a real number.** Extends the pattern A84/A85 already established (`{{DRAW_VALUE_EDGE_CLAUSE}}`) to all six: min/max odds bounds (A29), conditional floor/ceiling (A66/A84), the base value-edge floor, and the draw-specific floor (A85). New instruction alongside them: a candidate that would fail one of these checks may still appear in `candidates` for transparency, but must not be the `recommendation_pick` — pick a different eligible one, or if none clear every check, set `overall` to `no_bet` and leave `recommendation_pick` null.
2. **The balance instruction.** A smaller edge backed by a materially higher hit probability should generally beat a larger edge resting on high uncertainty; `composite_score` should reflect that judgment, not restate `value_edge`. This is guidance, not something code trusts blindly (see the self-consistency guardrail below) — this codebase has hit LLM numeric-rule unreliability before (BUG-019, BUG-023).
3. **Output structure.** Evaluate every market with real matched odds, write one `candidates` entry each (including ones being rejected, each with `composite_score` and a one-line `reason`), then set `recommendation_pick` to the winner. `explanation` must carry the actual cross-market tradeoff reasoning, not a per-market walkthrough — same discipline already enforced in the current prompt.

## Guardrails

The six existing `_downgrade_*` functions in `src/agent/schema.py` (`_downgrade_direct_bet_below_value_edge_floor`, `_downgrade_direct_bet_below_draw_value_edge_floor`, `_downgrade_direct_bet_with_null_odds`, `_downgrade_direct_bet_outside_odds_bounds`, `_downgrade_conditional_below_floor`, `_downgrade_conditional_above_ceiling`) change shape, not logic: today each loops over `data["markets"]`, downgrading whichever entries fail independently; each now runs **once**, against the single resolved `recommendation` object (after resolving `recommendation_pick` against `candidates`). Same thresholds, same rules — no loop.

**New: the self-consistency guardrail.** After resolving `recommendation`, compare its `composite_score` against every other listed candidate's. If some other candidate self-reports a *higher* `composite_score` than the one actually picked, downgrade straight to `no_bet` with a limitations note explaining why — the same downgrade-only philosophy every guardrail here already follows (never auto-substitute a different candidate, never upgrade). This catches "said X, but its own numbers favor Y" — the same class of self-contradiction BUG-027/BUG-019 already found this model prone to elsewhere.

**Deleted entirely, not adapted:** `_reconcile_overall_with_markets` (A65 — nothing left to reconcile), `pick_recommended_market()` (`src/agent/market_resolution.py`), and `bestMarket()` (`MatchUI.tsx`). Both reduction functions become unnecessary once there is only one real candidate to resolve, which also retires the two-hand-maintained-implementations-that-could-silently-drift problem those two functions represented (no shared source, no cross-language parity test, a duplicated "keep in sync" comment as the only enforcement).

`_attach_unit_bet_multiplier` (A82) and `_compute_target_odds` (A52) are unaffected in logic — both already operate on "the one picked market"; they simply read `recommendation` directly instead of calling `pick_recommended_market()` first.

## Testing

TDD, matching this codebase's existing convention:

- **Schema validation**: `MarketCandidateModel`/`RecommendationPick` field/enum tests, mirroring `test_agent_schema_validation.py`'s existing shape; a new case for "pick points at a candidate absent from `candidates`" degrading cleanly to `insufficient_data`.
- **Guardrails**: each of the six adapted `_downgrade_*` functions gets its existing test file reworked from "asserts on one downgraded entry in a list" to "asserts on the single resolved object" — same test intent, updated shape, not new coverage. New `tests/test_agent_self_consistency_guardrail.py`: picked candidate already has the top `composite_score` → unchanged; a higher-scoring rejected candidate exists → downgrades to `no_bet`.
- **Prompt-level**: extends `test_agent_prompt_thresholds.py`'s existing pattern to the newly-templated guardrail values (all six, not just A84/A85's two).
- **No cross-language parity test** — out of scope, since `bestMarket()`/`pick_recommended_market()` are being deleted, not kept in sync.

## Explicitly out of scope here

- Live serving wiring, settlement, frontend rendering, and cache-compatibility for rows already written under the old schema — sub-project #2.
- `agent-backtest`/`agent-train`, `BacktestRecord`, `evaluation.py`, `staking.py`, and the historical lessons adapter moving onto this shape — sub-project #3.
