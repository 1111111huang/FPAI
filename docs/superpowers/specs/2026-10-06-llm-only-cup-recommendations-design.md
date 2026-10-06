# LLM-Only Recommendations for Non-ML-Covered Competitions (Starting: UCL/UEL)

## 1. Problem

The frontend only ever shows matches from the 6 leagues with trained ML models (`E0, SWE, SP1, I1, D1, F1`) plus a generic `international` fallback (`config/competitions.yaml`). Cup competitions — specifically, to start, the UEFA Champions League and Europa League — are entirely unregistered today: no `competitions.yaml` entry, no `model_selection.yaml` entry, no historical `raw_matches` rows (results + odds). They are invisible to the system.

This extends coverage to UCL/UEL via a **pure-LLM recommendation path**: no trained model is ever consulted for this tier. The agent reasons directly from market odds (as context, not as a probability to beat) plus researched match context. Consequently there is no numeric ML-vs-market edge — only a recommendation, a self-reported confidence tier, and a rationale.

## 2. Scope

**In scope:**
- A new competition tier, `llm_only`, alongside the existing `general_purpose` / `competition_specific` tiers.
- UCL and UEL registered under this tier.
- A new agent tool that never looks up a model.
- A new prompt variant and a tier-aware lesson-reflection path.
- New ingestion for UCL/UEL fixtures, results, and odds (historical + live).
- Reuse of existing backtest/train tooling, frontend tier labeling, and a backtest-ROI launch gate before going live.

**Out of scope (explicitly not doing):**
- Domestic cup competitions (FA Cup, Copa del Rey, etc.) — a possible future extension of the same tier, not part of this pass.
- A feature store or any ML training pipeline for this tier — the entire point is to not need one.
- Lineup/injury data parity with ML leagues — ingestion only needs enough for live serving + backtest truth (results, odds, fixture metadata), not a full feature-engineering pipeline.
- Resolving exact fotmob/odds-provider competition codes for UCL/UEL — that's an implementation-time lookup, flagged as an open item (Section 9).

## 3. Tiering & routing

`competition_registry.py`'s `VALID_TIERS` (`competition_registry.py:41`) extends from `("general_purpose", "competition_specific")` to `("general_purpose", "competition_specific", "llm_only")`.

`config/competitions.yaml`: add `UCL` and `UEL` entries with `tier: llm_only`, `display_enabled: false` (flips to `true` only after the Section 8 launch gate passes).

Routing: `forecast_node` (`src/agent/pipeline.py`) gets a third branch — `tier == "llm_only"` dispatches to the new tool (Section 4) instead of `forecast_league`/`forecast_international`. Every tier-agnostic control already in place is untouched:
- No-odds abstain (`forecast_node` short-circuits to `status: "no_odds"`)
- `insufficient_data` backstop in `graph.py`'s `output_node` / `_apply_a30_backstop`
- The separate `no_bet` abstain path in `schema.py`

## 4. New forecast tool: `forecast_llm_only`

Added to `src/agent/tools.py`, parallel to `forecast_league`/`forecast_international`. Key difference: it **never calls `ForecastService.forecast_upcoming`** — no model lookup, no `FileNotFoundError` retry chain (that chain exists specifically to fall back *to* this tier from a missing league model; this tool is the fallback's destination, not another link in it).

It builds a context payload from:
- Odds-implied probabilities (reusing the existing, already model-independent `MKT_*` odds-feature computation)
- Fixture metadata and whatever stakes/context is available (round, leg number, group-stage vs. knockout)

Sets `prediction_basis = "llm_only_no_model"`, `tier = "llm_only"` on the payload for downstream schema/telemetry/frontend to key off.

## 5. Prompt design (`config/prompts/agent_llm_only_v1.txt`)

A **separate prompt file**, not a conditional branch inside `agent_v1.txt` — the differences are structural (see below), and reusing one file risks regressing the 6 proven leagues every time this experimental prompt iterates. Same output JSON *shape* as today (Section 6), so `schema.py`, the frontend, and the backtest harness need no shape-level branching.

**Removed / replaced (no model to anchor on):**
- No `FORECAST_PAYLOAD`, no per-target `metrics`/trust-calibration section — there is nothing to calibrate trust in.
- No `ml_probability` / `value_edge` / `composite_score`. Market odds become the **prior**, not a comparison point: deviating from the market-implied read requires a cited, concrete piece of evidence the market likely hasn't priced in (a just-announced lineup, a specific rotation signal, a stakes mismatch) — this replaces value-edge math as the discipline against confident-but-ungrounded picks.
- Confidence guidelines are rewritten around evidence quality instead of model feature coverage: **high** = clear, recent, multi-source confirmation for both sides; **medium** = partial coverage on one side; **low** = thin or conflicting sources.
- The value-edge floor/ceiling rules are replaced by a confidence-gated rule: confidence `high` → `direct_bet` eligible, `medium` → `conditional` eligible, below `medium` → forced `no_bet`. Odds sanity bounds remain — the same `{{MIN_ODDS_THRESHOLD}}`/`{{MAX_ODDS_THRESHOLD}}`/`{{MIN_CONDITIONAL_ODDS_THRESHOLD}}` template variables already used in `agent_v1.txt`, not new values to define — that's a trading-risk constraint, independent of where the edge comes from.

**Added (new structural requirements for this tier):**
- **Grounding requirement (load-bearing):** every claim in `team_evidence`/`the_read` must trace to an actual tool result from *this turn*. With no model number to sanity-check the narrative against, this is now the primary defense against confident hallucination, not an incidental rule.
- **Higher mandatory tool budget.** Today's prompt treats `web_search` as optional because pre-computed forecast + pre-gathered evidence usually cover it. Here there is no pre-gathered evidence besides odds — form/injury/stakes searches for both sides become mandatory, and the 2-tool-call stop rule is raised accordingly.
- **New "Cup/European Context" evidence-priority section**, parallel to the existing per-market list: squad rotation risk (fixture congestion, domestic-match proximity), dead-rubber group games, two-leg aggregate situations, away-goals-rule context — idiosyncrasies an ML model would normally encode from history, now reasoned about explicitly.

**Reused unchanged:** per-market evidence-priority ordering for result_3way/total_goals/btts/home_goals/away_goals, the injury-vs-benched distinction, the stale-vs-recent injury distinction, the FORM_SEARCH_RESULT competition-mixing warning, and the output JSON structure (`explanation`/`team_evidence`/`the_read`/`no_bet_read`/`limitations`).

## 6. Schema / output changes

For `tier == "llm_only"`:
- `ml_probability`, `value_edge`, `composite_score` are `null` on every candidate (never fabricated).
- `implied_probability` (from real odds) is still populated.
- `confidence` is the LLM's self-reported qualitative tier (low/medium/high), gating `recommendation_type` directly (Section 5) rather than a numeric-derived rank.
- `recommendation_pick` is chosen by the LLM directly among eligible (confidence ≥ medium) candidates — there is no composite-score ranking to mechanically enforce, since there's no numeric edge to rank by. The model justifies the pick in `the_read`.
- `prediction_basis` is fixed to `"llm_only_no_model"`.
- All other abstain/no-bet machinery (`no_odds`, `insufficient_data`) is unchanged and applies identically.

## 7. Lesson-generation loop

**Reused unchanged:** `generate_lesson_text` (deterministic fallback template) and the overall `agent_lessons`/`agent_telemetry` schema, approval flow (`approve_lesson`, `scope` of `competition`/`tier`), and the A127 staleness/fingerprint mechanism. Notably, the model-fingerprint half of that mechanism already "just works" for this tier for free: `compute_model_fingerprint` returns `None` consistently for a competition with no registered model, and `None == None` is not a mismatch — so `llm_only` lessons are naturally immune to model-retraining staleness (there's no model to go stale against). Only the agent-config-fingerprint half (prompt version) still invalidates them, correctly.

**Changed, with a `tier` parameter threaded into `generate_match_reflection`, `generate_batch_match_comparisons`, and `generate_rule_from_lesson` (`src/agent/lessons.py`):**

- `judge_lesson_candidate` / `classify_lesson_sensitivity` (`lessons.py:470-569`) both ask "would this rule survive a model retrain?" — nonsensical with no model. For `tier == "llm_only"`, skip that question and hardcode `survives_model_change=True`.
- **The reflection prompt for this tier defaults to assuming an evidence gap, not a reasoning gap** — the opposite of the ML-tier's neutral framing (`lessons.py:673`) — and only calls it a reasoning/interpretation gap when the trace clearly shows the relevant fact was already gathered this turn but misweighed. Rationale: with no model floor under the agent and a larger tool-call budget, the dominant, fixable failure mode is "didn't go look for X," not "misweighed X." This mirrors how the ML-tier's lesson loop already optimizes for interpretation fixes (e.g. the shot-volume-over-narrative rule) — this tier needs the opposite default.
- **The prescribed fix must be a specific, callable action** — a named search query, a named tool (`get_player_rating`, odds verification, a stakes/rotation check), or a named data source — never "weigh form more heavily." This is already item (4) in today's reflection prompt; for this tier it becomes the primary ask, not the trailing bullet.
- `generate_rule_from_lesson`'s distillation, for this tier, is biased toward "IF `<match situation>` THEN check/search `<specific evidence source>`" rules rather than "NEVER weigh X over Y" rules — without hard-banning the latter, since a genuine misweighting lesson should still be nameable when the trace actually shows one.
- **Default lesson scope is `tier`** (UCL and UEL share almost all relevant idiosyncrasies — rotation risk, mixed-league opponents, knockout stakes). Scope to one competition only when a reviewer sees a genuine UCL-vs-UEL divergence (e.g. UEL draws more squad rotation from clubs prioritizing UCL).

**Cold-start lesson-mining requirement:** because there is no model-promotion step to substitute for "did this get better," the lesson loop carries all of that weight for this tier. Run `agent-train-experiment` over the historical UCL/UEL backtest sample *first*, specifically to mine and get an initial batch of `scope='tier'` lessons reviewed and approved — **before**, not after, attempting the Section 8 backtest-ROI launch gate. A bad prompt has nothing to hide behind here the way it might behind a working model for ML leagues.

## 8. Data ingestion, backtesting, launch gate

**Ingestion (new work):** extend fotmob/odds-provider ingestion to pull UCL/UEL fixtures, results, and odds into `raw_matches`, following the existing per-league ingestion pattern (competition-code registration, provider mapping). Only enough for (a) historical backtest rows and (b) live fixture/odds serving — no feature store, no team-history rows at ML-league parity. Team/cross-league identity only needs name/ID mapping for display and odds correlation.

**Backtest tooling:** reused as-is. `BacktestHarness.load_matches()` already filters by `league` from `raw_matches` — works unchanged once UCL/UEL rows exist with the required result+odds columns. `agent-train-experiment` / `agent-backtest-run` are pipeline-version-agnostic and need no tooling changes beyond the registry + tool wiring above.

**Launch gate:** before `display_enabled: true`, run an agent-backtest over the historical sample; require ROI to beat (a) a flat-stake-every-match baseline and (b) an always-bet-market-favorite baseline. Same "real-edge before trust" discipline as `promoting-a-model`, measured as ROI since there's no numeric edge to check directly.

**Frontend labeling:** `MatchUI.tsx`'s `TierTag` gets a third label distinct from "Modeled" and the `general_purpose` fallback copy — suggested: "AI Judgment · No Model Backing" (copy detail, adjustable, not load-bearing for this design).

## 9. Open items (flagged, not resolved here)

- Exact frontend tier-label copy — a product/copy decision, not architectural.

**Resolved during plan-writing (2026-10-06):** investigated exact data sourcing. Live fixtures: fotmob's `/api/data/matches` already returns UCL/UEL fixtures/results unfiltered (`fetch_all_matches`, `src/ingestion/fotmob/fetcher.py:181-198`), but its parser currently extracts no score field — needs extending. Live + historical odds: OddsPapi (`app/backend/oddspapi_client.py`, `scripts/pull_oddspapi_btts_corners.py`) is the only viable source — tournament IDs for UCL/UEL aren't yet known and need discovery via `/v4/tournaments?sportId=10`, the same method already used for the 5 ML leagues' IDs. **Material constraint, accepted by direct user decision (2026-10-06):** OddsPapi's historical-odds endpoint only has data from 2026-01-01 onward (confirmed live, older fixtures 404) and a 250-req/month free-tier quota shared with other historical pulls — so the Section 8 backtest-ROI sample will be thin (a few months of matches, not a multi-season corpus) for a while. Decision: proceed anyway, treat the launch-gate result the same way the existing E0 lesson A/B (n=6-13 bets) was treated — inconclusive-not-negative on a small sample, not a blocker — and revisit the bar as more data accumulates month over month.

## 10. Testing

One runnable check per non-trivial piece of new logic:
- Registry: `llm_only` is a valid tier; `resolve_competition` routes it correctly.
- Routing: `forecast_node` dispatches `tier == "llm_only"` to `forecast_llm_only`, not `forecast_league`.
- `forecast_llm_only`: never attempts a model lookup; returns the expected schema shape with edge fields `null` and `prediction_basis = "llm_only_no_model"`.
- Schema/no-bet: confidence below `medium` forces `no_bet`; `no_odds`/`insufficient_data` abstain paths still fire unchanged.
- `judge_lesson_candidate`/`classify_lesson_sensitivity`: `tier == "llm_only"` short-circuits to `survives_model_change=True` without an extra LLM round-trip.
- Backtest harness: loads synthetic UCL/UEL rows correctly (no league-specific assumption breaks).

## 11. Documentation

Per project convention (`CLAUDE.md`):
- `documents/FRAI_TECHSPEC.md`: new section documenting the `llm_only` tier design, parallel to the existing §27.2/29.2 tier documentation.
- `documents/user_stories.md`: new `US#` entries for registry/tool/routing/ingestion work (forecast-engine side).
- `documents/agent_user_stories.md`: new entries in the relevant PHASE section for prompt/schema/lesson-loop work (agent side).
- Mark all new entries as the implementation backlog; close them out as work ships, per existing convention.
