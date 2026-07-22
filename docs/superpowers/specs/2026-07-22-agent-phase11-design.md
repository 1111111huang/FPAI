# Phase 11 Design: Deterministic Evidence Pipeline & Critic Mode

**Date:** 2026-07-22
**Status:** Approved, pre-implementation
**Covers:** A30, A31, A32, A33 (revised), A34 (new) in `documents/agent_user_stories.md`

## Motivation

A30 was opened after the sandbox investigation (`python scripts/launch_sandbox.py 2026-03-08 --precompute`) produced a Burnley vs Bournemouth recommendation with `overall: "no_bet"`, `markets: []`, an explanation claiming "Insufficient data," and `prediction_basis: "team_history_and_market"` — despite no forecast tool ever having been called. The root cause: `forecast_league`/`forecast_international`/`resolve_competition` are LLM-callable tools the model can simply choose not to invoke, and `extract_recommendation()` (A28/A29) validates shape and bounds but never checks that the evidence a recommendation claims to rest on actually exists.

Patching this downstream (A30 alone) treats a symptom. The design below moves forecast and baseline research out of LLM tool-choice entirely and into deterministic graph steps, closing the underlying gap rather than detecting it after the fact.

## Goals

- The agent cannot produce `direct_bet`/`conditional`/`no_bet` without a real forecast having actually run.
- The agent cannot claim an evidence basis it doesn't have.
- Minimum research coverage (availability, form, odds) is guaranteed structurally, not by prompt compliance.
- A repeatable, non-interactive way to generate and vet prompt-improvement lessons from completed matches, without ever letting unreviewed lessons reach live traffic.

## Non-goals

- No new reasoning-quality scoring metric — Critic mode reuses A13's existing ROI/hit-rate/drawdown evaluation.
- No lesson-review UI — a CLI command is sufficient (A33).
- No synthetic/config-driven fallback odds — if no real odds are found, the forecast runs odds-less and existing null-odds handling (BUG-013, A29) takes over.

## Architecture: new required node sequence

Today (`src/agent/graph.py`), the graph is `agent_node ⇄ tools_node → output_node`, where `agent_node` is the LLM deciding which of `web_search`, `resolve_competition`, `forecast_league`, `forecast_international` to call, if any. That last "if any" is the bug.

New shape:

```
resolve_competition_node   (deterministic)
        ↓
   research_node            (deterministic, A32)
        ↓
   forecast_node            (deterministic, A31)
        ↓  [on failure from either of the two nodes above: → output_node directly]
   llm_synthesis_node        (LLM; web_search tool only)
        ↓
   output_node               (A28/A29 validation + A30 backstop)
```

1. **`resolve_competition_node`** — calls `_resolve_competition_impl` (currently in `src/agent/tools.py`) as a plain function, not a `@tool`. Still passes through `SnapshotStore.wrap` for record/replay, same as today.
2. **`research_node`** (A32) — runs up to three templated `web_search` calls via `_web_search_impl`, each still wrapped by `SnapshotStore`:
   - availability/injury: `"{home_team} {away_team} injury suspension news"`
   - recent form: `"{home_team} {away_team} recent form last 5 matches"`
   - odds verification: `"{home_team} vs {away_team} odds {date}"` — **only if the caller didn't supply odds** (see A32 detail below)
   Each result is parsed into `AgentState["research_evidence"]` (new field, see Data Model). A failed/empty search doesn't raise — it's recorded as coverage-missing and surfaces later as a confidence downgrade / limitation.
3. **`forecast_node`** (A31) — calls `_forecast_league_impl`/`_forecast_international_impl` directly (not as tools), selecting the function per `resolve_competition_node`'s `recommended_tool`. Odds come from, in order: caller-supplied odds → `research_evidence.odds_verification` → none (forecast runs with odds fields omitted/None, per whatever `ForecastService.forecast_upcoming` already does for missing odds — no new fallback value is invented). Stores the raw payload in `AgentState["forecast_payload"]` (new field). A hard failure (exception, or a `{"error": ..., "status": "tool_error"}` payload) routes straight to `output_node`.
4. **`llm_synthesis_node`** (renamed from `agent_node`) — system prompt is rewritten so the workflow section no longer instructs tool selection; instead it's told the forecast and research evidence are already provided in context (injected as a formatted block ahead of the human turn) and its job is synthesis + value judgment only. `web_search` remains available as an optional tool for follow-up. `forecast_league`, `forecast_international`, and `resolve_competition` are removed from `get_default_tools()`'s LLM-facing tool list (the underlying `_impl` functions stay, now called by the deterministic nodes above).
5. **`output_node`** — unchanged A28/A29 validation, plus the new A30 backstop pass (below).

### AgentState changes (`src/agent/graph.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    match_info: dict
    recommendation: dict | None
    tool_call_count: int
    competition_resolution: dict | None   # new: {tier, recommended_tool}
    research_evidence: dict | None        # new: {availability, form_context, odds_verification}
    forecast_payload: dict | None         # new: raw forecast tool result, or None on failure
```

`_extract_forecast_diagnostics()` is updated to read from `AgentState["forecast_payload"]` directly instead of scanning `messages` for a `ToolMessage` — simpler now that the forecast isn't buried in tool-call history.

### Error handling

- `resolve_competition_node` cannot fail in a way that blocks the pipeline — unregistered/missing-registry already defaults to `general_purpose` (existing behavior, unchanged).
- `forecast_node` failure (exception or `tool_error` payload) → `output_node` is invoked directly with `forecast_payload = None`; A30's backstop forces `overall: "insufficient_data"` with a limitation naming the failure. The LLM synthesis node is skipped entirely — no point spending a model call on a guaranteed-insufficient outcome.
- `research_node` failures are non-fatal: missing availability/form coverage → confidence downgrade + limitation, added at `output_node` time (mirrors A32's original acceptance criteria). Missing odds (no caller odds AND search found none) → forecast runs odds-less → BUG-013/A29's existing null-odds downgrade to `no_bet`/`conditional` already handles blocking `direct_bet` on that market. No new blocking logic needed here — it composes with what A28/A29 already built.

## A30 — backstop (narrowed scope)

With forecast now structurally guaranteed, A30 becomes a small, purely defensive check in `output_node`, run after A28/A29's existing validation:

- If `AgentState["forecast_payload"]` is `None` or missing → force `overall: "insufficient_data"`, regardless of what the LLM wrote in `overall`/`markets`/`explanation`. This should be unreachable given the graph shape above (failed forecast already short-circuits before the LLM node runs) — it exists as insurance against a future graph change reintroducing the original bug class.
- `prediction_basis`, `cold_start_risk`, `feature_completeness`, `unknown_team` are always derived from `AgentState["forecast_payload"]`'s own `data_quality` block (already how `_extract_forecast_diagnostics` works today, per the W15 comment in `schema.py`) — never from the LLM's prose. This part *is* reachable in normal operation and stays load-bearing.

No string-matching on `explanation` text — the original concern about fragile keyword detection is resolved by keying purely on the structural presence of `forecast_payload`.

## A31 — mandatory deterministic forecast

Covered above in Architecture. Key acceptance changes from the original story:

- `forecast_league`/`forecast_international`/`resolve_competition` removed from the LLM's tool list (`get_default_tools()` in `src/agent/tools.py`), not merely "required" — the LLM has no path to skip or duplicate them.
- `config/prompts/agent_v1.txt` rewritten: remove the "CALL forecast_league or forecast_international" step and the "call resolve_competition first" step; replace with instructions to use the pre-supplied forecast/research context.
- `AgentConfig.max_tool_calls` semantics change slightly — it now only bounds optional `web_search` follow-ups in the synthesis node, not the whole workflow. Existing default value may need revisiting during implementation but isn't part of this design's scope to pre-decide.

## A32 — mandatory research coverage

Covered above (`research_node`). Additional details:

- New `research_evidence` structure (not part of the public `MatchRecommendation` schema — internal `AgentState` only, per the "leave overlap to A33" decision: raw evidence is persisted to DuckDB by A33's telemetry table, not by A32 itself).
- Confidence/limitation rules ported from the original A32 acceptance: missing availability or form coverage lowers `confidence` by one step (`high→medium→low`) and appends a limitation naming the missing category; this logic lives in `output_node` alongside A30's checks, reading `AgentState["research_evidence"]`.

## A33 — Critic / train mode

- New CLI entry point, e.g. `python main.py agent-train --from-date ... --to-date ... --league ...`, structurally parallel to `agent-backtest` (`src/agent/backtest.py`): loads completed matches, replays via `SnapshotStore` (or runs live), gets a `MatchRecommendation` per match, and runs the *same* `src/agent/evaluation.py` scoring (ROI, hit rate, drawdown) — no new metric.
- For each evaluated match, additionally generates one lesson candidate: `WHEN evaluating [League/Context]...`, written to a new DuckDB table (columns: `lesson_text`, `status` (`pending`/`approved`/`rejected`), `source_match_id`, `created_at`, `reviewed_at`, `reviewer`) with `status="pending"`.
- New DuckDB `agent_telemetry` table stores per-run evidence: `match_id`, `run_id`, `competition_resolution`, `research_evidence`, `forecast_payload`, final `recommendation` JSON, timestamps. This is where A32's raw research output actually gets persisted — `research_node` itself just returns data into `AgentState`; the persistence responsibility belongs to the train-mode run loop (or, if useful for future debugging, to every mode — but live mode never *reads* outcome data, only ever writes telemetry forward).
- New minimal CLI: `python main.py agent-lessons approve <id>` / `agent-lessons reject <id>` — updates `status`, `reviewed_at`. No UI.
- Live mode (`llm_synthesis_node`'s prompt construction) loads only `status="approved"` lessons from DuckDB and appends them to the system prompt loaded from `config/prompts/`. Live mode must never query match outcomes or pending/rejected lessons — enforced by simply not giving the live code path a function that can do so.
- **Open risk, not blocking:** approved lessons accumulate indefinitely with no cap or conflict resolution. Acceptable for initial land (lesson volume will be low early on); flagged here so it isn't forgotten if this becomes a real prompt-bloat problem later.

## A34 — rebaseline (new story)

Depends on A31 + A32. The tool-call sequence changes (tools removed, new deterministic nodes) invalidate the existing 24-match snapshot corpus for replay — same precedent as A23/BUG-011: don't trust partially-compatible snapshots.

- Discard `data/agent_snapshots/` pilot corpus (or move aside, don't delete outright — confirm with user before any destructive action).
- Recollect via `agent-snapshot` over the same or an expanded date range.
- Re-run `agent-backtest --stake-mode flat` and `--stake-mode kelly`.
- Document new baseline numbers in `agent_techspec.md`, same shape as A21's section, explicitly noting these supersede A21's pre-Phase-11 numbers.

## Testing considerations

- Node-level unit tests: each new deterministic node in isolation (success, empty-result, and error paths).
- `forecast_node` odds-sourcing precedence test: caller-supplied odds win over research-found odds win over odds-less forecast.
- Short-circuit test: a failing `forecast_node` never reaches `llm_synthesis_node` (mock the LLM and assert it's not invoked).
- A30 backstop test: forcibly construct a state with `forecast_payload=None` and an LLM output claiming `no_bet`/`team_history_and_market`, assert normalization to `insufficient_data`/`unknown`.
- Snapshot key stability test: two `agent-snapshot` runs over the same match produce identical keys for the new node sequence (extends A23's existing determinism test).
- `agent-train`/`agent-lessons` CLI tests: pending lesson not loaded live; approved lesson loaded; rejected lesson never loaded; live mode has no code path that can read match outcomes.
