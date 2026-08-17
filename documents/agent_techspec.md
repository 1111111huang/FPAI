# Technical Specification — FPAI Betting Agent

Authoritative implementation reference for the agent described in `agent_prd.md`. This document reflects the actual code as built through Phase 13 (stories A01–A33, A35, A37, A39–A47, excluding A18, A22–A26). Phase 6 (batch recommendation, A18), Phase 8 (full-season backtest expansion, A23–A26), A22 (config comparison against the Section 18 baseline), A34 (Phase 11 rebaseline), A36 (Swedish pilot backtest, blocked on BUG-018), and A38 (match/market content grounding) are **not yet implemented** — see [Implementation Status](#implementation-status) below.

---

## 1. Module Structure

```
src/agent/
  __init__.py
  agent_config.py     # AgentConfig dataclass + YAML loader
  schema.py            # MatchRecommendation/MarketRecommendation TypedDicts + JSON extraction
  tools.py             # web_search, forecast_league, forecast_international, resolve_competition — all routed through SnapshotStore
  pipeline.py          # resolve_competition_node, research_node, forecast_node, lessons_node — deterministic pre-LLM graph nodes (A31–A33, A41)
  graph.py             # AgentState, build_graph(), run_agent()
  schema.py            # MatchRecommendation/MarketRecommendation TypedDicts + MatchRecommendationModel/MarketRecommendationModel (Pydantic) + JSON extraction
  snapshot_store.py    # SnapshotStore, SnapshotMissingError — record/replay interceptor (A09), allow_lessons_in_replay flag (A41)
  backtest.py          # BacktestRecord, load_outcome(), process_match_row(), match_in_test_split(), BacktestHarness (A12, A40)
  lessons.py           # create_lessons_tables(), insert_lesson_candidate(), generate_lesson_text()/generate_batch_lesson_text(), generate_batch_reflection(), generate_rule_from_lesson(), find_conflicting_rule(), approve_lesson()/reject_lesson(), load_approved_lessons() (A33, A39, A43–A45)
  staking.py           # BetOutcome, BankrollResult, simulate_flat_stake(), simulate_kelly_stake() (A13, A15)
  evaluation.py        # compute_max_drawdown(), build_evaluation_report(), config_hash(), save_report(), print_report() (A13)
  comparison.py        # compare_configs(), print_comparison_table(), save_comparison() (A16)

config/
  agent_config.yaml            # tunable knobs (model, temperature, thresholds, markets) — default provider/model track budget constraints, see Section 2
  agent_config_deepseek.yaml   # DeepSeek variant (A42) — opt-in via --config, every other entry point stays on agent_config.yaml's default
  prompts/
    agent_v1.txt       # system prompt — see Section 5

data/
  agent_snapshots/<match_id>/   # SHA-256-keyed recorded tool responses + _complete.json markers (gitignored)

reports/
  agent_backtest/                # saved evaluation reports and config comparisons (gitignored)

documents/
  agent_prd.md         # product requirements
  agent_user_stories.md  # story tracking (A01–A18)
  agent_techspec.md   # this document
```

`SnapshotStore`, `BacktestHarness`, and all three backtest-related CLI commands (`agent-snapshot`, `agent-backtest`, `agent-compare`) are implemented — see Sections 9–14.

---

## 2. AgentConfig

`src/agent/agent_config.py`

```python
@dataclass
class AgentConfig:
    model: str
    provider: Literal["ollama", "anthropic", "groq", "gemini", "deepseek"]
    temperature: float
    max_tool_calls: int
    min_odds_threshold: float
    max_odds_threshold: float
    min_value_edge: float
    markets: list[str]
    system_prompt_version: str
```

- `AgentConfig.from_yaml(path)` — loads and validates against `_REQUIRED` field set; raises `ValueError` listing missing fields.
- `AgentConfig.default()` — loads `config/agent_config.yaml`.

Current default config:

```yaml
model: "llama3.1:8b"
provider: "ollama"
temperature: 0.1
max_tool_calls: 10
min_odds_threshold: 1.2
max_odds_threshold: 11.0
min_value_edge: 0.05
markets: [result_3way, btts, total_goals, home_corners, away_corners]
system_prompt_version: "v1"
```

**Provider/model is a moving target, tracked here rather than hardcoded into prose.** The default shown above (`llama3.1:8b`/`ollama`) is a 2026-07-25 budget-driven swap from `claude-haiku-4-5`/`anthropic` — the comment inline in `config/agent_config.yaml` records both entries and says to revert once Anthropic credits are topped up. `config/agent_config_deepseek.yaml` (A42, Section 21.3) is a separate, opt-in variant (`model: "deepseek-chat"`, `provider: "deepseek"`) selected via `--config` on any CLI command — it does not change the shared default, so `agent-recommend`/the webapp/plain `agent-backtest`/`agent-train` stay on whatever `agent_config.yaml` itself says.

2026-07-31: `scripts/launch_sandbox.py --precompute` and `scripts/scenario_runbook.py` gained the same `--config` opt-in (previously only the standalone `agent-*` CLI commands had it — `precompute_recommendations()`/`run_one_scenario()` both hardcoded `AgentConfig.default()`, so scenario testing could only ever exercise whatever `agent_config.yaml` currently said, i.e. `llama3.1:8b`). Motivated directly by BUG-023/024/027's live sandbox findings: local `llama3.1:8b` fabricated a completely different match/odds/markets on roughly half of a sample batch, whereas A42's own real DeepSeek train-split run showed correct canonical market naming on every inspected match and no comparable hallucination. `scripts/scenario_runbook.py --config config/agent_config_deepseek.yaml ...` now forwards `--config` through to each sampled date's `launch_sandbox.py ... --precompute --config ...` call; omitting it preserves the exact prior default. DeepSeek connectivity re-confirmed live (`_build_llm` + `.invoke()`) before relying on this.

**As of A29 (2026-07-11), `min_odds_threshold`/`max_odds_threshold` are code-enforced**, not just prompt-level instructions: `graph.py`'s `output_node` threads both values from `AgentConfig` into `extract_recommendation()` (Section 8), which downgrades any `direct_bet` market whose `current_odds` falls outside `[min_odds_threshold, max_odds_threshold]` to `conditional` regardless of what the LLM itself wrote. The floor widened from a bare `2.0` (previously prompt-only, with no ceiling at all) to `1.2`, and a `11.0` ceiling was added — see Section 19. `min_value_edge`, however, **remains prompt-only** — there is still no programmatic check that the model's `value_edge` computation actually respected the 0.05 minimum stated in the system prompt (Section 5).

---

## 3. StateGraph (`src/agent/graph.py`)

### 3.1 AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    match_info: dict
    recommendation: dict | None
    tool_call_count: int
```

### 3.2 Node Contracts

```
START → agent_node ──[should_continue]──► tools_node ──► agent_node (loop)
                              │
                              └──► output_node → END
```

**`agent_node`**
Invokes `llm_with_tools.invoke(state["messages"])` via `_invoke_with_retry()` (A64, Section 26.5) — up to 3 attempts, last exception re-raised if all fail. Increments `tool_call_count` by the number of tool calls in the response. Logs `tool_calls=[...]` or `raw_output_length=N` at INFO.

**`should_continue`** (conditional edge)
Routes to `"tools"` if the last message has tool calls AND `tool_call_count < max_tool_calls`; otherwise routes to `"output"`. Logs the routing decision at INFO every call — this was essential for diagnosing model failures (see Section 7) and should not be removed.

**`tools_node`**
A LangGraph `ToolNode(tools)` — standard library node, executes whatever tools were requested and appends `ToolMessage` results.

**`output_node`**
Takes the last message's text content and attempts `extract_recommendation()`.

- If content is empty (happens when `tool_call_count` hit the budget and the last message is a tool call, not text) — makes one additional LLM call with no tools bound, asking the model to synthesize a final answer from the conversation so far. This is the only place a "forced" non-tool LLM call happens.
- If parsing still fails, returns a hardcoded `insufficient_data` fallback embedding the raw text (first 800 chars) in `explanation`, so the caller can always inspect what the model produced even on failure. **The graph never raises on a parse failure** — `run_agent()` always returns a dict.

### 3.3 run_agent()

```python
def run_agent(
    match_info: dict,
    config: AgentConfig | None = None,
    tools: list | None = None,
    extra_system_instructions: str | None = None,
) -> MatchRecommendation
```

- `match_info` keys: `home_team`, `away_team`, `date`, optionally `league`, optionally `odds: {"home": float, "draw": float, "away": float}`.
- Builds the initial `HumanMessage` by interpolating match info directly into a sentence (not a template file). When `odds` is present, the exact decimal values are embedded in the prompt and the model is instructed to reuse them verbatim in forecast tool calls — without this, the model invents placeholder odds.
- Compiles a fresh graph per call (`build_graph()` is not cached/reused across calls).
- `extra_system_instructions` (added for A11): if given, appended to the loaded system prompt with a blank-line separator before building the `SystemMessage`. Used exclusively by `agent-snapshot` (Section 10) to inject a "this match is historical, ignore any web result that leaks the final score" addendum without forking the whole prompt file. `None`/omitted is fully backward compatible — no change to the prompt.

---

## 4. Tools (`src/agent/tools.py`)

All three tools are plain `@tool`-decorated functions, registered via `get_default_tools()`.

### 4.1 `web_search(query: str) -> str`

Wraps `tavily.TavilyClient`. Reads `TAVILY_API_KEY` from the environment — **this requires `load_dotenv()` to have run first** (see Section 6). If the key is missing, returns a fixed string:

```
TOOL_PERMANENTLY_UNAVAILABLE: web_search has no API key configured.
Do NOT call web_search again — it will always return this message.
Output your final JSON recommendation now using only the forecast data already retrieved.
```

This exact wording matters: an earlier, softer message (`"[web_search unavailable: ...]"`) caused `llama3.2:3b` to retry the tool repeatedly (observed up to 8 times) until the tool-call budget was exhausted. The directive phrasing plus the system prompt's STOP RULES (Section 5) reduced this to 1–2 calls.

### 4.2 `forecast_league(home_team, away_team, date, league, odds_h, odds_d, odds_a) -> str`

Calls `ForecastService.forecast_upcoming(match_type="league")`.

**Automatic fallback:** if the league context has no models registered in `config/model_selection.yaml` (BUG-010 — see `documents/bugs.md`), `ForecastService` raises `FileNotFoundError`. `forecast_league` catches this specifically and retries with `match_type="international"`, tagging the result:

```python
result.setdefault("data_quality", {})["prediction_basis"] = "market_odds_only_league_fallback"
```

This fallback exists so the agent never needs to make a second tool-call decision when league models are absent — it gets usable data from a single tool invocation regardless of model tier. Any other exception is caught by an outer `except Exception` and returned as `{"error": ..., "status": "tool_error"}`.

### 4.3 `forecast_international(home_team, away_team, date, odds_h, odds_d, odds_a) -> str`

Calls `ForecastService.forecast_upcoming(match_type="international")` directly. Same outer exception handling as above, no inner fallback (there is nothing to fall back to).

Both forecast tools return the full forecast JSON dict verbatim (`json.dumps(result, default=str)`) — they do not pre-summarize or strip fields before returning to the LLM.

### 4.4 `resolve_competition(competition_or_league: str) -> str` (A27)

Wraps `src.logic.competition_registry.get_competition_definition()` to give the agent a tool-callable way to know whether a league/competition has real historical-data model coverage, instead of relying on the LLM's own judgment about how "well-known" a league sounds.

```python
def _resolve_competition_impl(competition_or_league: str) -> str:
    try:
        tier = get_competition_definition(competition_or_league).tier
    except (ValueError, FileNotFoundError):
        tier = "general_purpose"
    recommended_tool = "forecast_league" if tier == "competition_specific" else "forecast_international"
    return json.dumps({"competition": ..., "tier": tier, "recommended_tool": recommended_tool})
```

Both a `ValueError` (competition not registered) and a `FileNotFoundError` (registry file itself missing) are caught identically and default to `tier="general_purpose"` — the always-safe fallback rather than guessing. Returns a JSON object with `tier` (`"competition_specific"` | `"general_purpose"`) and `recommended_tool` (`"forecast_league"` | `"forecast_international"`); the docstring instructs the model to follow `recommended_tool` exactly rather than deciding domestic-vs-international itself.

Registered first in `get_default_tools()`'s return list (`[resolve_competition, web_search, forecast_league, forecast_international]`) and wrapped by `_snapshot_store.wrap("resolve_competition", ...)` like every other tool (A10 convention, Section 9.1) — its snapshot key is a pure function of `competition_or_league`, stable across record and replay.

See Section 19 for the story behind this tool (US#107 dependency, and an honest caveat about `llama3.1:8b` not always calling it first in practice).

---

## 5. System Prompt (`config/prompts/agent_v1.txt`)

Loaded by `AgentConfig.system_prompt_version` → `agent_{version}.txt`. Structure, in order:

1. **CRITICAL RULE** — never narrate a tool call as text, always either call a tool or output JSON
2. **Workflow** — **(A27, 2026-07-11)** call `resolve_competition` first, with an explicit instruction not to decide domestic-vs-international itself based on how well-known the league sounds; follow its `recommended_tool` exactly to choose between `forecast_league`/`forecast_international` (with explicit fallback odds defaults if none provided), then web_search once, then output
3. **STOP RULES** — never call the same tool twice in a row; output JSON after **3** tool calls total (raised from 2 in A27, to account for the new `resolve_competition` step); once a tool reports `TOOL_PERMANENTLY_UNAVAILABLE`, never call it again
4. **Value Calculation** — implied probability, value edge formula, and **(A29, 2026-07-11)** both a floor and a ceiling on `direct_bet` odds — `[1.2, 11.0]` decimal (roughly −500 to +1000 American), replacing the old prompt-only `2.0`-floor-with-no-ceiling rule. The prompt now also states this is code-enforced at extraction time (Section 8), not just a suggestion, and that a market outside the range should be phrased as `conditional` rather than `direct_bet`.
5. **insufficient_data criteria** — explicit list of when to use this overall value
6. **Confidence Guidelines** — qualitative mapping of data completeness to `high`/`medium`/`low`
7. **Output Format** — the literal `MatchRecommendation` JSON schema as a fenced example

The STOP RULES section (item 3) was added after observing `llama3.2:3b` loop on `web_search` up to 10 times; it is the main lever for keeping weaker models within budget. If a future system prompt version removes it, re-test against a small model before assuming it's safe.

See Section 19 for A27's live-run finding that `llama3.1:8b` doesn't always follow the workflow's step order exactly (in particular, calling `resolve_competition` first).

---

## 6. Environment Setup

```bash
# .env at project root (gitignored)
TAVILY_API_KEY=...
DEEPSEEK_API_KEY=...   # only needed when running with --config config/agent_config_deepseek.yaml (A42)
```

`main.py` calls `load_dotenv()` at import time, before any agent code runs. **This was missing for most of the agent's development** — `TAVILY_API_KEY` was present in `.env` but never reached `os.environ` because nothing loaded the file, so `web_search` silently reported itself unavailable in every test run until this was added. If `web_search` reports unavailable unexpectedly, check that `load_dotenv()` is still called before `from src.agent... import` anywhere in the call path, not just in `main.py`.

Local models, pulled via `ollama pull <model>`:

| Model | Size | Status |
|---|---|---|
| `qwen2.5-coder:7b` | 4.7GB | Rejected — emits tool calls as plain-text JSON instead of structured calls |
| `llama3.2:3b` | 2.0GB | Rejected — see Section 7 |
| `llama3.1:8b` | 4.9GB | **Current default** — reliable enough for end-to-end runs |

---

## 7. Model Selection Findings

This section exists because the choice of local model was the single largest source of agent failures, and the failure modes were non-obvious. Recorded here so the next person tuning `agent_config.yaml` doesn't have to re-discover this.

`llama3.2:3b` (3B parameters) failed in **six distinct ways**, each only visible after fixing the previous one:

| # | Symptom | Root cause |
|---|---|---|
| 1 | Tool calls appear as text content, e.g. `{"name": "forecast_international", ...}` | Model doesn't reliably use structured tool-calling after the first call in a turn |
| 2 | Same error repeated after a tool error | Model writes the retry as text instead of an actual call |
| 3 | `web_search` called up to 8–10 times | Soft "unavailable" message read as a transient, retriable failure |
| 4 | `output_node` received empty string | Budget exhausted while last message was a tool call, not text |
| 5 | JSON rejected with "Extra data" | Model appended a stray trailing `}` |
| 6 | Final answer was the *forecast tool's* JSON schema, not `MatchRecommendation` | Model echoed the tool result instead of transforming it into the target schema — this requires arithmetic (value_edge) and cross-schema mapping that a 3B model could not do reliably |

Failures 1–5 were fixed with prompt and parsing changes (instrumented via INFO-level logging added to `agent_node`/`should_continue`/`output_node` — keep this logging; it was the only way to diagnose which node was misbehaving without LangSmith tracing configured). Failure 6 was not fixable by further prompt engineering — it required switching to `llama3.1:8b`, which performs the schema mapping and value-edge arithmetic correctly on the first attempt in testing.

Even with `llama3.1:8b`, the model occasionally produces a `markets` array with broken bracket nesting (splitting one array into multiple sibling array literals — `}], [` instead of `}, {`). This is handled by a `json_repair` fallback in `extract_recommendation` (Section 8) rather than further prompt tuning, since it is a known class of weak-model JSON error rather than a one-off typo.

**Practical takeaway:** if `system_prompt_version` or `model` changes in the future, re-run `agent-recommend` against the diagnostic logging in `graph.py` before assuming a fix worked — the JSON may *look* plausible while still failing schema validation.

---

## 8. Output Schema Parsing (`src/agent/schema.py`)

```python
class MarketRecommendation(TypedDict):
    market: str
    selection: str
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float
    min_odds: float
    ml_probability: float
    implied_probability: float
    value_edge: float

class MatchRecommendation(TypedDict):
    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    markets: list[MarketRecommendation]
    explanation: str
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str
```

```python
def extract_recommendation(
    text: str, min_odds_threshold: float = 1.2, max_odds_threshold: float = 11.0
) -> MatchRecommendation
```

1. Collects all fenced ` ```json ` blocks, tried **last-to-first** (the model sometimes echoes an earlier tool result inside a fenced block before its real final answer — the last block is the one that matters).
2. Falls back to the outermost bare `{...}` via regex if no fenced block parses.
3. For each candidate, parses with `json.JSONDecoder().raw_decode()` (not `json.loads`) — tolerates trailing characters after the JSON object closes.
4. If `raw_decode` still raises, falls back to `json_repair.loads()`, which tolerates structurally broken JSON (mismatched/duplicated brackets, etc.). Only accepted if the result is a `dict`.
5. Validates all 7 required keys are present and `overall` is one of the four valid literals (unchanged since before A28 — same error wording, so no pre-existing test needed to change).
6. **(A28, 2026-07-11) Structural/type validation via internal Pydantic v2 models.** `_MatchRecommendationModel`/`_MarketRecommendationModel` (module-private, not part of the public return type) are validated against the candidate dict *after* step 5 passes. This checks every market field's type (`current_odds`, `min_odds`, `ml_probability`, `implied_probability`, `value_edge` as `float`; `current_odds` nullable) and both enum fields (`recommendation_type` inside each market, top-level `confidence`) against their valid literals. A `pydantic.ValidationError` here is caught and folds into the same "try the next candidate, then raise `RecommendationParseError`" flow as steps 3–5 — the exception message is Pydantic's own field-path-qualified text (e.g. it names `value_edge` or `markets.0.confidence` directly), which is more diagnosable than the `TypeError` that used to surface later in `staking.py`. **This closes the gap this section previously documented** — a model writing `confidence: ""` or `value_edge: "high"` (a string) now fails extraction instead of silently passing through.

   `extract_recommendation` still returns a plain `dict`, not a Pydantic instance — the Pydantic models are used purely as a validation pass; no downstream caller (`graph.py`, `backtest.py`, `staking.py`) changed its dict-style access.

7. **(A28) Null-odds downgrade — `_downgrade_direct_bet_with_null_odds()`.** Applied after step 6 passes. This is the fix for **BUG-013** (`documents/bugs.md`, status `fixed`): a market with `recommendation_type == "direct_bet"` and `current_odds is None` is downgraded to `recommendation_type = "no_bet"` (the only other value valid for that field), with an explanatory note appended to top-level `limitations`. A `conditional`/`no_bet` market with null odds is a legitimate state and is left untouched.
8. **(A29, 2026-07-11) Odds-bounds downgrade — `_downgrade_direct_bet_outside_odds_bounds()`.** Applied last, after step 7. A market still marked `direct_bet` whose (non-null) `current_odds` falls outside `[min_odds_threshold, max_odds_threshold]` (inclusive bounds; defaults `1.2`/`11.0`, matching `config/agent_config.yaml`) is downgraded to `recommendation_type = "conditional"` — **not** `"no_bet"`, unlike step 7's null-odds case. The distinction is deliberate: here a real price exists, just outside the accepted range, matching the pre-existing prompt convention that a market with value but an unfavorable price is a "conditional" opportunity rather than a non-bet; step 7's null-odds case has no price to act on at all, so `"no_bet"` is the only coherent downgrade target. A market with `current_odds` already `None` is skipped by this pass (step 7 already handled it). `graph.py`'s `output_node` passes `config.min_odds_threshold`/`config.max_odds_threshold` explicitly rather than relying on the function's defaults, so a non-default `AgentConfig` is actually respected end-to-end (Section 2).
9. Raises `RecommendationParseError(raw_text, reason)` if no candidate validates through steps 3–6, carrying the original text for debugging.

`json_repair` is a hard dependency (`requirements.txt`), not optional — without it, the multi-array bracket-nesting failure (Section 7, #6 follow-on) causes silent `insufficient_data` fallbacks even when the model's numbers were correct.

**Residual gap, honestly noted:** the Pydantic validation in step 6 covers every *field on `MatchRecommendation`/`MarketRecommendation` itself*, but does not validate `match` (typed as a plain `dict`, no nested schema) or the contents of `limitations` (typed as `list[str]`, but a non-string element inside the list would still fail — this is enforced by Pydantic, not a gap) or cross-field semantic invariants beyond the two rules in steps 7–8 (e.g. nothing checks that `implied_probability` is actually `1 / current_odds`, or that `value_edge` is actually `ml_probability - implied_probability`). Those remain the LLM's responsibility, unchecked at extraction time.

### 8a. Schema-Constrained Structured Output for Ollama (`src/agent/graph.py`, A37, 2026-07-26)

Everything in this section up to here is a *validate-after-the-fact* pass: the LLM writes free text, and `extract_recommendation` regex-extracts + validates it. That works reliably for Anthropic (`claude-haiku-4-5`) but not for local Ollama models — live testing during a budget-motivated switch to `provider: "ollama"` (`config/agent_config.yaml`, scenario-testing use) showed `llama3.1:8b` emitting syntactically valid JSON that didn't match the schema at all (e.g. `"overall": {"odds": {...}}` instead of the enum string, `"confidence"` as a nested object, invented market field names) — an instruction-following gap, not a JSON-syntax defect, so `json_repair` (step 4 above) can't help.

**Fix: a dedicated, schema-constrained final-answer call, gated to `provider == "ollama"` only.**

- `MatchRecommendationModel`/`MarketRecommendationModel` in `src/agent/schema.py` were renamed from `_MatchRecommendationModel`/`_MarketRecommendationModel` (public now — used cross-module, not just internally by `extract_recommendation`).
- `_structured_output(llm, messages)` (`src/agent/graph.py`) calls `llm.with_structured_output(MatchRecommendationModel).invoke(messages)` — LangChain's provider-agnostic structured-output interface (`langchain-ollama>=1.1.0` supports it via Ollama's native JSON-schema-constrained decoding). Returns the validated `.model_dump()` dict on success, or `None` on any exception *or* an unexpected return shape (not an instance of `MatchRecommendationModel` — defensive, covers a provider/binding that doesn't raise but also doesn't cooperate).
- `output_node` (`build_graph`): when `config.provider == "ollama"`, tries `_structured_output` first, using the full `state["messages"]` conversation (system prompt + match context + tool-call history) as context — no extra instruction needed. On success, skips `extract_recommendation`/regex entirely. On `None`, falls through unchanged to the pre-existing free-text path (`last.content` → forced-synthesis call if empty → `_build_recommendation`).
- Both paths converge on a new shared `_finalize_recommendation()` (extracted from the old `_build_recommendation` body) so the A30/A31/A32 diagnostics/backstop/downgrade normalization applies identically regardless of which path produced the recommendation.
- **Deliberately not applied to other providers.** Anthropic/Groq/Gemini already produce reliable free-text JSON; unconditionally adding a second LLM call would silently double per-request cost on paid providers, directly conflicting with the budget motivation that drove this story. `test_run_agent_never_attempts_structured_output_on_non_ollama_providers` (`tests/test_agent_graph.py`) locks this in.
- **What this does and doesn't fix:** guarantees *shape* — correct field names/types/enum values, every time it succeeds — not semantic quality (a structurally valid but reasoning-poor answer is still possible). Verified live against real `llama3.1:8b` (not just mocked): a direct `_structured_output()` call and a full `POST /api/recommendations` round-trip through the real sandbox both returned fully schema-conformant recommendations on the first try, where the pre-fix free-text path had failed to parse on both of two consecutive live attempts for the same match.

TDD: `tests/test_agent_graph.py` — 6 new tests (`_structured_output` success/exception/wrong-shape, plus `run_agent()`-level: uses structured output on `ollama`, falls back to free text when structured output fails, never attempts it on non-`ollama` providers). Full suite: 806 passed / 1 skipped, zero regressions.

---

## 9. Snapshot Infrastructure (`src/agent/snapshot_store.py`, A09–A10)

Backtesting requires the agent to run against historical matches without ever calling a live API with knowledge of the future. `SnapshotStore` solves this with three modes:

```python
SnapshotMode = Literal["live", "record", "replay"]

class SnapshotStore:
    def __init__(self, base_dir: str | Path = "data/agent_snapshots") -> None: ...
    def set_mode(self, mode: SnapshotMode) -> None: ...
    def set_match(self, match_id: str, match_date: str | None = None) -> None: ...
    def wrap(self, tool: str, fn: Callable[..., str]) -> Callable[..., str]: ...
```

- **`live`** (default): `wrap()` calls `fn(**kwargs)` directly, no interception. This is the mode `agent-recommend` runs in.
- **`record`**: calls `fn(**kwargs)`, then writes `{tool, inputs, response, recorded_at}` to `data/agent_snapshots/<match_id>/<tool>_<key>.json`, where `key = SnapshotStore.key_for(kwargs)` is a SHA-256 hex digest of the canonical (`sort_keys=True`) JSON of the tool's input kwargs.
- **`replay`**: never calls `fn`. Reads the same path; if missing, raises `SnapshotMissingError(tool, match_id, key)` immediately — there is no silent fallback to a live call. The error message tells the operator to run `agent-snapshot` in record mode for that match.

**Thread-local state.** `mode`, `match_id`, and `match_date` are stored via `threading.local()`, not plain instance attributes — even though the store is a single module-level singleton (`src/agent/tools.py`'s `_snapshot_store = SnapshotStore()`, shared by all three tools). This was deliberately built in A09, before A14 existed, specifically so `agent-backtest --concurrency` could later run many matches' replay contexts on different OS threads (via `asyncio.to_thread`) without one match's `mode`/`match_id` clobbering another's. **Do not relax this to plain instance attributes** without re-verifying A14's concurrency story — see Section 13.

### 9.1 Tool Integration (A10)

All three tools in `src/agent/tools.py` route their real implementation through `_snapshot_store.wrap(tool_name, _impl_fn)`:

```python
@tool
def web_search(query: str) -> str:
    effective_query = query
    if _snapshot_store.mode in ("record", "replay") and _snapshot_store.match_date:
        effective_query = f"{query} before:{_snapshot_store.match_date}"
    return _snapshot_store.wrap("web_search", _web_search_impl)(query=effective_query)
```

`web_search` additionally appends `before:<match_date>` to the query during record/replay (never during live) to reduce the chance Tavily returns a result that leaks the actual final score. This filtering is applied **identically** in both record and replay mode — critical, because the SHA-256 key is computed from the (already-filtered) `effective_query`, so record and replay must compute the same `effective_query` for a given match/query or replay will always raise `SnapshotMissingError`.

`forecast_league` and `forecast_international` have no mode-dependent input transformation — their snapshot keys are a pure function of the forecast inputs (`home_team`, `away_team`, `date`, `league`, `odds_h/d/a`), stable across record and replay.

Module-level configuration helper (`src/agent/tools.py`):

```python
def configure_snapshot_store(mode: SnapshotMode, match_id: str | None = None, match_date: str | None = None) -> None
```

Callers (the `agent-snapshot`/`agent-backtest`/`agent-compare` CLI paths and `process_match_row`) call this before `run_agent()` to switch modes, and reset to `"live"` in a `finally` block afterward so a failure on one match never leaves a later, unrelated call running in `record`/`replay` mode by accident.

---

## 10. agent-snapshot CLI (A11)

```bash
python main.py agent-snapshot \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    [--league E0] [--config config/agent_config.yaml] [--dry-run]
```

`run_agent_snapshot()` (`main.py`):

1. Queries `raw_matches` for `match_id, league, date, home_team, away_team, odds_h, odds_d, odds_a` in the given date range (optionally filtered by `UPPER(league) = ?`), ordered by date.
2. Skips any match whose `data/agent_snapshots/<match_id>/_complete.json` marker already exists (resumable across interrupted runs) — printed as "already complete", not reprocessed.
3. `--dry-run` prints the count and fixture list for matches that *would* be processed, then returns without touching disk or calling the agent.
4. Otherwise, for each remaining match: builds `match_info` (with an `odds` dict only if all three of `odds_h/d/a` are truthy), calls `configure_snapshot_store("record", match_id=..., match_date=...)`, then `run_agent(match_info=..., config=cfg, extra_system_instructions=<snapshot addendum>)`. On success, writes the `_complete.json` marker. On any exception, prints an error to stderr and continues to the next match (does not abort the run) — mirrored later by A14's concurrent path (Section 13). The snapshot store is always reset to `"live"` in a `finally` block regardless of outcome.
5. Prints a final summary: `Processed: N | Errors: N | Skipped: N`.

The snapshot addendum text (passed via `run_agent`'s `extra_system_instructions`, Section 3.3):

```
## SNAPSHOT COLLECTION MODE

You are collecting training data from a historical match. Discard and ignore any
web_search result that mentions a final score, match result, or post-match analysis —
treat this match as still upcoming.
```

This is a second line of defense against outcome leakage on top of `web_search`'s `before:<date>` query filter (Section 9.1) — the filter reduces *what Tavily returns*, this instructs the model on *what to do if a leaked result slips through anyway*.

---

## 11. Backtest Harness (`src/agent/backtest.py`, A12)

### 11.1 Outcome Loading and Market Scoring

```python
def load_outcome(row: pd.Series) -> dict[str, Any]
```

Derives, from a finished match's `fthg`/`ftag` (full-time goals): `result` (`"home"`/`"draw"`/`"away"`), `btts` (`"yes"`/`"no"`), `total_goals`, `total_goals_side` (`"over_2.5"`/`"under_2.5"`).

```python
def _market_correct(market_rec: dict[str, Any], actual: dict[str, Any]) -> bool | None
```

Resolves whether one market entry in a `MatchRecommendation.markets` list matches the actual outcome:

| Market | Resolvable? | Logic |
|---|---|---|
| `result_3way` | Yes | `selection == actual["result"]` |
| `btts` | Yes | `selection == actual["btts"]` |
| `total_goals` | Yes | `selection == actual["total_goals_side"]` |
| `home_corners` / `away_corners` | **No — returns `None`** | `MatchRecommendation` has no numeric line field for corners (only `current_odds`/`min_odds`), so there is no way to know what threshold the agent's `selection` (e.g. `"over_4.5"`) actually refers to without that line value. |

**This `None`-vs-`False` distinction is the single most important contract in the backtest stack.** `None` means "unknown, cannot be scored" — not "wrong." Every downstream consumer (`staking.py`, `evaluation.py`) treats a market with `correct is None` as **skip entirely** (no bet recorded, no win/loss counted), never as a settled loss. This is a deliberate, permanent limitation, not a bug to be fixed by adding more `elif` branches — fixing it would require extending `MarketRecommendation` with a numeric line field and is out of scope for A12–A16.

### 11.2 process_match_row — the Single Shared Replay Path

```python
def process_match_row(row: pd.Series, config: AgentConfig) -> BacktestRecord
```

This is the **one and only** implementation of "replay one historical match through the agent and score it." It is called directly by `BacktestHarness.run()` (synchronous, used by `agent-backtest` without concurrency and by `agent-compare`) and via `asyncio.to_thread` by `_run_backtest_concurrent()` (`main.py`, A14, Section 13) — by design, there is no second copy of this logic, so the sync and concurrent paths can never drift apart.

Sequence: `configure_snapshot_store("replay", match_id=row["match_id"])` → `run_agent(match_info=..., config=config)` → (always, via `finally`) `configure_snapshot_store("live")` → `load_outcome(row)` → tag every entry in `recommendation["markets"]` with `correct: _market_correct(m, actual)` → return a `BacktestRecord`.

```python
@dataclass
class BacktestRecord:
    match_id: str
    home_team: str
    away_team: str
    date: str
    league: str
    recommendation: dict[str, Any]      # the raw MatchRecommendation dict
    actual: dict[str, Any]              # load_outcome() result
    market_results: list[dict[str, Any]]  # each market dict + "correct": bool | None
```

If `run_agent()` raises `SnapshotMissingError` (the match was never recorded, or the agent issued a tool call with inputs that don't match anything recorded — e.g. a different `web_search` query string than what was captured), the exception propagates out of `process_match_row` uncaught. Callers decide what to do with it (Section 13: the concurrent CLI path catches and skips; `BacktestHarness.run()` and `compare_configs()` do not, and will abort on the first missing snapshot).

### 11.3 BacktestHarness

```python
class BacktestHarness:
    def __init__(self, config: AgentConfig | None = None, db_path: str = "config.yaml") -> None
    def load_matches(self, from_date: str, to_date: str, league: str | None = None, sample: int | None = None) -> pd.DataFrame
    def run(self, from_date: str, to_date: str, league: str | None = None, sample: int | None = None) -> list[BacktestRecord]
```

`load_matches` queries `raw_matches` for `match_id, league, date, home_team, away_team, odds_h, odds_d, odds_a, fthg, ftag, hc, ac` where `fthg IS NOT NULL AND ftag IS NOT NULL` (only finished matches are backtestable) within the date range, optionally filtered by league, ordered by date. If `sample` is given and the result set exceeds it, applies `_stratified_sample`.

**Stratified sampling** (`_stratified_sample`, static method): stratifies by **actual result** (home/draw/away) — the only outcome dimension known *before* running the agent. (Bet/no-bet is the agent's own output and therefore cannot be used to pre-stratify the input sample — that would be looking at the answer before asking the question.) Seeded with `random_state=42` so re-running the same `from_date`/`to_date`/`league`/`sample` always produces the identical set of matches — this is what makes A16's config comparison meaningful (Section 14): different configs are compared over the literal same matches, isolating the config as the only varying factor.

`run()` simply does `[process_match_row(row, self.config) for _, row in matches.iterrows()]` — strictly sequential, one match at a time. There is no concurrency at this layer; concurrency is added one level up, in the CLI (Section 13).

---

## 12. Bankroll Simulation and Evaluation (`src/agent/staking.py`, `src/agent/evaluation.py`, A13 + A15)

### 12.1 Staking Modes

```python
@dataclass
class BetOutcome:
    match_id: str; market: str; selection: str
    odds: float; stake: float; won: bool
    payout: float  # net profit (+) or loss (-), already signed

@dataclass
class BankrollResult:
    starting_bankroll: float
    ending_bankroll: float
    equity_curve: list[float]   # starts with starting_bankroll, one entry appended per settled bet
    bets: list[BetOutcome]
```

Both staking functions iterate every `BacktestRecord`'s `market_results` and, for each market entry, skip it (no bet recorded) unless **both**: `recommendation_type == "direct_bet"` and `correct is not None`. This is where the `None`-vs-`False` contract from Section 11.1 is actually enforced — corners markets are never settled as wins or losses, they simply never become a bet.

```python
def simulate_flat_stake(records, starting_bankroll: float = 1000.0, stake_pct: float = 0.01) -> BankrollResult
```
Every qualifying bet stakes a fixed `flat_stake = starting_bankroll * stake_pct` (computed once, not re-derived per bet — so it does NOT compound, unlike Kelly). Win: `bankroll += flat_stake * (odds - 1)`. Loss: `bankroll -= flat_stake`.

```python
def simulate_kelly_stake(records, starting_bankroll: float = 1000.0, max_fraction: float = 0.10) -> BankrollResult
```
Per bet: `fraction = min(value_edge / (odds - 1), max_fraction)`, `stake = bankroll * fraction` — using the **current, running** bankroll, not the starting one, so Kelly stakes compound across the backtest. Skips any bet with `odds <= 1.0` or `value_edge <= 0` (a non-positive Kelly fraction means "don't bet," not "bet zero").

### 12.2 Evaluation Report

```python
def compute_max_drawdown(equity_curve: list[float]) -> float
```
Single-pass peak-tracking: the largest `(peak - value) / peak` observed at any point after a new peak, i.e. the worst peak-to-trough fractional decline in the bankroll's history.

```python
def build_evaluation_report(records: list[BacktestRecord], bankroll_result: BankrollResult) -> dict[str, Any]
```

| Field | Formula |
|---|---|
| `matches_evaluated` | `len(records)` |
| `bets_placed` | `len(bankroll_result.bets)` |
| `bets_won` | count where `bet.won` |
| `roi` | `total_profit / total_staked` (0.0 if nothing staked) |
| `hit_rate` | `bets_won / bets_placed` (0.0 if no bets) |
| `bet_frequency` | `bets_placed / matches_evaluated` |
| `max_drawdown` | `compute_max_drawdown(equity_curve)` |
| `insufficient_data_rate` | fraction of records whose `recommendation["overall"] == "insufficient_data"` |
| `starting_bankroll` / `ending_bankroll` | passed through / rounded to 2dp |

```python
def config_hash(config: AgentConfig) -> str   # 8-char sha256 prefix, sorted(markets) — order-independent
def save_report(report, config, base_dir="reports/agent_backtest") -> Path   # writes {timestamp}_{config_hash}.json
def print_report(report) -> None   # formatted stdout table
```

`config_hash` is computed over exactly `AgentConfig`'s tuning-relevant fields (`model`, `provider`, `temperature`, `max_tool_calls`, `min_odds_threshold`, `min_value_edge`, `markets` sorted, `system_prompt_version`). **Stale claim corrected (found during A27–A29 documentation pass, 2026-07-18):** this section previously stated that hash covers "every field on the dataclass" — that was true when written, but `AgentConfig` gained `max_odds_threshold` in A29 (Section 2) and `config_hash`'s field list in `src/agent/evaluation.py` was **not** updated to include it. Two configs differing only in `max_odds_threshold` therefore currently hash identically, and `save_report`'s `{timestamp}_{config_hash}.json` filename would not distinguish them. This is a real, currently-open gap, not a documentation error to just silently fix here — worth a small follow-up story to add `max_odds_threshold` to `config_hash`'s canonical field dict. `markets` order never matters regardless. `save_report` filenames also collide if two runs of the *same config* complete within the same UTC second — accepted as a non-issue for this tool's manual, human-paced usage pattern (a single backtest run takes at minimum seconds to minutes).

---

## 13. agent-backtest CLI (A14)

```bash
python main.py agent-backtest \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    [--league E0] [--stake-mode flat|kelly] [--sample 50] \
    [--concurrency 5] [--config config/agent_config.yaml]
```

`run_agent_backtest()` (`main.py`): loads config, builds a `BacktestHarness`, calls `load_matches()` (applying `--sample`'s stratified sampling if given), then drives `_run_backtest_concurrent()` via `asyncio.run()`, applies the chosen staking function over the resulting records, builds and prints the evaluation report, and saves it to `reports/agent_backtest/`.

```python
async def _run_backtest_concurrent(matches, config, concurrency: int) -> list[BacktestRecord]
```

- Bounds concurrency with `asyncio.Semaphore(concurrency)` (default 5; raises `ValueError` up front in `run_agent_backtest` if `concurrency < 1`, since `Semaphore(0)` would otherwise hang forever with no diagnostic).
- Each match runs `process_match_row` inside `asyncio.to_thread(...)`, since the agent graph and tool functions are entirely synchronous code — `to_thread` is what actually achieves parallelism here, the `async`/`await` machinery around it exists purely to let `asyncio.gather` orchestrate many of these thread-dispatches at once under one semaphore.
- **Per-match fault tolerance**: each `_run_one(row)` wraps its `process_match_row` call in `try/except Exception`, printing `SKIP {match_id}: {exc}` to stderr and returning `None` on failure rather than letting the exception propagate. This was added during code review — the original implementation let `asyncio.gather`'s default `return_exceptions=False` behavior abort the *entire* batch (discarding all other in-flight/completed work) on the first `SnapshotMissingError` from a single unrecorded match. The fix mirrors `agent-snapshot`'s (Section 10) existing per-match error tolerance, so a 500-match backtest with one bad match now returns 499 results plus a printed skip count instead of zero results plus a stack trace.
- A `tqdm` progress bar advances on every match (success or skip) via a `finally` inside `_run_one`, and is always closed via an outer `finally` even if something unexpected escapes `asyncio.gather`.
- `asyncio.gather` preserves input order in its returned list, so downstream chronological assumptions (if any) hold.

**Thread-safety**: this is the load-bearing consumer of A09's `threading.local()` design (Section 9). Because `asyncio.to_thread` uses Python's default `ThreadPoolExecutor` (which reuses worker threads across tasks), a thread that previously replayed match A may later be handed match B. This is safe because `process_match_row` unconditionally calls `configure_snapshot_store("replay", match_id=B...)` at the very start of every invocation, overwriting whatever stale `mode`/`match_id` that recycled thread was carrying — there is no window in which a stale value could be read, because nothing touches the snapshot store between thread pickup and that overwrite.

**Known scope limit**: effective parallelism is also bounded by Python's default thread-pool executor size (`min(32, os.cpu_count() + 4)`), not just `--concurrency`. Setting `--concurrency` above that has no further effect — extra tasks queue behind the executor rather than running. Not a bug, just a ceiling worth knowing about when tuning.

---

## 14. Config Comparison Framework (`src/agent/comparison.py`, A16)

```bash
python main.py agent-compare \
    --configs config/a.yaml config/b.yaml [config/c.yaml ...] \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    [--league E0] [--sample 50] [--stake-mode flat|kelly]
```

```python
def compare_configs(config_paths: list[str], from_date: str, to_date: str, league: str | None = None, sample: int | None = None, stake_mode: str = "flat") -> dict[str, dict[str, Any]]
```

For each config path: loads `AgentConfig.from_yaml(path)`, builds a **fresh** `BacktestHarness(config=cfg)`, calls `.run(from_date, to_date, league=league, sample=sample)` with the **exact same arguments for every config** (no per-config branching — the date range/league/sample are loop-invariant), applies the chosen staking function, and builds an evaluation report. Returns `{config_path: report}`.

The comparison is only meaningful because `BacktestHarness._stratified_sample`'s `random_state=42` (Section 11.3) guarantees every config gets the identical match sample for identical `from_date`/`to_date`/`league`/`sample` inputs — the agent's own tuning knobs (`AgentConfig`) play no role in which matches are selected, only in how the agent reasons about them.

`print_comparison_table(results)` prints a fixed-width table over `roi`, `hit_rate`, `bet_frequency`, `max_drawdown`, `insufficient_data_rate`. `save_comparison(results, base_dir="reports/agent_backtest")` writes `comparison_{timestamp}.json`.

**Scope note**: unlike `agent-backtest` (Section 13), `compare_configs` runs each config's matches strictly sequentially via `BacktestHarness.run()` — it does not reuse the concurrent, fault-tolerant `_run_backtest_concurrent()` helper (which lives in `main.py`, coupled to the CLI's `tqdm`/stderr conventions, not a clean library import). This means `agent-compare` is slower than `agent-backtest` for the same match count, and a single missing snapshot aborts the whole comparison rather than being skipped. This is a known, accepted asymmetry — not a defect — but worth revisiting if config comparison is ever run over large samples regularly.

---

## 15. CLI Reference

```bash
python main.py agent-recommend \
    --home "Manchester City" --away "Arsenal" --date 2026-06-21 \
    --league E0 \
    --odds-h 1.95 --odds-d 3.6 --odds-a 4.2 \
    [--config config/agent_config.yaml]
```

| Flag | Required | Notes |
|---|---|---|
| `--home`, `--away`, `--date` | Yes | |
| `--league` | No | Omit for international fixtures — agent uses `forecast_international` |
| `--odds-h`, `--odds-d`, `--odds-a` | No | All three must be supplied together (`run_agent_recommend` only attaches `match_info["odds"]` if none are `None`). Without odds, the agent cannot compute `implied_probability`/`value_edge` and will correctly return `no_bet`/`insufficient_data` — this is expected behavior, not a bug. |
| `--config` | No | Defaults to `config/agent_config.yaml` |

Output: prints `=== Explanation ===` (the `explanation` field, popped from the dict) followed by `=== Recommendation ===` (remaining fields as pretty JSON). Non-zero exit on `RecommendationParseError` (prints raw output to stderr) or `recommendation is None`.

```bash
python main.py agent-snapshot \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    [--league E0] [--config config/agent_config.yaml] [--dry-run]
```

| Flag | Required | Notes |
|---|---|---|
| `--from-date`, `--to-date` | Yes | Inclusive date range over `raw_matches.date` |
| `--league` | No | Omit to collect snapshots across all leagues |
| `--config` | No | Defaults to `config/agent_config.yaml` |
| `--dry-run` | No | Lists matches that would be processed, writes nothing to disk |

See Section 10 for full behavior (resumability, error handling, snapshot addendum).

```bash
python main.py agent-backtest \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    [--league E0] [--stake-mode flat|kelly] [--sample 50] \
    [--concurrency 5] [--config config/agent_config.yaml]
```

| Flag | Required | Notes |
|---|---|---|
| `--from-date`, `--to-date` | Yes | Inclusive date range |
| `--league` | No | Omit for all leagues |
| `--stake-mode` | No | `flat` (default) or `kelly` — see Section 12.1 |
| `--sample` | No | Stratified sample size; omit to run the full matched set |
| `--concurrency` | No | Max concurrent agent replays, default 5; must be `>= 1` |
| `--config` | No | Defaults to `config/agent_config.yaml` |

Prints a progress bar, then the evaluation report (Section 12.2), then saves it to `reports/agent_backtest/{timestamp}_{config_hash}.json`. Requires snapshots already collected via `agent-snapshot` for every match in range — see Section 13.

```bash
python main.py agent-compare \
    --configs config/a.yaml config/b.yaml [...] \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    [--league E0] [--sample 50] [--stake-mode flat|kelly]
```

| Flag | Required | Notes |
|---|---|---|
| `--configs` | Yes | Two or more paths to `agent_config.yaml`-shaped files (`nargs="+"`) |
| `--from-date`, `--to-date` | Yes | Inclusive date range — identical for every config compared |
| `--league` | No | Omit for all leagues |
| `--sample` | No | Stratified sample size — identical sample used for every config (Section 14) |
| `--stake-mode` | No | `flat` (default) or `kelly` |

Prints a comparison table (one row per config, columns `roi`/`hit_rate`/`bet_frequency`/`max_drawdown`/`insufficient_data_rate`) and saves `reports/agent_backtest/comparison_{timestamp}.json`. Like `agent-backtest`, requires snapshots already collected for every match in range.

---

## 16. Implementation Status

Reflects `documents/agent_user_stories.md` as of this writing.

| Phase | Stories | Status |
|---|---|---|
| 1 — Foundation | A01–A03 | ✅ Implemented |
| 2 — Live Recommendation | A04–A08 | ✅ Implemented |
| 3 — Snapshot Infrastructure | A09–A11 | ✅ Implemented — `SnapshotStore`, tool integration, `agent-snapshot` CLI (Sections 9–10) |
| 4 — Backtest Harness | A12–A14 | ✅ Implemented — `BacktestHarness`, evaluation report, `agent-backtest` CLI with concurrency (Sections 11–13) |
| 5 — Model & Prompt Tuning | A15–A16 | ✅ Implemented — Kelly staking, config comparison framework (Sections 12, 14) |
| 6 — Batch Recommendation | A18 | ⬜ Future |
| 7 — Backtest Execution Readiness | A19–A22 | A19–A21 ✅ Implemented — resolved BUG-010, collected the 24-match E0 pilot corpus, ran the first genuine backtest baseline (Section 18). A22 (config comparison against the baseline) ⬜ Future. |
| 8 — Full Season Backtest Expansion | A23–A26 | ⬜ Future — temperature=0 determinism fix, full-season data refresh, full-corpus snapshot collection, and full-season backtest report, in that dependency order |
| 9 — League-Aware Model Routing | A27 | ✅ Implemented — `resolve_competition` tool, updated tool-selection prompt and stop-rule budget (Sections 4.4, 5, 19.1) |
| 10 — Output Validation Hardening | A28–A29 | ✅ Implemented — Pydantic-backed field validation, BUG-013 null-odds downgrade, code-enforced odds bounds (Sections 2, 8, 19.2–19.3) |
| 11 — Deterministic Evidence Pipeline & Critic Mode | A30–A34, A39–A47 | A30–A32 ✅ Implemented — deterministic `resolve_competition`/`research`/`forecast` graph nodes replacing LLM tool-choice (Section 19 cross-references pending a future write-up). A33 ✅ Implemented — `agent-train`/`agent-lessons` CLIs, competition/tier-scoped live-mode lesson injection (Section 20). A34 (rebaseline of the pilot corpus/baseline against the Phase-11 graph shape) ⬜ Future — still open, not blocking anything downstream. A39–A45 ✅ Implemented — batched lesson candidates, stable train/test corpus split, lesson-driven held-out backtests, DeepSeek provider + a replay-mode `tools=[]` fix, LLM-synthesized batch reflections, distilled `rule_text` approval, pairwise conflict detection (Section 21). A46–A47 ✅ Implemented — confirmed web-search leakage into backtest/train replay, a prompt-level guard, then a code-enforced result-redaction filter plus deletion/re-recording of the E0 corpus's confirmed-leaky matches (Section 24). |
| 12 — Swedish League Verification | A35–A36 | A35 ✅ Implemented — regression coverage proving the deterministic pipeline/`resolve_competition` route a second real `competition_specific` competition (SWE) correctly, and that an unavailable target (corners) is cleanly omitted rather than guessed (Section 22). A36 (pilot SWE snapshot corpus + baseline backtest) ⬜ Blocked — `agent-snapshot --league SWE` produced a genuinely incomplete corpus (BUG-018, `documents/bugs.md`, open) under contended local-Ollama load; `agent-backtest` was correctly never run against it (Section 22.2). |
| 13 — Structured Output Reliability | A37–A38 | A37 ✅ Implemented — schema-constrained `with_structured_output()` final-answer call for `provider == "ollama"` (Section 8a). A38 (grounding the LLM's free-text match/market identity fields against `match_info` — A37 guarantees JSON shape, not that the content is about the requested match) ⬜ Future — a live-verified real gap, not yet fixed (Section 22.3). |

**A17 dependency resolved.** A17 (this document) formally depends on A08 *and* A14; both are now implemented, so this document covers the full backtesting/evaluation/comparison surface (Sections 9–14) it previously only described as design intent. `agent_prd.md` §7's CLI command examples (`agent-snapshot`, `agent-backtest`) match the flags actually implemented; §8 does not yet show the `agent-compare` example added in A16 — a documentation gap in `agent_prd.md`, not in this document or the code.

**LangSmith tracing:** referenced as an acceptance criterion in A01, but no `LANGCHAIN_API_KEY`/`LANGCHAIN_TRACING_V2` configuration exists in `.env` or anywhere in the codebase as of this writing. Diagnostic visibility currently comes entirely from the `_LOG.info()` calls in `graph.py` (Section 3), not from LangSmith. If LangSmith is set up later, the manual logging in `graph.py` should probably stay regardless — it proved more useful for diagnosing the model-selection issues in Section 7 than a trace UI would have been, since it could be grepped directly from CLI output.

---

## 17. Known Limitations

- ~~`min_odds_threshold` / `min_value_edge` are prompt-level instructions, not code-enforced gates.~~ **Partially resolved 2026-07-11 (A29).** `min_odds_threshold`/`max_odds_threshold` are now code-enforced — `extract_recommendation` downgrades an out-of-bounds `direct_bet` to `conditional` regardless of what the LLM wrote (Section 8, Section 19.3). `min_value_edge`, however, **remains prompt-only** — there is still no programmatic check that a recommendation's `value_edge` actually clears the configured 0.05 minimum.
- ~~BUG-010 (league-context models not trained)~~ — **fixed 2026-06-27** (A19). `forecast_league` now returns `data_quality.prediction_basis == "team_history_and_market"` for E0 matches. The `forecast_league` fallback (Section 4.2) still exists and still fires for leagues without a selected league-context model.
- No automated test exercises a live Ollama call — `tests/test_agent_graph.py` mocks the LLM and `ForecastService`. There is no CI coverage for "does the currently-configured model actually produce valid output," only for the graph routing logic and parsing functions in isolation. The same is true of the backtest stack: no test runs a real `agent-snapshot` → `agent-backtest` cycle end-to-end against a live model, only mocked unit tests per module. **This exact gap is what let BUG-011 (Section 18.4) ship undetected** — every `SnapshotStore` unit test calls `configure_snapshot_store()` and the tool function on the same thread, so none of them exercised the real `ToolNode`-driven thread-pool dispatch where the bug actually lived.
- **Snapshot replay key misses on LLM-regenerated tool arguments** (Section 18.6) — `agent-backtest` runs over the identical snapshot corpus can evaluate a different subset of matches each time, since the LLM regenerates its own tool-call arguments (not just its final answer) and a SHA-256 key match requires byte-identical inputs. Not a crash (A14's fault tolerance skips cleanly), but means bet counts and even which matches get scored can shift run to run — average over multiple runs before trusting any single comparison.
- ~~Result leakage can survive both leakage defenses~~ (Section 18.7) — observed on at least one match in the 24-match pilot despite the `before:<date>` web_search filter and the system-prompt instruction to discard final-score-bearing results. The model reported it honestly (`"Match result is known."` in `limitations`) rather than silently using it, but the defense was a mitigation, not a guarantee. **Substantially closed 2026-07-28/29 (A46–A47, Section 24).** A46 found the leak was structural, not a one-off (`process_match_row`'s shared replay path never applied the record-mode leakage instructions at all) and added a prompt-level guard; A47 went further with a code-enforced filter that drops individual leaked search results before they ever reach the LLM, and used it to re-scan and re-record the E0 corpus's confirmed-leaky matches. **Residual gap, honestly noted:** the filter is a title-pattern heuristic (`ponytail`-flagged in code), not a precise classifier — it will have residual false positives/negatives — and 21 of the corpus's re-recording attempts failed on a Tavily free-tier quota cap, leaving those 21 matches genuinely incomplete rather than re-verified clean.
- ~~`extract_recommendation` does not validate `recommendation_type`, `confidence`, or numeric field types beyond presence...~~ **Resolved 2026-07-11 (A28).** `extract_recommendation` now runs every candidate through internal Pydantic v2 models after the key-presence/`overall` checks, validating every market field's type and both `recommendation_type`/`confidence` enums — a model emitting `value_edge: "high"` now fails extraction with a field-path-qualified `RecommendationParseError` instead of silently passing through to a downstream `TypeError` (Section 8, Section 19.2). **Residual gap, honestly noted:** this validation does not cover `match` (still a plain untyped `dict`) or cross-field semantic invariants — nothing checks that `implied_probability` is actually `1 / current_odds`, or that `value_edge` is actually `ml_probability - implied_probability`; a model could still pass a self-inconsistent-but-well-typed set of numbers.
- **Corners markets (`home_corners`/`away_corners`) cannot be scored in backtests.** `_market_correct` (Section 11.1) always returns `None` for them, so they are never staked and never contribute to ROI/hit-rate — `min_odds_threshold`/`max_odds_threshold` enforcement only happens at recommendation time (code-enforced in `extract_recommendation` as of A29, Section 8, not just the LLM via the prompt), never validated against an actual outcome. Resolving this would require extending `MarketRecommendation` with a numeric line field and is out of scope for A09–A16.
- **`agent-compare` is slower and less fault-tolerant than `agent-backtest`** for equivalent match counts — it runs each config strictly sequentially via `BacktestHarness.run()` rather than reusing `agent-backtest`'s concurrent, per-match-fault-tolerant path (Section 14). A single missing snapshot aborts the whole comparison. Accepted scope boundary for A16, not a defect, but worth revisiting if comparison runs grow large.
- `save_report`/`save_comparison` filenames are timestamped to the second; two runs (of the same config, for `save_report`; of any size, for `save_comparison`) completing within the same UTC second will silently overwrite each other's report file. Low risk given this tool's manual, human-paced usage pattern.
- `agent-backtest --concurrency` is also capped by Python's default `ThreadPoolExecutor` size (`min(32, os.cpu_count() + 4)`) — values above that have no further effect on actual parallelism.
- **A37's structured-output path guarantees JSON shape, not content correctness (A38, open).** `with_structured_output()` (Section 8a) forces the model to emit a well-typed `MatchRecommendation` — it does not force the content to actually be about the requested match. Live-verified 2026-07-26: a fully schema-valid recommendation for a Burnley-vs-Bournemouth request came back describing an entirely different, invented match ("Manchester City" vs "Liverpool"). `MatchRecommendationModel.match` (`src/agent/schema.py`) is a bare, content-unchecked `dict`, and market `market`/`selection` fields are unconstrained strings — Pydantic checks type, never that the content matches `match_info`. Mirrors the class of gap A30 already closed for diagnostics (never trust the LLM's own prose over deterministic pipeline state); A38 (Section 22.3) is the open story to extend that same philosophy to match/market identity.
- **`recommendation_cache.db` (the app's cache, not this module, but populated by `run_agent()`'s output) has no awareness of agent/graph code changes.** Found during the same 2026-07-26 A38 investigation: `agent_config_hash` (Section 12.2) hashes only `config/agent_config.yaml`'s fields, never the graph/pipeline *code* — so Phase 11's 2026-07-22 restructure didn't invalidate any pre-restructure cache rows still sitting in the sandbox's `recommendation_cache.db`. 33 of 136 rows predated the A31 commit and were purged as a one-off cleanup, not a code fix. This blind spot is general and will recur on any future graph/pipeline change; worth its own story if it keeps causing confusion (e.g. stale `prediction_basis: "partial"` rows only the pre-A31 architecture could have produced).
- **BUG-018 (open, `documents/bugs.md`): `agent-snapshot` recording reliability under contention.** A pilot SWE snapshot run (A36) reported `Processed: 24 | Errors: 0 | Skipped: 0` but only 10/24 match directories actually had complete tool-response files on disk — 1/24 showed BUG-011's exact shape (a `_complete.json` marker with zero content) and 13/24 had no directory at all despite a clean-looking CLI exit. Not yet root-caused; suspected (not confirmed) to be related to a concurrently-running, unrelated `agent-snapshot --league E0` process contending for the same single-slot local Ollama instance. Blocks A36 until understood — `BacktestHarness`/`agent-backtest` correctly refuse to run against an incomplete corpus (`SnapshotMissingError`), so this didn't produce a wrong baseline, just no baseline yet.
- **`config_hash` (Section 12.2) does not include `max_odds_threshold`.** A29 added this field to `AgentConfig` but `src/agent/evaluation.py::config_hash`'s canonical field dict was not updated to hash it. Two `agent_config.yaml` variants differing only in `max_odds_threshold` currently produce the identical `config_hash` and, therefore, the identical `save_report` filename — discovered during this documentation pass (2026-07-18), not fixed here; a small follow-up story should add the field to the hash before `max_odds_threshold` is ever used as the varying knob in an `agent-compare` run.

---

## 18. First Real Backtest Baseline (2026-06-27, A19–A21)

Sections 9–14 describe a fully implemented backtest stack that, until this entry, had never actually been run — `data/agent_snapshots/` was empty and no report existed under `reports/agent_backtest/`. This section records the first real run and the bugs it surfaced, none of which were exercised by the mocked unit test suite (see Section 17's note on no live-model test coverage).

### 18.1 Prerequisite: resolving BUG-010

`select-best-models --context league` had never successfully populated anything, for two reasons neither obvious from reading the CLI surface alone:

1. **Tagging gap.** `run_train_target`/`ModelManager.run_pipeline()` never tagged `context` or `sweep_stage` on plain (non-sweep) training runs, so `ModelSelector._fetch_eligible_runs` (which filters on `tags.sweep_stage IN ('optuna','final')` and `tags.context`) could never find them. Fixed by tagging both inside `ModelManager.run_pipeline()` — confirmed safe because `src/utils/sweep_runner.py` never calls `run_pipeline()` (it calls `prepare_training_data`/`_evaluate_target` directly and tags `sweep_stage` itself), so sweep-based training is unaffected.
2. **Path-construction bug.** `ModelSelector` built `model_path` from the MLflow autolog artifact URI (`<artifact_uri>/model`) — an MLflow-flavor model directory, not a `joblib.load()`-able file. `ForecastService._load_context_models` silently skips any `model_path` that doesn't resolve via `Path(...).exists()`, so this failed with no error, just an empty league context. Fixed by logging an `artifact_filename` MLflow param at save time and pointing `model_path` at the real `models/*.joblib` artifact.

Also discovered: plain `train-forecast-suite --context league` (no `--model` override) defaults to `lr`/`rf_regressor` for all 8 targets, both of which require zero NaN across all 147 features — and 12 xG/LUCK rolling columns are currently 100% NaN in the feature store. Trained each target individually with `--model xgb`/`--model xgb_regressor` instead (XGBoost tolerates NaN natively, and is the model family that wins every target per Sections 22–24 anyway).

### 18.2 Pilot snapshot corpus

24 E0 matches, 2026-03-01 → 2026-03-16 (the most recent finished matches in `raw_matches`). First collection attempt was started in parallel with A19's last verification step and had to be discarded — it completed before the `model_path` fix landed, so it silently encoded `market_odds_only_league_fallback` for every match despite the league context technically existing. Re-collected cleanly after fix verification: 24/24 processed, 0 errors, 22/24 used `team_history_and_market`, 2 used `market_odds_only` (not investigated further).

### 18.3 Staking crash: `direct_bet` with null odds

The first `agent-backtest` run replayed all 24 matches successfully but crashed in `simulate_flat_stake` with `TypeError: float() argument must be ... not 'NoneType'`. Root cause: the agent sometimes marks a market `recommendation_type: "direct_bet"` while `current_odds` is `null` (no bookmaker odds found for that specific market, even though odds existed for others in the same match). `staking.py`'s skip-gate checked `recommendation_type == "direct_bet"` and `correct is not None` but never guarded `current_odds` itself. Fixed in both `simulate_flat_stake` and `simulate_kelly_stake` by skipping when `current_odds is None`, with two new regression tests in `tests/test_staking.py`. This is a model-output quality issue (the agent shouldn't call something a direct bet with no odds to act on), not purely a staking bug — worth tightening in a future prompt revision, but skipping rather than crashing is the correct defensive behavior either way.

### 18.4 BUG-011: the entire first run was secretly live, not replayed

The first `agent-backtest` run (18.3) replayed all 24 matches with zero `SnapshotMissingError`s and produced a report. That should have been impossible — `data/agent_snapshots/` turned out to contain 24 directories with **only a `_complete.json` marker each, zero actual tool-response files** (caught by manual inspection, not by anything in this pipeline). Root cause, traced via `superpowers:systematic-debugging`:

`SnapshotStore` stored `mode`/`match_id`/`match_date` in `threading.local()` (Section 9). LangGraph's `ToolNode` executes every tool call — even a single one — via `langchain_core.runnables.config.get_executor_for_config()`, which returns a `ContextThreadPoolExecutor`, and `run_agent()` calls the **sync** `compiled.invoke()`, which routes through `ToolNode._func()`'s `executor.map(self._run_one, ...)`. That dispatches the actual tool call onto a worker thread *different* from the one that called `configure_snapshot_store()`. `ContextThreadPoolExecutor` explicitly copies `contextvars.Context` into its worker (`copy_context().run(...)`) but never touches `threading.local()`, which is strictly per-OS-thread. So the worker thread's `SnapshotStore._local` was always a fresh, never-initialized `threading.local()`, and the `mode` property's `getattr(self._local, "mode", "live")` silently defaulted to `"live"` on **every single tool call, regardless of what the calling thread had configured**.

Effect: `agent-snapshot` ("record" mode) called the real `ForecastService`/Tavily every time — which is why it produced plausible, varied output and a `_complete.json` per match — but `wrap()`'s `mode == "live"` branch returns immediately without ever computing a path or writing a file, so nothing was ever persisted. `agent-backtest` ("replay" mode) suffered the identical bug: it never read a single recorded file, made fresh live calls every time, and could never raise `SnapshotMissingError` because the live branch never even checks whether a snapshot exists. The Section 18.3 staking crash was real and the fix for it is still correct, but the 7–8 bet "baseline" computed alongside it reflected nothing but live-call noise, not a frozen, reproducible replay. This also silently disabled the `before:<date>` web_search leakage filter the whole time, since it's gated on `mode in ("record", "replay")` — which never evaluated true either.

This was never caught by the existing test suite because every `SnapshotStore`/tool test calls `configure_snapshot_store()` and the tool function on the *same* thread — none of them exercise a real `ToolNode`-driven `graph.invoke()`, the only path where the thread boundary actually matters.

**Fix:** switched `SnapshotStore` from `threading.local()` to `contextvars.ContextVar` (same public API, zero changes needed in `tools.py`/`main.py`). `ContextThreadPoolExecutor` and `asyncio.to_thread` (the latter used by `agent-backtest --concurrency`, Section 13) both explicitly propagate `contextvars.Context` into worker threads via `copy_context()`, while a bare `threading.Thread()` — not used anywhere in this codebase, but exercised by the pre-existing `test_mode_and_match_are_thread_local` regression test — still correctly does *not* inherit it. So the fix closes the real gap without weakening the cross-match isolation A09/A14 were designed for.

Verified four ways before trusting it: (1) an isolated repro proving `threading.local()` reads back `"live"` and `contextvars.ContextVar` correctly reads back `"record"` across a real `ContextThreadPoolExecutor`; (2) a new permanent regression test, `test_mode_and_match_propagate_into_context_thread_pool_executor` in `tests/test_snapshot_store.py`; (3) a live single-match `agent-snapshot` run that produced real `forecast_league_*.json` and `web_search_*.json` files, with `before:2026-03-16` correctly appended to the recorded search query; (4) a canary-injection test — hand-edited a recorded `forecast_league` response with an obviously fake marker string, then called `agent_tools.forecast_league.func(...)` directly in `"replay"` mode and confirmed the doctored content came back verbatim, proving the live forecast service was never touched.

The entire A20 pilot corpus and both A21 backtest reports were discarded and regenerated from scratch after this fix landed.

### 18.5 Corrected results (post-BUG-011 fix)

Same pilot range (2026-03-01 → 2026-03-16, 24 E0 matches), genuinely replayed this time:

| Stake mode | Matches evaluated | Bets placed | Bets won | Hit rate | ROI | Max drawdown | Ending bankroll |
|---|---|---|---|---|---|---|---|
| `flat` | 23/24 | 20 | 11 | 0.55 | **+0.1845** | 0.0503 | 1036.90 |
| `kelly` | 20/24 | 5 | 2 | 0.40 | **−0.1476** | 0.1362 | 961.70 |

Both runs skipped some matches with `SnapshotMissingError` (flat: 1, kelly: 4 — different matches each time) — see 18.6, this is a separate, lower-priority finding, not a regression of the BUG-011 fix.

**Findings:**
- **Bet frequency jumped sharply once replay was genuine** (0.87 for flat vs. 0.29 in the bogus 18.3 run) — the original "baseline" wasn't just noisy, it was systematically more conservative, because every match was an independent fresh live call rather than a consistent replay of the same recorded forecast/search context. Any conclusion drawn from the original numbers would have been wrong, not just imprecise.
- Sample size (20–23 matches, 5–20 bets) is still far too small to claim the agent has real edge — this remains a pipeline validation, not a profitability verdict.
- Kelly's larger max drawdown (0.136 vs. flat's 0.050) is expected on a small sample — Kelly sizes by edge and compounds, so a short losing run moves the bankroll more than flat's fixed 1% stake.

### 18.6 Secondary finding: snapshot key misses from LLM-regenerated tool arguments

Flat and kelly evaluated a *different* number of matches (23 vs. 20) from the same 24-match corpus. `SnapshotStore`'s key is a SHA-256 hash of the tool call's input kwargs (Section 9) — but replay only freezes the tool *response*, not the request. The LLM regenerates its own tool-call arguments fresh on every `agent-backtest` invocation (team name spelling, exact odds precision, etc.), and at `temperature: 0.1` those arguments aren't always byte-identical to what was recorded, producing a different hash and a `SnapshotMissingError` that the harness correctly skips (per A14's fault-tolerance design — this did not crash anything). Combined with the pre-existing observation that final recommendations vary run-to-run (sampling noise in the LLM's output, not just its tool calls), this means **`agent-compare` (A22) should run each config multiple times and look at the distribution, not trust a single pass** — both the bet count and which matches even get evaluated can shift between runs. A future hardening pass could normalize/canonicalize tool inputs before hashing (e.g. resolve team names to canonical IDs before computing the key) to make replay matching more robust to this kind of LLM-output drift.

### 18.7 Secondary finding: result leakage survived both leakage defenses on at least one match

One recorded snapshot (Sunderland vs Brighton, 2026-03-14) produced a recommendation whose `explanation` field read *"The match has already occurred, and Brighton won with a score of 1-0"* with `limitations: ["Match result is known."]`. This is exactly what the two-layer defense in Section 9.1/10 (the `before:<date>` web_search query filter, plus the system-prompt instruction to discard any result mentioning a final score) was built to prevent, and it still got through on at least one match in a 24-match pilot. The model was at least honest about it in the structured output rather than silently incorporating it, but this confirms the techspec's own framing of that defense as a mitigation, not a guarantee. Not fixed here — out of scope for this pass — but worth a closer look (e.g. a stricter search backend constraint, or filtering Tavily results by URL-published-date metadata rather than trusting the `before:` query operator) before scaling the snapshot corpus up.

### 18.8 Recommended next step

Re-run `agent-compare` guidance from 18.6 applies before any config comparison work (A22). Scaling the pilot beyond 24 matches is reasonable now that record/replay isolation is genuinely verified, but each additional match still costs one full live Ollama+Tavily round trip at snapshot-collection time — there is no shortcut around that cost, and the 18.7 leakage finding is worth addressing first if the snapshot corpus is going to grow and get reused for repeated experiments.

---

## 19. League-Aware Routing and Output Validation Hardening (2026-07-11, A27-A29)

Three stories landed the same day, closing gaps this document had carried since Phase 5–6: the agent's own domestic-vs-international judgment was unreliable in exactly the cases where it mattered most (Section 19.1), and `extract_recommendation`'s validation stopped at key presence — a model could emit a structurally valid but semantically nonsensical `MatchRecommendation` and nothing downstream would catch it before a `TypeError` in `staking.py` (Section 19.2–19.3). All three are also motivated by the web app (`documents/app_user_stories.md`) becoming a second, independent consumer of `MatchRecommendation` output — a validation gap that only mattered to the backtest stack (which crashes loudly and gets skipped by A14's fault tolerance, Section 13) now also matters to a live-serving API and UI, which don't get that same safety net for free.

### 19.1 A27 — League-Aware Model Routing

Before A27, the agent decided between `forecast_league` and `forecast_international` purely from the system prompt's step ordering and its own judgment about whether a league "sounded" domestic. This was unreliable specifically for well-known leagues with zero actual historical-data coverage in this system (e.g. La Liga) — the model would guess `forecast_league`, hit `forecast_league`'s existing BUG-010 fallback (Section 4.2), and get a usable answer anyway, but only by accident of that fallback existing, not because the agent made an informed choice.

`resolve_competition` (Section 4.4) makes the underlying signal — whether a competition has `competition_specific` model coverage in the registry, per `US#107` on the forecast-engine side — directly tool-callable, and the system prompt (Section 5) now instructs the agent to call it first and follow its `recommended_tool` verbatim. The tool-call stop-rule budget was raised from 2 to 3 to accommodate the extra call without starving the agent of its `web_search` step.

**Verification.** TDD via `tests/test_agent_tool_selection.py` (4 tests: `competition_specific` resolution, `general_purpose` resolution, unregistered-competition resolution, and an end-to-end case showing that following the tool's advice for an unregistered league yields `market_odds_only`, not a cold-start `team_history_and_market` result) — all failed with `ImportError` before the tool existed, as expected. Full suite at merge: 335 passed / 1 skipped, zero regressions.

**Honest caveat from the live run.** A27 was verified against the real `llama3.1:8b` model, not just mocked tests, and that run surfaced a limitation worth recording rather than glossing over: the acceptance criterion's literal wording — "no case where the agent calls `forecast_league` for a competition with only `general_purpose` coverage" — **cannot be guaranteed** against a small local model. The live run showed `llama3.1:8b` calling `web_search` before `resolve_competition` (violating the prompt's own step order) and calling `forecast_league` multiple times after `resolve_competition` had already recommended `forecast_international`, eventually exhausting the tool-call budget. This is consistent with, not a new instance separate from, the weak-local-model behavior already documented in Section 7 — it is a pre-existing characteristic of this model tier, not a defect introduced by A27.

What *is* guaranteed regardless of the LLM's compliance: because of `US#107` on the forecast-engine side, even when the agent disobeys and calls `forecast_league` for an unregistered league anyway, the result it gets back is still the honest `market_odds_only` — never a silently mislabeled cold-start prediction. The regression suite covers `resolve_competition`'s own correctness deterministically; a small model's adherence to the prompt's tool-call ordering is a separate, ongoing concern, relevant to future prompt-tuning work (e.g. a future A22 config-comparison pass), not something this story could fully close by itself.

### 19.2 A28 — Output Validation Hardening (closes BUG-013)

Prior to A28, `extract_recommendation` (Section 8) validated only the 7 top-level key names and the `overall` enum — every other field, including every market's `recommendation_type`/numeric fields and the top-level `confidence` enum, passed through unchecked. A model could return `value_edge: "high"` (a string) or `confidence: ""` and extraction would succeed; the failure would only surface later, as a `TypeError`/`ValueError` inside `staking.py`'s `float(...)` calls at backtest time — or, post-A27's web-app integration, inside a second, independent consumer with its own (possibly absent) defensive handling.

A28 adds internal (non-return-type-changing) Pydantic v2 validation — `_MatchRecommendationModel`/`_MarketRecommendationModel` — applied after the pre-existing key-presence/`overall` checks (see Section 8 for the full field-by-field description). It also closes **BUG-013** (`documents/bugs.md`, status `fixed`): the agent occasionally emitted `recommendation_type: "direct_bet"` with `current_odds: null` — a logically incoherent combination that had previously only been patched at its downstream symptom (`staking.py`'s skip-gate, added during the Section 18.3 baseline). A28 fixes the actual root cause at extraction time via `_downgrade_direct_bet_with_null_odds()`.

**Verification.** TDD via `tests/test_agent_schema_validation.py` (6 tests: the three specific gaps this section previously documented in Section 17 — `value_edge` as a string, `confidence` as an empty string, an arbitrary `recommendation_type` string — plus the BUG-013 null-odds downgrade, a conditional-market-with-null-odds non-interference case, and a populated-markets-list regression), all failing correctly before the fix. Existing `tests/test_agent_schema.py` (9 tests) required zero changes — the pre-existing error wording for the key-presence/`overall` checks was preserved verbatim. Full suite at merge: 305 passed / 1 skipped, zero regressions. `documents/bugs.md` BUG-013 updated to `fixed`.

### 19.3 A29 — Widened, Code-Enforced Odds Bounds

A29 extends A28's validation layer with a second semantic rule, immediately after: the prompt's old `2.0`-decimal-floor-with-no-ceiling convention is replaced with an explicit, code-enforced band, `[1.2, 11.0]` decimal (roughly −500 to +1000 American) — see Section 2 for the `AgentConfig` field addition and Section 8 for `_downgrade_direct_bet_outside_odds_bounds()`'s exact behavior. The key design decision worth calling out here: **A29's downgrade target is `conditional`, not `no_bet`** — the opposite of A28's null-odds downgrade — because a real price exists in this case, just outside the accepted range, matching the prompt's pre-existing "conditional" convention for markets with value at an unfavorable price. A28's null-odds case has no price to act on at all, so `no_bet` is the only coherent target there. The two downgrade passes run in a fixed order (null-odds first, then bounds) and don't conflict: a market already downgraded to `no_bet` by A28's pass is no longer `direct_bet`, so A29's pass skips it.

**Verification.** TDD via `tests/test_agent_odds_bounds.py` (8 tests: below floor, above ceiling, both exact bounds accepted as inclusive, custom thresholds, an explicit regression proving the old `2.0`-only behavior is gone, a `conditional` market left untouched, and BUG-013 precedence — i.e. a null-odds market is downgraded to `no_bet` by A28's pass and never reaches A29's bounds check), plus 2 new/updated tests in `tests/test_agent_config.py` for the new required field. Every other `AgentConfig(...)` construction site in the test suite (`test_agent_graph.py`, `test_backtest.py`, `test_agent_evaluation.py`, `app/backend/tests/test_llm_check.py`) was updated to supply `max_odds_threshold`, since it is now a required field with no default in `AgentConfig.from_yaml`'s `_REQUIRED` set (Section 2). Full suite at merge: 344 passed / 1 skipped, zero regressions.

### 19.4 Cross-references

The factual specification of what changed lives in Sections 2 (`AgentConfig`), 4.4 (`resolve_competition`), 5 (system prompt workflow/stop-rule/value-calculation updates), and 8 (`extract_recommendation`'s new validation layer) — this section intentionally does not restate those details, only the narrative of why each story existed and what was learned running it live. Section 16's Implementation Status table and Section 17's Known Limitations have also been updated to reflect A27–A29.

## 20. Critic/Train Mode and Competition-Scoped Lessons (A33)

Design: `docs/superpowers/specs/2026-07-22-agent-phase11-design.md` (A33 section, revised 2026-07-24). Implementation plan: `docs/superpowers/plans/2026-07-24-agent-critic-mode-lessons.md`.

### 20.1 `agent-train` CLI

Structurally parallel to `agent-backtest` (Section 13): same `BacktestHarness.load_matches()` + `process_match_row()` replay path, same `src/agent/evaluation.py` ROI/hit-rate/drawdown scoring, same report shape (saved under `reports/agent_train/` instead of `reports/agent_backtest/` to keep the two apart — both directories are gitignored). Additionally, for every match that captured full graph state (`process_match_row(..., capture_state=True)`), `main.py`'s `_write_train_artifacts()` writes:

- One row to `agent_telemetry` (`match_id`, `run_id`, `competition_resolution`, `research_evidence`, `forecast_payload`, `recommendation` — JSON-serialized TEXT columns — `created_at`). `run_id` is a single `uuid4().hex` shared by every match in one `agent-train` invocation.
- One `status='pending'` row to `agent_lessons` (`lesson_text` from a deterministic template — `generate_lesson_text()` in `src/agent/lessons.py` — plus `competition_id`/`tier` recorded automatically from that match's `competition_resolution`).

```bash
python main.py agent-train --from-date 2026-01-01 --to-date 2026-01-31 --league E0 --stake-mode flat
```

All DB writes happen synchronously, single-threaded, strictly after `_run_backtest_concurrent`'s `asyncio.gather` over the concurrent per-match replay has fully completed — never from within a worker thread, so a single DuckDB connection is never touched concurrently.

### 20.2 `agent-lessons approve/reject` CLI

```bash
python main.py agent-lessons approve <id> --scope competition   # or --scope tier
python main.py agent-lessons reject <id>
```

`--scope` is required on `approve`, no default: `competition` pins the lesson to its recorded `competition_id`; `tier` widens it to every match resolving to its recorded `tier` (`general_purpose` / `competition_specific`), regardless of competition. This is the only point a human judges whether a lesson generalizes — `agent-train` itself makes no such judgment, it only records the two raw facts about its source match. `--reviewer` is optional and defaults to `getpass.getuser()`. An unknown lesson id raises `ValueError` (uncaught — surfaces as a traceback, matching this file's existing precedent for CLI-argument-driven errors, e.g. `agent-backtest --concurrency 0`).

### 20.3 Live-mode injection (`lessons_node`, `src/agent/pipeline.py`)

A new required graph node runs after `forecast_node` succeeds (`resolve_competition → research → forecast → lessons → agent`; `route_after_forecast` now returns `"lessons"` instead of `"agent"` on success). It loads `agent_lessons` rows where `status='approved'` AND (`scope='competition'` AND `competition_id` matches this match) OR (`scope='tier'` AND `tier` matches this match's tier), and injects them as a single `HumanMessage` ahead of the LLM's turn — the same mechanism `forecast_node` uses for forecast/research evidence (`_format_evidence_message`).

Gated on `SnapshotStore.mode == "live"` (`src/agent/tools.get_snapshot_store()`): `agent-backtest`/`agent-train` replay and `agent-snapshot` record never see lessons, since injecting anything approved after a historical match ran would leak future information into the A13/A21/A34 baseline scoring methodology `agent-backtest` and `agent-train` share. This gating decision was made during implementation, not specified in the original design doc language, precisely to protect that methodology.

`load_approved_lessons()` (`src/agent/lessons.py`) is the only function `lessons_node` imports from the lessons module — its SQL hardcodes `status='approved'` and never touches an outcome-bearing table, so as long as `lessons_node` itself only ever reads through that one function, live mode is structurally unable to read match outcomes or pending/rejected lessons (`tests/test_agent_pipeline.py::test_pipeline_module_never_imports_lesson_write_or_review_functions` asserts this at the source-text level). **Caveat, found in final review:** the guarantee is scoped to `load_approved_lessons`'s own signature and SQL, not to the raw `DuckDBManager` connection `lessons_node` opens — nothing stops a future edit to `lessons_node` from adding a direct `conn.execute(...)` against `raw_matches` or similar, since `conn` is in scope. The forbidden-function-name test would not catch that. Today's code doesn't do this; treat the isolation as a discipline `lessons_node` must keep honoring, not an invariant enforced by the type system.

**Failure handling, found and fixed during implementation review:** `lessons_node` opens its DuckDB connection with `read_only=True`. A missing `agent_lessons` *table* (e.g. `agent-train` has never been run) raises `duckdb.CatalogException`, which `load_approved_lessons` catches and treats as "no lessons." A missing DuckDB *file* (a fresh deployment, before anything has ever written to it) raises a different exception, `duckdb.IOException`, at connection-open time — before `load_approved_lessons` is even called. Code review caught that this second case was originally unhandled and would crash `run_agent` on a fresh environment's very first live recommendation for any match whose forecast never happens to touch DuckDB (e.g. odds-based international forecasting, which loads `.joblib` models directly). `lessons_node` now also catches `duckdb.IOException` and returns no lessons in that case, with a dedicated test (`test_lessons_node_returns_empty_dict_when_db_file_does_not_exist`) using a real, non-mocked `DuckDBManager` pointed at a genuinely nonexistent file.

### 20.4 Schema

```sql
CREATE SEQUENCE agent_lessons_id_seq START 1;
CREATE TABLE agent_lessons (
    id INTEGER PRIMARY KEY DEFAULT nextval('agent_lessons_id_seq'),
    lesson_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
    competition_id TEXT,                       -- NULL for leagueless internationals
    tier TEXT NOT NULL,                         -- general_purpose | competition_specific
    scope TEXT,                                 -- NULL until approved; competition | tier
    source_match_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    reviewed_at TIMESTAMP,
    reviewer TEXT
);

CREATE TABLE agent_telemetry (
    match_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    competition_resolution TEXT,   -- JSON
    research_evidence TEXT,        -- JSON
    forecast_payload TEXT,         -- JSON
    recommendation TEXT,           -- JSON
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, run_id)
);
```

Verified live against a real (non-`:memory:`) DuckDB file during implementation: `agent-train` → `agent-lessons approve` → `lessons_node` round-trips correctly end to end, independent of the unit test suite.

### 20.5 Known limitations, accepted as designed

- ~~Approved lessons accumulate indefinitely within each `competition_id`/`tier` bucket, with no cap or conflict resolution.~~ **Partially resolved 2026-07-28 (A45, Section 21.6).** `find_conflicting_rule()` now runs a pairwise LLM check against co-occurring approved `rule_text` rows at approval time, blocking approval on a detected conflict unless `--force`. Still no cap on volume, and A45's own live-verification found the LLM check's semantic recall is unreliable on paraphrased/indirectly-overlapping contradictions (Section 21.6) — a real mitigation, not a guarantee.
- `_write_train_artifacts` does not wrap its per-record writes in an explicit transaction; a mid-loop failure leaves earlier records committed. Low risk in practice (the values it writes are already JSON-safe by the time they reach it) and acceptable given this is an isolated review/training tool, not a production-critical path — a bad row is just a `pending` candidate a human can reject.
- `agent-train`/`agent-lessons`' unknown-id and CLI-argument errors surface as raw Python tracebacks rather than a friendly `[ERROR] ...` message (unlike `agent-recommend`). Matches this file's existing precedent elsewhere in `main.py`, not a regression introduced by A33.
- Running `agent-train` concurrently with live traffic is not coordinated. `run_agent_train` holds a read-write DuckDB connection on the shared database for the whole batch; a live `lessons_node` read that lands during that window can hit the same `duckdb.IOException` path as the missing-file case (Section 20.3) and silently degrade to "no lessons" for that one recommendation rather than crashing. Graceful, but means running `agent-train` in production temporarily and invisibly disables lesson injection — acceptable for a low-frequency, isolated training tool, but worth knowing before scheduling it during active serving hours.

---

## 21. Critic-Mode Batching, Train/Test Splitting, and Provider Diversification (A39–A45, 2026-07-28)

Seven stories landed the same day, all extending Section 20's critic/train-mode machinery: batching lesson candidates down to a reviewable volume (21.1), a stable train/test split so an unbiased backtest number survives critic review (21.2), letting approved lessons actually apply against that held-out split (21.3), a fifth LLM provider plus a replay-mode correctness fix it surfaced (21.4), an LLM-synthesized reflective narrative on top of the deterministic batch stats (21.5), distilling that narrative into a short prompt-ready rule at approval time (21.6), and a pairwise conflict check across approved rules (21.7). Design/motivation narrative for each lives in `documents/agent_user_stories.md` A39–A45; this section is the factual reference for the resulting code shape.

### 21.1 Batched Lesson Candidates (A39)

Before A39, `agent-train` wrote one `agent_lessons` row per scored match — a real E0 run produced 118 near-redundant pending candidates. `generate_batch_lesson_text(records: list[BacktestRecord]) -> str` (`src/agent/lessons.py`) is a pure, deterministic (zero-LLM-call) Counter-based aggregator: league label, date range, overall-recommendation distribution, per-market correct/incorrect/unresolved tallies (naming the single most-frequently-wrong market), confidence-vs-accuracy breakdown, and keyword-bucketed limitation themes (injury/availability, research-coverage gap, generic historical-data caveat).

`agent-train --batch-size N` (default `1`, preserving the original one-row-per-match behavior byte-for-byte at persisted row content). `main.py`'s `_write_train_artifacts()` takes a genuinely separate code path for `batch_size <= 1` (the original A33 per-record loop) vs. `batch_size > 1` (new chunking logic): consecutive same-`(competition_id, tier)` records — already date-ordered from `BacktestHarness.load_matches()` — are grouped into chunks of up to N, flushing a shorter final group rather than dropping remainder matches (batches never span a scope boundary, since `insert_lesson_candidate` takes one `competition_id`/`tier` per row). `agent_lessons.source_match_id` stores the batch's match_ids comma-joined rather than widening the schema. The CLI's stdout summary changed from a bare count to `(lessons_written, telemetry_written)`, since the two diverge once `batch_size > 1` — a cosmetic output change, not a data-format one.

### 21.2 Stable Train/Test Split (A40)

`match_in_test_split(match_id: str, test_fraction: float) -> bool` (`src/agent/backtest.py`) hashes `match_id` via SHA-256 and buckets the first 4 bytes into `[0, 1)`, comparing against `test_fraction`. Deliberately a per-id hash rather than `DataFrame.sample(frac=...)` — a given match's split assignment never shifts as the corpus grows or as different `--league`/date filters are applied, with no separate split-assignment table to persist.

`BacktestHarness.load_matches()`/`.run()` gain `split: Literal["all", "train", "test"] = "all"` and `test_fraction: float = 0.2`, applied to the DataFrame right after the DB query, before `--sample`'s stratified sampling; an invalid `split` value raises `ValueError` up front. `agent-backtest`/`agent-train` both expose `--split {all,train,test}` and `--test-fraction`. Live-verified against the real corpus: E0 (380 matches) split 298 train / 82 test (78.4/21.6%); SWE (24 matches, a much smaller population) split 21/3 (87.5/12.5% — off-target but expected, the same hash-bucket variance `_stratified_sample`'s per-stratum rounding also shows at small N). Zero `match_id` overlap between train/test confirmed for both leagues.

**Important scope boundary, stated plainly rather than implied:** `lessons_node` (Section 20.3) still only runs in `live` mode by default (Section 21.3 below is the narrow exception) — this split by itself does **not** let you measure "did lessons approved from train improve results on test" end-to-end. What it buys is a `--split test` ROI number the human reviewer never saw outcomes for while approving lesson candidates from `--split train`, removing reviewer-hindsight bias from that one number.

### 21.3 Lessons Applied During a Held-Out Test Backtest (A41)

`SnapshotStore` (`src/agent/snapshot_store.py`) gains a fourth context-scoped flag, `allow_lessons_in_replay` (default `False`, same `contextvars.ContextVar` pattern as `mode`/`match_id`/`match_date`), with a property and `set_allow_lessons_in_replay()` setter. `configure_snapshot_store(..., allow_lessons_in_replay: bool | None = None)` is sticky-if-omitted, matching `base_dir`'s existing convention. `lessons_node`'s gate widens from `mode == "live"` to `mode == "live" OR (mode == "replay" AND allow_lessons_in_replay)`.

`process_match_row()` (`src/agent/backtest.py`) gains an `allow_lessons_in_replay: bool = False` parameter, threaded into its `configure_snapshot_store()` call the same way A33's `capture_state` already is. `agent-backtest` gains `--use-lessons`; `agent-train` deliberately does **not** get this flag (its job is generating fresh candidates, not consuming approved ones). `run_agent_backtest()` hard-rejects `--use-lessons` combined with anything but `--split test` — a `ValueError` raised before any DB/async work — matching this codebase's established "structurally enforced, not just prompted" convention for leakage guards (A10's `before:<date>` filter, A31's tool removal).

### 21.4 DeepSeek Provider and the Replay-Mode `tools=[]` Fix (A42)

`_build_llm()` (`src/agent/graph.py`) gained a `deepseek` branch: DeepSeek exposes an OpenAI-compatible chat-completions endpoint, so it's implemented via `langchain_openai.ChatOpenAI(model=config.model, temperature=config.temperature, base_url="https://api.deepseek.com", api_key=os.environ.get("DEEPSEEK_API_KEY"))` — no dedicated `langchain-deepseek` package needed. `langchain-openai` added to `requirements.txt`. `AgentConfig.provider`'s `Literal` widened to `["ollama", "anthropic", "groq", "gemini", "deepseek"]` (Section 2). `config/agent_config_deepseek.yaml` is a separate, opt-in config (`model: "deepseek-chat"`) rather than an edit to the shared default (Section 2).

**Bug found and fixed while live-verifying DeepSeek against `agent-train`:** a smoke run hit a 100% `SnapshotMissingError` skip rate. Root cause: `process_match_row()` called `run_agent()` without overriding `tools`, so replay always got the same tool list as a genuine live run — `web_search`, the one LLM-callable tool remaining post-A31/A32. `research_node` already guarantees deterministic baseline evidence before the LLM's turn, but whenever the LLM chose to call `web_search` anyway (an optional follow-up, not the deterministic templated calls), its self-invented query text essentially never byte-matched what was recorded, aborting the whole match. **Fixed by passing `tools=[]` explicitly in both of `process_match_row()`'s `run_agent()` calls** — replay now structurally cannot exercise a tool that can never succeed in that mode, for any provider (this also explains, retroactively, part of `llama3.1:8b`'s earlier ~69% E0 skip rate, Section 17). Live mode (`agent-recommend`, the webapp) is untouched.

Verified live: the full E0 train split (298 matches, `--batch-size 80 --concurrency 8`, DeepSeek) completed in 2m42s with 29/298 skipped (9.7%, traced to pre-existing incomplete snapshot directories in the corpus itself, not the query-matching issue this fix addresses); 269 evaluated, 45 bets, 33 won, hit rate 73.3%, ROI +27.2% — not a controlled comparison against `llama3.1:8b`'s prior baseline (different model, different — post-fix — tool availability, a train-split subset), but a promising signal. DeepSeek also used canonical market names consistently (`result_3way`/`btts`) where `llama3.1:8b` sometimes drifted (`"1X2"`, `"Match Result"`).

### 21.5 LLM-Synthesized Batch Reflections (A43)

`generate_batch_reflection(records, stats_text, llm_invoke, n_examples=5)` (`src/agent/lessons.py`) layers a genuine narrative on top of 21.1's deterministic stats — the Counter-based numbers stay the trustworthy, unhallucinatable anchor; the reflection adds qualitative judgment the stats alone can't produce. `llm_invoke` is a plain `str -> str` callable, not a langchain object, so `lessons.py` stays decoupled from langchain and trivially testable. `_classify_and_rank()` splits a batch into misses/hits by whether more of a record's resolved markets were incorrect than correct, ranked highest-confidence-first; the top `n_examples` of each — with the agent's own original `explanation` text — go into the prompt alongside the stats, asking for 3–5 sentences on the systematic pattern behind the misses, what the hits got right, and one concrete adjustment, explicitly instructed not to invent facts or hedge generically.

Returns `None` on any failure (exception or empty response) — `_write_train_artifacts()` (gained a `config: AgentConfig | None = None` parameter) falls back to stats-only rather than losing the whole candidate over a transient API error. `_build_llm_invoke()` (new, `main.py`) wraps `_build_llm()`/`_extract_text()` into the plain callable, reusing whatever provider the run itself is configured for. Only wired into the `batch_size > 1` path — `batch_size <= 1` stays untouched, preserving A39's byte-identical-at-1 guarantee.

Live-verified against real DeepSeek output: a 6-match E0 smoke batch produced a reflection naming two specific matches with the actual forecast numbers involved and a concrete suggested adjustment — the qualitative judgment 21.1's stats-only version structurally cannot produce.

### 21.6 Distilled `rule_text` on Approval (A44)

`agent_lessons` gains a nullable `rule_text TEXT` column, added via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` inside `create_lessons_tables()` (verified idempotent against both a fresh and an already-populated table). `lesson_text` is untouched — it stays the full stats+reflection audit trail a reviewer judges *from*; it is **never read by the live path** anymore.

`agent-lessons approve <id> --scope ... [--rule "..."] [--config ...]`: `--rule` given stores it verbatim, zero LLM calls. `--rule` omitted auto-distills via `generate_rule_from_lesson(lesson_text, llm_invoke)` (same plain-callable design as 21.5), printed to the reviewer before being stored so a bad distillation is easy to notice and redo. Distillation happens at **approval** time, not generation time — not every batch lesson gets approved, so distilling every one at `agent-train` time would waste LLM calls on lessons that get rejected.

`approve_lesson()` now requires a non-empty `rule_text` — an approved lesson with no rule would otherwise silently vanish from live use. `load_approved_lessons()` (Section 20.3) switched from `SELECT lesson_text` to `SELECT rule_text ... WHERE rule_text IS NOT NULL` — excludes any approved row without one (only reachable for a hypothetical pre-A44 row), so the live prompt can never see raw match-specific `lesson_text` even by accident.

### 21.7 Conflicting-Rule Detection (A45)

`find_conflicting_rule(new_rule_text, existing_rules, llm_invoke)` (`src/agent/lessons.py`) — one LLM call per approval, comparing the candidate rule against every already-approved rule that could actually co-occur with it live. The co-occurrence set is fetched via `load_approved_lessons(conn, competition_id, tier)` itself (not a hand-rolled duplicate query), so it exactly matches what a real live match would ever load together, respecting the same `competition`- vs. `tier`-scope distinction. Unlike 21.5/21.6's helpers, `find_conflicting_rule` deliberately does **not** catch `llm_invoke` exceptions — a failed check and a clean "nothing found" result must stay distinguishable to the caller, since they're handled oppositely.

Wired into `run_agent_lessons_approve()` (`main.py`), which now always builds an LLM invoke callable (reused from 21.5/21.6) regardless of whether `--rule` was given manually, since the conflict check must run either way. Two failure modes, handled deliberately differently: the *check itself* raising fails **open** (warns, approval still proceeds — a transient API error shouldn't block a reviewer's workflow); the check *succeeding and finding a real conflict* fails **closed** (refuses to approve, `ValueError` naming the conflicting rule) unless `--force` is passed, which approves anyway with a printed warning.

**Honest live-verification finding.** Tested end-to-end against real DeepSeek using two disposable rules mirroring a real motivating contradiction observed in this session's own batch reflections (one rule: never recommend `result_3way` without a direct model probability; the other: fall back to odds-implied probability for `result_3way` when only BTTS is forecast) — DeepSeek's check returned `NONE` (no conflict found) for this pair, both through the full CLI and in an isolated re-test of the same prompt. The mechanism worked correctly throughout (prompt sent, response parsed, correctly treated as no-conflict) — but the LLM's semantic judgment missed a contradiction that requires a logical inference step ("only forecasts BTTS" implies "no `result_3way` probability exists") rather than direct textual overlap. This is real signal about design 3's reliability ceiling on paraphrased/indirectly-overlapping conditions, not a code defect — the check's silence should not be trusted as proof of no conflict. A visibility companion (printing co-occurring rules at approval time) or a periodic full-set audit are candidate follow-ups, not yet built.

---

## 22. Swedish League Verification and Structured-Output Content Grounding (A35–A38, 2026-07-23/26)

### 22.1 Second-Competition Routing Verification (A35)

Before A35, `resolve_competition_node`/`forecast_node` (Section 19, `src/agent/pipeline.py`) were built and tested only against `E0` (the sole real `competition_specific` competition) and an unregistered league (La Liga, the `general_purpose`-fallback case) — no test exercised a second genuinely-registered competition end-to-end, even though the forecast-engine side (`documents/user_stories.md` Phase 20) had already registered Sweden (`SWE`) as a second `competition_specific` competition with its own trained models.

`test_resolve_competition_node_uses_real_registry_for_second_competition_specific_league` (`tests/test_agent_pipeline.py`) confirms `resolve_competition_node` resolves `league="SWE"` to `tier="competition_specific"`/`recommended_tool="forecast_league"` against the real `config/competitions.yaml`, with zero code changes needed — `resolve_competition_node`/`forecast_node` already keyed off `competition_id` generically, with no E0-specific special-casing anywhere in that path. The deeper claim — that a SWE forecast actually scores against SWE's own `model_selection.yaml` context, distinct from E0's — was already covered by `tests/test_per_competition_context.py::test_forecast_upcoming_real_sweden_registration` (landed on the forecast-engine side under US#131 before this story was written); a near-duplicate test was written, confirmed passing, then deleted in favor of citing the existing one rather than duplicating coverage.

One genuine gap found and closed: `test_forecast_upcoming_swe_omits_a_target_it_has_no_registered_model_for` (`tests/test_per_competition_context.py`) proves that a target absent from a competition's registered model set (e.g. `home_corners` for SWE, which has no `hc` source column) is silently omitted from `forecast_upcoming()`'s result rather than raising — purely from `_load_context_models`'s existing "entry is `None` → skip" behavior, no production code changed.

### 22.2 Swedish Pilot Corpus — Blocked (A36, BUG-018)

A36 (pilot SWE snapshot corpus + baseline backtest, mirroring A20/A21 for E0) is **active, not completed**. `agent-snapshot --league SWE --from-date 2026-07-01 --to-date 2026-07-20 --dry-run` confirmed 24 fixtures; the real run reported a clean `Processed: 24 | Errors: 0 | Skipped: 0`. Directly inspecting `data/agent_snapshots/` (per this story's own acceptance criteria, not just trusting the exit code — A20's BUG-011 precedent) found the corpus is **not** actually clean: only 10/24 match directories have real tool-response files, 1/24 has the exact BUG-011 shape (`_complete.json` present, zero content), and 13/24 have no directory at all despite the CLI reporting `OK` for each.

Filed as **BUG-018** (`documents/bugs.md`, open) — not yet root-caused; suspected (not confirmed) to be related to this run sharing the machine's single-slot local Ollama instance with a separate, concurrently-running `agent-snapshot --league E0` process. `agent-backtest` was **correctly never run** against this corpus — `BacktestHarness` raises `SnapshotMissingError` by design for any match without a full recording, so a run over an incomplete corpus would abort, not silently produce a wrong baseline (the A21/BUG-011 lesson applied correctly here). The 10 genuinely-complete matches do independently confirm the SWE agent pipeline runs correctly end-to-end against real matches, consistent with 22.1's mocked coverage — real signal, but not sufficient to satisfy A36's acceptance criteria. Re-run once BUG-018 is understood or the local Ollama instance isn't contended by another job.

### 22.3 Structured-Output Content Grounding — Open Gap (A38)

Discovered 2026-07-26 investigating an `insufficient_data` report for Burnley vs Bournemouth. That specific report turned out to be an unrelated finding — a stale pre-A31 `recommendation_cache.db` row (Section 17) — but live-regenerating the same match through the current, correctly-routed pipeline (confirmed via `raw_matches`: `resolve_competition`/`research`/`forecast` all ran against the real fixture, a real XGBoost forecast was computed) returned a fully schema-valid recommendation entirely about a **different, invented match** (`"home": "Manchester City", "away": "Liverpool", "date": "April 17, 2026"`, with matching invented markets).

A37's `with_structured_output()` (Section 8a) guarantees JSON *shape* — correct field names/types/enums — never that the *content* is about the requested match: `MatchRecommendationModel.match` is a bare, content-unchecked `dict`, and market `market`/`selection` fields are unconstrained strings. This is the same class of gap A30 already closed for diagnostics (never trust the LLM's own `prediction_basis`/`cold_start_risk` prose over deterministic pipeline state, Section 19.2) — A38 is the still-open story to extend that same "deterministic state wins, LLM prose is never authoritative" philosophy to match/market identity: deterministically overwrite `match` from `match_info` (straightforward, since it's a wholesale-replaceable dict), and at minimum detect-and-flag (e.g. a `limitations` entry, a confidence downgrade, or forcing `insufficient_data`) for markets-list grounding, which is a harder problem than `match` since `selection`/`market` strings can't simply be overwritten with a known-correct value the way `match` can. Not yet implemented — tracked here rather than silently assumed fixed by A37.

## 23. La Liga Verification (A49–A51, 2026-08-07)

Mirrors Section 22 (Swedish League Verification) — motivated by the web app's La Liga integration (`documents/app_user_stories.md` Phase 15, W74–W82) and the ML-engine side's own La Liga registration (`documents/user_stories.md` Phase 21).

### 23.1 Third-Competition Routing Verification (A49)

`resolve_competition_node`/`forecast_node` (Section 19) needed no code changes to route `SP1` correctly — both already key off `competition_id` generically, confirmed by two new regression tests: `tests/test_agent_tool_selection.py::test_resolve_competition_recommends_forecast_league_for_sp1` (real, unmocked call against the live `config/competitions.yaml`) and `tests/test_agent_pipeline.py::test_forecast_node_calls_forecast_league_for_sp1` (mocked `ForecastService`, confirms `match_type="league"` is used, not the international fallback).

The other required half of this story: `"La Liga"` (the free-text name) was this codebase's standing example of an *unregistered* competition across 5 files — registering `SP1` doesn't retroactively change what those tests exercise (`gate_league`/`get_competition_definition` do exact-code lookups, no name normalization existed pre-A50), so they stayed technically valid but confusing (the codebase's own most prominent unregistered-league example was, in fact, now supported under a different string). Replaced with `"Bundesliga"`/`"D1"` (confirmed genuinely unregistered — the real registry has exactly `E0`, `SWE`, `SP1`, `international`) in 3 of the 5 files (`tests/test_agent_tool_selection.py`, `tests/test_agent_pipeline.py`, `tests/test_forecast_registry_fallback.py`); the 4th, `app/backend/tests/test_match_info_gating.py`, turned out to already be correctly handled by an earlier story (W75) — `gate_league` matches by code not free-text name, so `gate_league("La Liga") is None` stayed genuinely correct even with `SP1` registered, and W75 had already added `D1` in `SP1`'s old stock-example place. New explicit `SP1`-is-registered cases added alongside the retired examples, not just a find-and-replace.

### 23.2 Free-Text League-Name Normalization (A50)

`resolve_competition`'s own docstring told the calling LLM `'La Liga'` was a valid example input, but the registry only ever matched exact codes — an agent following the tool's own documented example for the very league this phase adds would get `general_purpose` back forever, even after `SP1` was fully registered and trained.

Chose the deterministic code fix over a prompt-only one: this environment's real provider (`config/agent_config.yaml`, local Ollama `llama3.1:8b`) is documented elsewhere in this file (Section 17/BUG-019) as unreliable at strict instruction-following, so depending on a *new* prompt instruction being reliably followed would have been a shakier fix than the existing evidence already argues against.

Root-cause placement mattered here: the naive fix (alias table only inside `_resolve_competition_impl`, `src/agent/tools.py`) would have left a real gap, because `ForecastService.forecast_upcoming()` (`src/forecast/forecast_service.py:356`) calls the *same* `get_competition_definition()` independently for its own tier lookup — fixing only `resolve_competition` would still leave `forecast_upcoming(league="La Liga", ...)` silently degrading to `market_odds_only`. The alias table (`COMPETITION_NAME_ALIASES`) lives in `get_competition_definition()` itself (`src/logic/competition_registry.py`) — the one choke point both callers route through — tried only as a case-insensitive fallback after the exact-code lookup misses, so a real registered code is never shadowed. `_resolve_competition_impl` now echoes the *resolved* code (e.g. `"SP1"`) in its `"competition"` JSON field rather than the caller's raw input, and its docstring tells the calling LLM to reuse that field for `forecast_league`'s own `league` argument (which has always required a code, never a free-text name, and was not changed).

Explicitly scoped out: `forecast_node`'s own deterministic-pipeline path (`src/agent/pipeline.py`) passes `match_info["league"]` straight to `forecast_league`, bypassing `resolve_competition`'s resolved-code field entirely — a non-issue in practice since `match_info["league"]` is always populated by the app's own code-matching `gate_league()` or the CLI's `--league` flag, never free text, but a real, intentionally-undone piece of full generality flagged here rather than silently expanded into.

### 23.3 La Liga Pilot Corpus and Baseline Backtest (A51)

Unlike A36's Sweden attempt (Section 22.2, still blocked by BUG-018's intermittent recording gap), this pilot ran clean. `agent-snapshot --from-date 2026-05-23 --to-date 2026-05-24 --league SP1 --dry-run` confirmed 10 fixtures (the season's final two matchdays, the most recent finished SP1 matches in `raw_matches`); the real run reported `Done. Processed: 10 | Errors: 0 | Skipped: 0`.

Per A51's explicit caution (BUG-018's lesson: never trust a clean CLI exit code alone), every one of the 10 resulting `data/agent_snapshots/SP1/<match_id>/` directories was directly inspected, not just the log: all 10 have exactly 4 real tool-response files (`resolve_competition`, `forecast_league`, 2×`web_search`) plus a non-empty `_complete.json` (52 bytes each, `{"completed_at": ...}`) — no BUG-018 recurrence this time, no partial/empty directories. Spot-checked one `resolve_competition` response directly: `{"competition": "SP1", "tier": "competition_specific", "recommended_tool": "forecast_league"}` — real, correct content, and (post-A50) already showing the resolved-code echo behavior.

`agent-backtest --stake-mode flat` and `--stake-mode kelly` both ran to completion against the corpus (no `SnapshotMissingError`, confirming the corpus really is complete):

| Metric | flat | kelly |
|---|---|---|
| matches_evaluated | 10 | 10 |
| bets_placed | 3 | 3 |
| bets_won | 0 | 1 |
| roi | -1.0 | -0.939 |
| hit_rate | 0.0 | 0.333 |
| bet_frequency | 0.3 | 0.3 |
| max_drawdown | 0.03 | 0.156 |
| ending_bankroll (start 1000) | 970.0 | 843.81 |

**Explicitly not compared apples-to-apples with E0 (Section 18) or SWE's own partial signal (Section 22.2)**, per this story's own acceptance criteria — different league, a 10-match sample sized for wall-clock feasibility (not a full-season backtest), and a genuinely weak baseline result (0-1 wins from 3 bets across both modes) that is not surprising or concerning at this sample size.

**One honest, unexplained-but-expected observation, not investigated further**: `bets_won` differs between the flat and kelly runs (0 vs 1) despite both replaying the *identical* recorded tool-response corpus for the *identical* 10 matches. This is consistent with the LLM's own final synthesis call being outside what `SnapshotStore` replays (only `resolve_competition`/`forecast_league`/`web_search` — the deterministic-pipeline tool calls — are snapshotted; the LLM's own recommendation-generation turn is a fresh call each run) combined with this environment's local Ollama model's known non-determinism (Section 17) — i.e., the same underlying evidence can still produce a different `overall`/market selection across two separate agent-backtest invocations. Not a defect in the SP1 integration itself; flagged here as a real characteristic of this environment's baseline-reproducibility ceiling, worth keeping in mind for any future SP1 backtest comparison.

**Secondary finding from the same investigation, not yet addressed:** `agent_config_hash` (Section 12.2) hashes only `config/agent_config.yaml`'s fields, never the graph/pipeline *code* itself, so the app's `recommendation_cache.db` has no way to detect that Phase 11's 2026-07-22 restructure changed what a cached row actually means. 33 of 136 sandbox cache rows predated the A31 commit and were purged as a one-off cleanup during this investigation, not a systematic fix — see Section 17.

---

## 24. Backtest/Train Replay Leakage: Guard Then Redaction (A46–A47, 2026-07-28/29)

### 24.1 Confirming the Leak and a Prompt-Level Guard (A46)

Triggered by a direct user request to check for data leakage after DeepSeek's E0 train-split ROI (Section 21.4) looked implausibly high. Two vectors were checked with hard evidence, not just code review. **Forecast model training-cutoff leakage — ruled out:** every E0 target's deployed model metadata (`.metadata.json`, not just MLflow tags — MLflow had a real logging gap for `btts` specifically) shows `training_cutoff` in 2023-04/05, well before the 2025/26 season being backtested. **Web-search leakage into replay — confirmed real and structural:** the recorded snapshot for a real backtested match (Sunderland vs Wolves, 2025-10-18) showed its "recent form" search, despite the `before:2025-10-18` query filter, returning a BBC Sport recap naming goalscorers and an ESPN result titled with the literal final score. Traced to a code gap, not bad luck: `agent-snapshot` (record mode) has always instructed the LLM via `extra_system_instructions` to discard any web_search result mentioning a final score, but `process_match_row()` — the one shared replay path for `agent-backtest`/`agent-train`/`agent-compare` — never passed it, so every backtest/train run to date had zero defense against leaked content already sitting in the recorded snapshots.

A held-out comparison on the same E0/DeepSeek config sharpened the concern: train ROI +22.0% (43 bets, 69.8% hit rate, from Section 21.4) vs. test ROI **-11.0%** (7 bets, 57.1% hit rate) — reported to the user with the caveat that 7 bets is too small to be conclusive on its own, and that the leakage mechanism isn't split-aware (it doesn't specifically enrich train over test), so this divergence doesn't by itself localize the leak. Both findings independently pointed the same direction: the train-set ROI number was not trustworthy.

**Fix:** new `LEAKAGE_GUARD_INSTRUCTIONS` constant (`src/agent/backtest.py`), passed as `extra_system_instructions` on both of `process_match_row()`'s `run_agent()` calls; `main.py`'s `run_agent_snapshot()` refactored to build its addendum from the same constant instead of an independent inline copy, so record and replay instructions can never drift apart again. Explicitly scoped as a mitigation for *future* replay runs, not a corpus cleanup — already-recorded leaked content stayed physically present in the snapshot files. 2 new tests in `tests/test_backtest.py`; full suite 900 passed / 1 skipped, zero regressions.

### 24.2 Code-Enforced Redaction and Corpus Remediation (A47)

Direct follow-up after quantifying A46's residual gap: scanning the full E0 corpus (hand-labeled 10-match sample, 6/10 genuine leaks, 4/10 false positives from coincidentally score-shaped head-to-head tables) found roughly 19% of matches had real leaked-result content in `research_node`'s deterministic evidence — content A46's prompt-level guard could only ask the model to disregard, not remove.

**Design:** new `_looks_like_post_match_result(title, content)` (`src/agent/tools.py`) filters individual Tavily results *before* they're joined into the response `_web_search_impl` returns — generic recap-language markers, or a score-shaped digit pattern in the **title** specifically (not the full body), deliberately never tied to any known score, so the same check applies identically to a genuinely-upcoming live match and to backtest/train replay of an already-played one. The title-only restriction is the precision lever: every genuine leak in the hand-labeled sample had the score in the title, a recap marker, or both; the false positives were score digits buried in table/list content, which the title-only check correctly ignores — verified against all 10 hand-labeled examples before wiring anything further (10/10 genuine leaks flagged, 4/4 known false positives left alone). Applied at the `_web_search_impl` level, so it covers both `research_node`'s deterministic calls and the LLM's own optional `web_search` call uniformly, live or replay. Marked `# ponytail: naive heuristic, not a precise classifier` in code — an upgrade path (a smarter title-vs-table classifier) is noted, not built.

**Remediation:** the new filter was re-run retroactively against the already-recorded E0 corpus — 159 matches flagged (134 train-split, 25 test-split) — those 159 directories were deleted entirely and `agent-snapshot` re-run over the full original date range, which (thanks to the existing `_complete.json`-marker skip logic) only re-recorded the deleted matches. 181/202 re-recorded successfully; the remaining 21 (clustered 2026-05-13 to 2026-05-24) failed on a hard Tavily free-tier plan cap (confirmed via a direct single-match retry, not a transient rate limit) and were left flagged as genuinely incomplete rather than retried indefinitely. Re-scanning the corpus post-remediation found **0 leaky matches remaining** among everything that did get re-recorded. Side effect: re-recording also incidentally repaired 43 pre-existing matches that had no `_complete.json` marker for reasons unrelated to leakage, so the corpus ended up more complete than before this story, not just cleaner.

7 new tests in `tests/test_agent_tools_snapshot.py` (5 unit tests against real leaked/clean examples from the A46 investigation, 2 integration tests on `_web_search_impl`); full suite 907 passed / 1 skipped, zero regressions.

**Fresh train/test backtest on the cleaned corpus:** train ROI **+25.9%** (285 evaluated, 29 bets, 75.9% hit rate) vs. test ROI **+46.9%** (75 evaluated, 8 bets, 87.5% hit rate) — both splits now agree in direction, a materially more coherent result than A46's train/test contradiction. **Explicitly not claimed as proof of a real edge:** bet counts remain tiny (29 and 8), and this same backtest has now produced three materially different ROI readings across this investigation on nominally the same corpus/config purely from LLM run-to-run variance at `temperature=0.1` (train: +22.0% → -0.4% → +25.9%; test: -11.0% → -9.1% → +46.9%) — that run-to-run swing is itself the dominant source of uncertainty in these numbers, larger than what fixing leakage alone changed. Establishing a genuine edge would need repeated backtests to see the ROI distribution, `temperature=0` for reproducibility, or a substantially larger evaluated sample — none of which this story attempted, since it was scoped to data quality, not edge validation.

## 25. Bullet-Point Explanation, and a Structural Snapshot-Staleness Finding (A55, 2026-08-08)

### 25.1 `explanation` as a List, Not a Paragraph (A55)

Direct user request: `explanation` read as one dense narrative paragraph mixing the value-edge math, team news, form, and market caveats into a single run-on block. `MatchRecommendation`/`MatchRecommendationModel` (`src/agent/schema.py`) both changed `explanation: str` → `list[str]`, mirroring the existing `limitations: list[str]` field's own shape rather than inventing a new convention. New public `normalize_explanation()` — moved out of `extract_recommendation`'s own pipeline specifically so `app/backend/recommendations.py` can reuse it — coerces a plain string to a single-item list and drops blank items; called *before* `MatchRecommendationModel.model_validate()`, not after, so a plain-string response (a pre-A55 cached row, or a model that ignores the updated prompt) doesn't fail structural type validation before it ever gets a chance to normalize.

`config/prompts/agent_v1.txt`'s schema block and instructions updated to ask for an array, one bullet per aspect. **Direct follow-up (same day)**: the first pass over-included — one bullet per aspect *evaluated*, not one bullet per aspect that actually justifies the recommendation (a real `no_bet` example produced 6 bullets walking through every market's own math individually). Revised to explicitly scope the array to *justifying* reasoning only: for `direct_bet`/`conditional`, just the reasoning behind the specific recommended market(s), not an analysis of every market considered and rejected; for `no_bet`/`insufficient_data`, the key reason(s) nothing cleared the bar, not a per-market walkthrough. Verified live: the same match class dropped from 6 bullets to 4 genuinely load-bearing ones. This narrowing is prompt-only, same residual instruction-following uncertainty as every other prompt-only rule in this doc (Section 17) — there is no code-side enforcement that the LLM actually keeps bullets scoped this way, only that they arrive as a well-typed list at all.

Full suite: 995 passed / 1 skipped; frontend `tsc --noEmit` clean, 100+/102 Vitest (one pre-existing, unrelated timing flake).

### 25.2 Structural Finding: `SnapshotStore` Has No Staleness Detection (BUG-036)

Found live, twice, in the same investigation session, via direct user screenshots of cards that should have looked healthy after this session's other fixes had already landed. Both instances had the identical shape: a `forecast_league` snapshot recorded *before* a real underlying fix (SP1's `model_selection.yaml` reaching a consistent state; BUG-029's model-path fix) kept replaying its now-disproven content indefinitely *after* the fix went live, because nothing about fixing the underlying model/code retroactively touches an already-recorded snapshot file, and `SnapshotStore` prefers a replay hit over ever calling the live tool again once a match has any recording on disk.

**(1)** A La Liga (SP1) card showed "Cold start — thin history" mid-season, when every other SP1 fixture correctly used the real trained model — its snapshot, recorded 03:11 that morning (before SP1's model config was consistent), kept serving `model_version: "forecast_suite_international_v1"`/`feature_completeness: 0.31` forever after. **(2)** An E0 card showed "Insufficient Data" ("the ML forecast only provides BTTS probabilities") — its snapshot was recorded during BUG-029's own open window (7 of 8 E0 targets silently missing). A full scan of `data/agent_snapshots/sandbox/` for the single-target signature found **43 E0 directories** carrying it, spanning 2026-07-25 through 2026-08-07.

**Fix applied to both, the only fix available today**: delete the stale snapshot directory/directories, re-run `--precompute` so the match(es) record fresh under current code. Verified live for both — real forecasts, real `direct_bet`/correct `cold_start_risk` afterward. **The structural gap itself is not fixed** — this is the exact same class of problem this doc already flags for `RecommendationCache`/`agent_config_hash` (Section 23.3/BUG-029's own investigation note: the cache hashes config fields, never pipeline code), just one layer lower, at the tool-response snapshot rather than the final recommendation. Any future fix to `forecast_league`/`forecast_international`/`resolve_competition`'s underlying behavior carries this same silent-staleness risk for every sandbox match already recorded before it, with no automated detection — the only mitigation available today is a manual scan-and-purge like the one that found these 43, which nothing in a normal `--precompute` run or dashboard session would ever trigger on its own. Worth a dedicated future story (e.g. recording each snapshot's own code/config fingerprint alongside it, mirroring what `agent_config_hash` already does one layer up) if this recurs.

## 26. Serie A, Bundesliga, Ligue 1 Verification, and a Full-Season Corpus (A58–A63, 2026-08-16/17)

Mirrors Section 22 (Swedish League Verification) and Section 23 (La Liga Verification) — motivated by the web app's three-league integration (`documents/app_user_stories.md` Phase 30) and the ML-engine side's own registration (`documents/user_stories.md` Phase 27). Same retirement problem Section 23.1 hit, one level deeper: A49 had replaced `"La Liga"` with `"Bundesliga"` as this codebase's stock unregistered-competition test fixture — once this phase registers `D1` (Bundesliga) for real, that fixture goes stale exactly the same way `"La Liga"` did.

### 26.1 Third-Competition-Set Routing Verification and Fixture Retirement (A58)

`resolve_competition_node`/`forecast_node` needed no code changes to route `I1`/`D1`/`F1` correctly, same generic `competition_id`-keyed behavior Section 23.1 already confirmed for `SP1` — one parametrized regression body covers all three rather than tripling the SP1-specific test shape. The retirement half: grepped fresh (not assumed unchanged from A49's own hit list) for every `"Bundesliga"` reference — 4 real stock-placeholder hits (`tests/test_agent_tool_selection.py`, `tests/test_agent_pipeline.py`, `tests/test_forecast_registry_fallback.py`, `tests/test_competition_registry.py`) plus 3 genuine data-source hits (`tests/test_understat.py`, `src/ingestion/understat/fetcher.py`, `src/ingestion/fotmob/fetcher.py`) correctly left untouched. Replaced with `"Eredivisie"` (confirmed genuinely still unregistered against the live registry) in the first 3 files — `test_competition_registry.py`'s own `"Bundesliga"` hit tests the alias-*rejection* mechanism itself, a distinct concern, deferred to A59.

### 26.2 Free-Text Aliases for All Three (A59)

Data-only change to `COMPETITION_NAME_ALIASES` (`src/logic/competition_registry.py`) — `"serie a": "I1"`, `"bundesliga": "D1"`, `"ligue 1": "F1"` — reusing A50's existing case-insensitive-fallback mechanism unchanged, no design decision needed. Expected-consequence fix: `test_get_competition_definition_still_rejects_a_genuinely_unregistered_free_text_name` asserted `"Bundesliga"` still raises — now false, since it's a real alias — retired the same way A58 retired it as the routing stock example, this time for alias-rejection specifically; replaced with `"Eredivisie"`.

### 26.3 Pilot Corpus and Baseline Backtest (A60)

Operational only (`agent-snapshot`/`agent-backtest` already `--league`-parameterized), gated on A58 so each corpus reflects verified-correct routing. Pilot window = each league's final real matchday: `I1` 2026-05-24 (7 fixtures), `D1` 2026-05-16 (9), `F1` 2026-05-17 (9) — 25 matches total, all processed with 0 errors, every directory directly verified to have real tool-response files plus a genuine `_complete.json` (BUG-018's standing lesson, per Section 23.3/A51's own precedent). Backtest results reported honestly per league, not averaged: `I1` — both stake modes placed 0 bets (a genuine "no market cleared the value-edge bar" result on this 7-match sample, not a data gap); `D1` — flat 3 bets/2 won/ROI +36.7%, kelly 1 bet/1 won/ROI +160.0%; `F1` — flat 6 bets/2 won/ROI +10.0%, kelly 5 bets/2 won/ROI +18.5%. Explicitly not treated as evidence of a real edge — same 7–9-match single-matchday caveat E0/SP1's own pilots (A21, A51) always carry.

### 26.4 Full-Season Corpus and I1/D1 Train/Test Evaluation (A61–A63)

Direct follow-up, same day — A60's pilot was an explicit smoke test, not the real corpus. `agent-snapshot` run to completion over each league's *entire* 2025/26 season (not just the final matchday): real, directly-verified totals `I1` 380, `D1` 306, `F1` 306 (992 combined), matching each league's real `raw_matches` row count exactly (US#163) — not a trusted CLI exit code. Recording was interrupted multiple times by Tavily web-search quota errors (`research_node`'s real API calls; the deterministic pipeline itself never failed) — per explicit user instruction, paused and reported at each quota error rather than silently retrying, resumed across 3 different user-supplied Tavily API keys. The `_complete.json` idempotency marker correctly skipped every already-finished match across all three key-swaps — no partial/false-complete directories found on resume.

`agent-train` (critic mode) + `agent-backtest --split test` then ran for `I1` and `D1` specifically (by explicit user scoping — `F1`'s corpus is recorded but deliberately left unprocessed): an ≈80/20 train/test split for each, generating one pending lesson candidate per train-split match (never auto-approved — same human-review gate A39 established). **`I1`**: 305 train / 75 test. Train-split backtest (byproduct of the critic-mode run): 170 bets, ROI −2.72%. Held-out 75-match test split — flat: 51 bets, 30 won, ROI **−2.65%**, hit rate 58.8%; kelly: 20 bets, 12 won, ROI **−24.23%**, hit rate 60.0%. Kelly's higher hit rate did not translate to a better ROI — its larger stake sizing amplified variance on a 20-bet sample into a real double-digit drawdown, consistent with kelly's known volatility profile rather than a bug. **`D1`**: 249 train / 57 test. Train-split backtest: 186 bets, ROI +15.22%. Held-out 57-match test split — flat: 39 bets, 28 won, ROI **+29.41%**, hit rate 71.8%; kelly: 15 bets, 11 won, ROI **+40.62%**, hit rate 73.3%. D1's result reads notably stronger than I1's on the identical methodology — consistent with US#167's own already-flagged observation that D1's `result_3way` model measures differently from I1's, not confirmed as a durable edge on a 57-match sample.

**Outstanding:** 554 pending lesson candidates (305 `I1` + 249 `D1`) sit in `agent_lessons` (`status=pending`) awaiting human review via `agent-lessons approve/reject` — explicitly left for the user, not auto-approved. `F1`'s 306-match corpus is complete but has no train/test pass yet.

Full story-level detail lives in `documents/agent_user_stories.md` Phase 19 and its full-season-corpus addendum.

## 27. Retry on the LLM Call, and EOD/Pregenerate Dedup (A64/W151, 2026-08-17)

Found live while investigating two real Dashboard fixtures stuck on "Not yet generated" (Atleti v Málaga, Marseille v Strasbourg) — every other external call in this graph already degrades gracefully on failure (`_web_search_impl`/A53, `_forecast_league_impl`/`_forecast_international_impl`, `resolve_competition`), but `agent_node`'s own call to the LLM provider (`llm_with_tools.invoke(...)`) and `output_node`'s forced-synthesis fallback call had none — a single transient provider error (timeout/rate limit/5xx) propagated all the way to `eod_batch.py`'s per-match `try/except`, silently skipping the match until its *next* scheduled window (T-30, potentially days away for an early fixture).

### 27.1 `_invoke_with_retry()` (`src/agent/graph.py`, A64)

Both call sites now go through a small module-level helper — `_invoke_with_retry(runnable, messages, attempts=3)` — that retries the identical `.invoke()` call up to 3 times, re-raising the last exception unchanged if every attempt fails (callers see the same failure mode as before, just less often). Deliberately a hand-rolled loop, not LangChain's own `Runnable.with_retry()`: the native wrapper returns a *new* `Runnable` instance, which is transparent in production but breaks every existing test double that mocks at `mock_llm.bind_tools.return_value.invoke` — the hand-rolled version keeps calling `.invoke()` on the exact same object, so it composes with the existing mock shape with zero collateral test changes. No backoff between attempts (batch concurrency is already bounded by `eod_batch.py`'s own semaphore, so this isn't hammering the provider) — flagged as the deliberate corner cut, add exponential backoff if a real sustained outage (not a one-off blip) starts exhausting all 3 attempts in practice.

### 27.2 EOD/pregenerate dedup — `already_fresh()` (`app/backend/eod_batch.py`, W151)

Separate but related root cause for the same symptom class: boot-time pregenerate (`main.py::_pregenerate_recommendations`, W103) generates every fixture in the next 5 days on deploy, and the nightly EOD batch (`scheduler_wiring.py`, W09) — which reuses `run_eod_batch()` internally — regenerated the *same* fixture again the night before kickoff regardless, a redundant (real-money) LLM call per fixture. `already_fresh(cache, match_id, date, agent_config_hash, odds)` extracts the odds-unchanged comparison `t30_refresh.py`'s `refresh_match_at_t30()` already used for its own "skip if nothing changed" check (Section 6.2 of `app_techspec.md`) into a function shared by both call sites — `eod_batch.py::_generate_one` checks it before calling `run_agent()`, and `t30_refresh.py` now calls the same helper instead of its own inline copy. `EodBatchResult` gained a third counter, `unchanged`, distinct from `skipped` (a real `run_agent()` failure) — surfaced through to `_pregenerate_recommendations()`'s per-league results dict so a batch-result log line tells the two apart without reading warnings. T-30 is still scheduled unconditionally for every fixture either way, unchanged.

Net effect of both fixes together: a fixture gets at most one real LLM call per scheduled window unless odds actually moved, and a transient failure within that one call is very likely absorbed by the retry instead of waiting for the next window — directly serving the original ask ("recommendation should show up ASAP after refresh").

Full story-level detail lives in `documents/agent_user_stories.md` A64 and `documents/app_user_stories.md` W151.
