# Technical Specification — FPAI Betting Agent

Authoritative implementation reference for the agent described in `agent_prd.md`. This document reflects the actual code as built through Phase 2 (M1–M2 / stories A01–A08). Phases 3–6 (snapshot infrastructure, backtest harness, model tuning, batch recommendation) are **not yet implemented** — see [Implementation Status](#implementation-status) below.

---

## 1. Module Structure

```
src/agent/
  __init__.py
  agent_config.py     # AgentConfig dataclass + YAML loader
  schema.py           # MatchRecommendation/MarketRecommendation TypedDicts + JSON extraction
  tools.py             # web_search, forecast_league, forecast_international
  graph.py             # AgentState, build_graph(), run_agent()

config/
  agent_config.yaml    # tunable knobs (model, temperature, thresholds, markets)
  prompts/
    agent_v1.txt       # system prompt — see Section 5

documents/
  agent_prd.md         # product requirements
  agent_user_stories.md  # story tracking (A01–A18)
  agent_techspec.md   # this document
```

No `SnapshotStore`, `BacktestHarness`, or backtest-related CLI commands exist yet (see Section 8).

---

## 2. AgentConfig

`src/agent/agent_config.py`

```python
@dataclass
class AgentConfig:
    model: str
    provider: Literal["ollama", "anthropic"]
    temperature: float
    max_tool_calls: int
    min_odds_threshold: float
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
min_odds_threshold: 2.0
min_value_edge: 0.05
markets: [result_3way, btts, total_goals, home_corners, away_corners]
system_prompt_version: "v1"
```

`min_odds_threshold` and `min_value_edge` are **not enforced in code** — they are values the LLM is instructed to apply via the system prompt (Section 5). There is no programmatic check that the model actually respected them.

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
Invokes `llm_with_tools.invoke(state["messages"])`. Increments `tool_call_count` by the number of tool calls in the response. Logs `tool_calls=[...]` or `raw_output_length=N` at INFO.

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
def run_agent(match_info: dict, config: AgentConfig | None = None, tools: list | None = None) -> MatchRecommendation
```

- `match_info` keys: `home_team`, `away_team`, `date`, optionally `league`, optionally `odds: {"home": float, "draw": float, "away": float}`.
- Builds the initial `HumanMessage` by interpolating match info directly into a sentence (not a template file). When `odds` is present, the exact decimal values are embedded in the prompt and the model is instructed to reuse them verbatim in forecast tool calls — without this, the model invents placeholder odds.
- Compiles a fresh graph per call (`build_graph()` is not cached/reused across calls).

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

---

## 5. System Prompt (`config/prompts/agent_v1.txt`)

Loaded by `AgentConfig.system_prompt_version` → `agent_{version}.txt`. Structure, in order:

1. **CRITICAL RULE** — never narrate a tool call as text, always either call a tool or output JSON
2. **Workflow** — call forecast tool first (with explicit fallback odds defaults if none provided), then web_search once, then output
3. **STOP RULES** — never call the same tool twice in a row; output JSON after 2 tool calls total; once a tool reports `TOOL_PERMANENTLY_UNAVAILABLE`, never call it again
4. **Value Calculation** — implied probability, value edge formula, the 2.0 minimum odds rule
5. **insufficient_data criteria** — explicit list of when to use this overall value
6. **Confidence Guidelines** — qualitative mapping of data completeness to `high`/`medium`/`low`
7. **Output Format** — the literal `MatchRecommendation` JSON schema as a fenced example

The STOP RULES section (item 3) was added after observing `llama3.2:3b` loop on `web_search` up to 10 times; it is the main lever for keeping weaker models within budget. If a future system prompt version removes it, re-test against a small model before assuming it's safe.

---

## 6. Environment Setup

```bash
# .env at project root (gitignored)
TAVILY_API_KEY=...
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

`extract_recommendation(text: str) -> MatchRecommendation`:

1. Collects all fenced ` ```json ` blocks, tried **last-to-first** (the model sometimes echoes an earlier tool result inside a fenced block before its real final answer — the last block is the one that matters).
2. Falls back to the outermost bare `{...}` via regex if no fenced block parses.
3. For each candidate, parses with `json.JSONDecoder().raw_decode()` (not `json.loads`) — tolerates trailing characters after the JSON object closes.
4. If `raw_decode` still raises, falls back to `json_repair.loads()`, which tolerates structurally broken JSON (mismatched/duplicated brackets, etc.). Only accepted if the result is a `dict`.
5. Validates all 7 required keys are present and `overall` is one of the four valid literals. Field values other than `overall` (e.g. `confidence`, `recommendation_type` inside markets) are **not validated** — a model could write `confidence: ""` and it would pass.
6. Raises `RecommendationParseError(raw_text, reason)` if no candidate validates, carrying the original text for debugging.

`json_repair` is a hard dependency (`requirements.txt`), not optional — without it, the multi-array bracket-nesting failure (Section 7, #6 follow-on) causes silent `insufficient_data` fallbacks even when the model's numbers were correct.

---

## 9. CLI Reference

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

---

## 10. Implementation Status

Reflects `documents/agent_user_stories.md` as of this writing.

| Phase | Stories | Status |
|---|---|---|
| 1 — Foundation | A01–A03 | ✅ Implemented |
| 2 — Live Recommendation | A04–A08 | ✅ Implemented (this document covers these) |
| 3 — Snapshot Infrastructure | A09–A11 | ⬜ Not started — no `SnapshotStore`, no `agent-snapshot` CLI |
| 4 — Backtest Harness | A12–A14 | ⬜ Not started — no `BacktestHarness`, no `agent-backtest` CLI |
| 5 — Model & Prompt Tuning | A15–A16 | ⬜ Not started — no Kelly staking, no config comparison framework |
| 6 — Batch Recommendation | A18 | ⬜ Future |

**Note on A17 dependency:** the user story A17 (this document) formally depends on A08 *and* A14. A14 (backtest harness CLI) does not exist yet, so Sections 7–8 of `agent_prd.md` (Backtesting & Evaluation, staking modes, evaluation metrics) describe **design intent only** — nothing in `src/agent/` implements them. This document was written early, against the live-recommendation path only, because that is what exists and what needed recording while the model-selection findings (Section 7) were fresh. **Re-open and extend this document once A09–A16 land** — do not treat it as covering backtesting just because the title says "techspec."

**LangSmith tracing:** referenced as an acceptance criterion in A01, but no `LANGCHAIN_API_KEY`/`LANGCHAIN_TRACING_V2` configuration exists in `.env` or anywhere in the codebase as of this writing. Diagnostic visibility currently comes entirely from the `_LOG.info()` calls in `graph.py` (Section 3), not from LangSmith. If LangSmith is set up later, the manual logging in `graph.py` should probably stay regardless — it proved more useful for diagnosing the model-selection issues in Section 7 than a trace UI would have been, since it could be grepped directly from CLI output.

---

## 11. Known Limitations

- `min_odds_threshold` / `min_value_edge` are prompt-level instructions, not code-enforced gates. A future stronger validation layer could reject/flag `direct_bet` recommendations that violate them programmatically.
- BUG-010 (league-context models not trained) is mitigated by the `forecast_league` fallback (Section 4.2) but not resolved. Every league-context call currently silently degrades to market-odds-only predictions. Run `python main.py train-forecast-suite --context league && python main.py select-best-models --context league` to fix at the source.
- No automated test exercises a live Ollama call — `tests/test_agent_graph.py` mocks the LLM and `ForecastService`. There is no CI coverage for "does the currently-configured model actually produce valid output," only for the graph routing logic and parsing functions in isolation.
- `extract_recommendation` does not validate `recommendation_type`, `confidence`, or numeric field types beyond presence — a model could emit `value_edge: "high"` (a string) and it would pass schema extraction.
