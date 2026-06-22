# Technical Specification — FPAI Betting Agent

Authoritative implementation reference for the agent described in `agent_prd.md`. This document reflects the actual code as built through Phase 5 (M1–M5 / stories A01–A16). Phase 6 (batch recommendation, A18) is **not yet implemented** — see [Implementation Status](#implementation-status) below.

---

## 1. Module Structure

```
src/agent/
  __init__.py
  agent_config.py     # AgentConfig dataclass + YAML loader
  schema.py            # MatchRecommendation/MarketRecommendation TypedDicts + JSON extraction
  tools.py             # web_search, forecast_league, forecast_international — all routed through SnapshotStore
  graph.py             # AgentState, build_graph(), run_agent()
  snapshot_store.py    # SnapshotStore, SnapshotMissingError — record/replay interceptor (A09)
  backtest.py          # BacktestRecord, load_outcome(), process_match_row(), BacktestHarness (A12)
  staking.py           # BetOutcome, BankrollResult, simulate_flat_stake(), simulate_kelly_stake() (A13, A15)
  evaluation.py        # compute_max_drawdown(), build_evaluation_report(), config_hash(), save_report(), print_report() (A13)
  comparison.py        # compare_configs(), print_comparison_table(), save_comparison() (A16)

config/
  agent_config.yaml    # tunable knobs (model, temperature, thresholds, markets)
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

`config_hash` is computed over exactly `AgentConfig`'s tuning-relevant fields (`model`, `provider`, `temperature`, `max_tool_calls`, `min_odds_threshold`, `min_value_edge`, `markets` sorted, `system_prompt_version`) — every field on the dataclass, so two configs differing in any tunable knob get distinguishable hashes, and `markets` order never matters. `save_report` filenames collide if two runs of the *same config* complete within the same UTC second — accepted as a non-issue for this tool's manual, human-paced usage pattern (a single backtest run takes at minimum seconds to minutes).

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

**A17 dependency resolved.** A17 (this document) formally depends on A08 *and* A14; both are now implemented, so this document covers the full backtesting/evaluation/comparison surface (Sections 9–14) it previously only described as design intent. `agent_prd.md` §7's CLI command examples (`agent-snapshot`, `agent-backtest`) match the flags actually implemented; §8 does not yet show the `agent-compare` example added in A16 — a documentation gap in `agent_prd.md`, not in this document or the code.

**LangSmith tracing:** referenced as an acceptance criterion in A01, but no `LANGCHAIN_API_KEY`/`LANGCHAIN_TRACING_V2` configuration exists in `.env` or anywhere in the codebase as of this writing. Diagnostic visibility currently comes entirely from the `_LOG.info()` calls in `graph.py` (Section 3), not from LangSmith. If LangSmith is set up later, the manual logging in `graph.py` should probably stay regardless — it proved more useful for diagnosing the model-selection issues in Section 7 than a trace UI would have been, since it could be grepped directly from CLI output.

---

## 17. Known Limitations

- `min_odds_threshold` / `min_value_edge` are prompt-level instructions, not code-enforced gates. A future stronger validation layer could reject/flag `direct_bet` recommendations that violate them programmatically.
- BUG-010 (league-context models not trained) is mitigated by the `forecast_league` fallback (Section 4.2) but not resolved. Every league-context call currently silently degrades to market-odds-only predictions. Run `python main.py train-forecast-suite --context league && python main.py select-best-models --context league` to fix at the source.
- No automated test exercises a live Ollama call — `tests/test_agent_graph.py` mocks the LLM and `ForecastService`. There is no CI coverage for "does the currently-configured model actually produce valid output," only for the graph routing logic and parsing functions in isolation. The same is true of the backtest stack: no test runs a real `agent-snapshot` → `agent-backtest` cycle end-to-end against a live model, only mocked unit tests per module.
- `extract_recommendation` does not validate `recommendation_type`, `confidence`, or numeric field types beyond presence — a model could emit `value_edge: "high"` (a string) and it would pass schema extraction. This flows downstream: `staking.py` calls `float(m["current_odds"])`/`float(m.get("value_edge", 0.0))`, which would raise `ValueError` at backtest time on such a malformed value rather than failing earlier at extraction time.
- **Corners markets (`home_corners`/`away_corners`) cannot be scored in backtests.** `_market_correct` (Section 11.1) always returns `None` for them, so they are never staked and never contribute to ROI/hit-rate — `min_odds_threshold` enforcement only happens at recommendation time (the LLM, via the prompt), never validated against an actual outcome. Resolving this would require extending `MarketRecommendation` with a numeric line field and is out of scope for A09–A16.
- **`agent-compare` is slower and less fault-tolerant than `agent-backtest`** for equivalent match counts — it runs each config strictly sequentially via `BacktestHarness.run()` rather than reusing `agent-backtest`'s concurrent, per-match-fault-tolerant path (Section 14). A single missing snapshot aborts the whole comparison. Accepted scope boundary for A16, not a defect, but worth revisiting if comparison runs grow large.
- `save_report`/`save_comparison` filenames are timestamped to the second; two runs (of the same config, for `save_report`; of any size, for `save_comparison`) completing within the same UTC second will silently overwrite each other's report file. Low risk given this tool's manual, human-paced usage pattern.
- `agent-backtest --concurrency` is also capped by Python's default `ThreadPoolExecutor` size (`min(32, os.cpu_count() + 4)`) — values above that have no further effect on actual parallelism.
