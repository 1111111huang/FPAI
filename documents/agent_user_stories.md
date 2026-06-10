# FPAI Betting Agent User Stories

This document tracks story-level actionable items for the FPAI Betting Agent. Stories are prefixed `A##` to distinguish them from the ML forecasting engine stories. Default status is `active`. Completed stories are archived in `documents/agent_techspec.md` when that document is created.

---

## Story Dependencies & Execution Order

```
PHASE 1 (Foundation)
  A01 → A02 → A03

PHASE 2 (Live Recommendation)
  A03 → A04
  A03 → A05
  A03 → A06
  A04 + A05 + A06 → A07
  A07 → A08

PHASE 3 (Snapshot Infrastructure)
  A01 → A09
  A04 + A05 + A06 + A09 → A10
  A10 → A11

PHASE 4 (Backtest Harness)
  A11 → A12
  A12 → A13
  A12 + A13 → A14

PHASE 5 (Model & Prompt Tuning)
  A13 → A15
  A14 + A15 → A16
  A08 + A14 → A17

PHASE 6 (Batch Recommendation — Future)
  A08 → A18
```

**Size key:** XS < 2 hrs · S ≈ half day · M ≈ 1 day · L ≈ 2–3 days

---

## PHASE 1: Foundation

### A01 — Set Up Development Environment
**Size:** M | **Status:** active | **Milestone:** M1

Set up all dependencies required to run a LangGraph agent locally.

**Acceptance criteria:**
- `langgraph`, `langchain`, `langchain-anthropic`, `langchain-ollama`, `tavily-python` added to `requirements.txt`
- Ollama installed locally with `qwen2.5:7b` pulled and verified
- LangSmith API key configured; a test run of any LangChain call produces a visible trace in the LangSmith dashboard
- `src/agent/__init__.py` package created

---

### A02 — Implement AgentConfig
**Size:** S | **Status:** active | **Milestone:** M1 | **Depends on:** A01

Create a typed configuration object and its corresponding YAML file covering all agent tuning knobs.

**Acceptance criteria:**
- `AgentConfig` dataclass in `src/agent/agent_config.py` with fields: `model`, `provider`, `temperature`, `max_tool_calls`, `min_odds_threshold`, `min_value_edge`, `markets`, `system_prompt_version`
- `config/agent_config.yaml` with sensible defaults (Ollama / qwen2.5:7b, temperature 0.1, max_tool_calls 10, min_odds 2.0)
- `AgentConfig.from_yaml(path)` loads and validates the file; raises a clear error on missing required fields

---

### A03 — Build Stub StateGraph
**Size:** M | **Status:** active | **Milestone:** M1 | **Depends on:** A01, A02

Implement the LangGraph `StateGraph` skeleton with the correct node structure and stub tools so the agent loop can be exercised end-to-end before real tools exist.

**Acceptance criteria:**
- `AgentState` TypedDict defined in `src/agent/graph.py` with `messages`, `match_info`, `recommendation`, `tool_call_count`
- `StateGraph` with `agent_node`, `tools_node`, `output_node` nodes wired correctly
- Conditional edge: tool calls present AND under budget → `tools_node`; otherwise → `output_node`
- Stub `web_search` and `forecast_league` tools return hardcoded strings
- Running the graph against a hardcoded match dict produces a trace in LangSmith showing the full loop

---

## PHASE 2: Live Recommendation

### A04 — Implement web_search Tool
**Size:** M | **Status:** active | **Milestone:** M2 | **Depends on:** A03

Connect the Tavily search API as the agent's primary web search tool.

**Acceptance criteria:**
- `web_search(query: str) -> str` defined in `src/agent/tools.py` with `@tool` decorator
- Tavily client reads API key from environment variable `TAVILY_API_KEY`
- Tool description clearly states its purpose (odds discovery, team name lookup, injury news) so the agent uses it correctly
- Registered with the `ToolNode` in `graph.py`
- Manual test: agent successfully searches for current odds for a named match

---

### A05 — Implement forecast_league and forecast_international Tools
**Size:** M | **Status:** active | **Milestone:** M2 | **Depends on:** A03

Expose the two ML model contexts as distinct, named tools so the agent cannot accidentally call the wrong one.

**Acceptance criteria:**
- `forecast_league(home_team, away_team, date, league, odds_h, odds_d, odds_a, ...) -> dict` makes a direct call to `ForecastService.forecast_upcoming(match_type="league")`
- `forecast_international(home_team, away_team, date, odds_h, odds_d, odds_a, ...) -> dict` makes a direct call to `ForecastService.forecast_upcoming(match_type="international")`
- Tool docstrings clearly state when each should be used (league context vs. no team history available)
- Both tools registered with the `ToolNode`
- Both tools return the full forecast JSON dict including `data_quality.prediction_basis`

---

### A06 — Write System Prompt v1 and MatchRecommendation Output Schema
**Size:** M | **Status:** active | **Milestone:** M2 | **Depends on:** A03

Define the agent's betting philosophy in the system prompt and enforce a structured JSON output at the end of every run.

**Acceptance criteria:**
- System prompt stored as a text file at `config/prompts/agent_v1.txt`; loaded by `AgentConfig`
- Prompt specifies: search for odds first → call appropriate forecast tool → search for news → evaluate value → output JSON
- Prompt includes the `MatchRecommendation` JSON schema the agent must output in its final turn
- `MatchRecommendation` and `MarketRecommendation` TypedDicts defined in `src/agent/schema.py`
- `output_node` in `graph.py` parses and validates the final message; raises `RecommendationParseError` with the raw text if JSON extraction fails
- `overall` field is one of: `direct_bet`, `conditional`, `no_bet`, `insufficient_data`
- `recommendation_type` per market is one of: `direct_bet`, `conditional`, `no_bet`

---

### A07 — Wire Full Agent Graph and Validate End-to-End
**Size:** M | **Status:** active | **Milestone:** M2 | **Depends on:** A04, A05, A06

Connect all real tools into the graph and validate the full live recommendation workflow.

**Acceptance criteria:**
- Graph runs with real Tavily search, real `ForecastService` calls, and real LLM (first with Ollama, then with Haiku)
- Agent produces a valid `MatchRecommendation` JSON for a known upcoming match
- LangSmith trace shows all tool calls, inputs, outputs, and reasoning turns
- Agent correctly selects `forecast_league` vs `forecast_international` for a league and international match respectively
- `insufficient_data` recommendation is produced when odds cannot be found by web search

---

### A08 — Add agent-recommend CLI Command
**Size:** S | **Status:** active | **Milestone:** M2 | **Depends on:** A07

Expose the live recommendation workflow as a CLI command.

**Acceptance criteria:**
- `python main.py agent-recommend --home "Man City" --away "Arsenal" --date 2026-06-15` runs the full agent and prints the `MatchRecommendation` JSON plus explanation
- `--config` flag allows specifying a non-default `agent_config.yaml` path
- Exit code 0 on success; non-zero on agent error or parse failure

---

## PHASE 3: Snapshot Infrastructure

### A09 — Implement SnapshotStore
**Size:** M | **Status:** active | **Milestone:** M3 | **Depends on:** A01

Build the record/replay interceptor that isolates backtest runs from live APIs.

**Acceptance criteria:**
- `SnapshotStore` class in `src/agent/snapshot_store.py` with `set_mode(mode: Literal["record", "replay", "live"])` method
- `record` mode: executes the wrapped callable, serialises `{tool, inputs, response, recorded_at}` to `data/agent_snapshots/<match_id>/<tool>_<sha256_of_inputs>.json`
- `replay` mode: loads from disk; raises `SnapshotMissingError` immediately if file not found — no silent fallback to live call
- `live` mode: passes through to the real callable with no interception (default for `agent-recommend`)
- Key is a SHA-256 hash of the canonically serialised input dict, ensuring the same query always maps to the same file

---

### A10 — Integrate SnapshotStore with All Tool Functions
**Size:** S | **Status:** active | **Milestone:** M3 | **Depends on:** A04, A05, A06, A09

Wrap every tool function with the `SnapshotStore` interceptor and add date-filtering to web search during snapshot collection.

**Acceptance criteria:**
- `web_search`, `forecast_league`, and `forecast_international` all route through `SnapshotStore`
- During snapshot collection (`agent-snapshot`), web search queries automatically have `before:<match_date>` appended to reduce post-match leakage
- System prompt for snapshot collection run instructs the agent to discard any result that references a final score
- Switching mode from `live` to `record` to `replay` requires no changes to tool function code

---

### A11 — Add agent-snapshot CLI Command
**Size:** S | **Status:** active | **Milestone:** M3 | **Depends on:** A10

Expose snapshot collection as a CLI command that drives the agent in record mode over historical matches.

**Acceptance criteria:**
- `python main.py agent-snapshot --from-date 2025-01-01 --to-date 2025-06-01 --league E0` loads historical matches from DuckDB and runs the agent in `record` mode over each
- Skips matches that already have a complete snapshot directory
- Prints progress (match count, skipped, errors)
- `--dry-run` flag lists matches that would be processed without executing

---

## PHASE 4: Backtest Harness

### A12 — Implement BacktestHarness and Outcome Loader
**Size:** L | **Status:** active | **Milestone:** M4 | **Depends on:** A11

Build the core backtest engine that replays snapshot episodes and compares agent recommendations against actual outcomes.

**Acceptance criteria:**
- `BacktestHarness` class in `src/agent/backtest.py`
- Loads historical matches from DuckDB for a given date range and league
- For each match: sets `SnapshotStore` to `replay` mode, runs the agent, loads the actual outcome from `raw_matches`
- `BacktestRecord` datatype holds: match info, `MatchRecommendation`, actual outcome, and whether each market recommendation was correct
- Raises `SnapshotMissingError` at the harness level if any match lacks snapshots — does not silently skip
- `--sample N` runs a stratified random sample (balanced across bet / no-bet outcomes) before a full run

---

### A13 — Implement Flat-Stake Bankroll Simulation and Evaluation Metrics
**Size:** M | **Status:** active | **Milestone:** M4 | **Depends on:** A12

Simulate bankroll evolution over backtest records and compute the evaluation report.

**Acceptance criteria:**
- `flat` staking: configurable fixed stake (default 1% of starting bankroll) applied to every `direct_bet` recommendation
- Bankroll updated per bet: win → bankroll += stake × (odds − 1); loss → bankroll −= stake
- Evaluation report computed by `src/agent/evaluation.py` with: ROI, hit rate, bet frequency, max drawdown, bets placed, insufficient data rate
- Report saved to `reports/agent_backtest/<timestamp>_<config_hash>.json`
- Report also printed to stdout in a human-readable table

---

### A14 — Add agent-backtest CLI with Parallelism
**Size:** M | **Status:** active | **Milestone:** M4 | **Depends on:** A12, A13

Expose the backtest harness as a CLI command with concurrent execution support.

**Acceptance criteria:**
- `python main.py agent-backtest --from-date 2025-01-01 --to-date 2025-06-01 --stake-mode flat --sample 50` runs and prints the evaluation report
- `--concurrency N` flag (default 5) controls how many agent runs execute concurrently via `asyncio.gather`
- `--config` flag allows specifying a non-default `agent_config.yaml` for model/prompt comparison
- Progress bar shown during run (matches completed / total)

---

## PHASE 5: Model & Prompt Tuning

### A15 — Implement Kelly Criterion Staking
**Size:** S | **Status:** active | **Milestone:** M5 | **Depends on:** A13

Add Kelly criterion as a second staking mode.

**Acceptance criteria:**
- `kelly` staking: stake = `value_edge / (odds − 1)` × current bankroll, capped at 10% of bankroll per bet
- Selectable via `--stake-mode kelly` on `agent-backtest`
- Both `flat` and `kelly` produce comparable report formats for side-by-side analysis

---

### A16 — Build Config Comparison Framework
**Size:** M | **Status:** active | **Milestone:** M5 | **Depends on:** A14, A15

Allow systematic comparison of agent configurations (model, prompt version, staking) over the same snapshot set.

**Acceptance criteria:**
- `python main.py agent-compare --configs config/agent_v1.yaml config/agent_v2.yaml --from-date ... --to-date ...` runs each config over the same snapshots and outputs a side-by-side comparison table
- Comparison table includes: ROI, hit rate, bet frequency, max drawdown per config
- Results saved to `reports/agent_backtest/comparison_<timestamp>.json`

---

### A17 — Create agent_techspec.md
**Size:** M | **Status:** active | **Milestone:** M5 | **Depends on:** A08, A14

Document the implementation details discovered during M1–M4 in a formal technical specification.

**Acceptance criteria:**
- `documents/agent_techspec.md` created as the authoritative implementation reference
- Covers: `src/agent/` module structure, `StateGraph` node contracts, `SnapshotStore` file layout, `BacktestHarness` data flow, CLI command reference, LangSmith configuration
- Reflects actual implementation (written after M4, not before)
- `agent_prd.md` Extension Points section updated to mark `agent_techspec.md` as complete

---

## PHASE 6: Batch Recommendation (Future)

### A18 — Implement Batch Recommendation for Weekend Fixtures
**Size:** L | **Status:** future | **Milestone:** M6 | **Depends on:** A08

Allow the user to request recommendations for all upcoming fixtures in a league over a given weekend.

**Acceptance criteria:**
- `python main.py agent-batch --league E0 --weekend` fetches upcoming fixtures, runs the agent over each in parallel, and produces a ranked recommendation report (best value bets first)
- Fixture discovery uses web search or a configured fixtures API
- Report groups matches by date and highlights `direct_bet` recommendations at the top
- Respects the same `agent_config.yaml` knobs as `agent-recommend`
