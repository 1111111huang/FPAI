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

PHASE 7 (Backtest Execution Readiness)
  A19 (independent — forecast-engine fix, blocks meaningful league results)
  A11 + A19 → A20
  A12 + A13 + A14 + A20 → A21
  A16 + A21 → A22 (Future)
```

**Size key:** XS < 2 hrs · S ≈ half day · M ≈ 1 day · L ≈ 2–3 days

---

## PHASE 1: Foundation

### A01 — Set Up Development Environment
**Size:** M | **Status:** completed | **Milestone:** M1

Set up all dependencies required to run a LangGraph agent locally.

**Acceptance criteria:**
- `langgraph`, `langchain`, `langchain-anthropic`, `langchain-ollama`, `tavily-python` added to `requirements.txt`
- Ollama installed locally with `qwen2.5:7b` pulled and verified
- LangSmith API key configured; a test run of any LangChain call produces a visible trace in the LangSmith dashboard
- `src/agent/__init__.py` package created

---

### A02 — Implement AgentConfig
**Size:** S | **Status:** completed | **Milestone:** M1 | **Depends on:** A01

Create a typed configuration object and its corresponding YAML file covering all agent tuning knobs.

**Acceptance criteria:**
- `AgentConfig` dataclass in `src/agent/agent_config.py` with fields: `model`, `provider`, `temperature`, `max_tool_calls`, `min_odds_threshold`, `min_value_edge`, `markets`, `system_prompt_version`
- `config/agent_config.yaml` with sensible defaults (Ollama / qwen2.5:7b, temperature 0.1, max_tool_calls 10, min_odds 2.0)
- `AgentConfig.from_yaml(path)` loads and validates the file; raises a clear error on missing required fields

---

### A03 — Build Stub StateGraph
**Size:** M | **Status:** completed | **Milestone:** M1 | **Depends on:** A01, A02

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
**Size:** M | **Status:** completed | **Milestone:** M2 | **Depends on:** A03

Connect the Tavily search API as the agent's primary web search tool.

**Acceptance criteria:**
- `web_search(query: str) -> str` defined in `src/agent/tools.py` with `@tool` decorator
- Tavily client reads API key from environment variable `TAVILY_API_KEY`
- Tool description clearly states its purpose (odds discovery, team name lookup, injury news) so the agent uses it correctly
- Registered with the `ToolNode` in `graph.py`
- Manual test: agent successfully searches for current odds for a named match

---

### A05 — Implement forecast_league and forecast_international Tools
**Size:** M | **Status:** completed | **Milestone:** M2 | **Depends on:** A03

Expose the two ML model contexts as distinct, named tools so the agent cannot accidentally call the wrong one.

**Acceptance criteria:**
- `forecast_league(home_team, away_team, date, league, odds_h, odds_d, odds_a, ...) -> dict` makes a direct call to `ForecastService.forecast_upcoming(match_type="league")`
- `forecast_international(home_team, away_team, date, odds_h, odds_d, odds_a, ...) -> dict` makes a direct call to `ForecastService.forecast_upcoming(match_type="international")`
- Tool docstrings clearly state when each should be used (league context vs. no team history available)
- Both tools registered with the `ToolNode`
- Both tools return the full forecast JSON dict including `data_quality.prediction_basis`

---

### A06 — Write System Prompt v1 and MatchRecommendation Output Schema
**Size:** M | **Status:** completed | **Milestone:** M2 | **Depends on:** A03

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
**Size:** M | **Status:** completed | **Milestone:** M2 | **Depends on:** A04, A05, A06

Connect all real tools into the graph and validate the full live recommendation workflow.

**Acceptance criteria:**
- Graph runs with real Tavily search, real `ForecastService` calls, and real LLM (first with Ollama, then with Haiku)
- Agent produces a valid `MatchRecommendation` JSON for a known upcoming match
- LangSmith trace shows all tool calls, inputs, outputs, and reasoning turns
- Agent correctly selects `forecast_league` vs `forecast_international` for a league and international match respectively
- `insufficient_data` recommendation is produced when odds cannot be found by web search

---

### A08 — Add agent-recommend CLI Command
**Size:** S | **Status:** completed | **Milestone:** M2 | **Depends on:** A07

Expose the live recommendation workflow as a CLI command.

**Acceptance criteria:**
- `python main.py agent-recommend --home "Man City" --away "Arsenal" --date 2026-06-15` runs the full agent and prints the `MatchRecommendation` JSON plus explanation
- `--config` flag allows specifying a non-default `agent_config.yaml` path
- Exit code 0 on success; non-zero on agent error or parse failure

---

## PHASE 3: Snapshot Infrastructure

### A09 — Implement SnapshotStore
**Size:** M | **Status:** completed | **Milestone:** M3 | **Depends on:** A01

Build the record/replay interceptor that isolates backtest runs from live APIs.

**Acceptance criteria:**
- `SnapshotStore` class in `src/agent/snapshot_store.py` with `set_mode(mode: Literal["record", "replay", "live"])` method
- `record` mode: executes the wrapped callable, serialises `{tool, inputs, response, recorded_at}` to `data/agent_snapshots/<match_id>/<tool>_<sha256_of_inputs>.json`
- `replay` mode: loads from disk; raises `SnapshotMissingError` immediately if file not found — no silent fallback to live call
- `live` mode: passes through to the real callable with no interception (default for `agent-recommend`)
- Key is a SHA-256 hash of the canonically serialised input dict, ensuring the same query always maps to the same file

---

### A10 — Integrate SnapshotStore with All Tool Functions
**Size:** S | **Status:** completed | **Milestone:** M3 | **Depends on:** A04, A05, A06, A09

Wrap every tool function with the `SnapshotStore` interceptor and add date-filtering to web search during snapshot collection.

**Acceptance criteria:**
- `web_search`, `forecast_league`, and `forecast_international` all route through `SnapshotStore`
- During snapshot collection (`agent-snapshot`), web search queries automatically have `before:<match_date>` appended to reduce post-match leakage
- System prompt for snapshot collection run instructs the agent to discard any result that references a final score
- Switching mode from `live` to `record` to `replay` requires no changes to tool function code

---

### A11 — Add agent-snapshot CLI Command
**Size:** S | **Status:** completed | **Milestone:** M3 | **Depends on:** A10

Expose snapshot collection as a CLI command that drives the agent in record mode over historical matches.

**Acceptance criteria:**
- `python main.py agent-snapshot --from-date 2025-01-01 --to-date 2025-06-01 --league E0` loads historical matches from DuckDB and runs the agent in `record` mode over each
- Skips matches that already have a complete snapshot directory
- Prints progress (match count, skipped, errors)
- `--dry-run` flag lists matches that would be processed without executing

---

## PHASE 4: Backtest Harness

### A12 — Implement BacktestHarness and Outcome Loader
**Size:** L | **Status:** completed | **Milestone:** M4 | **Depends on:** A11

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
**Size:** M | **Status:** completed | **Milestone:** M4 | **Depends on:** A12

Simulate bankroll evolution over backtest records and compute the evaluation report.

**Acceptance criteria:**
- `flat` staking: configurable fixed stake (default 1% of starting bankroll) applied to every `direct_bet` recommendation
- Bankroll updated per bet: win → bankroll += stake × (odds − 1); loss → bankroll −= stake
- Evaluation report computed by `src/agent/evaluation.py` with: ROI, hit rate, bet frequency, max drawdown, bets placed, insufficient data rate
- Report saved to `reports/agent_backtest/<timestamp>_<config_hash>.json`
- Report also printed to stdout in a human-readable table

---

### A14 — Add agent-backtest CLI with Parallelism
**Size:** M | **Status:** completed | **Milestone:** M4 | **Depends on:** A12, A13

Expose the backtest harness as a CLI command with concurrent execution support.

**Acceptance criteria:**
- `python main.py agent-backtest --from-date 2025-01-01 --to-date 2025-06-01 --stake-mode flat --sample 50` runs and prints the evaluation report
- `--concurrency N` flag (default 5) controls how many agent runs execute concurrently via `asyncio.gather`
- `--config` flag allows specifying a non-default `agent_config.yaml` for model/prompt comparison
- Progress bar shown during run (matches completed / total)

---

## PHASE 5: Model & Prompt Tuning

### A15 — Implement Kelly Criterion Staking
**Size:** S | **Status:** completed | **Milestone:** M5 | **Depends on:** A13

Add Kelly criterion as a second staking mode.

**Acceptance criteria:**
- `kelly` staking: stake = `value_edge / (odds − 1)` × current bankroll, capped at 10% of bankroll per bet
- Selectable via `--stake-mode kelly` on `agent-backtest`
- Both `flat` and `kelly` produce comparable report formats for side-by-side analysis

---

### A16 — Build Config Comparison Framework
**Size:** M | **Status:** completed | **Milestone:** M5 | **Depends on:** A14, A15

Allow systematic comparison of agent configurations (model, prompt version, staking) over the same snapshot set.

**Acceptance criteria:**
- `python main.py agent-compare --configs config/agent_v1.yaml config/agent_v2.yaml --from-date ... --to-date ...` runs each config over the same snapshots and outputs a side-by-side comparison table
- Comparison table includes: ROI, hit rate, bet frequency, max drawdown per config
- Results saved to `reports/agent_backtest/comparison_<timestamp>.json`

---

### A17 — Create agent_techspec.md
**Size:** M | **Status:** completed | **Milestone:** M5 | **Depends on:** A08, A14 (partial — see note)

Document the implementation details discovered during M1–M4 in a formal technical specification.

**Acceptance criteria:**
- `documents/agent_techspec.md` created as the authoritative implementation reference
- Covers: `src/agent/` module structure, `StateGraph` node contracts, `SnapshotStore` file layout, `BacktestHarness` data flow, CLI command reference, LangSmith configuration
- Reflects actual implementation (written after M4, not before)
- `agent_prd.md` Extension Points section updated to mark `agent_techspec.md` as complete

**Note:** written against A01–A08 only since A09–A16 (snapshot/backtest/tuning) are not yet implemented. The doc's Implementation Status section documents this gap explicitly and should be revisited once A14 actually lands.

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

---

## PHASE 7: Backtest Execution Readiness

The backtest harness (A09–A16) has been implemented and committed, but never actually run — `data/agent_snapshots/` is empty and no report exists under `reports/agent_backtest/`. This phase is the operational path from "harness exists" to "we have a first real backtest result."

### A19 — Train and Select League-Context Forecast Models (Resolve BUG-010)

**Size:** M | **Status:** completed | **Milestone:** M7 | **Depends on:** none (forecast-engine side)

`config/model_selection.yaml` currently only has an `international` (market-odds-only) context. Every `forecast_league` call for an E0 match silently falls back to the market-odds-only path and tags the result `market_odds_only_league_fallback`. Any snapshot or backtest collected before this is fixed will permanently encode odds-only forecasts, not the full 147-feature league models. This story must land before A20, or A20's snapshot corpus must be discarded and recollected afterward.

**Acceptance criteria:**
- `python main.py train-forecast-suite --context league` trains all 8 forecast targets under the league context
- `python main.py select-best-models --context league` populates `config/model_selection.yaml`'s `contexts.league` block for all 8 targets
- A manual `forecast_league` call for a known E0 match returns `data_quality.prediction_basis == "team_history_and_market"`, not the league-fallback tag
- BUG-010 in `documents/bugs.md` updated from `partial` to `fixed`

**Completion notes (2026-06-27):** Plain `train-forecast-suite --context league` defaults to `lr`/`rf_regressor`, which crash ("No rows left after dropping records with missing labels or features") because 12 xG/LUCK columns are 100% NaN in the current feature store and non-XGBoost models require zero NaN across all 147 features. Trained all 8 targets individually instead via `train-target --target <t> --model xgb|xgb_regressor --context league`, matching the XGBoost-wins-every-target conclusion already established in `agent_techspec.md` Sections 22–24. Discovered and fixed two latent bugs in the selection pipeline that would have silently no-opped otherwise: (1) plain (non-sweep) training runs were never tagged `context`/`sweep_stage`, so `select-best-models` could never find them — fixed in `ModelManager.run_pipeline()`. (2) `ModelSelector` built `model_path` from the MLflow autolog artifact URI, which isn't a `joblib.load()`-able file — fixed by logging an `artifact_filename` param and pointing `model_path` at the real `models/*.joblib` artifact. Verified live: `forecast_league` for an E0 match now returns `data_quality.prediction_basis == "team_history_and_market"`. `documents/bugs.md` BUG-010 updated to `fixed`.

---

### A20 — Collect a Pilot Snapshot Corpus for E0

**Size:** S | **Status:** completed | **Milestone:** M7 | **Depends on:** A11, A19

Record a real snapshot corpus over a deliberately small E0 date range to validate the end-to-end record path (live Ollama + live Tavily) before committing to a larger run. `agent-snapshot` is sequential (no concurrency flag), so range size should be chosen for wall-clock feasibility, not statistical completeness.

**Acceptance criteria:**
- `python main.py agent-snapshot --from-date <X> --to-date <Y> --league E0 --dry-run` run first to confirm fixture count for the chosen pilot range
- `agent-snapshot` run for real (no `--dry-run`) over that range; final summary shows `Errors: 0`, or any errors are investigated and resolved before proceeding
- Every match in the range has a `data/agent_snapshots/<match_id>/_complete.json` marker
- Corpus was collected after A19 lands, so `forecast_league` snapshots reflect league-context models, not the BUG-010 fallback

**Completion notes (2026-06-27):** Pilot range chosen: 2026-03-01 → 2026-03-16, E0 (the most recent 24 finished matches, the tail end of the available raw_matches data). Dry-run confirmed 24 fixtures. First real run was started in parallel with A19's last verification step and had to be discarded and redone — it completed before the `model_path` fix (mlflow artifact URI vs. loadable joblib) landed, so every match in it silently used the `market_odds_only_league_fallback` path. Cleared `data/agent_snapshots/` and re-ran after confirming the fix live. That second run reported 24/24 processed, 0 errors, all `_complete.json` markers present — **but this was also invalid**: the user inspected the snapshot directories directly and found every one contained only `_complete.json` and zero actual tool-response files, which led to discovering BUG-011 (`SnapshotStore`'s `threading.local()` never reaching the thread LangGraph's `ToolNode` actually runs tool calls on — see `agent_techspec.md` Section 18.4 for the full root-cause writeup). Fixed `SnapshotStore` to use `contextvars.ContextVar` instead. Cleared the corpus a third time and re-collected: 24/24 processed, 0 errors, and this time genuinely verified — 24 `forecast_league_*.json` + 11 `web_search_*.json` files actually present on disk, confirmed via direct inspection, not just log output.

---

### A21 — Run First Agent Backtest and Record Baseline Report

**Size:** S | **Status:** completed | **Milestone:** M7 | **Depends on:** A12, A13, A14, A20

Produce the first real evaluation report against the pilot corpus from A20, establishing a baseline ROI/hit-rate/drawdown reference for future prompt and model tuning.

**Acceptance criteria:**
- `python main.py agent-backtest --from-date <X> --to-date <Y> --league E0 --stake-mode flat` runs to completion against the A20 corpus and prints/saves an evaluation report under `reports/agent_backtest/`
- Same date range re-run with `--stake-mode kelly` for a side-by-side comparison
- Baseline findings (ROI, hit rate, bet frequency, insufficient_data_rate, notable failure patterns) written up as a new section in `documents/agent_techspec.md`

**Completion notes (2026-06-27):** First flat run crashed in `simulate_flat_stake` with `TypeError: float() argument must be ... not 'NoneType'` — the agent can mark a market `direct_bet` while `current_odds` is `null` (odds missing for that one market), and `staking.py`'s skip-gate never checked for it. Fixed in both `simulate_flat_stake` and `simulate_kelly_stake`, plus two new regression tests in `tests/test_staking.py`. Re-ran and got a "clean" 7-bet flat / 8-bet kelly result — **which was invalid**, because at that point `agent-backtest` was secretly making live calls instead of replaying anything (BUG-011, see A20's completion notes and `agent_techspec.md` Section 18.4). After fixing `SnapshotStore` and re-collecting a genuine A20 corpus, both reports were discarded and regenerated for real:

**flat** — 23/24 matches evaluated (1 skipped, `SnapshotMissingError`), 20 bets, 11 won, hit rate 0.55, ROI **+0.1845**, max drawdown 0.050, ending bankroll 1036.90.
**kelly** — 20/24 matches evaluated (4 skipped), 5 bets, 2 won, hit rate 0.40, ROI **−0.1476**, max drawdown 0.136, ending bankroll 961.70.

The genuinely-replayed bet frequency (0.87 for flat) is far higher than the bogus live-call run's (0.29) — confirms the original numbers weren't just imprecise, they were systematically wrong. Two further findings written up in `agent_techspec.md` Section 18.6–18.7 rather than fixed in this story (out of scope): (1) flat and kelly evaluated a *different* number of matches from the identical corpus, because the LLM regenerates its own tool-call arguments each run and the SHA-256 snapshot key occasionally misses on argument drift — A22 should run each config multiple times, not trust a single pass; (2) one match's recommendation explicitly stated the final score was known despite both leakage defenses (`before:<date>` filter + prompt instruction), meaning that protection is a mitigation, not a guarantee. Sample size (20–23 matches, 5–20 bets) is still far too small to draw any conclusion about agent edge — this story's goal was a working, reproducible, *genuinely isolated* pipeline and a documented baseline, not a profitability verdict.

---

### A22 — Compare Agent Configurations Against the Baseline

**Size:** S | **Status:** future | **Milestone:** M7 | **Depends on:** A16, A21

Once a baseline exists, use the config comparison framework to test whether a prompt or model change improves on it, over the identical match sample.

**Acceptance criteria:**
- At least one alternative `agent_config.yaml` variant created (e.g. a `v2` system prompt, or `provider: anthropic`)
- `python main.py agent-compare --configs config/agent_config.yaml config/agent_config_v2.yaml --from-date <X> --to-date <Y> --league E0 --sample N` run against the same snapshot corpus used in A21
- Comparison table reviewed; the better-performing config adopted as the new default if it wins on ROI without a worse max drawdown
