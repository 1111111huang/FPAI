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

PHASE 8 (Full Season Backtest Expansion)
  A21 → A23 (fix snapshot determinism before large-scale recording)
  A23 → A24 (refresh raw data before re-running feature pipeline)
  A24 → A25 (collect corpus only after data and temp fix are ready)
  A25 → A26 (run backtests only after full corpus exists)
```

**Size key:** XS < 2 hrs · S ≈ half day · M ≈ 1 day · L ≈ 2–3 days

---

## PHASE 1: Foundation

| ID | Status | Description | Comments |
|---|---|---|---|
| A01 | completed | Set up all dependencies required to run a LangGraph agent locally. **Acceptance:** `langgraph`, `langchain`, `langchain-anthropic`, `langchain-ollama`, `tavily-python` added to `requirements.txt`; Ollama installed locally with `qwen2.5:7b` pulled and verified; LangSmith API key configured with a visible test trace; `src/agent/__init__.py` created. | Size M · Milestone M1 · Depends on: none |
| A02 | completed | Create a typed configuration object and YAML file covering all agent tuning knobs. **Acceptance:** `AgentConfig` dataclass in `src/agent/agent_config.py` with fields `model`, `provider`, `temperature`, `max_tool_calls`, `min_odds_threshold`, `min_value_edge`, `markets`, `system_prompt_version`; `config/agent_config.yaml` with sensible defaults; `AgentConfig.from_yaml(path)` loads/validates, raising a clear error on missing fields. | Size S · Milestone M1 · Depends on: A01 |
| A03 | completed | Implement the LangGraph `StateGraph` skeleton with stub tools so the agent loop can be exercised end-to-end before real tools exist. **Acceptance:** `AgentState` TypedDict with `messages`, `match_info`, `recommendation`, `tool_call_count`; `StateGraph` with `agent_node`/`tools_node`/`output_node` wired correctly; conditional edge (tool calls under budget → tools, else → output); stub `web_search`/`forecast_league` tools; full-loop LangSmith trace against a hardcoded match. | Size M · Milestone M1 · Depends on: A01, A02 |

---

## PHASE 2: Live Recommendation

| ID | Status | Description | Comments |
|---|---|---|---|
| A04 | completed | Connect the Tavily search API as the agent's primary web search tool. **Acceptance:** `web_search(query: str) -> str` with `@tool` decorator; Tavily client reads `TAVILY_API_KEY`; description clearly states purpose (odds, team names, injury news); registered with `ToolNode`; manual test finds current odds for a named match. | Size M · Milestone M2 · Depends on: A03 |
| A05 | completed | Expose the two ML model contexts as distinct, named tools so the agent cannot call the wrong one. **Acceptance:** `forecast_league(...)` and `forecast_international(...)` call `ForecastService.forecast_upcoming(match_type=...)` directly; docstrings state when each should be used; both registered with `ToolNode`; both return full forecast JSON including `data_quality.prediction_basis`. | Size M · Milestone M2 · Depends on: A03 |
| A06 | completed | Define the agent's betting philosophy in the system prompt and enforce a structured JSON output every run. **Acceptance:** prompt at `config/prompts/agent_v1.txt`, loaded by `AgentConfig`; specifies search odds → forecast → news → evaluate value → output JSON; includes the `MatchRecommendation` schema; `MatchRecommendation`/`MarketRecommendation` TypedDicts in `src/agent/schema.py`; `output_node` parses/validates, raising `RecommendationParseError` with raw text on failure; `overall` ∈ {`direct_bet`, `conditional`, `no_bet`, `insufficient_data`}; `recommendation_type` ∈ {`direct_bet`, `conditional`, `no_bet`}. | Size M · Milestone M2 · Depends on: A03 |
| A07 | completed | Connect all real tools into the graph and validate the full live recommendation workflow end-to-end. **Acceptance:** graph runs with real Tavily, real `ForecastService`, real LLM (Ollama then Haiku); valid `MatchRecommendation` JSON for a known upcoming match; LangSmith trace shows all tool calls/inputs/outputs/reasoning; correct `forecast_league` vs `forecast_international` selection; `insufficient_data` produced when odds can't be found. | Size M · Milestone M2 · Depends on: A04, A05, A06 |
| A08 | completed | Expose the live recommendation workflow as a CLI command. **Acceptance:** `python main.py agent-recommend --home "Man City" --away "Arsenal" --date 2026-06-15` runs the full agent and prints the `MatchRecommendation` JSON plus explanation; `--config` flag for a non-default YAML path; exit code 0 on success, non-zero on agent error or parse failure. | Size S · Milestone M2 · Depends on: A07 |

---

## PHASE 3: Snapshot Infrastructure

| ID | Status | Description | Comments |
|---|---|---|---|
| A09 | completed | Build the record/replay interceptor that isolates backtest runs from live APIs. **Acceptance:** `SnapshotStore` class in `src/agent/snapshot_store.py` with `set_mode(mode: Literal["record", "replay", "live"])`; `record` serialises `{tool, inputs, response, recorded_at}` to `data/agent_snapshots/<match_id>/<tool>_<sha256_of_inputs>.json`; `replay` raises `SnapshotMissingError` immediately if file not found, no silent live fallback; `live` passes through with no interception (default for `agent-recommend`); key is a SHA-256 hash of the canonically serialised input dict. | Size M · Milestone M3 · Depends on: A01 |
| A10 | completed | Wrap every tool function with the `SnapshotStore` interceptor and add date-filtering to web search during snapshot collection. **Acceptance:** `web_search`, `forecast_league`, `forecast_international` all route through `SnapshotStore`; during `agent-snapshot`, web search queries automatically get `before:<match_date>` appended; snapshot-collection system prompt instructs discarding any result referencing a final score; switching `live`→`record`→`replay` requires no tool function code changes. | Size S · Milestone M3 · Depends on: A04, A05, A06, A09 |
| A11 | completed | Expose snapshot collection as a CLI command that drives the agent in record mode over historical matches. **Acceptance:** `python main.py agent-snapshot --from-date 2025-01-01 --to-date 2025-06-01 --league E0` loads historical matches from DuckDB and runs in `record` mode; skips matches with a complete snapshot directory already; prints progress (count/skipped/errors); `--dry-run` lists matches without executing. | Size S · Milestone M3 · Depends on: A10 |

---

## PHASE 4: Backtest Harness

| ID | Status | Description | Comments |
|---|---|---|---|
| A12 | completed | Build the core backtest engine that replays snapshot episodes and compares agent recommendations against actual outcomes. **Acceptance:** `BacktestHarness` class in `src/agent/backtest.py`; loads historical matches from DuckDB for a date range/league; per match sets `SnapshotStore` to `replay`, runs the agent, loads the actual outcome from `raw_matches`; `BacktestRecord` holds match info, `MatchRecommendation`, actual outcome, per-market correctness; raises `SnapshotMissingError` at the harness level (no silent skip); `--sample N` runs a stratified random sample before a full run. | Size L · Milestone M4 · Depends on: A11 |
| A13 | completed | Simulate bankroll evolution over backtest records and compute the evaluation report. **Acceptance:** `flat` staking (configurable, default 1% of starting bankroll) applied to every `direct_bet`; bankroll updates win → `+= stake × (odds − 1)`, loss → `−= stake`; `src/agent/evaluation.py` computes ROI, hit rate, bet frequency, max drawdown, bets placed, insufficient data rate; report saved to `reports/agent_backtest/<timestamp>_<config_hash>.json`; also printed as a human-readable table. | Size M · Milestone M4 · Depends on: A12 |
| A14 | completed | Expose the backtest harness as a CLI command with concurrent execution support. **Acceptance:** `python main.py agent-backtest --from-date 2025-01-01 --to-date 2025-06-01 --stake-mode flat --sample 50` runs and prints the evaluation report; `--concurrency N` (default 5) controls concurrent agent runs via `asyncio.gather`; `--config` flag for model/prompt comparison; progress bar shown during run. | Size M · Milestone M4 · Depends on: A12, A13 |

---

## PHASE 5: Model & Prompt Tuning

| ID | Status | Description | Comments |
|---|---|---|---|
| A15 | completed | Add Kelly criterion as a second staking mode. **Acceptance:** `kelly` staking = `value_edge / (odds − 1) × current bankroll`, capped at 10% of bankroll per bet; selectable via `--stake-mode kelly`; both `flat` and `kelly` produce comparable report formats. | Size S · Milestone M5 · Depends on: A13 |
| A16 | completed | Allow systematic comparison of agent configurations (model, prompt version, staking) over the same snapshot set. **Acceptance:** `python main.py agent-compare --configs config/agent_v1.yaml config/agent_v2.yaml --from-date ... --to-date ...` runs each config over the same snapshots and outputs a side-by-side table (ROI, hit rate, bet frequency, max drawdown per config); results saved to `reports/agent_backtest/comparison_<timestamp>.json`. | Size M · Milestone M5 · Depends on: A14, A15 |
| A17 | completed | Document the implementation details discovered during M1–M4 in a formal technical specification. **Acceptance:** `documents/agent_techspec.md` created covering `src/agent/` module structure, `StateGraph` node contracts, `SnapshotStore` file layout, `BacktestHarness` data flow, CLI command reference, LangSmith configuration; reflects actual implementation (written after M4, not before); `agent_prd.md` Extension Points section updated to mark `agent_techspec.md` complete. | Size M · Milestone M5 · Depends on: A08, A14 (partial). **Note:** written against A01–A08 only since A09–A16 weren't implemented yet at time of writing; the doc's Implementation Status section documents this gap explicitly and should be revisited once A14 actually lands. |

---

## PHASE 6: Batch Recommendation (Future)

| ID | Status | Description | Comments |
|---|---|---|---|
| A18 | future | Allow the user to request recommendations for all upcoming fixtures in a league over a given weekend. **Acceptance:** `python main.py agent-batch --league E0 --weekend` fetches upcoming fixtures, runs the agent over each in parallel, produces a ranked recommendation report (best value bets first); fixture discovery via web search or a configured fixtures API; report groups matches by date and highlights `direct_bet` recommendations at the top; respects the same `agent_config.yaml` knobs as `agent-recommend`. | Size L · Milestone M6 · Depends on: A08 |

---

## PHASE 7: Backtest Execution Readiness

> The backtest harness (A09–A16) has been implemented and committed, but never actually run — `data/agent_snapshots/` is empty and no report exists under `reports/agent_backtest/`. This phase is the operational path from "harness exists" to "we have a first real backtest result."

| ID | Status | Description | Comments |
|---|---|---|---|
| A19 | completed | Train and select league-context forecast models (resolve BUG-010). `config/model_selection.yaml` currently only has an `international` (market-odds-only) context — every `forecast_league` call for an E0 match silently falls back to market-odds-only and tags the result `market_odds_only_league_fallback`. Must land before A20, or A20's snapshot corpus must be discarded and recollected afterward. **Acceptance:** `python main.py train-forecast-suite --context league` trains all 8 forecast targets under the league context; `python main.py select-best-models --context league` populates `config/model_selection.yaml`'s `contexts.league` block for all 8 targets; a manual `forecast_league` call for a known E0 match returns `data_quality.prediction_basis == "team_history_and_market"`, not the league-fallback tag; BUG-010 in `documents/bugs.md` updated from `partial` to `fixed`. | Size M · Milestone M7 · Depends on: none (forecast-engine side). **Completion notes (2026-06-27):** Plain `train-forecast-suite --context league` defaults to `lr`/`rf_regressor`, which crash ("No rows left after dropping records with missing labels or features") because 12 xG/LUCK columns are 100% NaN in the current feature store and non-XGBoost models require zero NaN across all 147 features. Trained all 8 targets individually instead via `train-target --target <t> --model xgb\|xgb_regressor --context league`, matching the XGBoost-wins-every-target conclusion already established in `agent_techspec.md` Sections 22–24. Discovered and fixed two latent bugs in the selection pipeline that would have silently no-opped otherwise: (1) plain (non-sweep) training runs were never tagged `context`/`sweep_stage`, so `select-best-models` could never find them — fixed in `ModelManager.run_pipeline()`. (2) `ModelSelector` built `model_path` from the MLflow autolog artifact URI, which isn't a `joblib.load()`-able file — fixed by logging an `artifact_filename` param and pointing `model_path` at the real `models/*.joblib` artifact. Verified live: `forecast_league` for an E0 match now returns `data_quality.prediction_basis == "team_history_and_market"`. `documents/bugs.md` BUG-010 updated to `fixed`. |
| A20 | completed | Collect a pilot snapshot corpus for E0. Record a real snapshot corpus over a deliberately small E0 date range to validate the end-to-end record path (live Ollama + live Tavily) before committing to a larger run. `agent-snapshot` is sequential (no concurrency flag), so range size should be chosen for wall-clock feasibility, not statistical completeness. **Acceptance:** `agent-snapshot ... --dry-run` confirms fixture count for the pilot range; real run shows `Errors: 0`, or errors investigated/resolved before proceeding; every match has a `data/agent_snapshots/<match_id>/_complete.json` marker; corpus collected after A19 lands so `forecast_league` snapshots reflect league-context models, not the BUG-010 fallback. | Size S · Milestone M7 · Depends on: A11, A19. **Completion notes (2026-06-27):** Pilot range 2026-03-01 → 2026-03-16, E0 (most recent 24 finished matches). Dry-run confirmed 24 fixtures. First real run discarded — ran in parallel with A19's last verification step and completed before the `model_path` fix landed, so every match silently used `market_odds_only_league_fallback`. Cleared and re-ran after confirming the fix live: 24/24 processed, 0 errors — but **also invalid**: user inspected snapshot directories directly and found every one contained only `_complete.json` with zero actual tool-response files, leading to discovery of BUG-011 (`SnapshotStore`'s `threading.local()` never reaching the thread LangGraph's `ToolNode` actually runs tool calls on — see `agent_techspec.md` Section 18.4). Fixed `SnapshotStore` to use `contextvars.ContextVar` instead. Cleared a third time and re-collected: 24/24 processed, 0 errors, and this time genuinely verified — 24 `forecast_league_*.json` + 11 `web_search_*.json` files actually present on disk. |
| A21 | completed | Run first agent backtest and record baseline report against the A20 pilot corpus, establishing a baseline ROI/hit-rate/drawdown reference for future prompt and model tuning. **Acceptance:** `agent-backtest --stake-mode flat` runs to completion and prints/saves an evaluation report under `reports/agent_backtest/`; same date range re-run with `--stake-mode kelly` for side-by-side comparison; baseline findings (ROI, hit rate, bet frequency, insufficient_data_rate, notable failure patterns) written up as a new section in `documents/agent_techspec.md`. | Size S · Milestone M7 · Depends on: A12, A13, A14, A20. **Completion notes (2026-06-27):** First flat run crashed in `simulate_flat_stake` (`TypeError: float() argument must be ... not 'NoneType'`) — the agent can mark a market `direct_bet` while `current_odds` is `null`, and `staking.py`'s skip-gate never checked for it. Fixed in both `simulate_flat_stake`/`simulate_kelly_stake`, plus 2 regression tests in `tests/test_staking.py`. Re-ran and got a "clean" 7-bet flat / 8-bet kelly result — **invalid**, because `agent-backtest` was secretly making live calls instead of replaying (BUG-011, see A20). After fixing `SnapshotStore` and re-collecting a genuine A20 corpus, both reports were discarded and regenerated for real: **flat** — 23/24 evaluated (1 skipped, `SnapshotMissingError`), 20 bets, 11 won, hit rate 0.55, ROI **+0.1845**, max drawdown 0.050, ending bankroll 1036.90. **kelly** — 20/24 evaluated (4 skipped), 5 bets, 2 won, hit rate 0.40, ROI **−0.1476**, max drawdown 0.136, ending bankroll 961.70. Genuinely-replayed bet frequency (0.87 flat) far higher than the bogus live-call run's (0.29) — confirms the original numbers were systematically wrong, not just imprecise. Two further findings written up in `agent_techspec.md` Section 18.6–18.7 rather than fixed here (out of scope): (1) flat/kelly evaluated different match counts from the identical corpus because the LLM regenerates its own tool-call args each run and the SHA-256 snapshot key occasionally misses on arg drift — A22 should run each config multiple times; (2) one match's recommendation stated the final score was known despite both leakage defenses — that protection is a mitigation, not a guarantee. Sample size (20–23 matches, 5–20 bets) still far too small for any edge conclusion. |
| A22 | future | Once a baseline exists, use the config comparison framework to test whether a prompt or model change improves on it, over the identical match sample. **Acceptance:** at least one alternative `agent_config.yaml` variant created (e.g. a `v2` system prompt, or `provider: anthropic`); `agent-compare` run against the same snapshot corpus used in A21; comparison table reviewed, better-performing config adopted as new default if it wins on ROI without a worse max drawdown. | Size S · Milestone M7 · Depends on: A16, A21 |

---

## PHASE 8: Full Season Backtest Expansion

> The pilot backtest (A20–A21) covered 24 matches (E0, March 1–16 2026). This phase expands coverage to the full 2025/26 E0 season (~380 matches) to produce a statistically meaningful baseline. Four pre-conditions must be resolved before large-scale recording begins.

| ID | Status | Description | Comments |
|---|---|---|---|
| A23 | future | Fix snapshot recording to use temperature=0 so LLM tool-call arguments are deterministic across runs. The pilot revealed that at temperature=0.1 the LLM regenerates slightly different query strings each run, producing SHA-256 key misses on replay and dropping different match subsets per backtest run (flat evaluated 23/24, kelly 20/24 on the identical corpus). At scale this compounds to dozens of silently skipped matches per run. **Acceptance:** agent config used during `agent-snapshot` sets `temperature: 0`; re-run of the existing 24-match pilot produces zero `SnapshotMissingError` skips on first replay (both flat and kelly evaluate the full 24/24); unit test added confirming that back-to-back `agent-snapshot` runs on the same match produce identical snapshot hashes. | Size S · Milestone M8 · Depends on: A21 |
| A24 | future | Download complete 2025/26 season raw data and refresh the feature store. The current `data/raw/football_data/E0_2526.csv` ends at 2026-03-16 (301 rows); the season ran through May 2026 (~79 matches missing). **Acceptance:** updated `E0_2526.csv` covers the full season through the final matchday (≥ 380 rows for E0); feature engineering pipeline re-run over the full date range so `ForecastService` can produce predictions for all matches; `forecast_league` for a match in April or May 2026 returns a result without a `data_quality` fallback tag. | Size S · Milestone M8 · Depends on: A23 |
| A25 | future | Collect the full 2025/26 E0 snapshot corpus. Run `agent-snapshot` from 2025-08-15 to the final matchday with the temperature=0 fix (A23) and updated data (A24) in place. **Acceptance:** `agent-snapshot --from-date 2025-08-15 --to-date 2026-05-25 --league E0 --dry-run` confirms ≥ 370 fixtures; real run completes with 0 errors; every match directory contains at least one `forecast_league_*.json` file (zero tool-call files is a repeat of BUG-011 — must be caught before proceeding); total `web_search_*.json` count across all matches is non-zero. | Size M · Milestone M8 · Depends on: A24 |
| A26 | future | Run full-season backtests (flat and kelly) and produce the expanded baseline report. Supports `--sample N` (already wired in `agent-backtest`) for a quick sanity-check over a random subset of the collected corpus before committing to the full run. **Acceptance:** quick-test path: `agent-backtest --from-date 2025-08-15 --to-date 2026-05-25 --league E0 --sample 30 --stake-mode flat` completes with ≤ 2 `SnapshotMissingError` skips; full path: same command without `--sample` and with `--stake-mode flat` then `--stake-mode kelly` both complete with ≤ 5% matches skipped; reports saved under `reports/agent_backtest/`; findings written up in `documents/agent_techspec.md` as a new section covering ROI, hit rate, drawdown, and leakage observations across the full season; note that the training cutoff (2023-04-27) predates the entire 2025/26 season so evaluation is genuinely out-of-sample. | Size S · Milestone M8 · Depends on: A25 |
