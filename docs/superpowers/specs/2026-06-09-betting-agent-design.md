# Betting Agent Design Spec
**Date:** 2026-06-09
**Status:** Draft — approved for implementation planning

---

## 1. Overview

A LangGraph-based autonomous betting agent that sits on top of the existing FPAI ML forecasting engine. The agent uses web search and ML model forecasts as tool-callable signals, synthesises them with LLM reasoning (Claude or local model via Ollama), and produces structured bet recommendations with natural language explanations.

This is a separate product layer from the ML forecasting engine. It will have its own PRD and tech spec (`documents/agent_prd.md`, `documents/agent_techspec.md`) created during and after implementation.

### What it is not
- A replacement for the ML models — forecasts are one input signal, not the decision
- A guaranteed profitability system — it finds candidate value bets; outcomes are probabilistic
- A live odds monitor — conditional recommendations state a threshold; monitoring is manual for now

---

## 2. Milestones

| # | Name | Done When |
|---|---|---|
| M1 | Foundation | Agent loop runs with stub tools; LangSmith trace visible |
| M2 | Live Recommendation | Real tools, real model, structured JSON output for an upcoming match |
| M3 | Snapshot Infrastructure | Record episode to disk; replay produces same tool inputs deterministically |
| M4 | Backtest Harness | ROI report over 50+ historical snapshot matches |
| M5 | Model & Prompt Tuning | Documented config that beats baseline ROI on backtest set |
| M6 | Batch Recommendation | Weekend fixture card from one command (future) |

---

## 3. Architecture

```
src/agent/
  __init__.py
  graph.py              # LangGraph StateGraph
  agent_config.py       # AgentConfig dataclass
  tools.py              # @tool definitions
  snapshot_store.py     # record/replay interceptor
  backtest.py           # BacktestHarness + bankroll simulation
  evaluation.py         # ROI, drawdown, hit rate, bet frequency

config/
  agent_config.yaml     # all tunable knobs

data/agent_snapshots/   # stored tool responses per match episode
  <match_id>/
    web_search_<query_hash>.json
    forecast_league_<input_hash>.json
    forecast_international_<input_hash>.json
```

### System flow

```
CLI input
  └─► BacktestHarness (backtest) or direct call (live)
        └─► BettingAgent.run(match_info)
              └─► LangGraph StateGraph
                    ├─ agent_node   (LLM — Claude or Ollama)
                    ├─ tools_node   (web_search, forecast_league, forecast_international)
                    └─ output_node  (parse + validate MatchRecommendation)
                          ▲
                    SnapshotStore intercepts every tool call
                    (record: save response | replay: serve from disk)
```

LangSmith traces every node, tool call, and token automatically. No custom tracer needed.

---

## 4. Agent Core

### 4.1 AgentState

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]    # full conversation + tool call history
    match_info: dict               # home, away, date, league (injected at start)
    recommendation: dict | None    # populated by output_node when done
    tool_call_count: int           # enforces max_tool_calls budget
```

### 4.2 StateGraph

```
START → agent_node
          │
          ├─ has tool_calls AND tool_call_count < max → tools_node → agent_node
          │
          └─ end_turn OR budget exceeded → output_node → END
```

`output_node` extracts and validates the structured `MatchRecommendation` JSON from the final agent message. The raw message text is stored as the natural language explanation — no separate explanation tool is needed.

### 4.3 AgentConfig (`config/agent_config.yaml`)

```yaml
model: "qwen2.5:7b"                 # swap to claude-haiku-4-5-20251001 or claude-sonnet-4-6
provider: "ollama"                  # "ollama" | "anthropic"
temperature: 0.1                    # lower = more deterministic; use 0.0 for backtesting
max_tool_calls: 10                  # per-match tool budget
min_odds_threshold: 2.0             # +100 American; bets below this are never recommended
min_value_edge: 0.05                # model prob must exceed implied prob by ≥5%
markets:                            # targets the agent evaluates
  - result_3way
  - btts
  - total_goals
  - home_corners
  - away_corners
system_prompt_version: "v1"
```

Changing any knob and re-running the backtest harness is the equivalent of tuning hyperparameters.

### 4.4 Output Schema

```python
class MarketRecommendation(TypedDict):
    market: str                # "btts" | "result_3way" | "total_goals" | ...
    selection: str             # "yes" | "home" | "over_2.5" | ...
    recommendation_type: str   # "direct_bet" | "conditional" | "no_bet"
    current_odds: float
    min_odds: float            # value threshold
    ml_probability: float
    implied_probability: float
    value_edge: float          # ml_probability - implied_probability

class MatchRecommendation(TypedDict):
    match: dict                # home, away, date, league
    overall: str               # "direct_bet" | "conditional" | "no_bet" | "insufficient_data"
    markets: list[MarketRecommendation]
    explanation: str           # natural language explanation from final LLM turn
    confidence: str            # "low" | "medium" | "high"
    limitations: list[str]     # specific gaps (missing odds, cold-start, unknown team)
    prediction_basis: str      # forwarded from ML tool data_quality.prediction_basis: "team_history_and_market" | "market_odds_only" | "partial"
```

**Recommendation types:**
- `direct_bet`: current odds meet or exceed `min_odds_threshold` and `min_value_edge`
- `conditional`: value edge exists but current odds are below threshold — agent states the target odds and the gap
- `no_bet`: agent analysed the match fully and found no value in any market
- `insufficient_data`: agent could not complete analysis — distinct from `no_bet`

**Conditional bet note:** Full odds drift modelling (estimating how quickly odds will reach threshold) requires historical intra-day odds time series. This is an **extension point** for a future milestone when that data is available.

---

## 5. Tool Layer

All tools are decorated with `@tool` and registered with LangGraph's `ToolNode`. All calls pass through `SnapshotStore`.

### 5.1 `web_search(query: str) -> str`

- Backed by **Tavily API** (free tier: 1000 searches/month)
- In replay mode, Tavily is never called — SnapshotStore serves the recorded response
- Used for: discovering bookmaker odds, team name variants, injury news, team selection
- During snapshot collection, queries automatically get `before:<match_date>` appended to reduce post-match leakage. The system prompt also instructs the agent to ignore any result referencing a final score.

### 5.2 `forecast_league(home_team, away_team, date, league, odds_h, odds_d, odds_a, [over25_odds, ah_line, ah_home_odds, ah_away_odds, targets]) -> dict`

- Direct call to `ForecastService.forecast_upcoming(match_type="league")`
- Uses full rolling features (team history + market odds)
- Requires `league` to be known
- **Use when:** both teams play in a known league with historical data in the feature store

### 5.3 `forecast_international(home_team, away_team, date, odds_h, odds_d, odds_a, [targets]) -> dict`

- Direct call to `ForecastService.forecast_upcoming(match_type="international")`
- Uses market odds features only (MKT_* feature set)
- No team name lookup required
- **Use when:** international fixture, cup match, or team history unavailable

Two distinct tools prevent the agent from accidentally calling the wrong model context. The agent decides which to use based on what it learns from web search (can it identify a league context?).

The `data_quality` field in both tool responses signals ML-layer limitations back to the agent:
- `"partial"` or low `feature_completeness` → agent should flag `insufficient_data`
- Agent can also self-declare `insufficient_data` if web search yields nothing useful

---

## 6. Snapshot Store

### 6.1 Storage

```
data/agent_snapshots/<match_id>/
  web_search_<sha256_of_query>.json
  forecast_league_<sha256_of_inputs>.json
  forecast_international_<sha256_of_inputs>.json
```

Each file stores: `{tool, inputs, response, recorded_at}`.

### 6.2 Modes

| Mode | Behaviour |
|---|---|
| `record` | Execute live call → save to disk → return response |
| `replay` | Load from disk → return response. Raises `SnapshotMissingError` if not found. No silent fallback. |

`replay` mode never makes live API calls. `SnapshotMissingError` surfaces immediately so you know a snapshot gap exists — it does not degrade silently to a live call.

### 6.3 Snapshot Collection

```bash
python main.py agent-snapshot --from-date 2025-01-01 --to-date 2025-06-01 --league E0
```

Runs the agent in `record` mode over historical matches. Date-filtered queries reduce (but do not eliminate) leakage. Snapshot collection is a one-time cost per date range; once collected, backtests are free of live API calls.

---

## 7. Backtest Harness & Evaluation

### 7.1 BacktestHarness

```python
class BacktestHarness:
    def run(self, from_date, to_date, league, stake_mode, sample_n) -> BacktestReport:
        matches = load_historical_matches(from_date, to_date, league)
        if sample_n:
            matches = stratified_sample(matches, sample_n)

        results = asyncio.gather(*[
            self._run_match(m) for m in matches
        ], limit=5)   # max 5 concurrent agent runs

        return evaluate(results)

    async def _run_match(self, match):
        snapshot_store.set_mode("replay")
        recommendation = await agent.run(match)
        outcome = load_actual_outcome(match.match_id)  # from DuckDB raw_matches
        return BacktestRecord(match, recommendation, outcome)
```

### 7.2 Staking Modes

| Mode | Formula | Notes |
|---|---|---|
| `flat` | Fixed stake (e.g. 1% of starting bankroll) | Comparable across runs; good for M4 |
| `kelly` | `edge / (odds - 1)` × bankroll | Optimal long-term growth; aggressive; use in M5+ |

### 7.3 Evaluation Metrics

```
ROI:              (total_return - total_staked) / total_staked
Hit rate:         bets_won / bets_placed
Bet frequency:    bets_placed / matches_evaluated
Max drawdown:     largest peak-to-trough bankroll decline
Bets placed:      count
Insufficient data rate: matches where agent could not recommend
```

Report saved to `reports/agent_backtest/YYYY-MM-DD_<config_hash>.json`.

### 7.4 Efficient Testing

- `--sample N` runs a stratified random sample before a full backtest — quick sanity check
- Replay mode eliminates all live API costs during backtest
- `temperature: 0.0` in backtest config ensures deterministic LLM output for reproducibility
- Different prompt versions and model tiers can be compared by re-running over the same snapshots

---

## 8. CLI Commands

```bash
# Live single-match recommendation
python main.py agent-recommend \
    --home "Man City" --away "Arsenal" --date 2026-06-15

# Collect snapshots for a historical range (record mode)
python main.py agent-snapshot \
    --from-date 2025-01-01 --to-date 2025-06-01 --league E0

# Backtest over collected snapshots
python main.py agent-backtest \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    --stake-mode flat --sample 50
```

---

## 9. Extension Points (Out of Scope for Initial Build)

These are design decisions deferred to later milestones. The architecture is designed to accommodate them without structural changes.

| Extension | Notes |
|---|---|
| Odds drift modelling | Conditional bets currently state a target threshold only. Full estimation requires intra-day historical odds time series. |
| Batch recommendation | Weekend fixture card (`agent-batch --weekend --league E0`). Runs agent in parallel over all fixtures. |
| Richer conversation tools | If the agent needs to query structured data mid-conversation (e.g. H2H records, league tables), a `lookup_stats` tool can be added to `tools.py` without graph changes. |
| Automated odds monitoring | Polling bookmaker odds until threshold is reached. Requires integration with a live odds feed. |
| Agent PRD and tech spec | Formal product and technical documentation in `documents/agent_prd.md` and `documents/agent_techspec.md`, to be created during implementation. |

---

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over custom loop | Industry-standard, LangSmith tracing built in, model-agnostic via LangChain |
| Two distinct forecast tools | Prevents agent from calling wrong model context; clearer tool descriptions |
| SnapshotStore raises on miss in replay | Prevents silent live-call fallback contaminating backtest with real-time data |
| `insufficient_data` separate from `no_bet` | Enables diagnosis of agent capability gaps vs. genuine no-value decisions |
| Separate docs from ML tech spec | Agent is a distinct product layer; mixing specs creates confusion as both evolve |
