# Product Requirements Document - FPAI Betting Agent

## 1. Product Objective

The FPAI Betting Agent is an autonomous LangGraph-based agent that sits on top of the FPAI ML forecasting engine. It uses web search and ML model forecasts as tool-callable signals, synthesises them with LLM reasoning, and produces structured bet recommendations with natural language explanations.

The agent is the decision-making layer the forecasting engine was always designed to serve. ML model outputs are one quantitative input signal — not the final decision. The agent combines those priors with live bookmaker odds, news, injuries, and qualitative context before making any recommendation.

A separate technical specification (`documents/agent_techspec.md`) will be created during implementation and maintained as the authoritative implementation reference.

---

## 2. Product Positioning

### 2.1 Primary User
A bettor who wants a structured, reasoned recommendation for a specific upcoming match — including whether to bet, on which market, at what odds, and why.

### 2.2 Secondary Use
Systematic backtesting of the agent's decision-making over historical matches to evaluate profitability before committing real capital.

### 2.3 Relationship to Forecasting Engine
The FPAI forecasting engine (`FPAI_PRD.md`) remains a separate product. The agent consumes it as a tool. Changes to the forecasting engine do not require changes to the agent product contract, and vice versa.

---

## 3. Core Capabilities

### 3.1 Single-Match Recommendation (Point Prediction)
The primary workflow. Given a match, the agent:

1. Searches for current bookmaker odds and team name variants
2. Calls the ML forecasting tool appropriate to the match context
3. Searches for relevant news (injuries, suspensions, team selection hints)
4. Compares ML-implied probabilities against bookmaker-implied probabilities to identify value
5. Synthesises all signals into a structured recommendation with a natural language explanation

### 3.2 Recommendation Types
Each market analysed produces one of three recommendation types:

| Type | Meaning |
|---|---|
| `direct_bet` | Current odds meet the value threshold — act now |
| `conditional` | Value edge exists but current odds are below threshold — agent states the target odds and gap to current |
| `no_bet` | Agent analysed fully and found no value in this market |

The overall match-level recommendation is also one of:

| Overall | Meaning |
|---|---|
| `direct_bet` | At least one market has a direct bet opportunity |
| `conditional` | Best opportunity requires waiting for odds to move |
| `no_bet` | No value found across any market |
| `insufficient_data` | Agent could not complete analysis — distinct from `no_bet` |

`insufficient_data` is a first-class outcome. Some matches will fall outside the agent's capability (unknown teams, missing odds, very low ML feature completeness). The agent must say so explicitly rather than produce a low-confidence guess.

### 3.3 Odds Threshold
The user's minimum odds threshold is **2.0 (decimal) / +100 (American)**. The agent will never recommend a direct bet below this threshold regardless of value edge.

### 3.4 Per-Market vs Per-Match Output
The agent evaluates all configured markets and produces a recommendation per market. The match-level `overall` field reflects the strongest opportunity found. This allows the user to see, for example, a `conditional` recommendation on `btts` alongside a `no_bet` on `result_3way` for the same match.

### 3.5 Explanation
Every recommendation includes a natural language explanation of the agent's reasoning. No separate tool is needed — the LLM generates this as part of its final reasoning turn.

### 3.6 Batch Recommendation (Future)
Weekend fixture card: "give me recommendations for this weekend's Premier League." Deferred to a later milestone.

---

## 4. Output Schema

```python
class MarketRecommendation(TypedDict):
    market: str                # "btts" | "result_3way" | "total_goals" | "home_corners" | "away_corners"
    selection: str             # e.g. "yes", "home", "over_2.5"
    recommendation_type: str   # "direct_bet" | "conditional" | "no_bet"
    current_odds: float
    min_odds: float            # value threshold for this market
    ml_probability: float
    implied_probability: float
    value_edge: float          # ml_probability - implied_probability

class MatchRecommendation(TypedDict):
    match: dict                # home, away, date, league
    overall: str               # "direct_bet" | "conditional" | "no_bet" | "insufficient_data"
    markets: list[MarketRecommendation]
    explanation: str           # natural language explanation from final LLM turn
    confidence: str            # "low" | "medium" | "high"
    limitations: list[str]     # specific gaps: missing odds, cold-start, unknown team, etc.
    prediction_basis: str      # forwarded from ML tool: "team_history_and_market" | "market_odds_only" | "partial"
```

---

## 5. ML Model Contexts

The agent has access to two distinct ML forecasting tools. It selects the appropriate one based on what it learns during web search:

| Tool | Context | When to use |
|---|---|---|
| `forecast_league` | Full features: team history + market odds | Both teams play in a known league with historical data |
| `forecast_international` | Market odds only (MKT_* features) | International fixture, cup match, or team history unavailable |

These are separate tools — not a parameter — so the agent cannot accidentally call the wrong model context.

---

## 6. Agent Configuration

All tunable knobs live in `config/agent_config.yaml`. Changing knobs and re-running the backtest harness is the agent equivalent of tuning ML hyperparameters.

| Knob | Default | Effect |
|---|---|---|
| `model` | `qwen2.5:7b` | LLM powering the agent (`ollama` or Anthropic model ID) |
| `provider` | `ollama` | `"ollama"` for local; `"anthropic"` for API |
| `temperature` | `0.1` | Lower = more deterministic; use `0.0` for backtesting |
| `max_tool_calls` | `10` | Per-match tool budget — prevents runaway API costs |
| `min_odds_threshold` | `2.0` | Never recommend below this (decimal odds) |
| `min_value_edge` | `0.05` | ML probability must exceed implied probability by at least this |
| `markets` | all supported | Which targets the agent evaluates per match |
| `system_prompt_version` | `v1` | Which system prompt file to load |

**Model tiering:**

| Tier | Model | When |
|---|---|---|
| Local (free) | `qwen2.5:7b` or `llama3.2:3b` via Ollama | Development, prompt iteration |
| Cheap API | `claude-haiku-4-5-20251001` | Backtesting (many runs, cost matters) |
| Full API | `claude-sonnet-4-6` | Live recommendations (accuracy matters) |

---

## 7. Backtesting & Evaluation

### 7.1 Snapshot Store (Record / Replay)
Backtesting requires the agent to run against historical data without access to information published after each match. This is solved by a **Snapshot Store**:

- **Record mode:** during snapshot collection, every tool call (web search, ML forecast) is executed live and saved to `data/agent_snapshots/<match_id>/`
- **Replay mode:** during backtest, the snapshot store intercepts all tool calls and serves the recorded responses. Live APIs are never called. A `SnapshotMissingError` is raised immediately if a response is not found — no silent fallback.

Date-filtered web search queries (appending `before:<match_date>`) reduce post-match result leakage during the collection phase. The system prompt instructs the agent to discard any result that references a final score.

### 7.2 Backtest Harness
Runs the agent over collected snapshots in chronological order, compares each recommendation against the actual outcome in DuckDB `raw_matches`, and simulates bankroll evolution.

Two staking modes:

| Mode | Formula |
|---|---|
| `flat` | Fixed % of starting bankroll per bet |
| `kelly` | `value_edge / (odds - 1)` × current bankroll |

Evaluation report:

| Metric | Description |
|---|---|
| ROI | `(total_return - total_staked) / total_staked` |
| Hit rate | `bets_won / bets_placed` |
| Bet frequency | `bets_placed / matches_evaluated` |
| Max drawdown | Largest peak-to-trough bankroll decline |
| Insufficient data rate | Matches where agent returned `insufficient_data` |

### 7.3 Efficient Testing
- `--sample N` runs a stratified random sample before committing to a full backtest
- Up to 5 agent runs execute concurrently in replay mode
- `temperature: 0.0` in backtest config ensures reproducible LLM decisions across runs
- Different prompt versions and model tiers are compared by re-running over the same snapshots

---

## 8. CLI Commands

```bash
# Live single-match recommendation
python main.py agent-recommend \
    --home "Man City" --away "Arsenal" --date 2026-06-15

# Collect snapshots for a historical range
python main.py agent-snapshot \
    --from-date 2025-01-01 --to-date 2025-06-01 --league E0

# Backtest over collected snapshots
python main.py agent-backtest \
    --from-date 2025-01-01 --to-date 2025-06-01 \
    --stake-mode flat --sample 50
```

---

## 9. Delivery Milestones

| # | Name | Done When |
|---|---|---|
| M1 | Foundation | Agent loop runs with stub tools; LangSmith trace visible |
| M2 | Live Recommendation | Real tools, real model, structured JSON output for an upcoming match |
| M3 | Snapshot Infrastructure | Record episode; replay produces same inputs deterministically |
| M4 | Backtest Harness | ROI report over 50+ historical snapshot matches |
| M5 | Model & Prompt Tuning | Documented config with improved ROI on backtest set |
| M6 | Batch Recommendation | Weekend fixture card from one command *(future)* |

---

## 10. Extension Points

Features explicitly deferred. The architecture accommodates them without structural changes.

| Extension | Notes |
|---|---|
| Odds drift modelling | Conditional bets currently state a target threshold only. Full estimation requires intra-day historical odds time series not yet available. |
| Batch recommendation | M6 — parallel agent runs over weekend fixtures |
| Additional lookup tools | `lookup_stats` (H2H records, league tables) can be added to the tool layer without graph changes |
| Automated odds monitoring | Requires integration with a live odds feed |
| Agent tech spec | `documents/agent_techspec.md` — authoritative implementation reference, created during M1–M2 |
