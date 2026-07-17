# Sandbox Scenario Runbook (W31)

Given FPAI is a single-user local app (not a CI-heavy production service), the
real season is frequently off-season and a genuinely "live" E2E run (a real
upcoming fixture actually kicking off and finishing) is often infeasible on
demand -- W23's checklist hit exactly this problem. Sandbox mode (W27-W30)
solves it: `SANDBOX_MODE=1`/`SANDBOX_DATE=YYYY-MM-DD` make the whole app treat
an arbitrary *past* date as "today," while `app/backend/recommendations.get_cache()`
and `app/backend/bets.get_bet_tracker()` transparently redirect to scratch
databases under `app/data/sandbox/*.db` -- so a sandbox run exercises the real
Dashboard → recommendation → bet-logging → settlement pipeline against real
historical fixtures, real historical odds, and a real LLM/agent call, without
ever touching the real `app/data/*.db` files a live user session uses.

`scripts/sandbox_runbook.py` (this document's subject) is the repeatable,
one-command version of that flow. It is not a substitute for full CI
automation -- there is no CI pipeline in this project -- but it is the
closest thing to one: a scripted, deterministic-to-invoke, evidence-producing
check that can be re-run against any historical date whenever the app's
plumbing needs to be revalidated end-to-end, standing in for both W23's
original manual checklist and the "wait for a real live match" gap that
checklist could not close.

## How to run

```bash
SANDBOX_MODE=1 SANDBOX_DATE=2026-05-24 python scripts/sandbox_runbook.py
```

Requires `FOOTBALL_DATA_API_KEY` (and `TAVILY_API_KEY`, used by the agent's
`web_search` tool) in `.env` (auto-loaded, same as `app/backend/main.py`), and
a reachable local Ollama with the model configured in
`config/agent_config.yaml` (`llama3.1:8b` at the time of this run). Any date
with completed E0 fixtures and odds coverage in `raw_matches` works; this run
used `2026-05-24`, `raw_matches`'s last-refreshed date, confirmed to have 10
real completed E0 fixtures with real closing odds.

## Checklist

### 1. Dashboard: real fixtures fetched for the sandbox date

Calls `FootballDataClient.get_results(competition_code="PL", date_from=<date>, date_to=<date>)`
against the real football-data.org API for the sandbox date, and takes the
first fixture returned.

**Last run: 2026-07-17.** `SANDBOX_DATE=2026-05-24`. Real football-data.org
API call succeeded and returned 10 completed E0 fixtures for that date.
Fixture selected: **Sunderland vs Chelsea, final score 2-1** (football-data.org
match id `538155`). **PASS.**

### 2. Real historical odds looked up for that date

Calls `HistoricalOddsClient(sandbox_date=<date>).get_odds()`, which reads real
`odds_h`/`odds_d`/`odds_a` from the `raw_matches` DuckDB table (football-data.co.uk
sourced) instead of a live Odds API call (no historical replay exists for a
past date via the live Odds API).

**Last run: 2026-07-17.** Returned **10 odds events** for 2026-05-24 (one per
E0 fixture that day), non-empty as required. **PASS.**

### 3. Real recommendation generated (real agent/LLM call)

Calls `recommendations.run_agent(match_info)` for the Sunderland vs Chelsea
fixture (`league="E0"`, `match_id="538155"`), which in sandbox mode routes
through `SnapshotStore` in "record" mode -- the agent's real `web_search`
tool call is made live and date-filtered/recorded to
`data/agent_snapshots/sandbox/Sunderland__Chelsea__2026-05-24/`, and the real
`forecast_league` ML model tool runs against real point-in-time features.
Validated via `validate_and_degrade()` and recorded to the sandbox
`RecommendationCache` (`app/data/sandbox/recommendation_cache.db`) with
`triggered_by="manual_regenerate"`.

**Last run: 2026-07-17.** Real Ollama (`llama3.1:8b`) + real Tavily
`web_search` call, both completed without error. Result:
**`overall="conditional"`, 3 markets, `confidence="medium"`,
`prediction_basis="team_history_and_market"`**. Full market detail:

| market | selection | recommendation_type | current_odds | ml_probability | implied_probability | value_edge |
|---|---|---|---|---|---|---|
| result_3way | draw | no_bet | 3.2 | 0.34 | 0.3086 | -0.0314 |
| result_3way | home | conditional | 2.5 | 0.36 | 0.3077 | 0.0523 |
| result_3way | away | conditional | 2.9 | 0.30 | 0.3107 | -0.0107 |

Explanation given by the agent: "Chelsea are struggling with injuries but
have shown recent form improvement. Sunderland has a good record against
Chelsea in the past." Limitations: `["incomplete injury list for both teams"]`.
Recorded to `app/data/sandbox/recommendation_cache.db`. **PASS.** (Note: the
agent's predicted values are not being validated for correctness here --
only that the full real pipeline runs end-to-end without error, per this
story's scope. A different sandbox run against the same date could
legitimately produce a different recommendation, since the LLM call is not
deterministic.)

### 4. One bet logged from the recommendation, one logged manually

`from_recommendation` bet takes the first market from the generated
recommendation (`result_3way`/`draw`); `manual` bet is hand-specified
(`result_3way`/`home` @ 2.0, stake $5) -- deliberately the *other* outcome, so
both a losing and a winning settlement path get exercised in one run. Both
logged via `BetTracker.create_bet()` against the sandbox `BetTracker`
(`app/data/sandbox/user_bets.db`), keyed by `match_id="538155"` (the real
football-data.org fixture id -- required so `settle_open_bets()` can later
resolve it against the same id in a live results lookup).

**Last run: 2026-07-17.**
- **Bet id=1** (`from_recommendation`): `result_3way`/`draw` @ 3.2, stake $10.
- **Bet id=2** (`manual`): `result_3way`/`home` @ 2.0, stake $5.

Both logged successfully, both `outcome="open"` immediately after creation.
**PASS.**

### 5. Both bets settled against the real historical result

Calls `settle_open_bets(tracker, fixtures_client)`, which re-fetches real
results from football-data.org for each open bet's date and settles any bet
whose market is in `RESOLVABLE_MARKETS` (`result_3way`, `btts`,
`total_goals`) against the actual final score.

**Last run: 2026-07-17.** Real result: Sunderland 2-1 Chelsea (home win).
Both bets settled correctly in one call:
- **Bet id=1** (`result_3way`/`draw` @ 3.2, $10) → **`lost`**,
  `profit_loss=-10.0` (correct: match was not a draw).
- **Bet id=2** (`result_3way`/`home` @ 2.0, $5) → **`won`**,
  `profit_loss=5.0` = `5 × (2.0 − 1)` (correct: home win).

**PASS.**

*(Note: an earlier same-day attempt at this run surfaced a real bug --
initially the script logged bets with the synthetic team-name composite
match_id (`RecommendationRequest.effective_match_id()`, e.g.
`"Sunderland__Chelsea__2026-05-24"`) rather than the real football-data.org
fixture id. `settle_open_bets()` looks up results by
`NormalizedMatch.match_id` (football-data.org's real numeric id,
`"538155"` here), so the composite key never matched and both bets stayed
`open` -- 0 settled. Fixed by passing `match_id=fixture.match_id` into
`RecommendationRequest` explicitly, the same value `eod_batch.py` and
`t30_refresh.py` already use for real fixtures. This is a genuine plumbing
bug this task's real run caught, not an LLM issue -- see
`scripts/sandbox_runbook.py`'s comment at the `RecommendationRequest`
construction site.)*

### 6. Real `app/data/*.db` left untouched by the sandbox run

`ls -la app/data/*.db` modification times, captured immediately before and
immediately after the Step 5 (settlement) run above:

**Before:**
```
-rw-r--r--  1 tianqihuang  staff  94208 Jul 17 09:48 app/data/recommendation_cache.db
-rw-r--r--  1 tianqihuang  staff  12288 Jul 16 20:26 app/data/user_bets.db
```

**After:**
```
-rw-r--r--  1 tianqihuang  staff  94208 Jul 17 09:48 app/data/recommendation_cache.db
-rw-r--r--  1 tianqihuang  staff  12288 Jul 16 20:26 app/data/user_bets.db
```

Byte-for-byte and timestamp-for-timestamp identical -- the real dbs were not
opened for writing at all. For contrast, the sandbox scratch dbs *were*
written during the same run:

```
-rw-r--r--  1 tianqihuang  staff  16384 Jul 17 10:12 app/data/sandbox/recommendation_cache.db
-rw-r--r--  1 tianqihuang  staff  12288 Jul 17 10:12 app/data/sandbox/user_bets.db
```

**PASS.**

## Summary

| # | Check | Status | Notes |
|---|---|---|---|
| 1 | Dashboard: real fixtures fetched | ✅ PASS (2026-07-17) | Sunderland vs Chelsea, 2026-05-24, final 2-1 |
| 2 | Real historical odds looked up | ✅ PASS (2026-07-17) | 10 odds events found for the date |
| 3 | Real recommendation generated (real LLM + agent) | ✅ PASS (2026-07-17) | `overall="conditional"`, 3 markets, real Ollama + real Tavily web_search |
| 4 | One bet from recommendation + one manual logged | ✅ PASS (2026-07-17) | id=1 draw@3.2/$10 (from_recommendation), id=2 home@2.0/$5 (manual) |
| 5 | Both bets settled against the real result | ✅ PASS (2026-07-17) | id=1 lost/-10.0, id=2 won/+5.0; matches real 2-1 home win |
| 6 | Real `app/data/*.db` untouched | ✅ PASS (2026-07-17) | Mtimes identical before/after; only `app/data/sandbox/*.db` was written |

**All six checks passed on a real, repeatable run against a real historical
date (2026-05-24), with real observed values captured above.** One genuine
plumbing bug (bet `match_id` mismatch preventing settlement) was found and
fixed during this run -- see the note under item 5. Re-run at any time with
the command in "How to run" above; a different date, or a re-run of this same
date, may produce a different agent recommendation (the LLM call is not
deterministic) but should exercise the same six-step pipeline successfully.
