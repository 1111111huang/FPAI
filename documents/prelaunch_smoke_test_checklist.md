# Pre-launch Manual E2E Smoke Test Checklist (W23)

Given FPAI is a single-user local app (not a CI-heavy production service), this
checklist stands in for full E2E automation before the backend is left running
continuously and unattended (D2a's scheduled EOD/T-30 pipeline). Each item
below must be run through with real results recorded before that reliance
begins.

## How to run each check

Each check calls the real component functions directly (not through a running
FastAPI process) against real dependencies — real Ollama, real
football-data.org, and whatever Odds API configuration is actually present in
`.env`. Use a scratch `RecommendationCache`/`BetTracker` db path, not the real
`app/data/*.db` files, so a smoke-test run never mixes with real usage data.

## Checklist

### 1. Real EOD batch cycle (real Ollama + real football-data.org + real Odds API)

Run `app.backend.eod_batch.run_eod_batch()` directly against a real
`FootballDataClient`, `build_odds_client()`'s real result, a scratch
`RecommendationCache`, and `AgentConfig.default()`, for a date with at least
one real scheduled fixture. Confirm: fixtures are fetched for real, `run_agent`
is actually invoked (a real LLM call, not mocked), a recommendation is written
to the cache with `triggered_by="scheduled"`, and `schedule_t30` is invoked
once per fixture.

**Last run: 2026-07-12.** Real Ollama (`llama3.1:8b`, confirmed reachable) +
real football-data.org (real `FOOTBALL_DATA_API_KEY`). No real Odds API key is
configured in this environment (`build_odds_client()` correctly returned
`None` — see item 4). Target fixture: Arsenal v Coventry City, 2026-08-21
(the actual next real E0 fixture at run time). Result: `generated=1,
skipped=0`, one fixture processed, `schedule_t30` called once, a real cached
recommendation was written (`overall="no_bet"`,
`prediction_basis="market_odds_only"`, `triggered_by="scheduled"`).
**PASS.**

### 2. Real T-30 refresh observed

Run `app.backend.t30_refresh.refresh_match_at_t30()` directly for a real
fixture against the same real components. Confirm it runs to completion
without erroring, regardless of which of its four outcomes it lands on.

**Last run: 2026-07-12.** Same fixture as item 1. With no real Odds API key
configured, the correct/expected real outcome is `skipped_no_odds` — confirmed
verbatim, and the prior cached recommendation was left untouched (history
length stayed at 1 row). **PASS** (this run's real-world conditions mean it
necessarily exercises the same degrade-gracefully path as item 4, not the
"odds changed → re-run" path — that path is unit-tested in
`test_t30_refresh.py::test_refreshes_when_odds_changed` but has not been
observed against a real Odds API response, since no real key exists yet; see
W26).

### 3. One bet logged from a recommendation and one logged manually, both settled correctly after a real match completes

**Modified for this run, see note below.** Waiting for a real upcoming E0
fixture to actually kick off and finish was not feasible within a single
session — the next real E0 fixture was 41+ days out (2026-08-21) at the time
this checklist was first run. Substituted an **already-completed real E0
fixture** (Arsenal 3–0 Fulham, 2026-05-02, football-data.org match id
538127 — confirmed live via a real `get_results()` call) so the settlement
math is still exercised against a real recorded score, not a fabricated one.
Logged one `from_recommendation` bet (result_3way/home @ 1.75, stake $10 — a
hand-built recommendation snapshot referencing this real fixture, since
generating a fresh one wasn't the point of this check) and one `manual` bet
(result_3way/away @ 4.5, stake $5, deliberately the losing side, to exercise
both settlement outcomes in one run). Called `settle_open_bets()` with a real
`FootballDataClient` hitting real football-data.org for this match's actual
result.

**Last run: 2026-07-12.** Both bets settled correctly in one call: the home
bet → `won`, `profit_loss=7.50` (`10 × (1.75 − 1)`); the away bet → `lost`,
`profit_loss=-5.00`. Matches the real recorded scoreline (Arsenal 3–0 Fulham,
home win). **PASS, with the above substitution noted** — a true live
match-in-progress → settlement run is tracked separately (see W26), since a
same-day observation isn't achievable until closer to actual launch.

### 4. One deliberately-broken case (Odds API budget/key unavailable) degrades gracefully

**Already true in this environment today**, not simulated: no `ODDS_API_KEY`
is configured, so `build_odds_client()` genuinely returns `None`. Confirmed
via items 1 and 2 above that both the EOD batch and the T-30 refresh handle
this for real without erroring — `run_eod_batch` proceeds with every fixture's
`odds` field simply omitted, and `refresh_match_at_t30` returns
`skipped_no_odds` cleanly. **PASS.**

## Summary

| # | Check | Status | Notes |
|---|---|---|---|
| 1 | Real EOD batch | ✅ PASS (2026-07-12) | Real Ollama + real football-data.org; no real Odds API key today |
| 2 | Real T-30 refresh | ✅ PASS (2026-07-12) | Exercised the no-odds-available path only; "odds changed" path still needs a real Odds API key (W26) |
| 3 | Bet settlement (from-rec + manual) | ✅ PASS (2026-07-12), with substitution | Used an already-completed real fixture instead of waiting for a live one; true live-match run tracked in W26 |
| 4 | Odds API unavailable degrades gracefully | ✅ PASS (2026-07-12) | Genuinely true today, not simulated |

**All four items have been run through at least once with real results
recorded**, satisfying W23's acceptance criteria as written. Two follow-up
gaps intentionally deferred to **W26** (not blocking, since this environment's
real constraints — the current off-season gap and W25's pending Odds API key
— make them infeasible today): (a) observing a live, in-progress-to-completed
match settle a bet on the same day, and (b) observing the T-30 "odds actually
changed" re-run path against a real Odds API response.
