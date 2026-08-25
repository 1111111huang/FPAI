# Live Recommendation Outcome Tracking & Kelly Stake Sizing — Design

**Date:** 2026-08-25
**Status:** approved, pending implementation plan
**Origin:** direct user request — live recommendations are generated but never durably tracked against actual results, so there's no way to monitor the agent's live performance or (in a later phase) feed live mistakes back into the lessons/critic loop.

## Problem

`recommendation_cache.db` (`app/backend/recommendation_cache.py`) stores every live-generated recommendation, append-only, but nothing ever resolves those cached recommendations against real results:

- The "Hit / Not Hit" badge on a completed match card (`MatchUI.tsx`) is computed **client-side, per render**, and thrown away — it reflects only whatever match a user happens to be looking at, not a durable record.
- Server-side outcome tracking exists only for bets a user manually logs (`app/backend/bet_tracker.py`, settled by `app/backend/settlement.py`) — a tiny, self-selected, currently-unreachable subset (bet-logging UI is hidden, W106/W115).
- The lessons/critic loop (`src/agent/lessons.py`) only ever learns from offline `agent-train` backtests against historical corpora — its own docstring states the live agent path is "structurally unable to read match outcomes."

This design covers **Phase 1: diagnostics only** — durably resolving every live recommendation's actual pick against real results, for the user's own querying (not a public dashboard), plus Kelly-based stake sizing shown to the user in the app. Feeding live outcomes into `agent_lessons` (the critic loop) is an explicitly separate, later phase.

## Scope decisions (from user)

- Score only the agent's **actual pick** (the `direct_bet`/`conditional` market it surfaced), not every resolvable market in the recommendation — matches what a user actually sees today (`MatchUI.tsx`'s `HitBadge`).
- Diagnostics only, no new UI for it — a backend endpoint (or CLI) the user queries directly.
- Kelly-sized ROI simulation, reusing `src/agent/staking.py`'s existing `simulate_kelly_stake` formula rather than a new one.
- The live app **does** get one new user-facing element: a per-recommendation suggested-stake multiplier, expressed in "Unit Bets" (UB), with an explanatory line added under the existing "Daily Edges" header.
- **UB is an abstract unit, not a dollar figure.** Corrected mid-design (user): not an average of logged stakes, not a settings value, not anchored to any currency — it's the standard betting-unit convention (bet 2 UB at odds 3.0 → get 6 UB back), the same fraction-of-bankroll math `staking.py` already computes today (Kelly fraction / flat-stake %), just consistently labeled "UB" instead of implying dollars. No new storage, no per-user setting, no dollar fallback — this eliminates the earlier "average stake"/"$5 default" plumbing entirely.
- Multiplier cap: 10× (reuses `staking.py`'s existing `max_fraction=0.10` Kelly cap against a 1%-of-bankroll baseline — same ceiling as backtesting, so live and simulated sizing never drift apart).

## Architecture

### Agent-side (`src/agent/`, library changes — A80–A82)

1. **`kelly_fraction(value_edge, odds, max_fraction=0.10) -> float`** — extracted from `simulate_kelly_stake`'s existing inline computation (`fraction = min(value_edge / (odds - 1), max_fraction)`, 0 for non-positive edge or odds ≤ 1). `simulate_kelly_stake` calls it instead of duplicating the formula, so backtest sizing and live sizing can never drift apart.
2. **`pick_recommended_market(markets) -> MarketRec | None`** in `market_resolution.py` — ports `MatchUI.tsx`'s `bestMarket()` (prefer a non-`no_bet` market, break ties by `value_edge`) into Python, alongside that module's existing `RESOLVABLE_MARKETS`/`market_correct`/`build_actual_outcome`, which already exist specifically so the two language implementations don't drift.
3. **`unit_bet_multiplier` enrichment** — a new deterministic post-validation pass in `schema.py`, alongside the existing `_downgrade_direct_bet_*` passes: for the recommendation's picked market (via `pick_recommended_market`), compute `kelly_fraction(value_edge, current_odds) / 0.01`, attach as `unit_bet_multiplier` (float, capped at 10.0) or `null` (no priced pick). Deterministic, never LLM-generated.

### App-side (`app/backend/`, `app/frontend/` — W167–W169)

4. **`recommendation_outcomes` table** (new, in `recommendation_cache.db`):
   ```sql
   CREATE TABLE recommendation_outcomes (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       match_id TEXT NOT NULL,
       date TEXT NOT NULL,
       competition TEXT,
       market TEXT NOT NULL,
       selection TEXT NOT NULL,
       recommendation_type TEXT NOT NULL,   -- direct_bet | conditional
       confidence TEXT,
       odds REAL NOT NULL,
       value_edge REAL,
       correct INTEGER,                     -- 0/1; unresolvable markets are never inserted
       generated_at TEXT NOT NULL,          -- from the source cache row, for lead-time analysis
       resolved_at TEXT NOT NULL,
       UNIQUE(match_id, date)
   )
   ```
   One row per match, resolved from that match's **latest** cache entry. Only `overall in (direct_bet, conditional)` recommendations are resolved — `no_bet`/`insufficient_data` have no pick to score and are never inserted.

5. **Resolution job** — `app/backend/recommendation_outcomes.py::resolve_pending_recommendations(cache, client, sweden_client=None)`, structurally mirrors `settlement.py::settle_open_bets()`: find latest cache rows per `(match_id, date)` with a past date and no existing outcome row, group by date, fetch results via the same `FootballDataClient`/`sweden_client`, resolve the picked market via `pick_recommended_market` + `market_correct`, insert. New endpoint `POST /api/recommendations/settle-open` triggers it on demand — same trigger story as `/api/bets/settle-open`, no scheduler change. Idempotent: the query guard plus `UNIQUE(match_id, date)` mean a re-run is a no-op for anything already resolved.

6. **`GET /api/recommendations/stats?days=30`** — hit rate + sample size broken down by market, competition, and confidence bucket (adapted from `bet_stats.py`'s shape), **plus** a Kelly-sized ROI simulation over resolved outcomes: same mechanics as `simulate_kelly_stake`, but iterating `recommendation_outcomes` rows (ordered by `date`) instead of backtest records, reporting ROI/hit-rate/max-drawdown via the existing `evaluation.py` report shape. `starting_bankroll`/`ending_bankroll` are plain numbers denominated in UB (e.g. `starting_bankroll=1000`), not dollars — no unit conversion needed, `staking.py`'s math was already unit-agnostic.

7. **Frontend** — each actionable match card displays its `unit_bet_multiplier` (e.g. "Suggested: 2.3 UB"), sourced from the enriched recommendation payload (item 3). One explanatory line added under the existing "Daily Edges" subtitle stat row (`MatchUI.tsx:1317-1326`) explaining the abstract-unit convention with a worked example: *"UB = Unit Bet, your standard betting unit — bet 2 UB at odds 3.0, get 6 UB back."* No API call needed for this line (no dollar value to fetch); it's static copy.

## Data flow

```
EOD batch / T-30 refresh → run_agent() → schema.py enrichment (adds unit_bet_multiplier)
    → recommendation_cache.db (unchanged storage, new field flows through)
    → frontend card display (multiplier shown)

[separately, on demand]
POST /api/recommendations/settle-open
    → find unresolved past-date cache rows → fetch real results (FootballDataClient)
    → pick_recommended_market + market_correct → recommendation_outcomes row

GET /api/recommendations/stats
    → recommendation_outcomes → hit-rate breakdown + Kelly ROI simulation
```

## Error handling

- Unresolvable markets (e.g. corners) are never coerced to a miss — `market_correct` already returns `None` for them, and `recommendation_outcomes` simply never gets a row for a recommendation whose picked market isn't in `RESOLVABLE_MARKETS`.
- A cache entry with no actionable pick (`no_bet`) is never inserted — absence, not a null `correct`.
- Missing odds (`current_odds is None`) → `unit_bet_multiplier` is `null`, not zero (zero would misleadingly read as "Kelly says bet nothing" rather than "no price to size against").
- Resolution is idempotent by construction (`UNIQUE(match_id, date)` + query guard) — safe to call `settle-open` repeatedly.

## Testing

- `tests/test_staking.py`: `kelly_fraction` unit tests (positive edge, non-positive edge, odds ≤ 1, cap applied); `simulate_kelly_stake` regression confirming identical output pre/post extraction.
- `tests/test_agent_schema*.py`: new enrichment pass — multiplier computed correctly, capped at 10.0, `null` on no-pick/no-odds.
- New `tests/test_market_resolution.py` (or extend existing): `pick_recommended_market` cases (prefers actionable over no_bet, ties broken by value_edge), mirroring `MatchUI.tsx`'s own test coverage for `bestMarket`.
- New `app/backend/tests/test_recommendation_outcomes.py`, mirroring `test_settlement.py`'s cases: resolves a won/lost pick, skips unresolvable markets, skips `no_bet`/`insufficient_data`, skips not-yet-finished matches, idempotent re-run.
- Stats-aggregation test for `GET /api/recommendations/stats` (breakdown correctness, Kelly ROI simulation over a small fixed set of outcomes).

## Explicitly out of scope

- Feeding `recommendation_outcomes` into `agent_lessons`/the critic loop (Phase 2, separate spec).
- Any new dashboard UI beyond the two additions named above (per-card multiplier, one explainer line).
- Resolving markets beyond `RESOLVABLE_MARKETS` (home/away corners) — same accepted, documented gap `market_resolution.py` already carries.
- Converting the (currently hidden) Bet Tracker page's own stake/profit/bankroll display to UB — user scoped this design to the new recommendation-card suggestion only; revisit if/when that page is un-hidden (W106/W115).
- Any dollar-denominated setting, average, or conversion for UB — it's an abstract unit throughout this feature, not a currency figure.
