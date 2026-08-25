# Agent Performance Dashboard — Design

**Date:** 2026-08-25
**Status:** approved, pending implementation plan
**Origin:** direct user request — a local-only dashboard to view the agent's live performance, built on top of the recommendation-outcome tracking and Kelly stake sizing shipped earlier this session (A80–A82, W167–W169).

## Problem

`GET /api/recommendations/stats` (W168) already gives a lean diagnostics summary (overall hit rate, simple breakdowns, one Kelly ROI number) for querying directly. The user wants a real page to actually look at: headline P&L metrics, full metrics sliced by market/direction/league, distribution charts, and concrete best/worst examples — reachable locally, no production exposure needed.

## Scope decisions (from user)

- **Data source:** every resolved live recommendation (`recommendation_outcomes`, W167), not just user-logged bets (`bet_tracker` — near-empty, its UI is hidden). "Bet Placed" means a resolved `direct_bet` pick the Kelly simulation actually staked (`simulate_kelly_stake` already only stakes `direct_bet`, never `conditional`/`no_bet` — same convention this whole feature already established).
- **Time window:** all-time by default, with a range filter (`?days=N`, mirroring W168's existing param) to narrow it.
- **Local only:** the page ships normally (committed, merged to `main` like everything else this session) but is **not linked in `AppShell`'s nav** — reachable only by visiting its URL directly, the same "unlinked, not removed" precedent this codebase already uses for the bet-tracker page (W106/W115). The backend endpoint isn't environment-gated either — it's a read-only diagnostics query with no side effects, so there's no reason to special-case it out of the normal deploy; it just never gets a nav entry pointing at it.
- **Layout:** approved via the visual companion mockup (`.superpowers/brainstorm/.../dashboard-layout.html`, gitignored — not part of the deliverable). Four sections top to bottom: filter bar, Main Metrics (KPI row), Segmentation (3 breakdown tables + 3 distribution charts), Top Winning/Losing Bets (2 tables, 5 rows each).
- **Form decisions** (dataviz skill, since this app already ships that skill's exact reference palette — `--accent`/status colors in `globals.css` are that palette's dark steps, confirmed by the file's own header comment): the three segmentations (Market / Market+Direction / League) are 5 metrics × 3 dimensions — too many bars to read as charts, so they're **data tables**, not bar charts. The three distributions (odds, stake, league bet-count) are genuinely chart-shaped: two histograms (odds, stake — sequential blue) + one categorical bar chart (league bet-count, using the palette's categorical slots 1–5 in fixed order). Main metrics are single-number stat tiles, no chart.

## Architecture

### Backend — three additions, one extension

1. **`src/agent/evaluation.py::build_evaluation_report`** — additive extension only. `total_staked`/`total_profit` are already computed locally in the function but never returned; add both to the returned dict (`"total_staked": round(total_staked, 2)`, `"total_profit": round(total_profit, 2)`). Every existing caller (`main.py`'s `agent-backtest`/`agent-train` reporting, `src/agent/comparison.py`, `recommendation_stats.py`) reads specific keys or dumps the whole dict generically (`print_report`/`save_report` both iterate `report.items()`) — a pure addition, nothing to break. No signature change.

2. **`app/backend/recommendation_stats.py`** — extend `compute_recommendation_stats`:
   - New `_segment_kelly_report(outcomes, key_fn) -> dict[str, dict]`: groups outcomes by `key_fn(outcome)`, runs the same `_to_backtest_records` → `simulate_kelly_stake` → `build_evaluation_report` pipeline **per group**, returning `{segment_key: full_report_dict}` (roi/total_staked/total_profit/bets_placed/hit_rate, same shape `kelly_roi_simulation` already has).
   - Three new top-level keys in the return dict, distinct from the existing `by_market`/`by_competition`/`by_confidence` (which stay as-is — a different, still-useful "hit rate across every resolved pick including conditional" view): `by_market_metrics` (key = `market`), `by_market_selection_metrics` (key = `f"{market}:{selection}"`), `by_league_metrics` (key = `competition`) — each built via `_segment_kelly_report`.
   - New `"staked_bets"` key: the raw, unenriched list of `simulate_kelly_stake`'s own `BankrollResult.bets` (`match_id, market, selection, odds, stake, won, payout`) — feeds the frontend's odds/stake histograms directly (client-side bucketing, no server-side bucket-boundary logic to maintain) and is the source list the new dashboard module sorts for top/bottom examples.

3. **New `app/backend/agent_performance_dashboard.py`** — the only piece that needs `RecommendationCache` (DB I/O), kept out of `recommendation_stats.py` on purpose (that file's own docstring already commits to "pure aggregation, no DB I/O of its own," mirroring `bet_stats.py`'s separation from `bet_tracker.py`):
   - `compute_agent_performance_dashboard(outcomes, cache, top_n=5) -> dict`: calls `compute_recommendation_stats(outcomes)` for everything above, then sorts `staked_bets` by `payout` to slice the top-`top_n` and bottom-`top_n`. For just those (at most `2×top_n`, not every staked bet), looks up `date`/`competition` from the matching `RecommendationOutcome` (already in `outcomes`, keyed by `match_id`) and `home_team`/`away_team` via `cache.get_latest_any_config(match_id, date)` + `src.agent.schema.reported_teams()` (the existing, already-shared helper for exactly this `home`/`home_team` key-spelling ambiguity — not a new implementation). Returns everything from `compute_recommendation_stats` plus `top_winners`/`top_losers` (each bet enriched with `home_team`/`away_team`/`date`/`competition`).
   - A cache miss (recommendation since purged, or a race) degrades that one row to `home_team`/`away_team: None` rather than failing the whole dashboard — same "never let one bad row break the page" discipline `validate_and_degrade` already established elsewhere in this codebase.

4. **`app/backend/main.py`** — new `GET /api/recommendations/performance-dashboard?days=30&top_n=5`, `Depends`-wired to `recommendations.get_cache` and `get_recommendation_outcome_store` (both already exist). Same `days` bounds validation W168 already added (`Query(30, ge=0, le=3650)`) — reused, not reinvented. Registered before `GET /api/recommendations/{match_id}` in file order, same reason W167/W168 already had to be (Starlette route-matching order — `{match_id}` would otherwise swallow the literal `performance-dashboard` path segment, the exact BUG-051-adjacent class of bug already found and fixed once this session for `/stats`).

### Frontend

5. **New page** `app/frontend/app/agent-performance/page.tsx` — **not added to `AppShell`'s `NAV_ITEMS`**, matching the W106 precedent exactly (route exists, just unlinked). Renders the four approved sections: filter bar (All time / 90d / 30d / custom, driving `?days=N`), KPI row (5 stat tiles: ROI, Total Stake, Money Won, Bets Placed, Hit % — the last four pulled from `kelly_roi_simulation`, note **Hit % here is `kelly_roi_simulation.hit_rate`, not `overall.hit_rate`** — the former is scoped to the same staked-bet population as the other four tiles, the latter includes conditional picks and would be answering a different question sitting in the same row), three breakdown tables, three distribution charts (histograms bucketed client-side from `staked_bets`, league bar chart reading `by_league_metrics[*].bets_placed` directly — no separate count computation needed, it's already in that segment report), two top-N tables.
6. **`lib/api.ts`**: new `getAgentPerformanceDashboard(days?, topN?)`.
7. **`lib/types.ts`**: new response types matching the endpoint's shape.

## Data flow

```
GET /api/recommendations/performance-dashboard?days=30
  → RecommendationOutcomeStore.list_all(since=cutoff)
  → compute_agent_performance_dashboard(outcomes, cache, top_n=5)
      → compute_recommendation_stats(outcomes)
          → simulate_kelly_stake(records) → BankrollResult
          → build_evaluation_report(records, bankroll_result)  [main metrics + kelly_roi_simulation]
          → _segment_kelly_report × 3                          [by_market/selection/league_metrics]
          → staked_bets = bankroll_result.bets                 [raw, for histograms]
      → sort staked_bets by payout → top/bottom N
      → enrich only those via RecommendationCache + reported_teams()
  → JSON response
      → agent-performance/page.tsx renders all four sections
```

## Error handling

- Zero resolved outcomes (fresh install, nothing settled yet): every section renders its own empty state (`compute_recommendation_stats` already handles `[]` cleanly per its existing tests — `sample_size: 0`, `hit_rate: 0.0`, `bets_placed: 0`); no special-casing needed beyond what already exists.
- A cache miss during top/bottom enrichment degrades that one row's team names to `None` (frontend shows the match_id as a fallback label) rather than 500ing the whole dashboard.
- `days` bounds reuse W168's existing `Query(30, ge=0, le=3650)` validation — no new edge case to handle.

## Testing

- `tests/test_agent_evaluation.py`: extend the two existing `build_evaluation_report` tests to also assert `total_staked`/`total_profit`; no new test file needed for something this small.
- `app/backend/tests/test_recommendation_stats.py`: new tests for `_segment_kelly_report` (groups correctly, empty group handled) and the three new top-level keys on `compute_recommendation_stats`'s return.
- New `app/backend/tests/test_agent_performance_dashboard.py`: top/bottom sorting correct, enrichment populates team names for a real cache hit, degrades gracefully on a cache miss, respects `top_n`.
- New `app/backend/tests/test_agent_performance_dashboard_endpoint.py`: mirrors the W167/W168 endpoint-test pattern (`app.dependency_overrides` for cache/store) — 200 with real data, empty-state 200, route-ordering sanity check (the exact regression class already caught once).
- Frontend: component tests for the new page (renders all four sections, empty state, filter changes the `days` param) plus a `tsc`/build check — no changes needed to any *other* page's tests, since this page isn't linked from anywhere.

## Explicitly out of scope

- Any nav link / production visibility — direct URL access only, by design.
- Persisting dashboard state (filter selection) across visits — always defaults to all-time on load.
- Pagination or a "show more" beyond the top/bottom N examples — 5 and 5 is the whole feature; revisit only if the user asks for more.
- Any dollar-figure display anywhere — everything stays UB-denominated, same discipline as the rest of this feature.
