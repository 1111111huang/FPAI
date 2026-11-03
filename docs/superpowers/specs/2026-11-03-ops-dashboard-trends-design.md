# Ops Dashboard Trend Visualizations — Design

## 1. Problem

The `/ops` dashboard (shipped 2026-11-02) gives Failures, Odds Availability, Feature
Health, and Performance tabs, each showing a single windowed snapshot — "what does
this look like right now, in this date range." Using it live against production
surfaced that snapshots can't answer the actual question it was built for: *"is the
deployed app behaving correctly, over time?"*

Concretely, none of these are answerable from the current build:
- Which odds markets are **constantly** missing, not just missing today?
- Which features are **constantly** drifting to extreme values or going null, not
  just what today's per-league averages look like?
- Is agent performance trending up or down **week over week**, not just "in the
  last 30/90 days from right now"?

A secondary, concrete bug found during this same review: changing the shared
date-range filter had no effect on the Performance tab's numbers at all — it
renders the pre-existing `AgentPerformanceContent` component unmodified, which
manages its own independent "days" selector, completely decoupled from the rest
of the dashboard. (This was already known and explicitly documented in
`OpsDashboard.tsx`'s own header comment as of the last ship — the user flagged it
as unacceptable rather than an accepted tradeoff, once actually using the page.)

## 2. Goals

- Odds Availability, Feature Health, and Performance each get a genuine
  **week-over-week trend view** (Thursday-anchored week buckets) instead of a
  single-window snapshot, answering "is this a persistent problem or a one-off."
- The shared Window/League filter bar actually drives all three of these tabs,
  including Performance — closing the decoupling gap above.
- Feature Health's redesign must stay legible with 50+ tracked features: only
  flagged (drifting or elevated-null-rate) features ship to the frontend, sorted
  by severity, with the UI further capping/expanding/searching client-side.
- Performance supports comparing betting **strategies**, not just a single fixed
  definition of "performance": one-pick-per-match (today's existing behavior) vs.
  all direct-bet candidates per match, and direct-bets-only vs.
  direct-bets-plus-conditional (conditional staked at its `current_odds` when
  included — an explicit, deliberate simplification, not a claim that this
  reflects real tradeable conditional-bet economics).
- Visual language carries over from the "Weekend Pick Review" artifact (warm
  palette, Bricolage Grotesque/Source Serif 4/IBM Plex Mono type system,
  scorecard tiles, severity-colored pills) — chosen by direct reference, not
  reinvented. Every chart gets a real, labeled y-axis; no bare bars/sparklines
  with no scale.

## 3. Non-goals

- The **Failures** tab is unchanged — the trend-visualization problem was never
  raised against it. (Noted as a candidate follow-up if it turns out to matter
  once this ships, not bundled into this pass.)
- No changes to how recommendations are generated, staked, or resolved in
  production — this is read-only aggregation over existing stored data
  (`forecast_payload`, `recommendation_generations`, `recommendation_outcomes`).
  Nothing here touches `eod_batch.py`, `recommendation_stats.py`'s existing
  `compute_recommendation_stats`, or any live betting/staking path.
- No new feature-value instrumentation. Every signal used (odds-availability,
  `forecast_payload.explainability.top_features`, cached `candidates` lists) is
  already recorded today; this is new aggregation over old data, not new
  collection.
- `max_drawdown` and `insufficient_data_rate` (both real `SegmentMetrics` fields)
  are not wired into the Performance trend's metric toggle in this pass — the
  user picked hit_rate/ROI/bet_frequency/total_profit specifically. Easy to add
  later; the trend endpoint's per-week aggregation isn't metric-specific enough
  to make this costly to extend.

## 4. Shared filter bar change

`OpsDashboard.tsx`'s From/To date `<input type="date">` pair is replaced with a
**Window** chip selector: `4w | 8w | 12w | 26w`, defaulting to `8w`. Rationale:
arbitrary date-range pickers fight Thursday-anchored week buckets (a user could
pick a from/to pair that straddles partial weeks) — a week-count selector can't
produce that ambiguity. League filter is unchanged.

Week boundaries: given `asOf` (today, sandbox-aware per the existing
`useSandboxAsOf` convention), the most recent *complete* week ends the Wednesday
before (or today, if today is Wednesday) and starts the Thursday before that;
the window extends backward N such weeks. ("This week," the newest bucket, is
necessarily partial until the following Wednesday — shown as-is, not excluded,
since a partial-week's data is still informative, just not yet final; the UI
doesn't need to call this out specially, the same way the existing dashboard
doesn't distinguish "today's data is incomplete" anywhere else.)

The Failures tab keeps using `league` only (no date window — it already shows a
simple recent list, not a trend, per the non-goal above).

## 5. Odds Availability — trend view

**New backend**: `GET /api/admin/ops/odds-availability-trend?weeks=&league=`

Reuses the existing `get_ops_odds_availability` fixture-level logic internally —
for each of the N Thursday-anchored weeks in range, run the same
"fixture → missing markets → likely_cause" computation already shipped, then
roll up into a per-market, per-week count: `missing_count` / `total_fixtures`.
Response shape:

```json
{
  "weeks": [
    {
      "week_start": "2026-10-08",
      "total_fixtures": 112,
      "by_market": {
        "total_corners": {"missing": 76, "circuit_open": 61, "not_posted": 15},
        "asian_handicap": {"missing": 38, "circuit_open": 2, "not_posted": 36}
      }
    }
  ]
}
```

**Frontend**: the ranked horizontal-bar chart (already mocked and approved) —
one bar per market, summed `missing / total_fixtures` across the *whole*
selected window (not per-week), sorted descending, each with a labeled 0–100%
axis. Below it, a per-league breakdown table for the single worst-offending
market, columns: League / Fixtures / Missing % / dominant likely_cause —
answering "is this vendor-wide or one league's problem," using the same
per-week data already fetched (no second request).

## 6. Feature Health — trend view

**New backend**: `GET /api/admin/ops/feature-health-trend?weeks=&league=`

For every cached recommendation in the window with a non-null
`forecast_payload`, reads `forecast_payload.explainability.top_features` — a
list of `{feature, value, shap_value}` already recorded per recommendation
(confirmed present via the Weekend Pick Review artifact's own data — this is
historical, not new instrumentation). Buckets by week, by feature name:

- `null_rate`: fraction of this week's recommendations where that feature's
  `value` was null/missing (0 if the feature wasn't even in the top_features
  list that often — "not surfaced as a top feature" and "present but null" are
  collapsed for this purpose; a feature that never breaks into `top_features` at
  all for a match contributes 0 to both numerator and denominator, not treated
  as implicitly null).
- `mean`: average of non-null numeric values that week.

**Baseline**: the earlier half of the selected window's weeks (e.g. weeks 1–4 of
an 8-week window). **Drift flag**: a feature's back-half weekly `mean` deviates
from the baseline mean beyond a threshold — exact threshold tuned during
implementation (a magnitude-ratio or z-score approach, whichever is simpler to
compute correctly and explain; not committing to a specific formula here).
**Null-rate flag**: back-half `null_rate` is elevated relative to baseline
`null_rate` beyond a threshold, same tuning note. A feature can carry both flags.

**Minimum window**: if `weeks < 4` (not enough to split into a meaningful
baseline/recent half), the endpoint still runs but is noted as lower-confidence
— exact UX for this (a banner, or just fewer flags surviving) is an
implementation-time call, not a blocking design question.

Only flagged features are included in the response — not all 50+ tracked ones.
Response shape:

```json
{
  "features": {
    "XOC_HOME": {
      "category": "offense",
      "flags": ["drift"],
      "baseline_mean": 1.8,
      "recent_mean": 12.7,
      "weekly": [{"week_start": "2026-10-08", "mean": 1.9, "null_rate": 0.0}, ...]
    }
  }
}
```

**Frontend**: a 4-tile scorecard (tracked / drifting / elevated-null-rate / both)
+ a severity-sorted list (worst magnitude-ratio first), each row showing the
feature name, a category tag (derived client-side or server-side from the name's
existing prefix convention — `OFF_`/`DEF_`/`MKT_`/`CTX_`/`DIS_`/`SQUAD_`/`H2H_`/
`LUCK_`/`EFFICIENCY_`/etc.), flag badge(s), a small labeled-axis sparkline, and
the baseline→recent stat line. Capped to the top 8 rows with a "show N more"
client-side expand (no re-fetch — the full flagged list is already in the
response, this is a pure UI reveal). A client-side search/filter box narrows by
feature name substring.

## 7. Performance — trend + strategy view

**New backend**: `GET /api/admin/ops/performance-trend`
`?weeks=&league=&pick_scope=one_per_match|all_direct&include_conditional=false|true`

For each Thursday-anchored week in range:

- `pick_scope=one_per_match` (default, matches today's existing behavior):
  reads `RecommendationOutcomeStore`'s already-resolved outcome rows for
  matches in that week — the same data `compute_recommendation_stats` already
  consumes.
- `pick_scope=all_direct`: for every match in that week with a resolved outcome
  (so the real final score is already known, via the outcome row's stored
  `home_goals`/`away_goals`), loads that match's cached
  `recommendation_json.candidates`, filters to `recommendation_type ==
  "direct_bet"`, and grades **every** such candidate independently via the
  existing `market_correct()` (`src/agent/market_resolution.py`) against the
  known final score — each graded candidate becomes its own stakeable unit,
  not just the single historically-chosen pick.
- `include_conditional=true`: adds `recommendation_type == "conditional"`
  candidates to that same population (whichever `pick_scope` is active),
  staked at their own `current_odds` — the user's explicit, deliberate choice,
  made with the caveat understood that this doesn't reflect real
  price-triggered conditional-bet economics.

All four combinations feed the same existing Kelly-staking pipeline
(`simulate_kelly_stake` / `build_evaluation_report`) per week, one
`BacktestRecord` per stakeable unit (one per match in `one_per_match` mode, one
per graded candidate in `all_direct` mode) — reusing, not reimplementing, the
ROI/bankroll simulation machinery `compute_recommendation_stats` already relies
on. Response shape, one entry per week:

```json
{
  "weeks": [
    {
      "week_start": "2026-10-08",
      "hit_rate": 0.52, "roi": 0.182, "bets_placed": 8,
      "total_profit": 42.3, "ending_bankroll": 1042.3
    }
  ]
}
```

**Frontend**: the weekly bar chart (already mocked/approved, labeled 0–100% or
auto-scaled axis depending on the selected metric) with a metric toggle (Hit
rate / ROI / Bet frequency / Total profit) and the 2×2 strategy toggle
(pick_scope × include_conditional), all re-fetching and redrawing together. A
"this week's detail" by-market/by-league table below stays scoped to whichever
combination is currently selected, replacing the existing
`AgentPerformanceContent`'s own independent day-count selector for this tab
specifically (the standalone `/agent-performance` route is untouched — it still
renders `AgentPerformanceContent` with its own all-time/90/30-day selector,
unaffected by anything in this design).

## 8. What's explicitly reused vs. new

| Piece | Status |
|---|---|
| `market_correct()` (candidate grading) | Reused as-is |
| `simulate_kelly_stake` / `build_evaluation_report` (ROI/staking) | Reused as-is |
| `RecommendationOutcomeStore` (one-pick-per-match data) | Reused as-is, unchanged schema |
| `RecommendationCache.recommendation_json.candidates` (all-candidates data) | Reused, newly read for this purpose |
| `forecast_payload.explainability.top_features` | Reused, newly aggregated for this purpose |
| Existing fixture/odds-availability logic | Reused, newly bucketed by week |
| 3 new backend endpoints (`odds-availability-trend`, `feature-health-trend`, `performance-trend`) | New |
| Frontend: Window chip selector, 3 redesigned tab bodies, metric/strategy toggles | New |
| Weekend Pick Review's visual tokens (palette, type, scorecard/pill patterns) | Reused by direct reference |

## 9. Self-review notes

- **Spec coverage**: every item the user raised (constantly-missing markets,
  constantly-extreme/incorrect features, week-over-week performance, "the date
  range doesn't affect Performance" bug, the strategy-comparison request, the
  Thursday-week-start correction, the missing-y-axis correction) is addressed
  above.
- **Placeholder scan**: no TBD/TODO left in; the two explicitly-deferred
  numeric thresholds (drift/null-rate flag cutoffs) are called out as
  implementation-time tuning decisions, not missing requirements — the
  *mechanism* (baseline vs. recent-half comparison) is fully specified, just
  not the exact cutoff constant, which is the right level of detail for a
  design doc vs. an implementation plan.
- **Consistency**: `pick_scope`/`include_conditional` naming is consistent
  between the endpoint's query params and the frontend toggle description;
  response field names match between each endpoint's JSON shape and its
  frontend consumption description.
- **Scope**: this is one cohesive redesign (3 tabs sharing one filter-bar
  change and one visual language), appropriately sized for a single
  implementation plan — not decomposed further.
