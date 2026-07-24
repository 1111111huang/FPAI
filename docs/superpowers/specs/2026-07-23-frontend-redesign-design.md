# Frontend Redesign: Sidebar Shell & Sketch-Inspired Dashboard

**Date:** 2026-07-23
**Status:** Approved, pre-implementation
**Covers:** new `W##` stories to be appended to `documents/app_user_stories.md`

## Motivation

The user supplied a sketch (a SaaS-style trading/analytics dashboard mockup) and asked for the real `app/frontend` to adopt its overall UI style — sidebar shell, top search bar, denser match cards, right-rail analytics. The current app is a functional but visually minimal 3-page top-nav layout (`DraftNav` in `components/MatchUI.tsx`: Dashboard / Matches / Bets).

The sketch was designed against a much larger imagined product surface than this app actually has (5 leagues, live monitoring, alerts, per-team xG trend charts, a "Model Signals" categorization engine, user accounts). Per the user's explicit instruction, this design adopts the sketch's *structure and visual language* while dropping every element that has no real backend behind it, and rebuilds every element that does have real data honestly from that data — no fabricated numbers, no placeholder charts.

## Goals

- Rebuild the app frame around a left sidebar + right analytics rail, matching the sketch's structural identity, applied consistently across all 4 existing routes (`/`, `/matches`, `/matches/[id]`, `/bets`).
- Every new UI element is backed by a real field already returned by `app/backend` (`MatchRecommendationOut`, `Fixture`, `StatusResponse`, `BetStats`) or a pure client-side derivation over data the pages already fetch — no new backend endpoints.
- Preserve the existing locked dark palette (D6, `app/globals.css`) and existing status-color semantics (`STATUS_META` in `MatchUI.tsx`) rather than introducing a new color system.
- Preserve existing behavior test-observable via text content (team names/initials, `"Not yet generated"`, `"Log bet"`, status labels) — this is a visual/structural restyle, not a behavior change.

## Non-goals

- No new backend endpoints, no schema changes to `MatchRecommendationOut`/`Fixture`/`Bet`/`StatusResponse`.
- No accounts/auth, notifications, or multi-user affordances (avatar, bell) — still single-user per D6/"Confirmed So Far."
- No league expansion beyond the two currently supported (`E0`, `SWE`) — sketch's La Liga/Champions League/Serie A/Bundesliga sections are dropped, not stubbed.
- No xG trend charts, "Model Signals" categorized panel, Live Monitor, Models page, Leagues page, Alerts, Reports, or Settings — none of these have any real data or endpoint behind them; they are omitted entirely rather than shown disabled.
- No new color palette or chart-color decisions — the Edge Distribution donut reuses the existing `STATUS_META` status colors verbatim (already dataviz-status-safe: reserved, non-cycled, shipped with icon+label).

## Architecture

### New: `AppShell` component (`components/AppShell.tsx`)

Replaces `DraftNav` as the top-level chrome for all 4 routes. Three regions:

1. **Left sidebar** (fixed width, `--surface-1` background):
   - `FPAI` wordmark at top.
   - Nav: **Dashboard** (`/`), **All Matches** (`/matches`), **Bets** (`/bets`) — only routes that exist. Active-route styling ported from `DraftNav`'s existing `active === ...` pattern.
   - Footer stats block, stacked:
     - **Active Edges** — `count(matches where overall in {direct_bet, conditional})` over whatever match list the current page has loaded (Dashboard's `matches`/`nextMatches`, or Match Explorer's `matches`). Computed client-side, no new fetch.
     - **Model Status** — real per-league entries from `GET /api/status`'s `model_status.league`/`model_status.international` (same data `StatusFooter` already fetches: `model_type`, `primary_metric_value`, `metric_name`). Replaces the sketch's fabricated single "87% confidence" figure — there is no blended confidence score anywhere in the data model, so this slot shows what's actually real instead.
     - **Last updated** — `data_freshness.latest_match_date` / `days_since_update` / `is_stale`, same fields `StatusFooter` renders today. `StatusFooter` is retired as a standalone component; its data now lives in the sidebar footer.
2. **Top bar**: a search input, team-name substring match, promoted from `MatchExplorerPage`'s existing page-scoped search. Implementation: `AppShell` (or a small hook it owns) fetches the same 90-day-forward `getFixtures()` window Match Explorer already fetches, filters client-side on `home_team`/`away_team`, and renders a dropdown of results; selecting one navigates to `/matches/[id]?home=&away=&date=`. This duplicates one fetch already made elsewhere in the app, which is acceptable at this app's real scale (single league-pair, free-tier API, local single-user app — same tradeoff already accepted elsewhere per D2a/D2b).
3. **Main content slot**: existing page content renders here, restyled per-page below.
4. **Right rail** (Dashboard only — the sketch's rail is a dashboard-summary concept, not a per-page chrome element):
   - **Edge Distribution**: a donut over `{direct_bet, conditional, no_bet, insufficient_data}` counts among the Dashboard's currently-loaded matches (today's fixtures, or the W46 fallback window if today is empty). Colors are exactly `STATUS_META`'s existing `good`/`warning`/`muted`/`serious` — no new palette. Legend direct-labeled (4 categories), consistent with dataviz skill's ≤4-series direct-label rule.
   - **Top Edges**: top 5 matches by `bestMarket(match).valueEdge` (reusing the existing `bestMarket()` helper) among loaded matches, each linking to its Match Analysis page. Honestly scoped — this is "top edges among what's loaded today," not a global cross-league aggregate (no endpoint exists for that).

### Per-page changes

- **Dashboard (`/`)**: header/subtitle stay close to current copy. Add a client-side sort control (by edge %, by kickoff time) over already-fetched match state — no new fetch. Group matches by league (`E0` → "Premier League", `SWE` → "Allsvenskan") with a match-count badge per section, matching the sketch's grouped-section look. Match cards restyled to the sketch's denser layout (status pill top-right, team badges + names, best-market odds + edge % prominent) but **no xG trend row** — no per-team trend data exists anywhere in `MatchRecommendationOut`. Existing `MatchCard` expand-to-detail behavior (lazy recommendation generation, `TrustSignal` warnings, `LogBetButton`) is unchanged, just re-skinned.
- **All Matches / Match Explorer (`/matches`)**: same restyled card list; page-local search input can be removed since the top-bar search now covers this (or kept as a redundant affordance — implementation detail, not a design fork).
- **Match Analysis (`/matches/[id]`)**: restyled within the new shell; no functional change.
- **Bets (`/bets`)**: restyled within the new shell (stats row + bet list + manual-log form); no functional change.

### Data flow summary

No backend changes. All new visual elements are either:
1. A field the backend already returns, rendered somewhere new (Model Status, Last Updated), or
2. A pure client-side derivation over data a page already fetches (Active Edges, sort, league grouping, Edge Distribution, Top Edges), or
3. An existing fetch relocated/reused (top-bar search reuses Match Explorer's fixture fetch+filter).

### Testing

Existing component tests (`MatchUI.test.tsx` and siblings) assert on text content (`"ARS"`, `"Arsenal"`, `"Not yet generated"`, `"Log bet"`, status labels) and interaction flows, not on `DraftNav`/layout structure — grepped and confirmed no test references `DraftNav` or nav labels directly. Restyling is low-risk to these; the implementation plan should explicitly preserve the exact text strings/roles these tests query. New client-side derivations (Active Edges count, Edge Distribution counts, Top Edges ranking, league grouping, sort) get their own unit/component tests per this repo's existing test-strategy convention (W22).

## User stories

Append a new phase to `documents/app_user_stories.md` (`W##` continuing from the current max, `W63`), one story per: `AppShell` scaffold + sidebar nav, sidebar footer stats, top-bar global search, Dashboard league-grouping + sort, Dashboard right rail (Edge Distribution + Top Edges), match card restyle, All Matches/Match Analysis/Bets restyle within the shell. Each gets acceptance criteria following the doc's existing convention, and gets marked `completed` as implemented per `CLAUDE.md`'s workflow instructions.
