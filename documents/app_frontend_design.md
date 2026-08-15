# Frontend Design — FPAI Web App

> Design/UX reference: what each page and component is for, how they fit together, and the
> journeys a user actually walks through. Companion to `app_techspec.md` §11 (implementation
> detail — modules, work items) and `app_prd.md` (product scope). This doc describes the app as
> built, not a new proposal — treat it as the source of truth to update when the design changes.

## 1. Design Language

Dark, data-dense, terminal-adjacent. One accent color, a fixed 4-color status vocabulary, no
decoration beyond what encodes information.

| Token | Value | Use |
|---|---|---|
| `--page-plane` | `#0d0d0d` | page background |
| `--surface-1` | `#1a1a19` | cards, inputs, hover surfaces |
| `--text-primary` / `--text-secondary` / `--text-muted` | white → gray → dim gray | ink hierarchy |
| `--accent` | `#3987e5` (blue) | links, focus rings, active nav/tab, Edge Distribution's "Completed" slice |
| `--status-good` | `#0ca30c` (green) | Direct Bet / positive edge / won / Hit |
| `--status-warning` | `#fab219` (amber) | Conditional / wait / stale data |
| `--status-serious` | `#ec835a` (orange-red) | Insufficient Data / errors / data issues / Not Hit |
| `--status-critical` | `#d03b3b` (red) | `LiveBadge` (W144) |

The status colors are the app's one recurring visual system — `StatusBadge`, the Edge Distribution
donut, bet outcome text, and stale-data warnings all draw from this same 4-color palette rather
than each inventing their own. The Edge Distribution donut's one exception is its "Completed"
slice (W147), which reuses `--accent` rather than a status color — a completed match's outcome is
shown per-card via `HitBadge`'s green/orange, not by this panel, so the slice needed a neutral
"done" color rather than one of the 4 that already carry a good/bad meaning. Typography is
system-default sans + `font-mono` for every number (odds, probabilities, edges, dates) so figures
align and scan as data, not prose.

## 2. Layout Shell — `AppShell`

Every page renders inside `AppShell`: a left sidebar (nav + live status footer) and a top search
bar, wrapping page content in `<main>`. It is not a persisted layout component (no Next.js
`layout.tsx` shell) — each page mounts its own `AppShell` instance, so sidebar status data
re-fetches on every navigation.

**Sidebar**
- Nav links: Dashboard (`/`), All Matches (`/matches`). *(Bet Tracker exists and is fully wired
  — route, page, nav-highlight state — but its link is currently hidden per W106; see §7.)*
- Status footer (from `GET /api/status`): Active Edges count for the current page (only pages that
  compute one pass it in), Model Status (`league N · international M`), Last Updated freshness per
  competition with a stale warning in amber when data is old.

**Search bar**
- Single fixture search, present on every page. Lazy — does not fetch until first focus, then
  searches the next 90 days of fixtures by team name, showing up to 8 results as a dropdown of
  links straight into Match Analysis.

## 3. Pages

### 3.1 Dashboard (`/`) — "what should I look at today"

The landing page and default journey entry point. Shows the next 10 upcoming fixtures (E0, La
Liga, Allsvenskan) from today forward, grouped by date, each as a collapsed `MatchCard`.

- **Kickoff / Edge %** toggle re-sorts matches *within* each date group (date-group order itself
  always stays kickoff-order, so the toggle can't scramble which day a match appears under).
- Any match already recommendation-cached from the overnight batch job shows its verdict
  immediately, no click needed; everything else shows "Not yet generated" until expanded.
- **Dashboard Rail** (right sidebar, desktop only): an Edge Distribution donut chart (counts of
  Direct Bet / Conditional / No Edge / No Data across loaded matches) and a Top Edges list (top 5
  by value edge, linking straight to each match's analysis page).

This is the "check in once a day" surface — comes pre-populated by the backend's overnight
precompute so most matches need zero waiting.

### 3.2 Match Explorer (`/matches`) — "find a specific match"

A wider, flatter version of the Dashboard: same `MatchCard` list, no date grouping, no rail, no
sort toggle, but a 90-day fixture window with live text search instead of a fixed 10-match cap.
This is the browse/search journey — used when the user wants a fixture that isn't in the
Dashboard's next-10, e.g. planning a bet a month out.

### 3.3 Match Analysis (`/matches/:id`) — "why, exactly"

The deep-dive page, reached from a `MatchCard`'s "Full analysis" link, a search result, or a
direct URL with `?home=&away=&date=&league=` query params (all four required — missing any shows
a "Missing match details" fallback with a link back to Match Explorer).

Sections, top to bottom:
1. **Header** — teams, competition, date, and a large verdict (BET / WAIT / PASS / NO READ) with
   confidence and trust-signal badges.
2. **Model Probabilities** — one row per market: ML probability, current odds, value edge,
   recommendation status. A conditional market with a computed target price shows "Needs X.XX+ to
   clear edge" instead of just a static edge number. Each actionable row has an inline **Log bet**
   control (see §5).
3. **Squad Intelligence** — placeholder; the agent's player/squad reasoning exists in the engine
   but isn't yet exposed through this endpoint, so this always reads "Not yet exposed by the API
   for this view."
4. **Agent Reasoning** — the bulleted explanation and limitations the agent generated for this
   recommendation.

If no recommendation is cached yet, this page triggers a live agent call on load (same
cache-then-generate fallback `MatchCard` uses) rather than requiring an extra click.

### 3.4 Bet Tracker (`/bets`) — "what did I actually stake" *(nav-hidden, W106)*

Manual bet ledger: summary stats bar (bankroll, ROI, hit rate, max drawdown), a manual bet-logging
form (search a real fixture, fill in market/selection/odds/stake), the full bet list, and a
"Settle open bets" button that checks finished fixtures against open bets and marks them
won/lost/profit. Reachable today only by direct URL — the nav link is commented out, not deleted,
because the feature works but wasn't judged ready to surface. See §7 for what "ready" would mean.

## 4. Shared Components

| Component | Role |
|---|---|
| `MatchCard` | The one card used on both Dashboard and Match Explorer. Collapsed: kickoff time, tier tag, teams with color badges, best market + edge or day label. Click to expand: lazily fetches/generates the recommendation, then shows the explanation bullets and a link to full analysis. Live (W144): adds `LiveBadge` + a real-time score row, market/odds unchanged (no in-play odds feed). Completed-today (W145/W146): `StatusBadge` replaced by "FT" + `HitBadge`, pick struck through on a miss, edge relabeled "Pre-match edge". |
| `StatusBadge` | The 4-state verdict pill (Direct Bet / Conditional / No Bet / Insufficient Data) — same colors, same labels, everywhere a verdict appears. Not shown on a completed `MatchCard` — `HitBadge` takes its slot there instead (W146). |
| `LiveBadge` | (W144) Red pulsing dot + "LIVE", `--status-critical`. Renders alongside `StatusBadge`, not instead of it — pre-kickoff recommendation and in-progress state are two different, both-relevant facts. |
| `HitBadge` | (W145/W146) "Hit" (green `CheckCircle`) / "Not Hit" (orange `XCircle`) for a completed match's recommended market, resolved via the same rule as `src/agent/market_resolution.py`'s `market_correct()`. Shown both as the card's top badge and as an inline echo under the struck-through pick. |
| `TrustSignal` | A first-class warning badge for cold-start (thin history) or unknown-team matches — renders independently of the verdict, so a confident-looking recommendation still gets flagged if the underlying data is thin. |
| `TeamBadge` | Circular team initials badge; real club colors for known Premier League/La Liga/Allsvenskan teams, a deterministic hash-based fallback palette for anyone else — so a badge never looks broken for an unrecognized team. |
| `LogBetButton` | Inline "Log bet" → stake input → confirm, scoped to one market/selection from a specific recommendation snapshot. Used on both `MatchCard`'s expanded row and Match Analysis's probability rows. |
| `ErrorState` | One shared inline error row (message + optional Retry) used by every page instead of each rolling its own. |
| `DashboardRail` | Dashboard-only: edge distribution donut + top-5 edges list. Donut has a 5th "Completed" slice (W147, `--accent`) alongside the 4 status-colored recommendation-type slices. |

## 5. Cross-Cutting Interaction Patterns

- **Lazy recommendation generation.** A fixture has no recommendation until something asks for
  one. Expanding a `MatchCard` or opening Match Analysis triggers: check the precomputed cache →
  fall back to a live agent call on a miss. This keeps list pages fast (no N-way agent calls on
  load) while still resolving instantly for anything the overnight batch already covered.
- **Log a bet, from a recommendation.** The from-recommendation path (`LogBetButton`) locks
  match/market/selection/odds to the exact recommendation snapshot shown — only stake is
  free-entry. This is deliberately narrower than the Bet Tracker's manual form, which allows any
  real fixture + free-text market/selection/odds. Two paths, two trust levels: one is "log exactly
  what the agent told me," the other is "log something I did on my own."
  On `MatchCard`, the equivalent control lives inline in its expanded hover/expand row.
- **Sandbox-aware "today."** Every page resolves "now" through `useSandboxAsOf()` rather than a
  bare `new Date()` — in sandbox/testing mode this returns a pinned `as_of` date from the backend
  instead of the real clock, so the whole app (fixture windows, "today"/"tomorrow" labels,
  future-fixture leak guards) can be exercised at an arbitrary point in time without lying about
  match results. Invisible in normal use; this is what makes the sandbox runbook
  (`documents/sandbox_testing_runbook.md`) possible.
- **Data honesty over polish.** Several UI states exist purely to say "this isn't available" or
  "this is untrustworthy" rather than hide the gap: Squad Intelligence's placeholder, the
  "N markets omitted — malformed data" notes, the "Data issue" tag on a direct-bet with no odds,
  and `TrustSignal`'s cold-start/unknown-team badges. This is a deliberate product stance, not
  unfinished UI — see `app_techspec.md` §8.

## 6. User Journeys

**Daily check-in (primary journey).**
Land on Dashboard → scan today's/this week's date groups, most already showing verdicts from the
overnight batch → expand a card with an interesting edge → read the explanation bullets inline →
optionally jump to Full Analysis for the complete market breakdown → optionally Log Bet.
No search, no typing — the common case is fully passive scanning.

**Research a specific match.**
Use the top search bar or Match Explorer → find the fixture (up to 90 days out) → open Match
Analysis → read Model Probabilities + Agent Reasoning in full → Log Bet on a specific market if
convinced.

**Track what was actually bet.**
Open `/bets` directly (nav link hidden, W106) → check bankroll/ROI/hit-rate at a glance → either
log something the agent recommended (from the match page) or log a bet manually (search fixture →
fill market/selection/odds/stake) → periodically hit "Settle open bets" to reconcile against real
results.

**Recover from a stale or failed data source.**
Any page's fetch failure surfaces via `ErrorState` with a Retry button — no dead pages, no silent
blank states. Sidebar's Last Updated turns amber/warning-colored when a competition's data is
stale, visible on every page without needing to check a dedicated status page.

## 7. Open Design Question

Bet Tracker (`/bets`) is feature-complete but nav-hidden (W106) — "not ready yet" per the commit,
with no recorded reason in this doc's source material. Before re-enabling the nav link, decide
what "ready" means: is it a UX gap (e.g. no bet-editing/deletion), a trust gap (manual market/
selection are free-text, unvalidated against real markets), or just sequencing? Whoever picks this
back up should resolve that first — flip `NAV_ITEMS` in `AppShell.tsx` is the only code change
once it's decided.
