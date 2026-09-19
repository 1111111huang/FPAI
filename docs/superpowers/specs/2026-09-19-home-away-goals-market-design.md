# Home/Away Goals Market — Design

2026-09-19. Direct user request, following on from backlog story W199
(`documents/app_user_stories.md:713`): wire a real, live-bettable
home-goals/away-goals market end-to-end — live serving via The Odds API,
and backtest/agent-train parity via OddsPapi (data already pulled, no new
quota spend) — in one pass, not split into a live-only story plus a later
follow-up.

## 1. Scope

Add two new markets, `home_goals` and `away_goals` — each a fixed
**1.5-goal line** (same fixed-line convention as `total_goals`'s 2.5 and
corners' 9.5/2.5) — representing the bookmaker "team totals" market: how
many goals *one team alone* scores, independent of the other team's score
(distinct from `total_goals`, which is the combined score, and from `btts`,
which only asks whether both teams score at all).

Both sources are in scope, sharing the same market/schema/grading
definitions:

- **Live serving**: The Odds API's `team_totals` market, wired into the
  same real-time recommendation-candidate pipeline `total_goals`/`btts`
  already use (W164 precedent).
- **Backtest/agent-train**: OddsPapi's equivalent team-goals market,
  extracted from data already pulled for the existing corners/BTTS
  backtest corpus (A100/A101 precedent) — no new API pull needed.

Graded for Hit/Not-Hit everywhere `result_3way`/`btts`/`total_goals`/
`total_corners` already are (settlement + the frontend's resolvable-markets
list), since actual home/away goal counts are always available wherever
settlement or backtest scoring runs — no coverage-gap caveat like
`home_corners`/`away_corners` has.

No ML/model changes needed: `home_goals`/`away_goals` are already active
forecast targets (`src/logic/target_registry.py:63-76`, regression,
`primary_metric="mae"`) — the forecast payload already computes a Poisson
count distribution for each (`src/forecast/forecast_service.py:467-471`,
`poisson_count_distribution()`) and that payload is already dumped verbatim
into the LLM's evidence every single call
(`src/agent/pipeline.py:170`). They've simply never had a market slot to be
recommended through.

## 2. Live verification first (required before writing the live parser)

This codebase has an established habit of never shipping a new odds-JSON
parser against an assumed shape — W164's bookmaker-ordering fix and the
A100/A101 corners work were both built only after a real API response was
captured and inspected. The existing W199 investigation confirmed The Odds
API's `team_totals` market key *exists* and carries real prices (E0:
`uk,eu,us` needed for 3 bookmakers of coverage; SP1: `us`/`us2`
specifically), but never captured the actual outcome-object shape — how a
"Team A over 1.5" outcome names/tags which team it belongs to.

So the first implementation step is a one-off script hitting a real
fixture's `/events/{id}/odds?markets=team_totals&regions=uk,us,us2`,
capturing the raw JSON, and confirming the exact outcome field before the
parser is written against it (most likely a `description` key carrying the
team name alongside `name`: "Over"/"Under" and `point`: 1.5, mirroring the
publicly-documented shape for other per-team markets on this API — but not
assumed, confirmed).

## 3. Live odds fetching (`app/backend/odds_api_client.py`)

- `NormalizedSecondaryOdds` gains `home_goals: dict[str, float] | None` and
  `away_goals: dict[str, float] | None` (each `{"over_1.5": .., "under_1.5":
  ..}`, same shape as the existing `total_goals` field).
- `get_event_odds()` gains an optional `regions: tuple[str, ...] | None =
  None` override parameter, defaulting to `self._regions` when omitted.
  This lets one call keep using the client's normal cheap region set for
  `totals`/`btts` (unaffected, still `uk`-only by default), and a second
  call use a wider, fixed `("uk", "us", "us2")` set just for `team_totals` —
  confirmed live (W199 investigation) that UK bookmakers essentially never
  price this market, so paying the wider-region cost only where it's
  actually needed.
- New parsing logic (mirroring `_normalize_secondary`) matches each outcome
  to `home_team`/`away_team` by name via the same `TeamNameMapper` already
  used everywhere else in this codebase for name-variant matching, and
  buckets into over/under 1.5 per side.

## 4. Wiring into live recommendations (`app/backend/eod_batch.py`,
`src/agent/graph.py`)

- `add_secondary_odds()` gets a second `get_event_odds()` call (wider
  regions, `markets=("team_totals",)`), folding results into
  `match_info["home_goals_odds"]`/`match_info["away_goals_odds"]` and into
  the cached `odds` dict — same freshness-cache reuse pattern
  (`h2h_unchanged`/`already_checked_secondary`) already used for
  `total_goals`/`btts`.
- `graph.py`'s prompt builder (`run_agent()`, around
  `src/agent/graph.py:562-579`) gets a new block, same shape as the
  existing `total_goals_odds`/`btts_odds`/`corners_odds` blocks, surfacing
  real prices to the LLM as plain prompt text.

## 5. Schema & guardrails (`src/agent/schema.py`,
`config/prompts/agent_v1*.txt`)

- Add `"home_goals"`, `"away_goals"` to the `market` `Literal` on
  `MarketCandidate`, `MarketCandidateModel`, `RecommendationPick`, and
  `RecommendationPickModel`. Add `"over_1.5"`, `"under_1.5"` to the
  `selection` `Literal` on the same four.
- Add `("home_goals", "over_1.5")` and `("away_goals", "over_1.5")` to
  `_CONDITIONAL_ELIGIBLE_MARKETS` (`src/agent/schema.py:355-361`) —
  mirroring corners' own `over_2.5` entries: `"conditional"` (waiting for a
  better price) is only a coherent strategy on the "over" side, same
  reasoning A54 already established for every other market here. Under
  1.5 stays ineligible, same asymmetry as `under_2.5`/`btts:no`.
- Every prompt variant (`agent_v1.txt` and the aggressive/balanced/
  conservative variants under `config/prompts/`) gets the new markets added
  to: the market enum lines, the Evidence Priority section (a short
  "home_goals / away_goals: [team]'s own attacking output — same priority
  ordering rationale as total_goals" entry), and the conditional-eligibility
  line (currently `src/agent/schema.py`'s prompt line ~54 equivalent).

## 6. Grading (`src/agent/market_resolution.py`,
`app/frontend/components/MatchUI.tsx`)

- `RESOLVABLE_MARKETS` gains `"home_goals"`, `"away_goals"` — both the
  Python set (`src/agent/market_resolution.py:21`) and its TS mirror
  (`app/frontend/components/MatchUI.tsx:586`).
- `build_actual_outcome()` (`src/agent/market_resolution.py:50-84`) adds
  `home_goals_side`/`away_goals_side`: `"over_1.5" if <count> > 1 else
  "under_1.5"` for each side. Unconditional (not an optional
  keyword-argument gap like `home_corners`/`away_corners`), since
  `home_goals`/`away_goals` are always passed into this function already
  (they're the function's own required positional parameters).
- `market_correct()` gets two new branches:
  `selection == actual["home_goals_side"]` /
  `selection == actual["away_goals_side"]`.
- The TS mirror (`isMarketCorrect` around `MatchUI.tsx:617-621`) gets the
  matching branches.
- `MARKET_LABEL` (`MatchUI.tsx:2596-2602`) gains `home_goals: { label:
  "Home Goals", subtitle: "Full Time" }` and the away equivalent. The
  selection-label override map (`MatchUI.tsx:2448-2451`) gains
  `"home_goals:over_1.5": "Over 1.5"` etc. for all four combinations.

## 7. Backtest & agent-train
(`scripts/extract_oddspapi_odds_lookup.py`, `src/agent/backtest.py`)

Confirmed directly against the already-downloaded OddsPapi historical
snapshots (`data/oddspapi_snapshots/`, the corpus pulled for the existing
corners/BTTS backtest work): Pinnacle's team-goals-total markets are
**already present in that data** — 988 of 1,005 resolved-match snapshot
files (98.3%) carry at least one team-goals-total market ID. No new
OddsPapi API pull or quota spend is needed for the existing corpus.

The market IDs form two blocks under each fixture's `bookmakers.pinnacle.
markets`: `10224–10236` (7 IDs, coverage 752→6 matches as the line rises)
and `10240–10250` (6 IDs, coverage 876→10 matches) — each ID internally
bundles an over/under outcome pair exactly like the existing corners market
(e.g. market `10228`'s outcomes are `10228`/`10229`). This is consistent
with one block per team at ascending 0.5-goal lines, but — same as
corners' own `corners_line_map.json` precedent — the raw snapshot data
carries no team-name or line label, so this is a well-evidenced hypothesis,
not yet a confirmed mapping.

Steps:

1. One call to OddsPapi's `/v4/markets` reference endpoint (metadata-only,
   confirmed not counted against the 250/month quota) to pin which block is
   home vs. away and which specific ID is the 1.5 line. Saved as a small
   local map file, same convention as `corners_line_map.json`.
2. `extract_oddspapi_odds_lookup.py` gets two new market-ID constants
   (`HOME_GOALS_MARKET_ID`, `AWAY_GOALS_MARKET_ID`) and is re-run against
   the already-downloaded snapshot files to add `home_goals_1.5_odds`/
   `away_goals_1.5_odds` entries into `data/oddspapi_btts_corners_odds.json`
   — no re-pull, pure re-extraction.
3. `backtest.py::_build_match_info` (`src/agent/backtest.py:142-194`)
   threads `match_info["home_goals_odds"]`/`match_info["away_goals_odds"]`
   from that lookup — the same 3-line pattern used today for
   `corners_odds`.
4. `agent-train` needs no separate wiring: it shares `process_match_row`/
   `_build_match_info` with `agent-backtest`
   (`src/agent/backtest.py`'s own module docstring — "the single source of
   truth... used by both", specifically to prevent the two from drifting
   out of sync).
5. Grading is already covered by Section 6's `build_actual_outcome()`
   change — backtest's `load_outcome()` already calls that same function.

## 8. Testing

Unit tests mirroring existing coverage for each touched module:

- `test_odds_api_client.py`: parsing a captured real `team_totals`
  response (from Section 2's live-verification capture) into
  `NormalizedSecondaryOdds.home_goals`/`away_goals`, plus the
  `get_event_odds()` region-override parameter.
- `test_schema.py`: new market/selection values validate; the
  `_CONDITIONAL_ELIGIBLE_MARKETS` downgrade behaves correctly for
  `over_1.5` (allowed) vs. `under_1.5` (downgraded to `no_bet`, same as
  every other "under" selection).
- `test_market_resolution.py`: `build_actual_outcome()`'s new
  `home_goals_side`/`away_goals_side` fields; `market_correct()`'s new
  branches.
- `test_backtest.py`: `_build_match_info()` threads `home_goals_odds`/
  `away_goals_odds` from the OddsPapi lookup when present.
- `MatchUI.test.tsx`: a case exercising the new resolvable market end to
  end (label, selection formatting, Hit/Not-Hit grading).

## 9. Out of scope

- Any change to `total_goals`/`btts`/corners' existing (cheaper) region
  cost — only the new `team_totals` fetch uses the wider region set.
- A configurable goal line — fixed at 1.5, matching this codebase's
  existing one-fixed-line-per-market convention.
- `home_corners`/`away_corners` gaining a real odds source (tracked
  separately as backlog story W201) — unrelated to this market, not
  touched here.
