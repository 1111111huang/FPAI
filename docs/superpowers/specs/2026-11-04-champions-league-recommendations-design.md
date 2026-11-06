# Champions League / Europa League Recommendations (Phase 2 of the Cup Tier)

## 1. Problem

UCL/UEL fixtures routinely pair clubs from completely different training
populations — exactly the problem domestic cups face, but with a wider and
more international opponent pool. The prior attempt at this
(`worktree-llm-only-ucl-uel`, the unmerged `llm_only` tier) tried to paper
over that gap with pure-LLM confidence instead of a real numeric edge. Its
own launch-gate backtest actually ran
(`reports/agent_backtest/20261008_*_chunk.raw.log`): 9 UCL + 18 UEL matches
evaluated, **ROI = 0.0 on both**, because the agent returned `no_bet` on 26
of 27 matches — `web_search` repeatedly returned nothing match-specific or
contaminated results for lower-profile clubs, and with no model underneath,
low confidence always forced abstention. It failed on volume, not on bad
picks.

This design does not invent new methodology. It is
`docs/superpowers/specs/2026-11-04-domestic-cup-recommendations-design.md`'s
two-engine approach (a dedicated cup-transfer model for teams with full
domestic feature histories, Dixon-Coles as a fallback for everyone else),
extended to a wider opponent pool. UCL/UEL and domestic cups should
ultimately share one `cup` tier's engine, not duplicate it — this spec is
phase 2 of that same architecture, not a parallel design.

## 2. Scope

**In scope:** UEFA Champions League + Europa League, men's senior
competition, league-phase and knockout rounds.

**Out of scope (explicitly):**
- UEFA Europa Conference League (UECL) — the third tier, with the heaviest
  minnow/thin-data exposure of the three UEFA club competitions. Same
  "accept partial coverage over forcing bad coverage everywhere" logic as
  domestic cups' early-round exclusion. A possible future phase, not part
  of this pass.
- Club World Cup, Super Cup — never in scope.
- Women's competitions — excluded, consistent with how OddsPapi's own
  tournament categories were filtered during the domestic-cup pre-flight
  spike.

## 3. Reusing the abandoned branch

Cherry-pick from `worktree-llm-only-ucl-uel` rather than rebuild:
- OddsPapi tournament IDs (UCL=7, UEL=679)
- The 368-match historical odds backfill (189 UCL + 179 UEL, zero gaps) —
  `scripts/pull_oddspapi_ucl_uel_1x2.py`
- FotMob score extraction for finished UCL/UEL matches
- The `raw_matches` join script (`scripts/build_ucl_uel_raw_matches.py`)

**One thing was checked directly and confirmed fine; one thing must still be
fixed when reusing:**

1. **Live vs. historical odds-response shape — checked, confirmed not an
   issue for the cherry-picked code.** The domestic-cup pre-flight spike
   (2026-11-04) found that OddsPapi's live `/v4/odds` endpoint and its
   historical `/v4/historical-odds` endpoint have genuinely different real
   shapes: live uses top-level key `bookmakerOdds` with `players["0"]` as a
   single price dict; historical uses `bookmakers` with `players["0"]` as a
   list of ticks. Traced directly: the backfill script
   (`scripts/pull_oddspapi_ucl_uel_1x2.py`) calls `/v4/historical-odds` and
   its `extract_1x2_odds()` helper (`app/backend/oddspapi_client.py`) uses
   the `bookmakers`/list shape — correct for that endpoint. `extract_1x2_odds`
   has no other caller in the abandoned branch (confirmed via `git grep`) —
   there is **no live-serving 1X2 fetch for UCL/UEL anywhere in that branch
   at all**, buggy or otherwise; `forecast_llm_only`'s odds parameters are
   passed in from upstream, and that upstream live-wiring was simply never
   built. This needs to be built fresh when live serving is added (Section
   8), not inherited — and must use the live shape (`bookmakerOdds`/dict)
   confirmed today, not the historical one, to avoid repeating the same
   mistake `get_corners_odds()` made elsewhere.
2. **Team-name crosswalk duplication — not yet checked, do before trusting.** The abandoned branch's 112-entry
   bespoke `ODDSPAPI_TO_FOTMOB_ALIASES` dict duplicates `TeamNameMapper`
   (flagged as a bug on that branch only — its bug numbering diverged from
   `main`'s and collides with an unrelated, already-fixed bug there; cite by
   description, not number, if this resurfaces). Migrate onto
   `TeamNameMapper`/`config/team_mapping.json` when cherry-picking, per the
   same discipline the domestic-cup design already committed to.

## 4. Engine extension

- **Cup-transfer model**: unchanged mechanism, same 11-league universe (6
  modeled leagues + the 5 second divisions once those ship per the
  domestic-cup design). Many UCL/UEL fixtures already fall under this for
  free — Bayern, PSG, Real Madrid, Inter, Arsenal, Tottenham, etc. are
  already in the 6 modeled leagues, and their domestic rolling-form
  features need no new ingestion to use here.
- **Dixon-Coles fallback**: the pool widens from "5 second divisions" to
  roughly the 20-30 UEFA countries whose top-flight clubs actually reach
  the UCL/UEL league phase (Portugal, Netherlands, Belgium, Scotland,
  Croatia, Norway, Austria, Czech Republic, etc.). Same mechanism, bigger
  pool, no new methodology — but needs its own pre-flight data check
  (Section 8) before committing to the full breadth, since score-only
  coverage for this wider set hasn't been verified yet.

## 5. What's genuinely new here, not already covered by the domestic-cup design

- **Swiss-format league-phase stakes.** The 36-team league phase (each club
  plays 8 different opponents once) makes dead-rubber/qualification-scenario
  detection more central than in a single-elimination domestic cup bracket,
  where "stakes" is usually binary (win or go home). Needs group/league-phase
  standings ingested and computed structurally — this is exactly the failure
  mode the old `llm_only` attempt's per-match lessons already demonstrated
  repeatedly (contaminated/absent web-search results for stakes context,
  e.g. the Ferencváros/Panathinaikos and Dinamo Zagreb/FCSB lessons).
- **Two-legged knockout ties.** Round of 16 onward is first-leg/second-leg
  aggregate. Per-leg markets (1X2, totals) fit the existing schema fine; an
  aggregate-progression market is explicitly out of scope, same decision
  already made for domestic cups' two-leg/replay cases.

## 6. Validation against the already-measured failure

The old branch's actual backtest (9 UCL + 18 UEL matches, January 2026) is
preserved in `reports/agent_backtest/20261008_*_chunk.raw.log` with real
fixtures, real odds, and real outcomes already known. Once the two-engine
approach is built, re-running that *exact* historical sample through it and
comparing directly against the pure-LLM result (ROI = 0.0, 26/27 `no_bet`)
is a clean, apples-to-apples validation — new-approach-vs-already-measured
-failure on the same matches, not just a baseline comparison.

## 6a. Data budget policy

**Odds (training and live) have exactly one source, and it's metered.**
Confirmed live, 2026-11-06: football-data.co.uk has zero UCL/UEL coverage
of any kind — no dedicated page, no CSV under either of its two formats
(`mmz4281`/`new`), no mention of "champions"/"europa"/"uefa" anywhere on
its site index. OddsPapi is the only source for UCL/UEL odds, live or
historical, with no free fallback. Given OddsPapi's 250-req/month budget
is shared with live production serving for the 6 leagues' `total_corners`
market, **no further OddsPapi spend for this effort without a deliberate,
budgeted decision** — the ingestion cherry-pick (Section 3) costs nothing
new, since it reuses credits already spent on the abandoned branch's
backfill.

**The existing 368-match odds-backed sample is the real ceiling for
odds-anchored training/validation, not the Swiss-format boundary.**
Confirmed live: 189 UCL + 179 UEL fixtures, 2026-01-20 to 2026-09-17 —
spanning the tail of the 2025-26 knockout rounds plus the 2026-27 season's
qualifying rounds and opening league-phase matchdays, not one clean
continuous sample. OddsPapi's own historical floor (2026-01-01) is already
16 months inside the Swiss-format era (started August 2024), so the format
change never actually constrains the odds-anchored bucket — OddsPapi's own
floor is the binding constraint. The sample is also stale by about 7 weeks
as of 2026-11-06 (last backfilled 2026-10-08) — a bounded, cheap extension
(fixture discovery is near-free; only the incremental odds pull for the
gap would spend real credits) worth doing deliberately when ingestion work
starts, not reflexively here.

**Score-only data for the Dixon-Coles pool has no such ceiling, and is
free.** fotmob and football-data.co.uk's "new leagues" format (both used
for the pool-widening check, Section 8) carry results with no odds
dependency, so they aren't bound by OddsPapi's floor at all. Capped
instead at the Swiss-format boundary (August 2024) as a deliberate policy
— not reaching further back for marginally more Dixon-Coles rows that
would need separate, format-aware feature handling (a different
standings shape, a different per-team fixture count, non-comparable
group-vs-league-phase stakes semantics).

## 7. Launch gate

Same volume-floor + ROI-floor requirement as the domestic-cup design,
applied here — doubly important given the prior attempt's specific failure
mode was volume (zero bets), not bad picks. A gate that only checks ROI
could pass trivially at zero volume again if not made explicit.

## 8. Open items (flagged, not resolved here)

- **Dixon-Coles pool breadth for the wider UEFA country set — resolved
  2026-11-04.** See
  `docs/superpowers/plans/2026-11-04-champions-league-preflight-findings.md`:
  15 additional countries confirmed on football-data.co.uk beyond the
  11-league universe, covering a clear majority of the real 150-club
  UCL/UEL opponent pool; fotmob's unfiltered payload covers most of the
  remainder as a lighter secondary signal. Replaces the abstract "20-30
  countries" estimate with a real, data-backed answer.
- **Live UCL/UEL odds-serving needs to be built fresh** (Section 3, item 1)
  — confirmed it doesn't exist yet anywhere in the abandoned branch, not a
  bug to fix. Must use the live `/v4/odds` response shape
  (`bookmakerOdds`/dict) confirmed during the domestic-cup spike, not the
  historical one.
- **Group/league-phase standings source — resolved 2026-11-04.** See the
  same findings doc: `GET /api/data/tltable?leagueId=<id>` (fotmob),
  confirmed live for both competitions (36 teams each, matching the
  Swiss-format league phase exactly), with qualification-zone and per-team
  standings data needed for Section 5's stakes/dead-rubber detection.

## 9. Testing

One runnable check per non-trivial new piece: cherry-picked ingestion
scripts produce the same row counts/odds coverage as originally recorded
(368/368) after the live/historical shape re-verification; Dixon-Coles pool
widening doesn't regress the 5-second-division fit from the domestic-cup
design; two-legged fixture rows resolve as independent per-leg matches, not
a single aggregate outcome, in the backtest harness.

## 10. Documentation

Per `CLAUDE.md`: `documents/FRAI_TECHSPEC.md` gets a new section for the
`cup` tier's UCL/UEL extension, parallel to the domestic-cup section;
`documents/user_stories.md` / `documents/agent_user_stories.md` get new
entries for the cherry-pick/re-verification/pool-widening work, marked as
backlog.
