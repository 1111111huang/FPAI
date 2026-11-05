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

**Two things must be re-verified when reusing, not assumed correct:**

1. **Live vs. historical odds-response shape.** The domestic-cup pre-flight
   spike (2026-11-04) found that OddsPapi's live `/v4/odds` endpoint and its
   historical `/v4/historical-odds` endpoint have genuinely different real
   shapes: live uses top-level key `bookmakerOdds` with `players["0"]` as a
   single price dict; historical uses `bookmakers` with `players["0"]` as a
   list of ticks. The backfill script calls `/v4/historical-odds`, so its
   parsing is **likely already correct** (matches the shape production's
   own historical-odds handling already assumes). But if any live-serving
   code path in the abandoned branch calls `/v4/odds` instead, it needs the
   live shape — check directly before trusting it, the same way a
   production bug (`get_corners_odds()` in `app/backend/oddspapi_client.py`,
   flagged separately, not fixed by this spec) was found doing exactly this
   wrong.
2. **Team-name crosswalk duplication.** The abandoned branch's 112-entry
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

## 7. Launch gate

Same volume-floor + ROI-floor requirement as the domestic-cup design,
applied here — doubly important given the prior attempt's specific failure
mode was volume (zero bets), not bad picks. A gate that only checks ROI
could pass trivially at zero volume again if not made explicit.

## 8. Open items (flagged, not resolved here)

- **Dixon-Coles pool breadth for the wider UEFA country set** — needs its
  own pre-flight data-availability spike, same shape as
  `docs/superpowers/plans/2026-11-04-domestic-cup-preflight-spike.md`, before
  committing to which countries' score-only data is actually reachable.
- **Live vs. historical odds-shape re-verification** (Section 3, item 1) —
  a concrete check against the cherry-picked code, not yet done.
- **Group/league-phase standings source** — not yet identified; needed for
  Section 5's stakes/dead-rubber detection.

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
