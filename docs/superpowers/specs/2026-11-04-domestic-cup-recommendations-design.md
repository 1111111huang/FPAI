# Domestic Cup Recommendations (FA Cup, Carabao Cup, Copa del Rey, DFB-Pokal, Coupe de France, Coppa Italia)

## 1. Problem

Domestic cup competitions are entirely unregistered today: no `competitions.yaml`
entry, no historical data, invisible to the system. A prior pure-LLM attempt at
cup coverage — the unmerged `llm_only` tier, built for UCL/UEL on
`worktree-llm-only-ucl-uel` (last commit 2026-10-08, never merged) — stalled at
its own launch gate. The backtest it was gated on actually ran
(`reports/agent_backtest/20261008_*_chunk.raw.log`): 9 UCL + 18 UEL matches
evaluated, ROI = 0.0 on both, because the agent returned `no_bet` on 26 of 27
matches. Per-match lessons show a consistent cause: `web_search` repeatedly
returned nothing match-specific, or contaminated results (one search for "Go
Ahead Eagles" pulled NFL and Nigeria "Super Eagles" content), for lower-profile
clubs. With no model underneath to anchor a number, low-confidence evidence
always forced abstention via the confidence-gated downgrade rule.

This design replaces LLM-self-reported confidence with a real numeric edge
(`ml_probability` vs. market, the same `value_edge` formula the 6 leagues
already use), computed by one of two engines depending on data depth per
fixture, rather than asking an LLM to reason its way to a number from search
results alone.

## 2. Scope

**In scope:** FA Cup, Carabao Cup, Copa del Rey, DFB-Pokal, Coupe de France,
Coppa Italia.

**Out of scope (explicitly):**
- Champions League / Europa League / continental cups — a well-positioned
  phase 2. The odds-ingestion plumbing and team-mapping discipline below carry
  over directly; the engine itself extends with no new methodology, just a
  wider Dixon-Coles pool covering more countries' top flights. Not part of
  this pass.
- Club World Cup, Super Cup — never in scope.
- Early-round non-league/minnow fixtures where neither side has any
  resolvable feature history — explicit no-coverage, not a degraded LLM
  fallback. This is deliberate: the prior attempt's failure mode was exactly
  "force an answer anyway." This design accepts partial cup coverage over
  forcing bad coverage everywhere.

## 3. Architecture: one new tier, two internal engines

A new tier (`cup` — exact name TBD at implementation time) alongside
`general_purpose`/`competition_specific`/`llm_only`. Routing inside the
forecast dispatch (`forecast_node`, `src/agent/pipeline.py`) checks, per side
of the fixture, whether that club has a resolvable domestic feature history
within an **11-league universe**: the 6 already-modeled leagues
(E0/SWE/SP1/I1/D1/F1) plus 5 newly-onboarded second divisions (Championship,
Segunda, Serie B, 2.Bundesliga, Ligue 2).

- **Both sides resolvable in the 11-league universe** → the cup-transfer
  model (Section 5).
- **At least one side unresolvable there, but resolvable in the broader
  Dixon-Coles score-only pool** → the Dixon-Coles fallback (Section 5).
- **Neither side resolvable anywhere** → no coverage. Same no-odds/
  insufficient-data abstain machinery already in place (`forecast_node`'s
  `no_odds` short-circuit, `output_node`'s `insufficient_data` backstop) —
  not a new silent failure mode.

**Decision note — why not just add cup rows + a `cup_indicator` feature to
the existing per-league models (E0, etc.)?** Considered and rejected: it only
works cleanly when both cup opponents are in the *same* league, and it would
require routing a cup fixture to "whichever league's model owns it" — a
registry ambiguity that doesn't exist today (routing is currently by
`competition_id`, not by each side's home league). A single dedicated
cup-transfer model, fed each side's domestic rolling-form features regardless
of which of the 11 leagues they come from, handles same-league and
cross-league cup ties through one path, with no new registry routing
complexity, and lets the model learn cup-specific effects (rotation discount,
giant-killing variance) directly from actual cup results instead of hoping a
league-tuned model generalizes to them.

## 4. Data ingestion (three buckets)

1. **Five new league onboardings** (feeds the cup-transfer model's universe)
   — the existing [`adding-a-league`](../../../.claude/skills/adding-a-league/SKILL.md)
   ~20-file checklist, run 5 times for Championship/Segunda/Serie
   B/2.Bundesliga/Ligue 2. Full feature engineering, live-verified IDs
   (fotmob/football-data.org/Odds API), per the skill's documented gotchas
   (never guess an external ID — wrong guesses fail silently by returning
   another league's data).
2. **A new, lighter score-only ingestion path** (feeds the Dixon-Coles pool)
   — just `(date, home_team, away_team, fthg, ftag, competition)` for clubs
   below that depth, none of the feature-engineering plumbing. Source TBD —
   needs a live check of whether fotmob's general match-list endpoint
   (already used unfiltered for UCL/UEL, per the old branch's findings) has
   adequate depth before assuming this bucket is cheap.
3. **The cup competitions' own fixtures/results/odds** — reuses the pattern
   already built on the abandoned UCL/UEL branch (OddsPapi tournament-ID
   discovery via `/v4/tournaments?sportId=10`, a `raw_matches` join script,
   historical odds backfill), run per domestic cup competition.

**Cross-cutting requirement:** all team-name resolution across all three
buckets routes through `TeamNameMapper`/`config/team_mapping.json` — not a
bespoke per-script crosswalk. The unmerged UCL/UEL branch built a duplicate
112-entry crosswalk instead of reusing this (logged as `BUG-082` on that
branch only — its bug numbering diverged from `main`'s and was never
reconciled, so that ID does not resolve in `documents/bugs.md` on `main`; cite
by description, not number, if this gets re-surfaced). At least 8 of those
112 entries were pure accent/diacritic variants `TeamNameMapper`'s existing
folding already resolves for free.

**Pre-flight check before committing further build time:** verify an odds
source actually carries markets for the 5 second divisions and the domestic
cups themselves. Unlike UCL/UEL (already confirmed: 368/368 real odds, zero
gaps, per the old branch's historical backfill), domestic lower-tier odds
liquidity is an open unknown — if it's thin, the no-odds abstain path fires
constantly regardless of model quality, and that's worth knowing before the
onboarding work, not after.

## 5. Edge computation

**Cup-transfer model:** a dedicated model (not shared with any league's
`competition_specific` model), trained on actual historical cup match
results, with inputs = each side's domestic rolling-form features (R3/R5
stats, rest days) pulled from whichever of the 11 leagues they belong to,
plus cup-context features (round, two-legged flag, days-since-last-match for
both sides as a fixture-congestion proxy). Follows the existing
`ModelManager`/`feature_subset` convention. Targets: `result_3way`,
`total_goals`, `btts`, `home_goals`, `away_goals`, and `total_corners`
**only** for this path, where real per-team corner features
(`OFF_HOME_HC_R3`/`DEF_HOME_AC_R3`-style) already exist from the onboarded
leagues' feature stores.

**Dixon-Coles fallback** (`src/models/dixon_coles.py`, already exists,
already used as a statistical baseline for the 6 leagues — US#73): joint MLE
fit of attack/defence parameters across the broader score-only pool
(`neg_ll` in `fit()`, lines 115-134), giving `λ_home`/`λ_away` per fixture
and, from the Poisson/tau-corrected joint distribution, `result_3way`,
`total_goals`, `btts`, `home_goals`, `away_goals`. Refit cadence is periodic
batch MLE (there's already a `warm_start` param, US#177, suggesting
incremental refitting across walk-forward windows has been explored), not a
live per-match update — scheduling the refit frequency is an open
implementation detail.

**`total_corners` is explicitly excluded from the Dixon-Coles fallback path.**
The model's existing corner handling (`fit()` lines 175-182, `predict_match()`
lines 260-276) is not opponent-adjusted at all — it's each team's own
historical mean corners won at home/away (`groupby("home_team")["hc"].mean()`),
with no defence term, so prediction is literally
`home_team.avg_home_corners + away_team.avg_away_corners` regardless of
opponent. For a team unseen during fit, it falls back further, to the
*global* average across the whole pool (`.get(team, self._global_corner_home)`)
— not even that team's own history. This is categorically different from how
goals are handled (an unseen team there falls back to `self._mean_attack`/
`self._mean_defence`, fitted population means within the same identifiability
framework the model actually solved for). Serving `total_corners` from this
path would be a number with the shape of a real prediction but effectively no
signal — the same silent-but-wrong failure shape as `BUG-079`'s NaN-cleaned
market features. Extending Dixon-Coles to true corner attack/defence
parameters (a second joint Poisson process, fit alongside goals) is a real
fix, not assumed to work by extension — tracked as a future item, not a
blocker for this launch.

## 6. Rotation/stakes — explicitly not fully solved

Neither engine knows a club is about to rest 7 starters for a
league-prioritized cup tie. Where computable without web search, feed it as
a feature (fixture-congestion/rest-days, Section 5). Confirmed-lineup data
typically isn't available until close to kickoff, which may be later than
current pregeneration timing assumes — flagged as an open item, not solved
here. Documented limitation, not a silent gap.

## 7. Schema/output

Real `ml_probability`/`value_edge`/`composite_score` on every candidate —
reusing the existing `general_purpose`/`competition_specific` validation
models and downgrade chain, **not** the nullable-edge
`LLMOnlyMatchRecommendationModel` pair built for the old tier (that schema
exists specifically because edge fields are genuinely absent there; here they
aren't).

## 8. Launch gate — volume floor + ROI floor, not ROI alone

Direct lesson from the stalled prior attempt: "beat baseline ROI" is
meaningless at n=0 bets. The gate needs **both** a minimum bet-count
threshold **and** ROI beating flat-stake/always-favorite baselines, over the
historical backtest sample, before `display_enabled: true`.

## 9. Risks

- Volume floor must be explicit in the launch gate (Section 8) or this can
  stall exactly like the last attempt did.
- Rotation is the dominant unknown and isn't fully computable (Section 6).
- Early-round data cliff for non-league minnows — explicit no-coverage, not
  forced (Section 2).
- Odds liquidity for the 5 second divisions and the domestic cups themselves
  is unverified — check before committing to the onboarding cost (Section 4).
- Team-name mapping must route through `TeamNameMapper` from the start, not
  repeat the unmerged branch's duplicate-crosswalk mistake (Section 4).
- Two-leg/replay semantics: some domestic cups retain two-legged ties or
  replays — schema/backtest resolution needs to know a single match isn't
  always the terminal outcome, unlike every league match today.
- `total_corners` has no real signal on the Dixon-Coles-fallback path
  (Section 5) — must stay excluded there, not silently served.
- Sample size for the launch-gate backtest will be thin for a genuinely new
  competition set — same "inconclusive-not-negative" posture used for the E0
  lesson A/B applies here too.

## 10. Testing

One runnable check per non-trivial new piece:
- Tier routing: both-resolvable → cup-transfer model; one-side-unresolvable
  → Dixon-Coles; neither → no-coverage.
- Cup-transfer model's feature-subset resolution across cross-league inputs
  (a Championship-side feature set combined with an E0-side feature set in
  one training row).
- Dixon-Coles joint fit on a synthetic multi-league sample — unseen-team
  fallback uses fitted population means, not per-team history.
- `total_corners` is absent/not served for Dixon-Coles-fallback matches.
- Schema validation: real edge fields (`ml_probability`/`value_edge`)
  populated, never null, for this tier.
- Launch-gate check enforces both the volume floor and the ROI-vs-baseline
  comparison, not ROI alone.

## 11. Documentation (per CLAUDE.md)

`documents/FRAI_TECHSPEC.md` — new section for the `cup` tier, parallel to
the existing tier documentation (§30.2's `llm_only` section is the template
to follow, with the engine swapped out). `documents/user_stories.md` +
`documents/agent_user_stories.md` — new US#/A# entries for registry/
ingestion/model/schema work, marked as backlog, closed out as shipped, per
existing convention.

## 12. Relationship to the abandoned `llm_only` branch

The `worktree-llm-only-ucl-uel` branch's ingestion work (OddsPapi tournament
ID discovery, historical odds backfill pattern, `raw_matches` join script
shape) is reusable infrastructure for continental cups in a later phase — not
wasted, even though this design doesn't use its `llm_only` tier or schema for
domestic cups. Its decision-layer (the pure-LLM confidence-gated tool) is
superseded by this design's two-engine approach for any match either engine
can cover; it would only remain relevant as a last-resort path for matches
neither engine can price — which this design deliberately treats as
no-coverage instead (Section 2), given the prior attempt's demonstrated
failure mode on exactly that kind of match.
