# Player-Quality Proxy (SoFIFA Ratings) — Design

**Date:** 2026-09-22
**Status:** approved, pending implementation plan
**Origin:** direct user idea — the agent's existing injury/absence reasoning has no real way to weigh "is there an adequate replacement," and A121's workaround (spend one optional `web_search` call comparing FIFA/EA Sports FC card ratings) was always framed as a temp fix, not the actual answer. The user proposed a real player-quality proxy data source, initially suggesting Football Manager ratings, to serve both that agent-side need and as a potential new ML feature.

## Problem

Two related but distinct gaps:

1. **Agent-side:** `A121` covers the case where an absence's real impact depends on squad depth (e.g. Man City missing Foden/Doku but having real cover), but its fix is an ad-hoc, budget-gated `web_search` call per match, with the LLM reading and interpreting unstructured search snippets. Slow, expensive (spends the agent's limited tool-call budget), and unreliable (LLM-parsed text, not a real number).
2. **ML feature-side:** the existing player-level features (`SQUAD_HOME/AWAY_RATING_MEAN_R3/R5`, `feature_factory.py`) are a *rolling post-match performance* signal (FotMob's own per-match player ratings, averaged and rolled forward) — real signal, but it answers "how have they been playing," not "how good are they on paper." No static ability-rating signal exists in the feature set at all.

## Data source decision

Football Manager ratings were the user's original idea, but were found impractical to source in bulk: no API, and the only real bulk-export path (`FM Player Export`, a mod that runs inside the licensed game client) requires owning the game and manually exporting through the GUI — not automatable, not repeatable on a schedule. One community post even shows a full-database export having to be individually requested from site admins for academic use — bulk access isn't self-serve.

**Decision: use SoFIFA (EA FC) ratings instead.** Same "ability rating" flavor the user wanted, but with real prior art for automated bulk collection (multiple existing open-source scrapers; the `soccerdata` Python library ships a SoFIFA scraper module already), deep coverage (18k+ players, reaching backups — the exact population this feature cares about), and a better update cadence (EA FC patches ratings a few times per season vs. FM's yearly release cycle).

Transfermarkt market values were considered as a third option (the standard football-analytics proxy for squad depth) but not chosen — SoFIFA's ability-rating framing matches the user's original FM intent more directly; nothing here rules out adding Transfermarkt values later as a second, complementary signal.

## Player identity crosswalk

Joining SoFIFA data onto this project's existing FotMob-sourced player/roster data needs a name/identity match — the player-level equivalent of `config/team_mapping.json`, which already solves this exact class of problem for team names.

**Decision: adopt REEP (`github.com/withqwerty/reep`) rather than building matching from scratch.** REEP is a CC0/public-domain (Wikidata-derived) crosswalk that already maps FotMob IDs directly to SoFIFA/EA FC IDs (confirmed — both providers are explicitly listed in its schema), distributed as CSV/DuckDB. Known limitation: its public snapshot is frozen as of ~April 2026, so this season's new transfers/breakout players may be missing at first — treated the same non-blocking way `BUG-057` already established for unmapped team names (log it, degrade, don't crash; patch coverage gaps as they're found in production, not as a blocking prerequisite).

## Storage schema

```
sofifa_ratings(fotmob_player_id, sofifa_player_id, snapshot_date, overall_rating, potential_rating, position, ...)
```

A dated-snapshot table from day one — each refresh inserts a new row set stamped with that run's date, never overwrites. This costs nothing in Phase 1 (the agent tool only ever wants "most recent snapshot ≤ today"), but is exactly what Phase 2 needs: a backtest for a September match must read the rating that existed *in* September, not one revised in December — the same lookahead-bias class of bug `W179` already had to fix once for closing-line odds features. Getting this right now avoids a painful migration later.

## Ingestion pipeline

New module `src/ingestion/sofifa/`, mirroring the existing `src/ingestion/fotmob/` shape (`fetcher.py`, `merge.py`) rather than inventing a new convention:
- `fetcher.py` — fetches SoFIFA player data, rate-limited politely (mirrors `W243`'s `_throttle_oddspapi_request()` pattern — a deliberate delay between requests, since this isn't an official API).
- `merge.py` — joins fetched rows onto the REEP crosswalk (attaching `fotmob_player_id`), then upserts into `sofifa_ratings` with the run's date as `snapshot_date`.

**Scope:** not all ~18k SoFIFA players — only players already appearing in this project's own `raw_player_match_stats` (i.e. players who've actually played for one of the 5 tracked leagues' teams). That list already exists from FotMob ingestion and is the seed for which SoFIFA lookups to do. Keeps this bounded, avoiding the same "why fetch more than we need" mistake `BUG-042` already found once for FotMob's own season range.

**Cadence:** EA FC only patches ratings a few times per season — this does not belong in the nightly EOD scheduler. A standalone CLI command (`python main.py refresh-sofifa-ratings`, following the existing `main.py` subcommand convention), run manually or on a slow cadence (monthly is plenty). The REEP crosswalk itself refreshes on an even slower, separate cadence.

**Degradation:** a player with no crosswalk entry or no SoFIFA rating is a non-blocking gap — `NaN`/`null`, logged, never a failure — the same discipline every other optional feature/lookup in this codebase already follows.

## Phase 1: agent tool

New tool `get_player_rating(player_name: str) -> {overall_rating, potential_rating, position, matched: bool}`, registered alongside `web_search` in the agent's existing tool-calling setup.

**Match context is bound automatically, not supplied by the LLM.** The tool is scoped to the two teams' current rosters (~40-50 names, from the same FotMob roster data `SQUAD_*`/lineup features already use) for whichever match the agent is currently processing — narrowing the candidate pool this way is what makes reliable fuzzy matching possible at all; matching a free-text name against a ~50-name pool is a fundamentally safer problem than matching against a global 18k-player database.

**Matching, layered cheapest/most-certain first** (mirrors `team_mapping.json`'s own proven layering):
1. Exact match (case-insensitive) on full name.
2. Accent-folded match — reuse `BUG-044`'s existing, already-tested `_fold_accents()` utility rather than writing a second one.
3. Surname-only match (`web_search` text usually says "De Bruyne," not the full name) — match against the last token of each roster name.
4. Fuzzy fallback via stdlib `difflib.get_close_matches` against the ~50-name pool only, with a similarity floor.

**Ambiguity is a non-match, never a guess.** Two plausibly-matching roster players (shared/common surname) → `matched: false` with an "ambiguous" reason, not a silent pick — a wrong confident number here actively misleads betting reasoning, worse than admitting uncertainty.

**Prompt change:** rewrites A121's instruction (in all 4 posture files) from "spend the one optional `web_search` call comparing FIFA/EA Sports FC card ratings" to calling this tool instead. A structured tool call is far cheaper than a live search + LLM-parsed snippet, so this no longer needs to be budget-gated as "the one optional call" — the agent can check replacement quality routinely, not only when it can spare a search.

## Phase 2: ML feature

`SQUAD_OVR_MEAN_HOME/AWAY_R3/R5` — the identical rolling-window computation `feature_factory.py`'s existing `_squad_rolling_from_data` already does for `SQUAD_HOME/AWAY_RATING_MEAN_R3/R5`, pointed at `sofifa_ratings.overall_rating` (joined via the crosswalk) instead of `raw_player_match_stats.rating`. Result: two complementary per-team signals computed identically — "how have they been *playing*" (existing) vs. "how good are they *on paper*" (new). Gated behind the same `SQUAD` feature-group toggle already governing `FRDS_*`/`DEF_ANCHOR_*`/`XOC_*`.

**Deliberately does not need today's confirmed lineup.** Like the existing feature, this rolls over each team's *past* matchday squads, so it's computable at EOD-generation time same as everything else — no dependency on knowing who's actually starting today.

**Out of scope, named for later, not built now:** a lineup-aware refinement (today's *actual* confirmed starting XI's OVR vs. the team's own rolling average — the real "weakened by absences" signal), naturally a T-30-refresh addition once lineups are confirmed, mirroring how `total_goals`/`btts` odds already get layered in at T-30 (`BUG-052`'s fix).

**Point-in-time correctness:** the rolling window for a historical match must use, for each *past* match in that window, the snapshot dated at-or-before that match's own date — never today's. This is the entire reason Section "Storage schema" stores dated snapshots instead of overwriting.

**Historical backfill.** Live snapshotting alone means months of `NaN`-filled runway before there's enough real coverage to retrain on. Multiple already-scraped, publicly available SoFIFA datasets exist per yearly EA FC/FIFA edition (FIFA 18 through FC 26, confirmed via Kaggle). Using one dataset per edition as one coarse snapshot per season unlocks real multi-year historical coverage immediately, instead of waiting for live collection to catch up — switch to this project's own finer-grained live scrapes going forward as they accumulate.

## Testing

- Crosswalk ingestion: a known mapping resolves; an unmapped player doesn't crash.
- Rolling-feature computation: mirrors the existing `SQUAD_RATING` tests, swapped input source.
- Point-in-time correctness: a rolling window never reads a snapshot dated after the match it's computing for (highest-value test, given `W179`'s precedent).
- Agent tool matching: each layer (exact/accent/surname/fuzzy) resolves correctly in isolation; an ambiguous case returns `matched: false`, not a guess.
- Prompt: the standard "instruction present and correct in all 4 posture files" check already used for A118-122.

## Sequencing

Two phases against one shared data/crosswalk layer, not two independent builds (rejected as its own option — would duplicate the player-matching problem twice):

1. **Phase 1 (agent tool)** ships first — smaller, no retrain required, immediately retires A121's `web_search` workaround.
2. **Phase 2 (ML feature)** follows, once Phase 1 has exercised the ingestion/crosswalk plumbing on a lower-stakes consumer. The harder problem (point-in-time correctness, historical backfill) is de-risked by having already proven the crosswalk and ingestion work for real, live matches first.
