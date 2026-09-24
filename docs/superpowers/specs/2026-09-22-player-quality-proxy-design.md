# Player-Quality Proxy (Transfermarkt Market Values) — Design

**Date:** 2026-09-22, revised 2026-09-23
**Status:** approved, implementation in progress
**Origin:** direct user idea — the agent's existing injury/absence reasoning has no real way to weigh "is there an adequate replacement," and A121's workaround (spend one optional `web_search` call comparing FIFA/EA Sports FC card ratings) was always framed as a temp fix, not the actual answer. The user proposed a real player-quality proxy data source, initially suggesting Football Manager ratings, to serve both that agent-side need and as a potential new ML feature.

**Revision (2026-09-23): data source switched from SoFIFA to Transfermarkt market values, mid-implementation.** SoFIFA was found to be behind Cloudflare's JS-challenge bot protection — confirmed live that a plain `requests.get()` (this repo's usual ingestion pattern) is blocked outright. The only real workaround, the `soccerdata` library's own SoFIFA reader, requires `seleniumbase`'s undetected-Chrome mode (a real local Chrome install, unreliable even then per that library's own docs), and installing it into this project's shared venv created a genuine dependency conflict with `google-genai`/`langgraph-sdk`'s `websockets` requirement — a real risk to the live agent stack, not a cosmetic warning. **Transfermarkt market values, tested live, have neither problem**: a plain `requests.get()` with a browser User-Agent returns HTTP 200 (confirmed against a real player page, market value parsed out via a `class="data-header__market-value-wrapper"` div), `robots.txt` allows general crawling, and REEP's own crosswalk already carries a `key_transfermarkt` column (same file, same shape as `key_sofifa` below) — so the crosswalk layer barely changes. Every section below is updated in place for this switch; the underlying signal is now "market value" (transfer/wage-market perception) rather than an EA FC-style ability rating — a different flavor, same underlying purpose (a player-quality/depth proxy), and the standard football-analytics proxy this design's own original "Data source decision" section had already named and considered before picking SoFIFA.

## Problem

Two related but distinct gaps:

1. **Agent-side:** `A121` covers the case where an absence's real impact depends on squad depth (e.g. Man City missing Foden/Doku but having real cover), but its fix is an ad-hoc, budget-gated `web_search` call per match, with the LLM reading and interpreting unstructured search snippets. Slow, expensive (spends the agent's limited tool-call budget), and unreliable (LLM-parsed text, not a real number).
2. **ML feature-side:** the existing player-level features (`SQUAD_HOME/AWAY_RATING_MEAN_R3/R5`, `feature_factory.py`) are a *rolling post-match performance* signal (FotMob's own per-match player ratings, averaged and rolled forward) — real signal, but it answers "how have they been playing," not "how good are they on paper." No static ability-rating signal exists in the feature set at all.

## Data source decision

Football Manager ratings were the user's original idea, but were found impractical to source in bulk: no API, and the only real bulk-export path (`FM Player Export`, a mod that runs inside the licensed game client) requires owning the game and manually exporting through the GUI — not automatable, not repeatable on a schedule. One community post even shows a full-database export having to be individually requested from site admins for academic use — bulk access isn't self-serve.

**Original decision (2026-09-22): use SoFIFA (EA FC) ratings.** Same "ability rating" flavor the user wanted, with real prior art for automated bulk collection cited (the `soccerdata` Python library ships a SoFIFA scraper module). **Superseded 2026-09-23**: sofifa.com is behind Cloudflare's JS-challenge bot protection — confirmed live that a plain `requests.get()` is blocked outright. `soccerdata`'s own SoFIFA reader only gets through via `seleniumbase`'s undetected-Chrome mode (a real local Chrome install; even that library's own docs say headless mode "might" get blocked anyway), and installing it conflicted with `google-genai`/`langgraph-sdk`'s `websockets` requirement in this project's shared venv — a real risk to the live agent stack that outweighs the benefit here.

**Decision (2026-09-23): use Transfermarkt market values instead.** Considered as a third option in the original pass and not chosen only because "SoFIFA's ability-rating framing matches the user's original FM intent more directly" — not because of any feasibility problem, and feasibility is exactly what broke on the SoFIFA side. Tested live: a plain `requests.get()` with a browser User-Agent returns the real page (HTTP 200, no Cloudflare block), the market value is parseable directly (`class="data-header__market-value-wrapper"`), and `robots.txt` allows general crawling. Same `requests`-only pattern this repo's other ingestion modules (FotMob, Understat) already use — no new dependency, no browser automation, runs fine anywhere including inside the deployed backend if that's ever wanted later.

## Player identity crosswalk

Joining Transfermarkt data onto this project's existing FotMob-sourced player/roster data needs a name/identity match — the player-level equivalent of `config/team_mapping.json`, which already solves this exact class of problem for team names.

**Decision: adopt REEP (`github.com/withqwerty/reep`) rather than building matching from scratch.** REEP is a CC0/public-domain (Wikidata-derived) crosswalk that already maps FotMob IDs directly to Transfermarkt IDs (confirmed live — `data/people.csv`'s `key_fotmob`/`key_transfermarkt` columns, both populated for the same rows), distributed as CSV. Known limitation: its public snapshot is a point-in-time export, so this season's new transfers/breakout players may be missing at first — treated the same non-blocking way `BUG-057` already established for unmapped team names (log it, degrade, don't crash; patch coverage gaps as they're found in production, not as a blocking prerequisite).

## Storage schema

```
player_market_values(fotmob_player_id, transfermarkt_player_id, snapshot_date, market_value_eur, ...)
```

A dated-snapshot table from day one — each refresh inserts a new row set stamped with that run's date, never overwrites. This costs nothing in Phase 1 (the agent tool only ever wants "most recent snapshot ≤ today"), but is exactly what Phase 2 needs: a backtest for a September match must read the value that existed *in* September, not one revised in December — the same lookahead-bias class of bug `W179` already had to fix once for closing-line odds features. Getting this right now avoids a painful migration later.

## Ingestion pipeline

New module `src/ingestion/transfermarkt/`, mirroring the existing `src/ingestion/fotmob/` shape (`fetcher.py`, `merge.py`) rather than inventing a new convention:
- `fetcher.py` — fetches each player's Transfermarkt profile page via plain `requests` + a browser User-Agent, rate-limited politely (mirrors `W243`'s `_throttle_oddspapi_request()` pattern — a deliberate delay between requests, since this isn't an official API; `robots.txt`'s own `Crawl-delay: 2` for other bots is a reasonable floor to match).
- `merge.py` — joins fetched rows onto the REEP crosswalk (attaching `fotmob_player_id`), then inserts into `player_market_values` with the run's date as `snapshot_date`.

**Scope:** not every Transfermarkt-listed player — only players already appearing in this project's own `raw_player_match_stats` (i.e. players who've actually played for one of the 5 tracked leagues' teams). That list already exists from FotMob ingestion and is the seed for which Transfermarkt lookups to do. Keeps this bounded, avoiding the same "why fetch more than we need" mistake `BUG-042` already found once for FotMob's own season range.

**Cadence:** market values update irregularly (Transfermarkt's own periodic revaluation cycles, not a fixed schedule) — this does not belong in the nightly EOD scheduler. A standalone CLI command (`python main.py refresh-market-values`, following the existing `main.py` subcommand convention), run manually or on a slow cadence (monthly is plenty). The REEP crosswalk itself refreshes on an even slower, separate cadence.

**Degradation:** a player with no crosswalk entry or no listed market value (common for fringe/lower-profile players — confirmed live, not every Transfermarkt profile has one) is a non-blocking gap — `null`, logged, never a failure — the same discipline every other optional feature/lookup in this codebase already follows.

## Phase 1: agent tool

New tool `get_player_rating(player_name: str) -> {market_value_eur, position, matched: bool}`, registered alongside `web_search` in the agent's existing tool-calling setup. (Tool name kept as `get_player_rating` for prompt-instruction continuity even though the underlying signal is now market value, not an ability rating — the prompt instruction below spells out what the number means.)

**Match context is bound automatically, not supplied by the LLM.** The tool is scoped to the two teams' current rosters (~40-50 names, from the same FotMob roster data `SQUAD_*`/lineup features already use) for whichever match the agent is currently processing — narrowing the candidate pool this way is what makes reliable fuzzy matching possible at all; matching a free-text name against a ~50-name pool is a fundamentally safer problem than matching against a global 18k-player database.

**Matching, layered cheapest/most-certain first** (mirrors `team_mapping.json`'s own proven layering):
1. Exact match (case-insensitive) on full name.
2. Accent-folded match — reuse `BUG-044`'s existing, already-tested `_fold_accents()` utility rather than writing a second one.
3. Surname-only match (`web_search` text usually says "De Bruyne," not the full name) — match against the last token of each roster name.
4. Fuzzy fallback via stdlib `difflib.get_close_matches` against the ~50-name pool only, with a similarity floor.

**Ambiguity is a non-match, never a guess.** Two plausibly-matching roster players (shared/common surname) → `matched: false` with an "ambiguous" reason, not a silent pick — a wrong confident number here actively misleads betting reasoning, worse than admitting uncertainty.

**Prompt change:** rewrites A121's instruction (in all 4 posture files) from "spend the one optional `web_search` call comparing FIFA/EA Sports FC card ratings" to calling this tool instead (comparing `market_value_eur`, not an ability rating — the prompt text says so explicitly). A structured tool call is far cheaper than a live search + LLM-parsed snippet, so this no longer needs to be budget-gated as "the one optional call" — the agent can check replacement quality routinely, not only when it can spare a search.

## Phase 2: ML feature

`SQUAD_MKT_VALUE_MEAN_HOME/AWAY_R3/R5` — the identical rolling-window computation `feature_factory.py`'s existing `_squad_rolling_from_data` already does for `SQUAD_HOME/AWAY_RATING_MEAN_R3/R5`, pointed at `player_market_values.market_value_eur` (joined via the crosswalk) instead of `raw_player_match_stats.rating`. Result: two complementary per-team signals computed identically — "how have they been *playing*" (existing) vs. "how much is the squad *worth*" (new, a market-perception-of-quality/depth proxy). Gated behind the same `SQUAD` feature-group toggle already governing `FRDS_*`/`DEF_ANCHOR_*`/`XOC_*`.

**Deliberately does not need today's confirmed lineup.** Like the existing feature, this rolls over each team's *past* matchday squads, so it's computable at EOD-generation time same as everything else — no dependency on knowing who's actually starting today.

**Out of scope, named for later, not built now:** a lineup-aware refinement (today's *actual* confirmed starting XI's OVR vs. the team's own rolling average — the real "weakened by absences" signal), naturally a T-30-refresh addition once lineups are confirmed, mirroring how `total_goals`/`btts` odds already get layered in at T-30 (`BUG-052`'s fix).

**Point-in-time correctness:** the rolling window for a historical match must use, for each *past* match in that window, the snapshot dated at-or-before that match's own date — never today's. This is the entire reason Section "Storage schema" stores dated snapshots instead of overwriting.

**Historical backfill.** Live snapshotting alone means months of `NaN`-filled runway before there's enough real coverage to retrain on. Transfermarkt itself shows each player's own market-value history over time on their profile page (a "Market value development" chart/table), and multiple already-scraped bulk Transfermarkt datasets exist publicly (Kaggle) covering several past seasons — not yet confirmed in as much depth as the original SoFIFA-per-edition claim was, so this needs its own quick verification pass when Phase 2 actually starts, not assumed to carry over unchanged from the superseded SoFIFA framing above.

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
