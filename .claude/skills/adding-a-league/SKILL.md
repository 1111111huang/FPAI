---
name: adding-a-league
description: Use when adding a new football league/competition to FPAI — registering it in the ML pipeline, agent, web app, and deployed instance. Triggers on requests like "add [league name]", "support a new competition", "onboard a new league".
---

# Adding a League

## Overview

FPAI has done this 3 times (Sweden, La Liga, then Serie A+Bundesliga+Ligue1 together).
Same ~20-file checklist every time. Follow it in order — each layer depends on the
previous one having real data to work against. Don't guess IDs/codes; every external
source has its own naming and its own numeric ID for the same league, and guessing
has caused real bugs (see Gotchas).

## Order of operations

1. **User stories first** — append a new phase to `documents/user_stories.md` (ML),
   `documents/app_user_stories.md` (app), `documents/agent_user_stories.md` (agent),
   mirroring the most recent completed phase's structure exactly. Mark each story
   `completed` with notes as you finish it, per CLAUDE.md.
2. **Data/ML layer** (below) — must land and be verified with real data before app/agent work starts.
3. **App layer** (below) — wires the now-real data into the API/frontend.
4. **Agent layer** (below) — free-text aliases, optionally a snapshot/backtest pass.
5. **Deploy** — the deployed instance has its own DB; local commits don't touch it (see Gotchas).

## Data/ML layer checklist

| File | Change |
|---|---|
| `config/competitions.yaml` | Register `competition_specific` tier, `enabled_feature_groups` (copy an existing competition's list; drop `SQUAD`-prefixed groups if no reliable player-stats source) |
| `config/league_coefficients.yaml` | Add a coefficient if missing |
| `config/team_mapping.json` | Add mappings — **verify against real fetched names per source**, don't guess (see Gotchas) |
| `src/logic/competition_registry.py` | `COMPETITION_NAME_ALIASES` entry |
| `src/ingestion/understat/fetcher.py` | `LEAGUE_MAP` entry |
| `src/ingestion/fotmob/fetcher.py` | `LEAGUE_IDS` entry — **live-verify the numeric ID** against `/api/data/matches`, several similarly-named entries exist (e.g. 2nd division, other countries' "Bundesliga") |
| `main.py` | Add to `_ALL_BIG_FIVE_LEAGUES` (name goes stale past 5 leagues, rename if it bothers you) **and** `_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE` — the latter holds the football-data.co.uk page URL (pattern is `<countryname>m.php`, e.g. `germanym.php`, but **live-verify**, don't assume the pattern holds) |
| Then: scrape → ingest → verify row counts → train → `select-best-models` (budget ~30min/context, it's MLflow filesystem-history growth, not a hang) | |

## App layer checklist

| File | Change |
|---|---|
| `app/backend/match_info.py` | `COMPETITION_ALLOWLIST` |
| `app/backend/football_data_competition_codes.py` | `FOOTBALL_DATA_CODE_BY_LEAGUE` — **live-verify** the football-data.org code. This is a *different provider* than the `main.py` scrape-source URL above (`.org`'s live-fixtures REST API vs `.co.uk`'s historical CSV site) — don't reuse one code for the other |
| `app/backend/odds_sport_keys.py` | `ODDS_SPORT_KEY_BY_COMPETITION` — **live-verify** against The Odds API |
| `app/backend/main.py` | **5 separate edit sites, not one** — `get_<league>_fixtures_client()` wrapper + `<LEAGUE>_COMPETITION_CODE` const, the `register_eod_job(...)` call, `/api/fixtures`'s results-range branch, its fixtures-range branch, **and `_REFRESH_LEAGUE_NAMES` Literal** (easiest one to skip) |
| `app/backend/scheduler_wiring.py` | `COMPETITIONS` tuple, a `<LEAGUE>_COMPETITION_CODE` const, a branch in `_fetch_fixtures_for_league`, a new param on `register_eod_job` |
| `app/frontend/lib/dashboardMetrics.ts` | `LEAGUE_LABEL` |

## Agent layer checklist

- Free-text aliases so the agent recognizes the league from natural language.
- Optional: full-season snapshot corpus + train/test backtest + lesson generation
  (only if you want agent-side lessons for this league specifically — can be deferred).

## Deploy

- `data/fpai_core.db` is **gitignored** — merging to `main` does not update the deployed
  instance's data. Call `POST /api/admin/trigger-data-refresh?league=<CODE>` (header
  `x-app-token`, not `Authorization`) once per league to backfill it live.
- Match ingestion (scrape+ingest) finishes in minutes; the FotMob player-stats step is
  the real bottleneck (~1-2hrs, rate-limited to ~1 call/match) — don't assume it's stuck.
- Verify with `GET /api/status` (`by_league.<CODE>.match_count` should match the real
  season total) then a real `POST /api/recommendations` call — `unknown_team: false`
  and a non-null `feature_completeness` confirm the whole chain actually works, a
  status check alone doesn't.

## Gotchas (each one caused a real bug or false alarm)

- **Check test suites for the league name/code first.** Several test files use a
  currently-unregistered league (e.g. a name like "Eredivisie"/"La Liga" retired
  after it got registered for real) as their stock "this one isn't registered yet"
  fixture — `grep -rl "<your league name>"` across `tests/` and `app/backend/tests/`
  before you start. If you're about to register a name that's load-bearing as a
  fixture, those tests need to move to a still-unregistered name first, or they'll
  fail for the right reason with no obvious fix path.
- **`_REFRESH_LEAGUE_NAMES` is easy to miss.** Everything else works but the admin
  refresh endpoint 404s/422s for the new league until this Literal is updated (W143).
- **Never guess an external ID/code.** FotMob league IDs, football-data.org
  competition codes, and Odds API sport keys must each be confirmed with a live API
  call — they don't follow a predictable pattern and wrong guesses fail silently
  (return another league's data) rather than erroring.
- **`ModelManager.prepare_training_data` scopes by league automatically** (US#131) —
  don't add a manual filter, it's already there for any registered competition.
- **"Unmapped team" warnings aren't automatically your live-forecast path's problem.**
  `feature_factory.py`'s `build_for_match()` calls `TeamNameMapper.map_team()` with no
  candidates; ingestion merge steps (FotMob/Understat) call it *with* a candidate pool.
  Both log the identical "Add mapping to..." text on a miss, so the log alone can't
  tell you which fired — check whether the affected feature group
  (e.g. `SQUAD_*`) is even in `enabled_feature_groups` before treating it as urgent,
  and confirm with a real `/api/recommendations` call for that exact team pair.

## What's automatable vs. what isn't

Mechanical (safe to script/generate): the config/registry table edits above — they're
all "add a key to a dict/enum" once you have the real IDs.

Needs a live check every time, can't be templated: the three external IDs (Gotchas),
and team-name mappings (each source has its own spelling; only real fetched data tells
you what needs mapping — see the SC Freiburg/TSG Hoffenheim FotMob-naming episode).
