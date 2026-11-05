# Domestic Cup Pre-Flight Findings

Live-verified 2026-11-04. Each section records what the real API output
showed, not an assumption.

## OddsPapi tournament IDs (domestic cups)

A real bug was caught and fixed while running this: the discovery script's
first draft assumed field names `id`/`name`, matching the comments on
existing `LEAGUE_TOURNAMENT_IDS` dicts elsewhere in this codebase -- the
real OddsPapi response uses `tournamentId`/`tournamentName`/`categoryName`
instead, so the first run silently matched nothing (0/2016). Also: OddsPapi
reuses generic names ("FA Cup", "League Cup") across dozens of countries
(14 "FA Cup"-named tournaments alone -- England, China, South Africa, Ghana,
Thailand, Malaysia, Fiji, Hong Kong, plus women's/qualification variants) --
name alone isn't enough to disambiguate; `categoryName` (confirmed real
field, not guessed) is required too.

| Cup | Tournament ID | OddsPapi name | Category | Note |
|---|---|---|---|---|
| FA Cup | 19 | FA Cup | England | Excludes id=1696 (Qualification), id=18488 (Women) |
| Carabao Cup | 21 | EFL Cup | England | Single unambiguous match |
| Copa del Rey | 329 | Copa del Rey | Spain | Single unambiguous match |
| DFB-Pokal | 217 | DFB Pokal | Germany | Excludes id=686 (Women) |
| Coupe de France | 335 | Coupe de France | France | Excludes id=18364 (Women) |
| Coppa Italia | 328 | Coppa Italia | Italy | Excludes id=824/1554/39221 (Serie C/Primavera/Serie D feeders), id=28127 (Women) |

## OddsPapi live odds coverage (domestic cups)

Two real bugs caught and fixed while running this check (not pre-existing
knowledge — found live against a real payload): the odds response's actual
top-level key is `bookmakerOdds`, not `bookmakers`; and `outcome["players"]["0"]`
is a single price object, not a list of historical "ticks". **Both bugs also
exist in production code** (`app/backend/oddspapi_client.py`'s
`_parse_corners_odds`, plus `scripts/extract_oddspapi_odds_lookup.py` and
`scripts/pull_oddspapi_btts_corners.py`) and mask each other there — the
wrong key always produces an empty `outcomes` dict, so the list-indexing bug
is never reached with real data. Practical effect: production's
`get_corners_odds()` very likely has never successfully parsed a real
corners price from a live payload, silently falling through to its
documented "no line posted yet" branch every time instead. This is a
pre-existing production issue unrelated to domestic cups — **not fixed as
part of this plan**, flagged here because it was found while verifying the
same endpoint for cup coverage.

| Cup | Upcoming fixtures (statusId=0) | Nearest fixture start time | Has real 1X2 odds |
|---|---|---|---|
| FA Cup | 40 | 2026-11-06 | **True** (2.95 / 3.65 / 2.23) |
| Carabao Cup | 4 | 2026-12-15 | **True** (12.33 / 6.22 / 1.215) |
| Copa del Rey | 8 | 2026-10-04 (⚠ in the past — likely postponed/rescheduled, still flagged pre-game) | False on this sample |
| DFB-Pokal | 8 | 2026-12-01 | False on this sample (3-4 weeks out — may simply be too early for a line) |
| Coupe de France | 86 | 2026-11-14 | False on this sample (same caveat) |
| Coppa Italia | 17 | 2026-12-01 | False on this sample (same caveat) |

All 6 cups have real fixture schedules in OddsPapi. Only 2/6 confirmed a
real posted price on a single-sample check — not conclusive for the other
4 (re-check closer to each one's matchday before treating as "no coverage,"
per the plan's own guidance).
