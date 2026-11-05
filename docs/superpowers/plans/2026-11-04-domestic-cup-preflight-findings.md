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
knowledge — found live against a real payload): the LIVE `/v4/odds`
response's actual top-level key is `bookmakerOdds`, not `bookmakers`; and
`outcome["players"]["0"]` there is a single price object, not a list of
historical ticks.

**This also surfaces a likely pre-existing production bug, precisely
scoped after checking both endpoints' real shapes (Task 3 below) —
narrower than first suspected.** `app/backend/oddspapi_client.py`'s
`get_corners_odds()` calls the **live** `/v4/odds` endpoint, but parses the
response with `bookmakers`/list-of-ticks logic — which is correct for the
**historical** `/v4/historical-odds` endpoint, not the live one it actually
calls. The file's own header comment cites a specific prior investigation
("W199's investigation notes confirm the live /v4/odds endpoint returns the
identical shape") as the reason the historical parser was reused for live
calls — that claim is what's actually false, confirmed against a real live
payload today. `scripts/extract_oddspapi_odds_lookup.py` and
`scripts/pull_oddspapi_btts_corners.py` both correctly target the
historical endpoint (confirmed, Task 3) — they are **not** bugged, an
earlier draft of this note wrongly lumped them in. Practical effect:
production's `get_corners_odds()` very likely has never successfully
parsed a real corners price from a live payload, silently falling through
to its documented "no line posted yet" branch every time instead. This is
a pre-existing production issue unrelated to domestic cups — **not fixed as
part of this plan**, flagged here because it was found while verifying the
same provider for cup coverage.

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

## OddsPapi historical odds coverage (domestic cups, for training)

The historical endpoint (`/v4/historical-odds`) has a genuinely different
real shape from the live endpoint above — confirmed live, 2026-11-04:
top-level key `bookmakers` (not `bookmakerOdds`), and `players["0"]` is a
real list of price ticks over time (not a single dict). This matches what
production's `_parse_corners_odds` already assumes — correct for *this*
endpoint, just wired to the wrong one elsewhere (see note above).

Each cup's oldest finished fixture (2024) returned 404 — consistent with
the already-known UCL/UEL cutoff (historical odds only exist from
2026-01-01 onward). The newest finished fixture per cup (checked instead of
assuming the cutoff is identical for every competition):

| Cup | Finished fixtures (statusId=2) | Oldest sample (2024) | Newest sample |
|---|---|---|---|
| FA Cup | 571 | 404 (no odds) | **True** (4.91 / 5.01 / 1.408), 2026-10-20 |
| Carabao Cup | 235 | 404 (no odds) | **True** (1.077 / 10.89 / 44.16), 2026-10-29 |
| Copa del Rey | 323 | 404 (no odds) | False, 2026-10-29 (one sample, not conclusive) |
| DFB-Pokal | 148 | 404 (no odds) | **True** (1.089 / 9.67 / 34.62), 2026-10-28 |
| Coupe de France | 411 | 404 (no odds) | 404 (no odds), 2026-10-31 (one sample, not conclusive) |
| Coppa Italia | 98 | 404 (no odds) | **True** (1.196 / 5.31 / 30.8), 2026-09-15 |

4/6 cups confirmed real historical 1X2 odds within the known coverage
window — a real training-data foundation exists for most of this scope.
Copa del Rey and Coupe de France's single newest-sample check came back
empty; worth a wider sample before concluding no coverage, same caveat as
the live-odds section above.

## The Odds API sport_keys (second divisions)

One real naming miss caught while running this: matching on "segunda"
alone found nothing — the real title is "La Liga 2 - Spain" (key
`soccer_spain_segunda_division`), confirmed live, 2026-11-04. All 5 resolved
to a single unambiguous key once corrected. Incidental finding: The Odds
API also carries `soccer_spain_copa_del_rey` — a second potential live-odds
source for that cup alongside OddsPapi, not pursued further here.

| Division | sport_key | Real title | active |
|---|---|---|---|
| Championship | soccer_efl_champ | Championship | True |
| Segunda División | soccer_spain_segunda_division | La Liga 2 - Spain | True |
| Serie B | soccer_italy_serie_b | Serie B - Italy | True |
| 2. Bundesliga | soccer_germany_bundesliga2 | Bundesliga 2 - Germany | True |
| Ligue 2 | soccer_france_ligue_two | Ligue 2 - France | True |
