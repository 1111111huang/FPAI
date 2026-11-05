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
