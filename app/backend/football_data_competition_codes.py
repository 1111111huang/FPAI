"""W76 (La Liga app-layer wiring): competition-id -> football-data.org
competition_code mapping, mirroring odds_sport_keys.py's existing pattern
for the same underlying problem. E0 and SP1 are both covered by
football-data.org (confirmed live) -- SWE is not (W55), so it has its own
SwedenFixturesClient (Odds-API-backed) instead and is deliberately absent
here.

Every fixture/results call site previously either hardcoded "PL"
(eod_batch.py's COMPETITION_CODE) or omitted competition_code entirely
(FootballDataClient.get_fixtures()'s own "PL" default) -- harmless while E0
was the only football-data.org-sourced competition, silently wrong the
moment a second one (La Liga) needs its own code."""

from __future__ import annotations

FOOTBALL_DATA_CODE_BY_LEAGUE: dict[str, str] = {
    "E0": "PL",
    "SP1": "PD",
    # W134, live-verified 2026-08-15 against GET /v4/competitions.
    "I1": "SA",
    "D1": "BL1",
    "F1": "FL1",
}

# Matches the pre-existing get_fixtures()/get_results() "PL" default this
# mapping replaces.
DEFAULT_FOOTBALL_DATA_CODE = FOOTBALL_DATA_CODE_BY_LEAGUE["E0"]
