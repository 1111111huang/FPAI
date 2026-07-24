"""W58: competition-id -> Odds-API sport_key mapping. Every real odds-fetch
call site (eod_batch.py, t30_refresh.py, main.py's manual-request path,
scheduler_wiring.py) previously relied on OddsAPIClient.get_odds()'s own
"soccer_epl" default parameter rather than an explicit lookup -- harmless
while EPL was the only competition, but silently wrong the moment a second
competition (Sweden) needs its own sport_key.

SWE's value confirmed live (W55, 2026-07-23) against The Odds API's own
/v4/sports/?all=true listing: {"key": "soccer_sweden_allsvenskan", ...,
"active": true} -- not a guess."""

from __future__ import annotations

ODDS_SPORT_KEY_BY_COMPETITION: dict[str, str] = {
    "E0": "soccer_epl",
    "SWE": "soccer_sweden_allsvenskan",
}

# Matches the pre-existing get_odds(sport_key="soccer_epl") default this
# mapping replaces -- an unmapped/unrecognized competition falls back to
# EPL's sport_key rather than silently fetching no odds at all.
DEFAULT_SPORT_KEY = ODDS_SPORT_KEY_BY_COMPETITION["E0"]
