"""W58: competition-id -> Odds-API sport_key mapping. Every real odds-fetch
call site (eod_batch.py, t30_refresh.py, main.py's manual-request path,
scheduler_wiring.py) previously relied on OddsAPIClient.get_odds()'s own
"soccer_epl" default parameter rather than an explicit lookup -- harmless
while EPL was the only competition, but silently wrong the moment a second
competition (Sweden) needs its own sport_key.

SWE's value confirmed live (W55, 2026-07-23) against The Odds API's own
/v4/sports/?all=true listing: {"key": "soccer_sweden_allsvenskan", ...,
"active": true} -- not a guess.

SP1's value confirmed live (W74/W77, 2026-08-06) the same way:
{"key": "soccer_spain_la_liga", "title": "La Liga - Spain", "active": true}
-- not a guess.

I1/D1/F1's values confirmed live (W134/W137, 2026-08-15) the same way.
D1 needed extra care disambiguating from the same /v4/sports/?all=true
listing's "soccer_germany_bundesliga2" (2. Bundesliga) and
"soccer_austria_bundesliga" (Austria's own top flight) -- confirmed against
the real title "Bundesliga - Germany" specifically, not just a name-pattern
guess."""

from __future__ import annotations

ODDS_SPORT_KEY_BY_COMPETITION: dict[str, str] = {
    "E0": "soccer_epl",
    "SWE": "soccer_sweden_allsvenskan",
    "SP1": "soccer_spain_la_liga",
    "I1": "soccer_italy_serie_a",
    "D1": "soccer_germany_bundesliga",
    "F1": "soccer_france_ligue_one",
}

# Matches the pre-existing get_odds(sport_key="soccer_epl") default this
# mapping replaces -- an unmapped/unrecognized competition falls back to
# EPL's sport_key rather than silently fetching no odds at all.
DEFAULT_SPORT_KEY = ODDS_SPORT_KEY_BY_COMPETITION["E0"]
