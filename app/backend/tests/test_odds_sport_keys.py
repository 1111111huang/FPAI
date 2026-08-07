"""W58: competition-id -> Odds-API sport_key mapping, consulted at every
odds-fetch call site instead of relying on OddsAPIClient.get_odds()'s own
"soccer_epl" default parameter."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.odds_sport_keys import DEFAULT_SPORT_KEY, ODDS_SPORT_KEY_BY_COMPETITION


def test_e0_maps_to_soccer_epl():
    assert ODDS_SPORT_KEY_BY_COMPETITION["E0"] == "soccer_epl"


def test_swe_maps_to_the_real_confirmed_sport_key():
    """W55 confirmed this live against The Odds API's own /v4/sports listing
    (active: true) -- not a guess."""
    assert ODDS_SPORT_KEY_BY_COMPETITION["SWE"] == "soccer_sweden_allsvenskan"


def test_sp1_maps_to_the_real_confirmed_sport_key():
    """W74 confirmed this live against The Odds API's own /v4/sports listing
    (active: true) -- not a guess."""
    assert ODDS_SPORT_KEY_BY_COMPETITION["SP1"] == "soccer_spain_la_liga"


def test_default_sport_key_matches_e0():
    """A caller with no/unrecognized league falls back to EPL's sport_key --
    matches the pre-existing get_odds(sport_key="soccer_epl") default this
    mapping replaces, so an unmapped competition doesn't silently degrade to
    no odds at all."""
    assert DEFAULT_SPORT_KEY == ODDS_SPORT_KEY_BY_COMPETITION["E0"]
