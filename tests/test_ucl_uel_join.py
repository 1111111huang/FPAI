"""Tests for the UCL/UEL team-name canonicalization step -- resolving a
fotmob-sourced team name into the same canonical identity the domestic
feature pipeline already uses, via the existing TeamNameMapper mechanism
(src/ingestion/fotmob/merge.py's own pattern), not a bespoke crosswalk."""
from __future__ import annotations

from src.ingestion.ucl_uel_join import canonicalize_team_name
from src.ingestion.common.team_mapping import TeamNameMapper


def test_canonicalize_resolves_exact_alias_already_in_the_shared_mapping() -> None:
    mapper = TeamNameMapper()
    pool = {"Ath Madrid", "Barcelona", "Real Madrid"}

    assert canonicalize_team_name("Atletico Madrid", mapper, pool) == "Ath Madrid"


def test_canonicalize_resolves_accent_variant_via_folding() -> None:
    mapper = TeamNameMapper()
    pool = {"Ath Madrid"}

    assert canonicalize_team_name("Atlético Madrid", mapper, pool) == "Ath Madrid"


def test_canonicalize_resolves_via_token_match_fuzzy_fallback() -> None:
    mapper = TeamNameMapper()
    pool = {"Sociedad", "Betis", "Celta"}

    assert canonicalize_team_name("Real Sociedad", mapper, pool) == "Sociedad"
    assert canonicalize_team_name("Real Betis", mapper, pool) == "Betis"
    assert canonicalize_team_name("Celta Vigo", mapper, pool) == "Celta"


def test_canonicalize_returns_the_name_unchanged_for_a_genuinely_new_club() -> None:
    """A club with no existing representation anywhere in raw_matches --
    e.g. Besiktas, never seen before -- has nothing to resolve against.
    Returns the (normalized) input itself, which becomes the de-facto new
    canonical identity from this point forward, same as how every other
    first-time club already bootstraps through TeamNameMapper elsewhere."""
    mapper = TeamNameMapper()
    pool = {"Ath Madrid", "Barcelona"}

    assert canonicalize_team_name("Besiktas Istanbul", mapper, pool) == "Besiktas Istanbul"
