"""Tests for the UCL/UEL team-name canonicalization step -- resolving a
fotmob-sourced team name into the same canonical identity the domestic
feature pipeline already uses, via map_team()'s existing exact/accent-fold/
conservative-fuzzy mechanism, not a bespoke crosswalk. use_token_match is
deliberately not enabled here -- see
test_canonicalize_does_not_falsely_resolve_inter_club_de_escaldes_to_inter
below for the confirmed real-data false positive that ruled it out."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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


def test_canonicalize_resolves_via_existing_exact_alias_for_known_la_liga_clubs() -> None:
    """These three names are already exact keys in the real
    config/team_mapping.json ("Real Sociedad" -> "Sociedad", "Real Betis"
    -> "Betis", "Celta Vigo" -> "Celta"), so map_team() resolves them on
    its exact-match check before `candidates`/fuzzy matching is even
    consulted -- same path as the first test above, just confirming it
    holds for more of the pre-existing La Liga roster. This does NOT
    exercise the token-match fuzzy fallback; see the dedicated test below
    for that."""
    mapper = TeamNameMapper()
    pool = {"Sociedad", "Betis", "Celta"}

    assert canonicalize_team_name("Real Sociedad", mapper, pool) == "Sociedad"
    assert canonicalize_team_name("Real Betis", mapper, pool) == "Betis"
    assert canonicalize_team_name("Celta Vigo", mapper, pool) == "Celta"


def test_canonicalize_does_not_falsely_resolve_inter_club_de_escaldes_to_inter() -> None:
    """Regression test for a confirmed real-data false positive:
    canonicalize_team_name used to pass use_token_match=True, and against
    this module's real-world pool -- every value in config/team_mapping.json
    combined across leagues, the shape `raw_matches` naturally has, since
    there is no per-league scoping here (see team_mapping.py's own US#141
    docstring on why an unscoped multi-league candidate pool is already a
    known risk factor for fuzzy matching) -- "Inter Club de Escaldes" (a
    real, tiny Andorran club) wrongly resolved to "Inter" (Inter Milan's
    canonical name) via a token-containment score of 0.95.

    use_token_match is no longer passed (defaults to False), which uses
    plain Levenshtein similarity instead: the closest candidate scores only
    0.27, well below the 0.82 threshold, so the name correctly falls
    through to the unmapped/unchanged branch. This test uses the real
    TeamNameMapper (real config/team_mapping.json) and the real full pool
    of every canonical name in it, not a synthetic fixture, since the bug
    only reproduced against real data."""
    mapper = TeamNameMapper()
    full_pool = set(mapper.mapping.values())

    assert canonicalize_team_name("Inter Club de Escaldes", mapper, full_pool) == "Inter Club de Escaldes"


def test_canonicalize_returns_the_name_unchanged_for_a_genuinely_new_club() -> None:
    """A club with no existing representation anywhere in raw_matches --
    e.g. Besiktas, never seen before -- has nothing to resolve against.
    Returns the (normalized) input itself, which becomes the de-facto new
    canonical identity from this point forward, same as how every other
    first-time club already bootstraps through TeamNameMapper elsewhere."""
    mapper = TeamNameMapper()
    pool = {"Ath Madrid", "Barcelona"}

    assert canonicalize_team_name("Besiktas Istanbul", mapper, pool) == "Besiktas Istanbul"
