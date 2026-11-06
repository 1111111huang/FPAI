"""Tests for the UCL/UEL team-name canonicalization step -- resolving a
fotmob-sourced team name into the same canonical identity the domestic
feature pipeline already uses, via the existing TeamNameMapper mechanism
(src/ingestion/fotmob/merge.py's own pattern), not a bespoke crosswalk."""
from __future__ import annotations

import json
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


def test_canonicalize_resolves_via_token_match_fuzzy_fallback(tmp_path: Path) -> None:
    """Genuinely exercises the fuzzy/token-match path: map_team() checks
    self.mapping (loaded from mapping_path) before ever looking at
    `candidates`, so a name that's already a real exact key -- like
    "Real Sociedad" in the actual config/team_mapping.json -- would never
    reach suggest(), no matter what pool is passed. Pointing TeamNameMapper
    at an empty mapping file here guarantees the exact/accent-fold checks
    miss, so this falls through to suggest()/_token_containment_score.

    Verified directly (same TeamNameMapper, same empty-mapping setup):
    mapper.suggest("Real Sociedad", pool, use_token_match=True) ==
    ("Sociedad", 0.95) -- a strict token-subset match ({"sociedad"} subset
    of {"real", "sociedad"}), uniquely highest-scoring against "Betis" and
    "Celta" (both score far below the 0.82 min_similarity via plain
    Levenshtein), and well above threshold so map_team() returns it rather
    than falling back to the unmapped/unchanged branch."""
    empty_mapping_path = tmp_path / "empty_team_mapping.json"
    empty_mapping_path.write_text(json.dumps({}))
    mapper = TeamNameMapper(mapping_path=str(empty_mapping_path))
    pool = {"Sociedad", "Betis", "Celta"}

    assert canonicalize_team_name("Real Sociedad", mapper, pool) == "Sociedad"


def test_canonicalize_returns_the_name_unchanged_for_a_genuinely_new_club() -> None:
    """A club with no existing representation anywhere in raw_matches --
    e.g. Besiktas, never seen before -- has nothing to resolve against.
    Returns the (normalized) input itself, which becomes the de-facto new
    canonical identity from this point forward, same as how every other
    first-time club already bootstraps through TeamNameMapper elsewhere."""
    mapper = TeamNameMapper()
    pool = {"Ath Madrid", "Barcelona"}

    assert canonicalize_team_name("Besiktas Istanbul", mapper, pool) == "Besiktas Istanbul"
