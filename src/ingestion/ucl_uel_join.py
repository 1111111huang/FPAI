"""UCL/UEL-specific ingestion helpers. The team-name canonicalization step
here is the one piece of new logic in this cherry-pick -- everything else
(fixture pairing, the historical odds backfill) ports the abandoned
branch's already-proven code largely unchanged. See
docs/superpowers/specs/2026-11-04-champions-league-recommendations-design.md
Section 3 for why fotmob's raw spelling must not be written directly into
raw_matches."""
from __future__ import annotations

from src.ingestion.common.team_mapping import TeamNameMapper


def canonicalize_team_name(name: str, mapper: TeamNameMapper, team_pool: set[str]) -> str:
    """Resolve a fotmob-sourced team name into the same canonical identity
    the domestic feature pipeline already uses for that club, via
    map_team()'s existing exact/accent-fold/conservative-fuzzy mechanism --
    not a bespoke OddsPapi/fotmob-specific crosswalk. Returns the name
    unchanged when no existing representation is found, which is correct:
    that becomes the club's first-time canonical identity, not an error.

    Deliberately does NOT pass use_token_match=True. This module's real
    candidate pool is unscoped across every competition (every value in
    config/team_mapping.json, the shape `raw_matches` naturally has here --
    see the US#141 precedent in team_mapping.py's docstring for why an
    unscoped multi-league pool is already a known risk factor). Against
    that real pool, token-containment scoring produces a confirmed false
    positive on real data: "Inter Club de Escaldes" (a real Andorran club)
    resolves to "Inter" (Inter Milan) at score 0.95. Plain Levenshtein
    (use_token_match's default, False) correctly leaves it unchanged
    (closest score 0.27, well below the 0.82 threshold)."""
    return mapper.map_team(name, team_pool)
