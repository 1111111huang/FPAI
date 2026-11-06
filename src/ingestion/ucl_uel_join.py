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
    the domestic feature pipeline already uses for that club, via the
    existing exact/accent-fold/token-match fuzzy mechanism
    (src/ingestion/fotmob/merge.py's own pattern) -- not a bespoke
    OddsPapi/fotmob-specific crosswalk. Returns the name unchanged when no
    existing representation is found, which is correct: that becomes the
    club's first-time canonical identity, not an error."""
    return mapper.map_team(name, team_pool, use_token_match=True)
