"""League tier mapping shared across ingestion sources."""

from __future__ import annotations

LEAGUE_TIER_MAP: dict[str, int] = {
    "E0": 1,  # Premier League
    "E1": 2,  # Championship
    "E2": 3,  # League One
}


def map_league_code_to_tier(league_code: str) -> int:
    """Map league code to its tier integer."""
    code = str(league_code).strip().upper()
    return LEAGUE_TIER_MAP.get(code, 4)
