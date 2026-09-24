"""Layered, cheapest-first player-name matching against a small (~40-50
name) roster pool -- the player-level analog of
src/ingestion/common/team_mapping.py's TeamNameMapper, per the design spec.

Ambiguity is a non-match, never a guess (mirrors TeamNameMapper.suggest()'s
own tie-breaking-to-None precedent) -- a wrong confident number here actively
misleads betting reasoning, worse than admitting uncertainty."""

from __future__ import annotations

import difflib

from src.ingestion.common.team_mapping import _fold_accents

_FUZZY_SIMILARITY_FLOOR = 0.75


def _surname(full_name: str) -> str:
    return full_name.strip().split()[-1]


def match_player_name(query: str, roster: list[str]) -> str | None:
    """Returns the single roster name query most plausibly refers to, or
    None if there's no match or more than one equally plausible match."""
    query_norm = query.strip().casefold()

    # Layer 1: exact (case-insensitive)
    exact = [name for name in roster if name.casefold() == query_norm]
    if len(exact) == 1:
        return exact[0]

    # Layer 2: accent-folded
    folded_query = _fold_accents(query_norm)
    folded = [name for name in roster if _fold_accents(name.casefold()) == folded_query]
    if len(folded) == 1:
        return folded[0]

    # Layer 3: surname-only
    surname_query = _fold_accents(_surname(query).casefold())
    surname_matches = [name for name in roster if _fold_accents(_surname(name).casefold()) == surname_query]
    if len(surname_matches) == 1:
        return surname_matches[0]
    if len(surname_matches) > 1:
        return None  # ambiguous surname -- never guess

    # Layer 4: difflib fuzzy fallback against the full roster
    close = difflib.get_close_matches(query, roster, n=2, cutoff=_FUZZY_SIMILARITY_FLOOR)
    if len(close) == 1:
        return close[0]
    return None
