"""Fuzzy team-name resolution shared across ingestion sources.

Maps a source's team-name spelling (Understat, FotMob, etc.) onto the
canonical names already used in raw_matches, via an explicit JSON mapping
file with a Levenshtein-distance fallback for names not yet mapped.

League-scoping (US#141): `TeamNameMapper`/`map_team` deliberately has no
`league` parameter and `config/team_mapping.json` remains one flat
`{name: short_name}` namespace shared by every competition -- this was
reviewed, not overlooked. The mapper's exact-lookup path (no `candidates`
passed) has negligible cross-league collision risk in practice (it would
require two competitions using the literal same full club name string,
which we checked and found no instance of between EPL and Sweden's
Allsvenskan). The real risk was in the *fuzzy-match candidate pool* two call
sites built from an unscoped `SELECT ... FROM raw_matches` (no `league`
filter): `resolve_match_ids` (`src/ingestion/fotmob/merge.py`) and
`update_raw_matches_xg` (`src/ingestion/understat/merge.py`). Both are fixed
to accept an optional `league` parameter that scopes the raw_matches query
(and therefore the fuzzy candidate pool) to one competition -- see the
regression tests in `tests/test_fotmob_merge.py` /
`tests/test_understat.py` proving a same-date, similarly-named team from an
unrelated competition can otherwise steal a real match away from its correct
resolution (not just theoretically collide -- a legitimate row silently
turns up unmatched).

A `(league, name)` composite key inside `TeamNameMapper`/`team_mapping.json`
itself was considered and deferred (mirroring the "reserve the seam, don't
build it yet" precedent from US#90, `documents/FRAI_TECHSPEC.md` Section
27.2): every real call site now scopes its own candidate pool by league
before calling `map_team`, which closes the practical risk without needing
the mapping file itself to become league-aware. Revisit this if a 3rd/4th
competition's club names turn up an actual exact-lookup collision the
scoped-candidate-pool fix above doesn't cover.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _fold_accents(value: str) -> str:
    """Strip diacritics via stdlib NFKD decomposition (e.g. 'Atlético' ->
    'Atletico', 'Coruña' -> 'Coruna') -- an accented letter decomposes into
    its base letter plus a separate combining-mark codepoint, which this
    then drops. Case/punctuation/everything else is untouched, so this adds
    accent-insensitivity on top of map_team()'s existing exact-match
    semantics without changing anything else about it.

    Root cause for BUG-043-adjacent live warnings (2026-08-13): SP1's Odds-
    API-sourced full club names ("Atlético Madrid", "Deportivo La Coruña")
    carry diacritics that config/team_mapping.json's existing entries
    ("Atletico Madrid", "Deportivo La Coruna", added by earlier
    football-data.org-shortName-only audits, W78) don't -- a plain exact
    match can never bridge that, no matter how many accented variants get
    manually enumerated one club/source at a time (already done twice: W59
    for Sweden's Odds-API names, W78 for La Liga's football-data.org
    shortNames). Folding both sides once, here, closes the whole class for
    every existing and future competition instead of a third manual audit."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    left = left.lower()
    right = right.lower()
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, start=1):
        current = [i]
        for j, rchar in enumerate(right, start=1):
            insert_cost = previous[j] + 1
            delete_cost = current[j - 1] + 1
            replace_cost = previous[j - 1] + (lchar != rchar)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _similarity_score(left: str, right: str) -> float:
    """Convert Levenshtein distance into a 0-1 similarity score."""
    if not left and not right:
        return 1.0
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(left, right)
    return 1.0 - (distance / max_len)


def _tokenize(value: str) -> set[str]:
    """Lowercase, accent-folded, punctuation-stripped word set -- 'TSG
    Hoffenheim' -> {'tsg', 'hoffenheim'}. Used by `_token_containment_score`
    below, not by the exact/accent-fold lookup paths in `map_team()`, which
    stay untouched."""
    folded = _fold_accents(value).lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in folded)
    return set(cleaned.split())


def _token_containment_score(left: str, right: str) -> float | None:
    """Token-containment score for `use_token_match=True` callers, or None
    if the two names don't relate this way (caller falls back to
    `_similarity_score`). Equal token sets (a pure word-order variant) score
    1.0; a genuine strict subset -- one side is the other plus extra words,
    e.g. a club-type prefix like 'TSG Hoffenheim' vs 'Hoffenheim' -- scores
    0.95. Ranking equal-set above strict-subset means an exact token match in
    a candidate list always outright wins over a merely-prefixed one, rather
    than tying with it under suggest()'s ambiguity guard. Two names that just
    happen to share one common word ('Manchester United' vs 'West Ham
    United') relate neither way -- neither full token set is a subset of the
    other -- so this returns None for them, same as any unrelated pair."""
    left_tokens, right_tokens = _tokenize(left), _tokenize(right)
    if not left_tokens or not right_tokens:
        return None
    if left_tokens == right_tokens:
        return 1.0
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return 0.95
    return None


class TeamNameMapper:
    """Map a source's team names to the CSV canonical names."""

    def __init__(self, mapping_path: str = "config/team_mapping.json", min_similarity: float = 0.82) -> None:
        self.mapping_path = Path(mapping_path)
        self.min_similarity = min_similarity
        self.mapping = self._load_mapping()
        # setdefault: first-listed key wins on a fold collision (stable,
        # deterministic on the file's own order) -- not expected in
        # practice (two distinct real club names folding to the same
        # string would already be a data problem), just a defined tie-break
        # rather than silent last-wins.
        self._folded_mapping: dict[str, str] = {}
        for key, value in self.mapping.items():
            self._folded_mapping.setdefault(_fold_accents(key), value)

    def _load_mapping(self) -> dict[str, str]:
        if not self.mapping_path.exists():
            LOGGER.warning("Team mapping file not found: %s", self.mapping_path)
            return {}
        try:
            with self.mapping_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            LOGGER.warning("Invalid JSON in team mapping file: %s", self.mapping_path)
            return {}
        if not isinstance(payload, dict):
            LOGGER.warning("Team mapping file must be a JSON object: %s", self.mapping_path)
            return {}
        return {str(key): str(value) for key, value in payload.items()}

    def map_team(
        self,
        team_name: str,
        candidates: Iterable[str] | None = None,
        use_token_match: bool = False,
    ) -> str:
        """Map a team name using explicit mappings or a fuzzy fallback."""
        normalized = " ".join(str(team_name).strip().split())
        if not normalized:
            return normalized
        if normalized in self.mapping:
            return self.mapping[normalized]

        folded = _fold_accents(normalized)
        if folded in self._folded_mapping:
            return self._folded_mapping[folded]

        if candidates is None:
            LOGGER.warning(
                "Unmapped team '%s'. Add mapping to %s.",
                normalized,
                self.mapping_path,
            )
            return normalized

        suggestion, score = self.suggest(normalized, candidates, use_token_match=use_token_match)
        if suggestion is None:
            LOGGER.warning(
                "Unmapped team '%s'. Add mapping to %s.",
                normalized,
                self.mapping_path,
            )
            return normalized

        if score >= self.min_similarity:
            LOGGER.warning(
                "Unmapped team '%s'. Using fuzzy match '%s' (score=%.2f). "
                "Add mapping to %s.",
                normalized,
                suggestion,
                score,
                self.mapping_path,
            )
            return suggestion

        LOGGER.warning(
            "Unmapped team '%s'. Closest match '%s' (score=%.2f). "
            "Add mapping to %s.",
            normalized,
            suggestion,
            score,
            self.mapping_path,
        )
        return normalized

    def suggest(
        self, team_name: str, candidates: Iterable[str], use_token_match: bool = False
    ) -> tuple[str | None, float]:
        """Suggest the closest mapping candidate for a new team name.

        `use_token_match=True` (opt-in) additionally scores each candidate by
        `_token_containment_score` before falling back to Levenshtein --
        closes the club-type-prefix gap ("TSG Hoffenheim" vs "Hoffenheim")
        plain Levenshtein can't bridge (0.71 similarity, below the 0.82
        threshold). Left False (the default) for every pre-existing caller
        (fotmob/understat merge.py), whose candidate pools span whole-league
        history and can contain genuinely distinct, differently-named clubs a
        token-subset match would wrongly conflate (see
        tests/test_three_leagues_team_mapping.py's "GFC Ajaccio" case).

        If two or more candidates tie for the best score, returns (None,
        best_score) rather than guessing one -- more likely to matter once
        token matching makes an exact-token tie between two real candidates
        possible (e.g. two "Real ..." clubs sharing the word "Real")."""
        best_name: str | None = None
        best_score = -1.0
        tied = False
        for candidate in candidates:
            candidate_name = standardize_team_name(str(candidate))
            token_score = _token_containment_score(team_name, candidate_name) if use_token_match else None
            score = token_score if token_score is not None else _similarity_score(team_name, candidate_name)
            if score > best_score:
                best_score = score
                best_name = candidate_name
                tied = False
            elif score == best_score:
                tied = True
        if tied and best_score >= self.min_similarity:
            return None, best_score
        return best_name, best_score
