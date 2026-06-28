"""Fuzzy team-name resolution shared across ingestion sources.

Maps a source's team-name spelling (Understat, FotMob, etc.) onto the
canonical names already used in raw_matches, via an explicit JSON mapping
file with a Levenshtein-distance fallback for names not yet mapped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


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


class TeamNameMapper:
    """Map a source's team names to the CSV canonical names."""

    def __init__(self, mapping_path: str = "config/team_mapping.json", min_similarity: float = 0.82) -> None:
        self.mapping_path = Path(mapping_path)
        self.min_similarity = min_similarity
        self.mapping = self._load_mapping()

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

    def map_team(self, team_name: str, candidates: Iterable[str] | None = None) -> str:
        """Map a team name using explicit mappings or a fuzzy fallback."""
        normalized = " ".join(str(team_name).strip().split())
        if not normalized:
            return normalized
        if normalized in self.mapping:
            return self.mapping[normalized]

        if candidates is None:
            LOGGER.warning(
                "Unmapped team '%s'. Add mapping to %s.",
                normalized,
                self.mapping_path,
            )
            return normalized

        suggestion, score = self.suggest(normalized, candidates)
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

    def suggest(self, team_name: str, candidates: Iterable[str]) -> tuple[str | None, float]:
        """Suggest the closest mapping candidate for a new team name."""
        best_name: str | None = None
        best_score = -1.0
        for candidate in candidates:
            candidate_name = standardize_team_name(str(candidate))
            score = _similarity_score(team_name, candidate_name)
            if score > best_score:
                best_score = score
                best_name = candidate_name
        return best_name, best_score
