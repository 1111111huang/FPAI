"""Utility helpers for ingestion workflows."""

from __future__ import annotations

import hashlib


def _normalize(value: str) -> str:
    """Normalize text for stable hashing by trimming and lowercasing."""
    return " ".join(value.strip().lower().split())


def generate_match_id(date: str, home_team: str, away_team: str) -> str:
    """Generate a deterministic SHA-256 ID from date, home team, and away team."""
    normalized_parts = (_normalize(date), _normalize(home_team), _normalize(away_team))
    payload = "|".join(normalized_parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
