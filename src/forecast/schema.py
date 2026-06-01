"""Forecast payload schema and lightweight validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FORECAST_PAYLOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "match_id",
        "date",
        "league",
        "home_team",
        "away_team",
        "forecast",
        "explainability",
        "diagnostics",
    ],
    "properties": {
        "match_id": {"type": "string"},
        "date": {"type": "string"},
        "league": {"type": "string"},
        "home_team": {"type": "string"},
        "away_team": {"type": "string"},
        "forecast": {"type": "object"},
        "explainability": {"type": "object"},
        "diagnostics": {"type": "object"},
    },
}


def validate_forecast_payload(payload: Mapping[str, Any]) -> None:
    """Validate the stable forecast payload contract used by agents."""
    required = FORECAST_PAYLOAD_SCHEMA["required"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Forecast payload missing required keys: {', '.join(missing)}")
    if not isinstance(payload["forecast"], Mapping) or not payload["forecast"]:
        raise ValueError("Forecast payload must contain a non-empty forecast object.")
    explainability = payload["explainability"]
    if not isinstance(explainability, Mapping) or not isinstance(explainability.get("top_features"), list):
        raise ValueError("Forecast payload explainability.top_features must be a list.")
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValueError("Forecast payload diagnostics must be an object.")
    for key in ["model_version", "target_versions", "feature_completeness", "cold_start_risk", "generated_at"]:
        if key not in diagnostics:
            raise ValueError(f"Forecast diagnostics missing required key: {key}")
