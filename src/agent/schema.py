from __future__ import annotations

import json
import re
from typing import Literal, TypedDict


class MarketRecommendation(TypedDict):
    market: str
    selection: str
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float
    min_odds: float
    ml_probability: float
    implied_probability: float
    value_edge: float


class MatchRecommendation(TypedDict):
    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    markets: list[MarketRecommendation]
    explanation: str
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str


_REQUIRED_KEYS = {"match", "overall", "markets", "explanation", "confidence", "limitations", "prediction_basis"}
_VALID_OVERALL = {"direct_bet", "conditional", "no_bet", "insufficient_data"}


class RecommendationParseError(Exception):
    def __init__(self, raw_text: str, reason: str = ""):
        self.raw_text = raw_text
        msg = f"Failed to parse MatchRecommendation from agent output"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


def extract_recommendation(text: str) -> MatchRecommendation:
    """Extract and validate a MatchRecommendation JSON block from agent output text."""
    # Prefer explicit ```json ... ``` fenced block
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        json_str = fenced.group(1)
    else:
        # Fall back to outermost { ... } object
        obj = re.search(r"\{.*\}", text, re.DOTALL)
        if not obj:
            raise RecommendationParseError(text, "no JSON object found")
        json_str = obj.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise RecommendationParseError(text, f"invalid JSON: {exc}") from exc

    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        raise RecommendationParseError(text, f"missing fields: {sorted(missing)}")

    if data["overall"] not in _VALID_OVERALL:
        raise RecommendationParseError(text, f"invalid overall value: {data['overall']!r}")

    return data  # type: ignore[return-value]
