"""Tests for MatchRecommendation schema parsing (A06)."""
import json
import pytest

from src.agent.schema import extract_recommendation, RecommendationParseError

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "no_bet",
    "markets": [],
    "explanation": "No value found.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_extract_fenced_json():
    rec = extract_recommendation(_wrap_json(_VALID))
    assert rec["overall"] == "no_bet"
    assert rec["match"]["home"] == "Arsenal"


def test_extract_bare_json():
    text = "Analysis done. " + json.dumps(_VALID)
    rec = extract_recommendation(text)
    assert rec["confidence"] == "medium"


def test_all_valid_overall_values():
    for val in ("direct_bet", "conditional", "no_bet", "insufficient_data"):
        data = {**_VALID, "overall": val}
        rec = extract_recommendation(_wrap_json(data))
        assert rec["overall"] == val


def test_invalid_overall_raises():
    bad = {**_VALID, "overall": "maybe_bet"}
    with pytest.raises(RecommendationParseError, match="invalid overall"):
        extract_recommendation(_wrap_json(bad))


def test_missing_field_raises():
    bad = {k: v for k, v in _VALID.items() if k != "explanation"}
    with pytest.raises(RecommendationParseError, match="missing fields"):
        extract_recommendation(_wrap_json(bad))


def test_no_json_raises():
    with pytest.raises(RecommendationParseError, match="no JSON"):
        extract_recommendation("The agent could not produce a recommendation.")


def test_invalid_json_raises():
    with pytest.raises(RecommendationParseError):
        extract_recommendation("```json\n{bad json here\n```")


def test_trailing_brace_tolerated():
    """Model output with extra }} at end should still parse."""
    text = json.dumps(_VALID) + "}"  # simulate llama3.2:3b adding an extra }
    rec = extract_recommendation(text)
    assert rec["overall"] == "no_bet"
