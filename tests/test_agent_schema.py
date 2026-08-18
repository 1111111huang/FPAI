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


def test_malformed_markets_array_repaired():
    """llama3.1:8b splits a multi-element markets array into separate bracket groups
    instead of comma-separating objects within one array. json_repair fallback fixes it."""
    text = (
        '{"match": {"home": "City", "away": "Arsenal", "date": "2026-06-21", "league": "E0"}, '
        '"overall": "conditional", '
        '"markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "conditional", '
        '"current_odds": 1.95, "min_odds": 1.95, "ml_probability": 0.48, "implied_probability": 0.51, '
        '"value_edge": -0.03}], ["market": "result_3way", "selection": "away", "recommendation_type": "conditional", '
        '"current_odds": 4.2, "min_odds": 4.2, "ml_probability": 0.26, "implied_probability": 0.24, '
        '"value_edge": 0.02]], '
        '"explanation": "test", "confidence": "medium", "limitations": [], "prediction_basis": "market_odds_only"}'
    )
    rec = extract_recommendation(text)
    # A54 downgrades both (result_3way isn't conditional-eligible) to
    # no_bet, and A65 reconciles 'overall' to match -- this test's own
    # concern is json_repair recovering both markets, not this value, but
    # it must reflect what the markets actually ended up as.
    assert rec["overall"] == "no_bet"
    assert len(rec["markets"]) == 2
    assert rec["markets"][1]["selection"] == "away"
