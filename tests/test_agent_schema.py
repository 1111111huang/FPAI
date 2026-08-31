"""Tests for MatchRecommendation schema parsing (A06).

A90 (2026-08-31 design): reworked for the single-recommendation schema --
`markets` is now `candidates` plus a `recommendation_pick` pointer."""
import json
import pytest

from src.agent.schema import extract_recommendation, RecommendationParseError

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "no_bet",
    "candidates": [],
    "recommendation_pick": None,
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


def test_all_valid_overall_values_are_accepted_as_input():
    """Every value in the enum parses without raising -- distinct from
    whether it survives unchanged. A90's _resolve_recommendation_pick caps
    an unsupported claim down to 'no_bet' when candidates is empty (no pick
    to back it up, same downgrade-only direction A65 already established),
    so 'direct_bet'/'conditional' correctly land on 'no_bet' here -- only
    'no_bet' and 'insufficient_data' themselves (at or below that floor)
    pass through unchanged. See test_agent_schema_validation.py's own
    test_overall_downgrades_from_direct_bet_when_pick_is_null_with_no_candidates
    for the dedicated regression test of that specific behavior."""
    expected = {
        "direct_bet": "no_bet",
        "conditional": "no_bet",
        "no_bet": "no_bet",
        "insufficient_data": "insufficient_data",
    }
    for val, want in expected.items():
        data = {**_VALID, "overall": val}
        rec = extract_recommendation(_wrap_json(data))
        assert rec["overall"] == want


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


def test_malformed_candidates_array_repaired():
    """llama3.1:8b splits a multi-element candidates array into separate bracket
    groups instead of comma-separating objects within one array. json_repair
    fallback fixes it."""
    text = (
        '{"match": {"home": "City", "away": "Arsenal", "date": "2026-06-21", "league": "E0"}, '
        '"overall": "conditional", '
        '"candidates": [{"market": "result_3way", "selection": "home", "recommendation_type": "conditional", '
        '"current_odds": 1.95, "min_odds": 1.95, "ml_probability": 0.48, "implied_probability": 0.51, '
        '"value_edge": -0.03, "composite_score": 0.1, "reason": "Priced too tight."}], '
        '["market": "result_3way", "selection": "away", "recommendation_type": "conditional", '
        '"current_odds": 4.2, "min_odds": 4.2, "ml_probability": 0.26, "implied_probability": 0.24, '
        '"value_edge": 0.02, "composite_score": 0.3, "reason": "Modest edge, away form unclear."]], '
        '"recommendation_pick": {"market": "result_3way", "selection": "away"}, '
        '"explanation": "test", "confidence": "medium", "limitations": [], "prediction_basis": "market_odds_only"}'
    )
    rec = extract_recommendation(text)
    # A54 downgrades both (result_3way isn't conditional-eligible) to
    # no_bet, and _resolve_recommendation_pick (A90) syncs 'overall' to
    # match -- this test's own concern is json_repair recovering both
    # candidates, not this value, but it must reflect what they actually
    # ended up as.
    assert rec["overall"] == "no_bet"
    assert len(rec["candidates"]) == 2
    assert rec["candidates"][1]["selection"] == "away"
