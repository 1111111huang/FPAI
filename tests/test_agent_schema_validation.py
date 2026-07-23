"""Regression tests for A28: extract_recommendation must validate field types/
enums beyond key presence, and close BUG-013's root cause (a market marked
direct_bet with a null current_odds) at extraction time rather than passing
it through to crash downstream. Covers the three specific gaps documented in
agent_techspec.md Section 17 (value_edge as a string, confidence as an empty
string, an arbitrary recommendation_type string) plus BUG-013's null-odds
case."""

from __future__ import annotations

import json

import pytest

from src.agent.schema import RecommendationParseError, extract_recommendation

_VALID_MARKET = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 2.1,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "markets": [_VALID_MARKET],
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_fully_valid_output_with_a_real_market_still_parses_unchanged():
    """Regression: existing valid outputs (now including a populated markets
    list, not just the empty-list case in test_agent_schema.py) must be
    completely unaffected by the new validation layer."""
    rec = extract_recommendation(_wrap_json(_VALID))
    assert rec["overall"] == "direct_bet"
    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["markets"][0]["current_odds"] == 2.1
    assert rec["limitations"] == []


def test_value_edge_as_string_raises():
    bad_market = {**_VALID_MARKET, "value_edge": "high"}
    bad = {**_VALID, "markets": [bad_market]}
    with pytest.raises(RecommendationParseError, match="value_edge"):
        extract_recommendation(_wrap_json(bad))


def test_confidence_empty_string_raises():
    bad = {**_VALID, "confidence": ""}
    with pytest.raises(RecommendationParseError, match="confidence"):
        extract_recommendation(_wrap_json(bad))


def test_arbitrary_recommendation_type_raises():
    bad_market = {**_VALID_MARKET, "recommendation_type": "maybe_bet"}
    bad = {**_VALID, "markets": [bad_market]}
    with pytest.raises(RecommendationParseError, match="recommendation_type"):
        extract_recommendation(_wrap_json(bad))


def test_direct_bet_with_null_odds_downgraded_to_no_bet():
    """BUG-013: a market marked direct_bet with current_odds=null must be
    downgraded to no_bet with an explanatory limitations note, not passed
    through as-is and not raised as a parse error."""
    bad_market = {**_VALID_MARKET, "current_odds": None}
    bad = {**_VALID, "markets": [bad_market]}

    rec = extract_recommendation(_wrap_json(bad))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["markets"][0]["current_odds"] is None
    assert any("direct_bet" in note and "no_bet" in note for note in rec["limitations"])


def test_conditional_market_with_null_odds_is_not_touched():
    """The downgrade rule is specific to direct_bet -- a conditional market
    with null current_odds (a legitimate state) must be left alone."""
    market = {**_VALID_MARKET, "recommendation_type": "conditional", "current_odds": None}
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_overall_as_unhashable_dict_raises_parse_error_not_type_error():
    """BUG-020: a real qwen2.5-coder:7b response returned `overall` as a dict
    (e.g. {"liverpool_form": ..., "bournemouth_form": ...}) instead of a
    string. `data["overall"] not in _VALID_OVERALL` requires hashing the LHS
    for the set membership test -- a dict/list there raised an unhandled
    `TypeError: unhashable type: 'dict'` that crashed the whole agent-recommend
    call instead of being treated as an invalid `overall` value and falling
    through to RecommendationParseError like any other malformed candidate."""
    bad = {**_VALID, "overall": {"liverpool_form": "mixed", "bournemouth_form": "worse"}}
    with pytest.raises(RecommendationParseError):
        extract_recommendation(_wrap_json(bad))


def test_overall_as_unhashable_list_raises_parse_error_not_type_error():
    bad = {**_VALID, "overall": ["direct_bet", "no_bet"]}
    with pytest.raises(RecommendationParseError):
        extract_recommendation(_wrap_json(bad))
