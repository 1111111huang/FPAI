"""Regression tests for A52: code-compute a `target_odds` field for
'conditional' markets, so a "wait" recommendation states what odds would
actually clear the value threshold -- not left to the LLM's own (unverified)
min_odds arithmetic. See documents/agent_user_stories.md A52."""

from __future__ import annotations

import json

import pytest

from src.agent.schema import extract_recommendation

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


def test_floor_downgraded_market_gets_a_target_odds_above_current_odds():
    """A29's floor-downgrade case: current_odds (1.05) is below min_odds_threshold
    (1.2), so this market becomes conditional. Hand-computed: needed_prob =
    0.55 - 0.05 = 0.5, candidate = 1 / 0.5 = 2.0, and 2.0 > 1.05 -- a genuine
    forward target. A54: market/selection overridden to an eligible pair --
    result_3way (the shared _VALID_MARKET default) would be further
    downgraded to no_bet by that pass, which isn't this test's concern."""
    market = {**_VALID_MARKET, "market": "total_goals", "selection": "over_2.5", "current_odds": 1.05, "ml_probability": 0.55}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["markets"][0]["target_odds"] == pytest.approx(2.0)
    assert rec["markets"][0]["target_odds"] > rec["markets"][0]["current_odds"]


def test_ceiling_downgraded_market_gets_no_target_odds():
    """A29's ceiling-downgrade case: current_odds (15.0) is above
    max_odds_threshold (11.0). The break-even price (2.0, same formula as
    above) is *below* current_odds here -- 'wait for it to rise' would be
    backwards, so target_odds must be None, not a nonsensical lower number.
    A54: see test_floor_downgraded_market_gets_a_target_odds_above_current_odds's comment."""
    market = {**_VALID_MARKET, "market": "total_goals", "selection": "over_2.5", "current_odds": 15.0, "ml_probability": 0.55}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["markets"][0]["target_odds"] is None


def test_llm_originated_conditional_market_gets_a_correctly_computed_target_odds():
    """A market the LLM itself marked 'conditional' from the start (never
    touched by either downgrade pass) still gets target_odds computed.
    Hand-computed: needed_prob = 0.5 - 0.05 = 0.45, candidate = 1 / 0.45 =
    2.2222..., and 2.2222 > 1.5 (current_odds). A54: market/selection
    overridden to an eligible pair (btts/yes) -- an LLM-originated
    'conditional' on result_3way would be downgraded to no_bet by that pass,
    tested separately in test_agent_conditional_market_eligibility.py."""
    market = {
        **_VALID_MARKET,
        "market": "btts",
        "selection": "yes",
        "recommendation_type": "conditional",
        "current_odds": 1.5,
        "ml_probability": 0.5,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["target_odds"] == pytest.approx(1 / 0.45)


def test_direct_bet_market_left_untouched():
    """A market that stays direct_bet (within bounds) never gets a real
    target_odds -- there's nothing to wait for."""
    market = {**_VALID_MARKET, "current_odds": 2.1, "ml_probability": 0.55}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["markets"][0]["target_odds"] is None


def test_no_bet_market_left_untouched():
    market = {**_VALID_MARKET, "recommendation_type": "no_bet", "current_odds": None}
    data = {**_VALID, "overall": "no_bet", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["target_odds"] is None


def test_conditional_market_with_null_current_odds_gets_no_target_odds():
    """No price to solve a forward target against."""
    market = {**_VALID_MARKET, "recommendation_type": "conditional", "current_odds": None}
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["target_odds"] is None


def test_conditional_market_with_ml_probability_already_below_min_value_edge_gets_no_target_odds():
    """needed_prob <= 0 -- no price fixes an ml_probability that's already
    below the edge floor on its own."""
    market = {
        **_VALID_MARKET,
        "recommendation_type": "conditional",
        "current_odds": 1.5,
        "ml_probability": 0.03,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge=0.05)

    assert rec["markets"][0]["target_odds"] is None


def test_custom_min_value_edge_is_respected_not_hardcoded():
    # A54: market/selection overridden to an eligible pair, same reasoning
    # as test_llm_originated_conditional_market_gets_a_correctly_computed_target_odds.
    market = {
        **_VALID_MARKET,
        "market": "btts",
        "selection": "yes",
        "recommendation_type": "conditional",
        "current_odds": 1.5,
        "ml_probability": 0.5,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge=0.1)

    # needed_prob = 0.5 - 0.1 = 0.4, candidate = 1 / 0.4 = 2.5
    assert rec["markets"][0]["target_odds"] == pytest.approx(2.5)
