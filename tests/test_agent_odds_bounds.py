"""Regression tests for A29: code-enforce widened direct_bet odds bounds
[1.2, 11.0] (decimal) in the same extract_recommendation validation pass A28
added, replacing the old prompt-only 2.0-minimum rule."""

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


def test_direct_bet_below_floor_downgraded_to_conditional():
    # A54: 'conditional' only stays conditional for an eligible (over/yes)
    # market -- result_3way (the shared _VALID_MARKET default) would be
    # further downgraded to no_bet by that pass, which isn't this test's
    # concern.
    market = {**_VALID_MARKET, "market": "total_goals", "selection": "over_2.5", "current_odds": 1.05}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert any("1.05" in note and "conditional" in note for note in rec["limitations"])


def test_direct_bet_above_ceiling_downgraded_to_conditional():
    # A54: see test_direct_bet_below_floor_downgraded_to_conditional's comment.
    market = {**_VALID_MARKET, "market": "total_goals", "selection": "over_2.5", "current_odds": 15.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert any("15.0" in note and "conditional" in note for note in rec["limitations"])


def test_direct_bet_at_exact_floor_is_accepted():
    market = {**_VALID_MARKET, "current_odds": 1.2}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


def test_direct_bet_at_exact_ceiling_is_accepted():
    market = {**_VALID_MARKET, "current_odds": 11.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


def test_old_2_0_only_floor_behavior_is_gone():
    """Odds of 1.5 -- below the OLD 2.0 floor but within the new [1.2, 11.0]
    band -- must now be accepted as direct_bet, not downgraded."""
    market = {**_VALID_MARKET, "current_odds": 1.5}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"


def test_custom_thresholds_are_respected_not_hardcoded():
    # A54: see test_direct_bet_below_floor_downgraded_to_conditional's comment.
    market = {**_VALID_MARKET, "market": "total_goals", "selection": "over_2.5", "current_odds": 3.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_odds_threshold=3.5, max_odds_threshold=11.0)

    assert rec["markets"][0]["recommendation_type"] == "conditional"


def test_conditional_market_outside_bounds_is_not_touched():
    """The bounds rule only governs direct_bet -- a conditional market's own
    odds are none of this rule's business. A54: market/selection overridden
    to an eligible pair so A54's own restriction pass doesn't also fire here
    -- that's tested separately in test_agent_conditional_market_eligibility.py."""
    market = {**_VALID_MARKET, "market": "btts", "selection": "yes", "recommendation_type": "conditional", "current_odds": 50.0}
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_direct_bet_with_null_odds_still_downgrades_to_no_bet_not_conditional():
    """BUG-013's null-odds rule (A28) takes precedence over the bounds rule --
    a market can't be bounds-checked if it has no odds to check."""
    market = {**_VALID_MARKET, "current_odds": None}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
