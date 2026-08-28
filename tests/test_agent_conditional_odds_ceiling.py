"""Regression tests for the direct-user-requested (2026-08-28) odds
ceiling: a 'conditional' market's current_odds above
max_conditional_odds_threshold (default unbounded -- only set, to 4.0/+300
American, in the live production config) is downgraded to 'no_bet'. Mirrors
A66's own floor check (test_agent_conditional_odds_floor.py) symmetrically
-- together with the direct_bet bounds (A29) and this floor (A66), this
closes the loop so nothing above the ceiling is ever recommended, direct or
conditional."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_VALID_MARKET = {
    "market": "total_goals",
    "selection": "over_2.5",
    "recommendation_type": "conditional",
    "current_odds": 1.8,
    "min_odds": 1.5,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "conditional",
    "markets": [_VALID_MARKET],
    "explanation": "Value found if the price improves.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_conditional_above_ceiling_downgraded_to_no_bet():
    market = {**_VALID_MARKET, "current_odds": 4.5}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), max_conditional_odds_threshold=4.0)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["markets"][0]["target_odds"] is None
    assert any("4.5" in note and "no_bet" in note for note in rec["limitations"])


def test_conditional_at_exact_ceiling_is_accepted():
    market = {**_VALID_MARKET, "current_odds": 4.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), max_conditional_odds_threshold=4.0)

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_conditional_below_ceiling_is_untouched():
    market = {**_VALID_MARKET, "current_odds": 3.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), max_conditional_odds_threshold=4.0)

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_conditional_with_null_current_odds_is_not_touched():
    market = {**_VALID_MARKET, "current_odds": None}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), max_conditional_odds_threshold=4.0)

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_direct_bet_and_no_bet_markets_are_never_touched_by_this_pass():
    direct = {**_VALID_MARKET, "recommendation_type": "direct_bet", "current_odds": 2.1}
    no_bet = {**_VALID_MARKET, "recommendation_type": "no_bet", "current_odds": 9.0}
    data = {**_VALID, "overall": "direct_bet", "markets": [direct, no_bet]}

    rec = extract_recommendation(_wrap_json(data), max_conditional_odds_threshold=4.0)

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["markets"][1]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_default_ceiling_is_unbounded_so_no_config_change_is_backward_compatible():
    """Every config that doesn't explicitly set max_conditional_odds_threshold
    (every posture/backtest config today) must keep today's real behavior:
    no ceiling at all."""
    market = {**_VALID_MARKET, "current_odds": 999.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))  # no override -- uses the default

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_custom_threshold_is_respected_not_hardcoded():
    market = {**_VALID_MARKET, "current_odds": 3.5}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), max_conditional_odds_threshold=3.0)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
