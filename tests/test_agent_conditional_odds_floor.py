"""Regression tests for A66: a 'conditional' market's current_odds below
min_conditional_odds_threshold (default 1.5 decimal, roughly -200 American)
is downgraded to 'no_bet' -- either the price is already too short for
"wait for it to improve" to be a realistic strategy, or current_odds isn't
a real price at all (e.g. 0.0, confirmed live on a corners market with no
real bookmaker feed to ground it). See documents/agent_user_stories.md A66."""

from __future__ import annotations

import json

import pytest

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


def test_conditional_below_floor_downgraded_to_no_bet():
    market = {**_VALID_MARKET, "current_odds": 1.13}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["markets"][0]["target_odds"] is None
    assert any("1.13" in note and "no_bet" in note for note in rec["limitations"])


def test_zero_current_odds_downgraded_to_no_bet():
    """The concrete production case: current_odds=0.0 on a corners market
    with no real bookmaker feed to ground it -- not a real price at all,
    caught by the same floor as any other implausibly short one."""
    market = {**_VALID_MARKET, "current_odds": 0.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["markets"][0]["target_odds"] is None


def test_conditional_at_exact_floor_is_accepted():
    market = {**_VALID_MARKET, "current_odds": 1.5}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_conditional_above_floor_is_untouched():
    market = {**_VALID_MARKET, "current_odds": 3.0}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_conditional_with_null_current_odds_is_not_touched():
    """Out of scope for this pass -- a null current_odds already means no
    target_odds gets computed (A52's own guard); nothing here to downgrade."""
    market = {**_VALID_MARKET, "current_odds": None}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_direct_bet_and_no_bet_markets_are_never_touched_by_this_pass():
    direct = {**_VALID_MARKET, "recommendation_type": "direct_bet", "current_odds": 2.1}
    no_bet = {**_VALID_MARKET, "recommendation_type": "no_bet", "current_odds": 0.0}
    data = {**_VALID, "overall": "direct_bet", "markets": [direct, no_bet]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["markets"][1]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_custom_threshold_is_respected_not_hardcoded():
    market = {**_VALID_MARKET, "current_odds": 1.6}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_conditional_odds_threshold=2.0)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
