"""Regression tests for A82: attach a deterministic unit_bet_multiplier to
every extracted recommendation -- a Kelly-derived stake-sizing suggestion
for the recommendation's actual pick (A81's pick_recommended_market),
expressed as a multiple of an abstract "Unit Bet" (UB), not a dollar
figure. See documents/agent_user_stories.md A82."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_VALID_MARKET = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 3.0,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.33,
    "value_edge": 0.10,
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


def test_direct_bet_gets_a_positive_multiplier():
    # kelly_fraction(0.10, 3.0) = 0.10 / 2.0 = 0.05 -> 0.05 / 0.01 = 5.0
    rec = extract_recommendation(_wrap_json(_VALID))
    assert rec["unit_bet_multiplier"] == 5.0


def test_multiplier_capped_at_ten():
    market = {**_VALID_MARKET, "current_odds": 1.5, "value_edge": 0.9}
    data = {**_VALID, "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] == 10.0


def test_no_bet_overall_gets_no_multiplier():
    market = {**_VALID_MARKET, "recommendation_type": "no_bet"}
    data = {**_VALID, "overall": "no_bet", "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] is None


def test_missing_odds_gets_no_multiplier():
    # A67/BUG-013 already forbid direct_bet with null odds, but a
    # conditional market can legitimately have current_odds=None.
    market = {
        **_VALID_MARKET, "market": "btts", "selection": "yes",
        "recommendation_type": "conditional", "current_odds": None,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] is None


def test_conditional_below_value_floor_gets_zero_not_null():
    # A real price exists but doesn't clear the value bar yet -- "wait, 0
    # UB for now" is meaningfully different from "no price at all".
    market = {
        **_VALID_MARKET, "market": "btts", "selection": "yes",
        "recommendation_type": "conditional", "current_odds": 1.6, "value_edge": -0.02,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["unit_bet_multiplier"] == 0.0
