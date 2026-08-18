"""Regression tests for A67: recommendation_type='direct_bet' requires the
market's own self-reported value_edge to actually clear min_value_edge --
confirmed live, a market reported 'direct_bet' with value_edge=-0.138, an
incoherent combination nothing before this checked. See
documents/agent_user_stories.md A67."""

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
    "ml_probability": 0.4,
    "implied_probability": 0.48,
    "value_edge": -0.08,
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


def test_direct_bet_with_negative_value_edge_downgraded_to_no_bet():
    """The concrete production case: current_odds=1.30, value_edge=-0.138."""
    market = {**_VALID_MARKET, "current_odds": 1.30, "value_edge": -0.138}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert any("-0.138" in note and "no_bet" in note for note in rec["limitations"])
    # A65: overall must be reconciled down alongside the market.
    assert rec["overall"] == "no_bet"


def test_direct_bet_at_exact_floor_is_accepted():
    market = {**_VALID_MARKET, "value_edge": 0.05}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


def test_direct_bet_above_floor_is_untouched():
    market = {**_VALID_MARKET, "value_edge": 0.12}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


def test_conditional_market_below_the_floor_is_not_touched():
    """Out of scope for this pass -- a 'conditional' market's whole premise
    is "not enough edge yet, wait for a better price," so a low/negative
    current value_edge there is expected, not incoherent."""
    market = {
        **_VALID_MARKET, "market": "total_goals", "selection": "over_2.5",
        "recommendation_type": "conditional", "value_edge": -0.2,
    }
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_no_bet_market_below_the_floor_is_not_touched():
    market = {**_VALID_MARKET, "recommendation_type": "no_bet", "value_edge": -0.3}
    data = {**_VALID, "overall": "no_bet", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_custom_min_value_edge_is_respected_not_hardcoded():
    market = {**_VALID_MARKET, "value_edge": 0.06}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge=0.1)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"


def test_runs_before_a29_so_a_negative_edge_market_never_becomes_conditional():
    """current_odds (1.05) is also below min_odds_threshold (1.2), which
    A29 alone would reclassify as 'conditional' -- but this pass runs
    first, so a market with no genuine underlying value never reaches
    that reclassification at all."""
    market = {**_VALID_MARKET, "current_odds": 1.05, "value_edge": -0.138}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
