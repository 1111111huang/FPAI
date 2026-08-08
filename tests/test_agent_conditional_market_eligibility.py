"""Regression tests for A54: 'conditional' is only a coherent recommendation
for over/yes-type markets (total_goals over, corners over, btts yes) -- the
markets where waiting for a better price is a directional strategy, not a
coin flip. See documents/agent_user_stories.md A54."""

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


@pytest.mark.parametrize(
    "market,selection",
    [
        ("total_goals", "over_2.5"),
        ("home_corners", "over_2.5"),
        ("away_corners", "over_2.5"),
        ("btts", "yes"),
    ],
)
def test_eligible_market_stays_conditional_after_a29_downgrade(market, selection):
    """A29's floor-downgrade case, on an eligible market -- unchanged
    behavior, still conditional, still gets a real target_odds (A52)."""
    m = {**_VALID_MARKET, "market": market, "selection": selection, "current_odds": 1.05, "ml_probability": 0.55}
    data = {**_VALID, "markets": [m]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["markets"][0]["target_odds"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "market,selection",
    [
        ("result_3way", "home"),
        ("result_3way", "draw"),
        ("result_3way", "away"),
        ("total_goals", "under_2.5"),
        ("home_corners", "under_2.5"),
        ("away_corners", "under_2.5"),
        ("btts", "no"),
    ],
)
def test_ineligible_market_downgraded_to_no_bet_after_a29_downgrade(market, selection):
    """A29's floor-downgrade case, on an ineligible market -- must not stay
    conditional. Downgrades to no_bet, and correspondingly never gets a
    target_odds."""
    m = {**_VALID_MARKET, "market": market, "selection": selection, "current_odds": 1.05, "ml_probability": 0.55}
    data = {**_VALID, "markets": [m]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["markets"][0]["target_odds"] is None
    assert any("conditional to no_bet" in note for note in rec["limitations"])


def test_llm_originated_conditional_on_ineligible_market_is_downgraded():
    """Not just A29's algorithmic downgrade -- an LLM free-text 'conditional'
    call on an ineligible market (never touched by A29, since it wasn't
    direct_bet to begin with) must also be corrected."""
    m = {**_VALID_MARKET, "recommendation_type": "conditional", "current_odds": 1.5, "ml_probability": 0.5}
    data = {**_VALID, "overall": "conditional", "markets": [m]}  # market=result_3way, selection=home

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["markets"][0]["target_odds"] is None


def test_llm_originated_conditional_on_eligible_market_is_untouched():
    m = {
        **_VALID_MARKET,
        "market": "btts",
        "selection": "yes",
        "recommendation_type": "conditional",
        "current_odds": 1.5,
        "ml_probability": 0.5,
    }
    data = {**_VALID, "overall": "conditional", "markets": [m]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "conditional"
    assert rec["markets"][0]["target_odds"] == pytest.approx(1 / 0.45)


def test_direct_bet_and_no_bet_markets_are_never_touched_by_this_pass():
    """This pass only ever looks at markets already 'conditional' -- a
    direct_bet within bounds, or a no_bet, must be completely unaffected."""
    direct = {**_VALID_MARKET, "current_odds": 2.1, "ml_probability": 0.55}
    no_bet = {**_VALID_MARKET, "market": "btts", "selection": "no", "recommendation_type": "no_bet", "current_odds": None}
    data = {**_VALID, "markets": [direct, no_bet]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["markets"][1]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []
