"""Regression tests for A65: `overall` (the LLM's own top-level self-report)
is reconciled against `markets`' *final* recommendation_type (after every
code-enforced downgrade pass above has run), so it can never claim a
stronger state than any market actually supports."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_BASE_MARKET = {
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
    "markets": [_BASE_MARKET],
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_overall_left_unchanged_when_a_market_actually_supports_it():
    rec = extract_recommendation(_wrap_json(_VALID))

    assert rec["overall"] == "direct_bet"
    assert rec["limitations"] == []


def test_overall_downgraded_to_the_strongest_market_present():
    """Two markets: the direct_bet one gets bounds-downgraded to
    conditional (A29's *ceiling* case -- odds too high, not too low; A66's
    floor doesn't apply to it, unlike the floor case), the other was
    already no_bet -- overall must land on 'conditional' (the strongest of
    the two survivors), not 'no_bet'."""
    downgraded_market = {
        **_BASE_MARKET, "market": "total_goals", "selection": "over_2.5",
        "current_odds": 15.0, "ml_probability": 0.10,
    }
    no_bet_market = {**_BASE_MARKET, "market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 1.8}
    data = {**_VALID, "markets": [downgraded_market, no_bet_market]}

    rec = extract_recommendation(_wrap_json(data))

    assert {m["recommendation_type"] for m in rec["markets"]} == {"conditional", "no_bet"}
    assert rec["overall"] == "conditional"
    assert any("direct_bet" in note and "conditional" in note for note in rec["limitations"])


def test_overall_untouched_when_markets_list_is_empty():
    """Nothing to reconcile against -- 'overall' is left exactly as the
    LLM reported it rather than forced toward some default."""
    data = {**_VALID, "overall": "no_bet", "markets": []}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["overall"] == "no_bet"
    assert rec["limitations"] == []


def test_overall_already_no_bet_is_never_upgraded_by_a_direct_bet_market():
    """The reconciliation only ever downgrades -- it must never raise
    'overall' to match a stronger market than the LLM itself reported."""
    data = {**_VALID, "overall": "no_bet"}  # markets still has the direct_bet _BASE_MARKET

    rec = extract_recommendation(_wrap_json(data))

    assert rec["overall"] == "no_bet"
