"""Tests for A107: extract_recommendation() snapshots each candidate's own
self-reported recommendation_type into initial_recommendation_type before
any downgrade pass can overwrite recommendation_type in place -- so a
guardrail-caused change is a plain field comparison instead of needing a
full reasoning-trace read (BUG-054's own investigation)."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_VALID_CANDIDATE = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 2.1,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
    "composite_score": 0.6,
    "reason": "Clears the odds bounds at a realistic price.",
}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [_VALID_CANDIDATE],
    "recommendation_pick": {"market": "result_3way", "selection": "home"},
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_survives_untouched_when_no_guardrail_fires():
    result = extract_recommendation(_wrap_json(_VALID))
    candidate = result["candidates"][0]
    assert candidate["initial_recommendation_type"] == "direct_bet"
    assert candidate["recommendation_type"] == "direct_bet"


def test_records_original_type_when_a_guardrail_downgrades_it():
    # current_odds outside [1.2, 11.0] -- A29 downgrades direct_bet -> conditional,
    # then A54 downgrades it further to no_bet (result_3way is never conditional-eligible).
    candidate = {**_VALID_CANDIDATE, "current_odds": 15.0}
    data = {**_VALID, "candidates": [candidate]}

    result = extract_recommendation(_wrap_json(data))
    resolved = result["candidates"][0]

    assert resolved["initial_recommendation_type"] == "direct_bet"
    assert resolved["recommendation_type"] == "no_bet"


def test_only_the_downgraded_candidate_differs_from_its_own_initial_type():
    downgraded = {**_VALID_CANDIDATE, "current_odds": 15.0}
    untouched = {
        **_VALID_CANDIDATE, "market": "btts", "selection": "yes",
        "recommendation_type": "conditional", "current_odds": 1.8,
    }
    data = {**_VALID, "candidates": [downgraded, untouched], "recommendation_pick": {"market": "btts", "selection": "yes"}}

    result = extract_recommendation(_wrap_json(data))
    by_market = {c["market"]: c for c in result["candidates"]}

    assert by_market["result_3way"]["initial_recommendation_type"] == "direct_bet"
    assert by_market["result_3way"]["recommendation_type"] == "no_bet"
    assert by_market["btts"]["initial_recommendation_type"] == "conditional"
    assert by_market["btts"]["recommendation_type"] == "conditional"
