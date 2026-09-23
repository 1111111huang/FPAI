"""US#210 (2026-09-22): btts/no is suppressed project-wide as a direct_bet
recommendation. Three independent training-time levers (calibration,
recency, class-weight) are all ruled out with real held-out evidence
(documents/user_stories.md US#210/US#207) -- real edge is severely negative
(-10% to -35%) in all 5 leagues. Unlike the retired result_3way/draw floor
(test_agent_draw_value_edge_floor.py), this is a flat, always-on
suppression -- there's no value_edge threshold that makes a btts/no pick
trustworthy, so no config knob to gate it on."""
from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_BTTS_NO_CANDIDATE = {
    "market": "btts",
    "selection": "no",
    "recommendation_type": "direct_bet",
    "current_odds": 1.9,
    "min_odds": 1.6,
    "ml_probability": 0.51,
    "implied_probability": 0.35,
    "value_edge": 0.16,
    "composite_score": 0.6,
    "reason": "Both defenses have looked solid recently.",
}

_VALID = {
    "match": {"home": "Roma", "away": "Inter", "date": "2026-09-22", "league": "I1"},
    "overall": "direct_bet",
    "candidates": [_BTTS_NO_CANDIDATE],
    "recommendation_pick": {"market": "btts", "selection": "no"},
    "explanation": "Both teams unlikely to score.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_btts_no_direct_bet_is_always_downgraded_even_with_a_large_value_edge():
    """Unlike every other guardrail in this file, there's no threshold to
    clear -- a large, apparently-attractive value_edge (0.16, well above the
    general 0.05 floor) is still suppressed."""
    rec = extract_recommendation(_wrap_json(_VALID))

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert any("btts" in note and "no" in note for note in rec["limitations"])
    assert rec["overall"] == "no_bet"


def test_btts_yes_direct_bet_is_not_affected():
    candidate = {**_BTTS_NO_CANDIDATE, "selection": "yes", "value_edge": 0.10}
    data = {**_VALID, "candidates": [candidate], "recommendation_pick": {"market": "btts", "selection": "yes"}}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["candidates"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


def test_other_markets_no_named_selection_not_affected():
    """Scoped to market == 'btts' specifically -- a coincidentally
    'no'-named selection elsewhere shouldn't be possible given the schema's
    Literal constraints, but this pass must not key off selection alone."""
    candidate = {**_BTTS_NO_CANDIDATE, "market": "total_goals", "selection": "under_2.5"}
    data = {**_VALID, "candidates": [candidate], "recommendation_pick": {"market": "total_goals", "selection": "under_2.5"}}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["candidates"][0]["recommendation_type"] == "direct_bet"


def test_no_bet_btts_no_market_is_not_touched_by_this_pass():
    candidate = {**_BTTS_NO_CANDIDATE, "recommendation_type": "no_bet", "value_edge": -0.3}
    data = {**_VALID, "overall": "no_bet", "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_conditional_btts_no_is_left_to_the_existing_conditional_pipeline():
    """This pass is scoped to 'direct_bet' only, same precedent as the
    other downgrade passes -- a self-reported 'conditional' pick runs
    through the existing conditional-eligibility checks untouched by this
    one, since those already decide whether over/yes-type markets alone can
    be conditional."""
    candidate = {**_BTTS_NO_CANDIDATE, "recommendation_type": "conditional", "value_edge": 0.16}
    data = {**_VALID, "overall": "conditional", "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["candidates"][0]["recommendation_type"] != "direct_bet"
