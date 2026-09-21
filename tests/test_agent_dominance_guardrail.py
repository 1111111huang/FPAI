"""Direct user finding (2026-09-20): composite_score is the LLM's own
self-report with no fixed formula (see
_downgrade_recommendation_below_top_composite_score's own docstring in
src/agent/schema.py) -- the existing self-consistency check only catches a
contradiction WITHIN the composite_score column itself, never a
contradiction between composite_score and the ml_probability/value_edge
numbers it's supposed to summarize. Two real matches showed the gap: a
candidate with both a higher ml_probability AND a higher value_edge than
the actual pick, yet a lower self-reported composite_score, sailed straight
through untouched. This guardrail catches strict dominance on both axes
directly, independent of composite_score."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_AWAY = {
    "market": "result_3way", "selection": "away", "recommendation_type": "direct_bet",
    "current_odds": 2.5, "min_odds": 1.8, "ml_probability": 0.45, "implied_probability": 0.4,
    "value_edge": 0.05, "composite_score": 0.35, "reason": "Away side looks the likelier winner.",
}
_OVER = {
    "market": "total_goals", "selection": "over_2.5", "recommendation_type": "direct_bet",
    "current_odds": 2.35, "min_odds": 1.8, "ml_probability": 0.546, "implied_probability": 0.426,
    "value_edge": 0.12, "composite_score": 0.3, "reason": "Over has a real edge.",
}

_BASE = {
    "match": {"home": "Parma", "away": "Genoa", "date": "2026-09-20", "league": "I1"},
    "overall": "direct_bet",
    "explanation": "Value found.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_pick_dominated_on_both_axes_is_downgraded_regardless_of_composite_score():
    # _OVER beats _AWAY on both ml_probability (0.546 > 0.45) and value_edge
    # (0.12 > 0.05), yet self-reports a LOWER composite_score (0.3 < 0.35) --
    # exactly the real Parma vs Genoa case this guardrail exists for.
    data = {**_BASE, "candidates": [_AWAY, _OVER], "recommendation_pick": {"market": "result_3way", "selection": "away"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] is None
    assert rec["overall"] == "no_bet"
    assert any("dominates" in note for note in rec["limitations"])


def test_dominating_candidate_is_never_auto_promoted():
    """The dominated pick is downgraded to no_bet, not silently swapped for
    the dominating candidate -- that candidate's own reason/team_evidence
    text was never vetted as the actual pick."""
    data = {**_BASE, "candidates": [_AWAY, _OVER], "recommendation_pick": {"market": "result_3way", "selection": "away"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] != {"market": "total_goals", "selection": "over_2.5"}


def test_pick_dominant_on_both_axes_is_unaffected():
    # _OVER's composite_score must also be >= _AWAY's here, or A91's own
    # self-consistency check (composite_score column only) fires first --
    # this test isolates the new dominance guardrail, not A91.
    dominant_over = {**_OVER, "composite_score": _AWAY["composite_score"]}
    data = {**_BASE, "candidates": [_AWAY, dominant_over], "recommendation_pick": {"market": "total_goals", "selection": "over_2.5"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "total_goals", "selection": "over_2.5"}
    assert rec["overall"] == "direct_bet"


def test_only_higher_on_one_axis_is_not_dominance():
    """A genuine tradeoff (higher probability, lower edge, or vice versa) is
    the legitimate judgment call the prompt asks composite_score to make --
    this guardrail must not fire for it."""
    higher_prob_lower_edge = {**_OVER, "ml_probability": 0.5, "value_edge": 0.03}
    data = {
        **_BASE, "candidates": [_AWAY, higher_prob_lower_edge],
        "recommendation_pick": {"market": "result_3way", "selection": "away"},
    }
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "away"}
    assert rec["overall"] == "direct_bet"


def test_a_no_bet_dominating_candidate_does_not_count():
    """Already-disqualified by an earlier guardrail -- was never a real
    alternative, so out-dominating the pick isn't a contradiction."""
    disqualified = {**_OVER, "recommendation_type": "no_bet"}
    data = {**_BASE, "candidates": [_AWAY, disqualified], "recommendation_pick": {"market": "result_3way", "selection": "away"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "away"}


def test_exact_tie_on_either_axis_is_not_dominance():
    """Strict dominance requires being higher on BOTH axes -- a tie on
    either one is not a contradiction, same >-not->= convention as the
    composite_score self-consistency guardrail."""
    tied_probability = {**_OVER, "ml_probability": _AWAY["ml_probability"]}
    data = {
        **_BASE, "candidates": [_AWAY, tied_probability],
        "recommendation_pick": {"market": "result_3way", "selection": "away"},
    }
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "away"}
    assert rec["overall"] == "direct_bet"
