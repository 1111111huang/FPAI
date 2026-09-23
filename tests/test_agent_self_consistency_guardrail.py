"""A91 (2026-08-31 design): the LLM self-reports a composite_score per
candidate meant to balance value_edge against ml_probability. Code can't
verify the *number* itself -- there's no fixed formula -- but it can catch
self-contradiction: did the model's own stated pick actually have the best
score among its own listed candidates?"""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_HOME = {
    "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
    "current_odds": 2.1, "min_odds": 1.8, "ml_probability": 0.55, "implied_probability": 0.48,
    "value_edge": 0.07, "composite_score": 0.4, "reason": "Modest edge, high uncertainty.",
}
_BTTS = {
    # selection="yes", not "no": US#210 unconditionally suppresses btts/no
    # direct_bet regardless of composite_score, which would trip this
    # fixture's own guardrail before the self-consistency logic under test
    # ever ran.
    "market": "btts", "selection": "yes", "recommendation_type": "direct_bet",
    "current_odds": 2.2, "min_odds": 1.8, "ml_probability": 0.6, "implied_probability": 0.45,
    "value_edge": 0.15, "composite_score": 0.8, "reason": "Large edge with strong hit probability.",
}

_BASE = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "explanation": "Value found.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_pick_with_the_top_composite_score_is_unaffected():
    data = {**_BASE, "candidates": [_HOME, _BTTS], "recommendation_pick": {"market": "btts", "selection": "yes"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "btts", "selection": "yes"}
    assert rec["overall"] == "direct_bet"


def test_pick_with_a_lower_composite_score_than_a_rejected_candidate_is_downgraded():
    data = {**_BASE, "candidates": [_HOME, _BTTS], "recommendation_pick": {"market": "result_3way", "selection": "home"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] is None
    assert rec["overall"] == "no_bet"
    assert any("composite_score" in note for note in rec["limitations"])


def test_a_no_bet_rejected_candidates_own_higher_score_does_not_count():
    """A no_bet candidate was already disqualified by an earlier guardrail --
    it was never a real alternative, so out-scoring it isn't a
    contradiction."""
    disqualified = {**_BTTS, "recommendation_type": "no_bet", "composite_score": 0.9}
    data = {**_BASE, "candidates": [_HOME, disqualified], "recommendation_pick": {"market": "result_3way", "selection": "home"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "home"}


def test_an_exact_tied_composite_score_is_not_a_contradiction():
    """Code-quality review gap (2026-08-31, Task 5): the design spec and the
    implementation both use strict greater-than -- only a candidate that
    self-reports a *higher* score than the pick counts as contradicting it,
    per docs/superpowers/specs/2026-08-31-single-market-recommendation-design.md's
    own "if some other candidate self-reports a higher composite_score"
    wording. An exact tie is not a contradiction (there's no basis to say
    the LLM's own numbers favor the *other* candidate over the pick when
    they're equal) -- this pins that boundary down explicitly, since the
    original 3-test suite never exercised it. Deliberately not just _BTTS
    with composite_score overridden: _BTTS's own ml_probability/value_edge
    both strictly exceed _HOME's, which would trip the separate Pareto-
    dominance guardrail (_downgrade_pick_dominated_by_another_candidate)
    regardless of composite_score -- this fixture isolates the
    composite_score-tie behavior alone by giving `tied` a higher
    ml_probability but a lower value_edge than _HOME (neither dominates)."""
    tied = {**_BTTS, "composite_score": _HOME["composite_score"], "ml_probability": 0.6, "value_edge": 0.05}
    data = {**_BASE, "candidates": [_HOME, tied], "recommendation_pick": {"market": "result_3way", "selection": "home"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "home"}
    assert rec["overall"] == "direct_bet"
