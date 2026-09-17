"""Regression tests for A113: team_evidence/the_read/no_bet_read, additive
structured-reasoning fields for the redesigned "Why This Pick" UI. Lenient
normalization (never raises, never fails parsing) -- a malformed or absent
value degrades to None, the frontend's own signal to fall back to the flat
explanation/limitations rendering. See src/agent/schema.py's
normalize_structured_reasoning() and documents/agent_user_stories.md A113."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_CANDIDATE = {
    "market": "btts",
    "selection": "yes",
    "recommendation_type": "direct_bet",
    "current_odds": 1.73,
    "min_odds": 1.71,
    "ml_probability": 0.672,
    "implied_probability": 0.578,
    "value_edge": 0.094,
    "composite_score": 0.6,
    "reason": "Both sides have leaky defenses and sharp enough attacks.",
}

_VALID = {
    "match": {"home": "Espanyol", "away": "Elche", "date": "2026-06-15", "league": "SP1"},
    "overall": "direct_bet",
    "candidates": [_CANDIDATE],
    "recommendation_pick": {"market": "btts", "selection": "yes"},
    "explanation": ["BTTS yes reads as the strongest value on this fixture."],
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_valid_team_evidence_and_the_read_pass_through():
    data = {
        **_VALID,
        "team_evidence": {"home": "Espanyol have not kept a clean sheet in 5 matches.", "away": "Elche have conceded 16 in 6."},
        "the_read": "Neither side looks capable of keeping a clean sheet right now.",
        "no_bet_read": None,
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["team_evidence"] == {
        "home": "Espanyol have not kept a clean sheet in 5 matches.",
        "away": "Elche have conceded 16 in 6.",
    }
    assert rec["the_read"] == "Neither side looks capable of keeping a clean sheet right now."


def test_fields_absent_entirely_degrade_to_none_not_a_parse_failure():
    """Backward compat: a non-compliant model (or a pre-A113 caller/test)
    that never produces these fields must still parse successfully --
    they're additive, not required."""
    data = {**_VALID}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["team_evidence"] is None
    assert rec["the_read"] is None
    assert rec["no_bet_read"] is None


def test_team_evidence_missing_one_side_degrades_to_none():
    """A partial object is treated the same as absent -- the frontend
    renders both sides together as one block, so home-only is unusable."""
    data = {**_VALID, "team_evidence": {"home": "Some fact."}}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["team_evidence"] is None


def test_team_evidence_wrong_type_degrades_to_none_not_a_parse_failure():
    data = {**_VALID, "team_evidence": "not a dict"}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["team_evidence"] is None


def test_the_read_blank_string_degrades_to_none():
    data = {**_VALID, "the_read": "   "}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["the_read"] is None


def test_no_bet_read_passes_through_for_no_bet_overall():
    no_bet_candidate = {**_CANDIDATE, "recommendation_type": "no_bet", "value_edge": 0.01}
    data = {
        **_VALID, "overall": "no_bet", "candidates": [no_bet_candidate], "recommendation_pick": None,
        "no_bet_read": "Neither team profiles as a bet worth putting money on tonight.",
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["no_bet_read"] == "Neither team profiles as a bet worth putting money on tonight."


# ---------------------------------------------------------------------------
# Pick-switch staleness: team_evidence/the_read must be cleared, same class
# of bug the explanation-rebuild fix already covers.
# ---------------------------------------------------------------------------

_DRAW = {
    "market": "result_3way", "selection": "draw", "recommendation_type": "direct_bet",
    "current_odds": 3.3, "min_odds": 1.71, "ml_probability": 0.3, "implied_probability": 0.2,
    "value_edge": 0.1, "composite_score": 0.6, "reason": "Draw edge.",
}
_BTTS_CONDITIONAL = {
    "market": "btts", "selection": "yes", "recommendation_type": "conditional",
    "current_odds": 1.8, "min_odds": 1.71, "ml_probability": 0.55, "implied_probability": 0.51,
    "value_edge": 0.04, "composite_score": 0.5, "reason": "Qualifies once the price improves.",
}


def test_team_evidence_and_the_read_cleared_when_pick_switches():
    data = {
        **_VALID, "overall": "direct_bet",
        "candidates": [_DRAW, _BTTS_CONDITIONAL],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
        "team_evidence": {"home": "Draw-specific home fact.", "away": "Draw-specific away fact."},
        "the_read": "The draw looks live given how evenly matched both sides are.",
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["recommendation_pick"] == {"market": "btts", "selection": "yes"}
    assert rec["team_evidence"] is None
    assert rec["the_read"] is None
    # explanation still gets rebuilt around the new pick, same as before
    assert any("Qualifies once the price improves" in line for line in rec["explanation"])
