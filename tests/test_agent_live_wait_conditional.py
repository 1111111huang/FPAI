"""Regression tests for A112: a favorite (ml_probability > 0.5) priced at
live_wait_min_odds or better gets promoted/kept as 'conditional' -- the
user's own live-wait strategy (e.g. btts priced ~-150, wait ~20min for a
scoreless start to drift the price out toward +100 with the true
probability barely changed). Unlike every other downgrade pass in
schema.py, this one PROMOTES: it can turn 'no_bet' into 'conditional' and
can override an existing 'direct_bet' too. See
documents/agent_user_stories.md A112 and
src/agent/schema.py's _promote_favorite_to_conditional_for_live_wait."""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_VALID_CANDIDATE = {
    "market": "btts",
    "selection": "yes",
    "recommendation_type": "no_bet",
    "current_odds": 1.75,
    "min_odds": 1.5,
    "ml_probability": 0.6,
    "implied_probability": 0.57,
    "value_edge": 0.03,
    "composite_score": 0.5,
    "reason": "Short price now, but a likely scoreline either way.",
}

_VALID_PICK = {"market": "btts", "selection": "yes"}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "no_bet",
    "candidates": [_VALID_CANDIDATE],
    "recommendation_pick": _VALID_PICK,
    "explanation": "Live-wait candidate.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}

_LIVE_WAIT_MIN_ODDS = 1.6667  # -150 American


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_disabled_by_default_no_bet_untouched():
    """live_wait_min_odds=None (the default) is a no-op -- every existing
    caller/test that doesn't pass it keeps today's exact behavior."""
    data = {**_VALID}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_promotes_no_bet_to_conditional_when_favorite_and_price_ok():
    data = {**_VALID}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "conditional"
    assert any("no_bet" in note and "conditional" in note for note in rec["limitations"])


def test_overrides_direct_bet_to_conditional_per_user_preference():
    """The user's own stated preference: given both a direct_bet and this
    condition, prefer waiting for the better number on a likely winner."""
    candidate = {**_VALID_CANDIDATE, "recommendation_type": "direct_bet"}
    data = {**_VALID, "overall": "direct_bet", "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "conditional"


def test_not_promoted_when_probability_not_a_favorite():
    candidate = {**_VALID_CANDIDATE, "ml_probability": 0.5}
    data = {**_VALID, "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_not_promoted_when_price_shorter_than_floor():
    """-200 American (decimal 1.5) is shorter than the -150 floor -- too
    short a price even for this strategy."""
    candidate = {**_VALID_CANDIDATE, "current_odds": 1.5}
    data = {**_VALID, "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"


def test_ineligible_market_never_promoted():
    """result_3way has no one-directional drift property (same A54
    restriction this rule reuses) -- never a coherent 'wait' candidate."""
    candidate = {
        **_VALID_CANDIDATE, "market": "result_3way", "selection": "home",
    }
    data = {**_VALID, "candidates": [candidate], "recommendation_pick": {"market": "result_3way", "selection": "home"}}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_already_conditional_is_left_alone_no_duplicate_note():
    candidate = {**_VALID_CANDIDATE, "recommendation_type": "conditional"}
    data = {**_VALID, "overall": "conditional", "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_promoted_candidate_gets_a_real_target_odds():
    """Must run before A52's target_odds computation -- a newly-conditional
    candidate is still a real conditional pick, not a second-class one."""
    data = {**_VALID}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS, min_value_edge=0.05)

    assert rec["candidates"][0]["target_odds"] is not None


def test_survives_the_stricter_general_conditional_floor():
    """The core conflict this rule resolves: the general conditional floor
    (min_conditional_odds_threshold, prod default 1.71 / -140) is STRICTER
    than this strategy's own explicit floor (1.6667 / -150) -- a price of
    1.68 (-147) sits between the two. Without running after A66's floor
    check, this promotion would be immediately undone."""
    candidate = {**_VALID_CANDIDATE, "current_odds": 1.68}
    data = {**_VALID, "candidates": [candidate]}

    rec = extract_recommendation(
        _wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS, min_conditional_odds_threshold=1.71,
    )

    assert rec["candidates"][0]["recommendation_type"] == "conditional"
