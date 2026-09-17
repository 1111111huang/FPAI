"""Regression tests for A112: a favorite (ml_probability > 0.5) priced at
live_wait_min_odds or better gets promoted to 'conditional' from 'no_bet' --
the user's own live-wait strategy (e.g. btts priced ~-150, wait ~20min for a
scoreless start to drift the price out toward +100 with the true
probability barely changed).

Two-part design, direct user clarification (2026-09-17) after a live card
(Espanyol v Elche btts) exposed a flaw in the first version: overriding an
ALREADY-qualifying direct_bet to conditional (throwing away a certain edge
for an uncertain one) is wrong -- the live-wait trade-off is a CROSS-
candidate decision, not a same-candidate one. (1)
_promote_favorite_to_conditional_for_live_wait fills the 'no_bet' gap only
(a candidate whose current price/edge doesn't clear the bar, but whose
target price would). (2) _prefer_higher_probability_conditional_pick
separately compares an existing direct_bet pick against a DIFFERENT,
higher-probability conditional candidate and switches the recommendation
if the conditional is a meaningfully more likely winner. See
documents/agent_user_stories.md A112 and
src/agent/schema.py's docstrings on both functions."""

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


# ---------------------------------------------------------------------------
# _promote_favorite_to_conditional_for_live_wait -- fills the no_bet gap only
# ---------------------------------------------------------------------------


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


def test_direct_bet_candidates_are_never_touched_by_this_rule():
    """Direct user correction (2026-09-17), found via a real Espanyol v
    Elche card: an ALREADY-qualifying direct_bet (real edge clearing
    min_value_edge at the current price) must never be relabeled by this
    rule alone -- that threw away a certain edge for an uncertain one, with
    no comparison of which was actually better, and left the LLM's own
    explanation text (written before the override) contradicting the
    badge. value_edge=0.1 here genuinely clears the floor, unlike this
    file's default fixture -- a real direct_bet, not one masquerading as
    one."""
    candidate = {**_VALID_CANDIDATE, "recommendation_type": "direct_bet", "value_edge": 0.1}
    data = {**_VALID, "overall": "direct_bet", "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


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


def test_not_promoted_when_already_priced_past_the_target():
    """A no_bet candidate already priced longer than the target has nothing
    to wait for -- structurally can't clear edge at the target either, by
    the same monotonic-implied-probability reasoning; this just confirms
    it's correctly skipped rather than mishandled."""
    candidate = {**_VALID_CANDIDATE, "current_odds": 2.5}
    data = {**_VALID, "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS)

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"


def test_ineligible_market_never_promoted():
    """result_3way has no one-directional drift property (same A54
    restriction this rule reuses) -- never a coherent 'wait' candidate."""
    candidate = {**_VALID_CANDIDATE, "market": "result_3way", "selection": "home"}
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


def test_not_promoted_when_edge_at_target_price_does_not_clear_min_value_edge():
    """Direct user refinement (2026-09-17): a bare ml_probability > 0.5
    isn't enough -- it must clear min_value_edge AT the live-wait target
    price (default 2.0 / +100). 0.52 - implied(2.0)=0.5 = 0.02, below the
    default 0.05 min_value_edge -- a coin-flip-ish favorite not worth
    recommending a wait on."""
    candidate = {**_VALID_CANDIDATE, "ml_probability": 0.52}
    data = {**_VALID, "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS, min_value_edge=0.05)

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_promoted_when_edge_at_target_price_clears_min_value_edge():
    """Mirror of the above at the threshold: 0.56 - 0.5 = 0.06 >= 0.05."""
    candidate = {**_VALID_CANDIDATE, "ml_probability": 0.56}
    data = {**_VALID, "candidates": [candidate]}

    rec = extract_recommendation(_wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS, min_value_edge=0.05)

    assert rec["candidates"][0]["recommendation_type"] == "conditional"


def test_custom_target_odds_is_respected_not_hardcoded():
    """A longer custom target (e.g. +150/2.5) needs more edge to clear at
    that price -- ml_probability 0.6's edge at 2.5 (implied 0.4) is 0.20,
    comfortably above 0.05, so this should still promote with a
    non-default target."""
    data = {**_VALID}

    rec = extract_recommendation(
        _wrap_json(data), live_wait_min_odds=_LIVE_WAIT_MIN_ODDS, live_wait_target_odds=2.5,
    )

    assert rec["candidates"][0]["recommendation_type"] == "conditional"


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


# ---------------------------------------------------------------------------
# _prefer_higher_probability_conditional_pick -- the cross-candidate switch
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


def test_switches_pick_to_higher_probability_conditional():
    """The exact scenario that prompted this redesign: a draw direct_bet
    (10% edge, 30% probability) vs. a btts conditional (4% edge now, would
    clear 5% at target, 55% probability) -- the far more likely winner
    should become the recommendation, even though it isn't bettable at the
    current price and has smaller edge."""
    data = {
        **_VALID, "overall": "direct_bet",
        "candidates": [_DRAW, _BTTS_CONDITIONAL],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["recommendation_pick"] == {"market": "btts", "selection": "yes"}
    assert rec["overall"] == "conditional"
    assert any("draw" in note and "btts" in note for note in rec["limitations"])


def test_does_not_switch_when_conditional_probability_is_not_higher():
    lower_prob_conditional = {**_BTTS_CONDITIONAL, "ml_probability": 0.25}
    data = {
        **_VALID, "overall": "direct_bet",
        "candidates": [_DRAW, lower_prob_conditional],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "draw"}
    assert rec["overall"] == "direct_bet"


def test_does_not_switch_when_no_eligible_conditional_exists():
    """The original Espanyol v Elche scenario: a single, uniquely-qualifying
    direct_bet with no other conditional candidate on the match -- nothing
    to switch to, stays direct_bet."""
    data = {
        **_VALID, "overall": "direct_bet",
        "candidates": [_DRAW],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "draw"}
    assert rec["overall"] == "direct_bet"


def test_does_not_switch_when_conditional_market_is_not_eligible():
    """A higher-probability 'conditional' in a non-eligible market (e.g.
    result_3way, which A54 already restricts) doesn't count -- waiting
    isn't a coherent strategy there regardless of probability."""
    ineligible = {**_BTTS_CONDITIONAL, "market": "result_3way", "selection": "home", "ml_probability": 0.8}
    data = {
        **_VALID, "overall": "direct_bet",
        "candidates": [_DRAW, ineligible],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "draw"}


def test_switching_pick_rebuilds_explanation_around_the_new_candidate():
    """Found live (2026-09-17): explanation is the LLM's own prose written
    to justify the ORIGINAL pick (the draw) -- left untouched, it would
    keep reading like a case for the draw while the badge/market shown is
    now btts. Must be rebuilt from the new pick's own reason, not left
    stale."""
    data = {
        **_VALID, "overall": "direct_bet",
        "candidates": [_DRAW, _BTTS_CONDITIONAL],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
        "explanation": ["The draw is well-supported by both teams' recent head-to-head record."],
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["recommendation_pick"] == {"market": "btts", "selection": "yes"}
    assert not any("draw is well-supported" in line for line in rec["explanation"])
    assert any("Qualifies once the price improves" in line for line in rec["explanation"])
    assert any("Switched" in line and "draw" in line and "btts" in line for line in rec["explanation"])


def test_explanation_untouched_when_pick_is_not_switched():
    data = {
        **_VALID, "overall": "direct_bet",
        "candidates": [_DRAW],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
        "explanation": ["The draw is well-supported by both teams' recent head-to-head record."],
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["explanation"] == ["The draw is well-supported by both teams' recent head-to-head record."]


def test_does_not_switch_when_pick_is_not_a_direct_bet():
    """Scoped to a currently-direct_bet pick only -- a no_bet/conditional
    pick has nothing to compare a 'certain edge in hand' against."""
    no_bet_pick = {**_DRAW, "recommendation_type": "no_bet", "value_edge": -0.1}
    data = {
        **_VALID, "overall": "no_bet",
        "candidates": [no_bet_pick, _BTTS_CONDITIONAL],
        "recommendation_pick": {"market": "result_3way", "selection": "draw"},
    }

    rec = extract_recommendation(_wrap_json(data))

    assert rec["overall"] != "direct_bet"
