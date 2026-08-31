"""Regression tests for A28: extract_recommendation must validate field types/
enums beyond key presence, and close BUG-013's root cause (a market marked
direct_bet with a null current_odds) at extraction time rather than passing
it through to crash downstream. Covers the three specific gaps documented in
agent_techspec.md Section 17 (value_edge as a string, confidence as an empty
string, an arbitrary recommendation_type string) plus BUG-013's null-odds
case.

A88 (2026-08-31): reworked for the single-recommendation schema -- `markets`
is now `candidates` (richer: adds composite_score/reason) plus a
`recommendation_pick` pointer naming which candidate is the actual pick."""

from __future__ import annotations

import json

import pytest

from src.agent.schema import RecommendationParseError, extract_recommendation

_VALID_CANDIDATE = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 2.1,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
    "composite_score": 0.62,
    "reason": "Clears the edge floor with a well-supported home win probability.",
}

_VALID_PICK = {"market": "result_3way", "selection": "home"}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [_VALID_CANDIDATE],
    "recommendation_pick": _VALID_PICK,
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_fully_valid_output_with_a_real_candidate_still_parses_unchanged():
    """Regression: a valid single-recommendation output (candidates +
    recommendation_pick) must parse cleanly with no downgrades."""
    rec = extract_recommendation(_wrap_json(_VALID))
    assert rec["overall"] == "direct_bet"
    assert rec["candidates"][0]["recommendation_type"] == "direct_bet"
    assert rec["candidates"][0]["current_odds"] == 2.1
    assert rec["candidates"][0]["composite_score"] == 0.62
    assert rec["recommendation_pick"] == _VALID_PICK
    assert rec["limitations"] == []


def test_missing_composite_score_raises():
    """composite_score/reason are new, required MarketCandidateModel fields
    -- a candidate missing either fails validation the same way a missing
    ml_probability already does, not silently defaulted (unlike min_odds,
    BUG-032 -- composite_score is new and load-bearing for A91's
    self-consistency guardrail, not vestigial)."""
    bad_candidate = {k: v for k, v in _VALID_CANDIDATE.items() if k != "composite_score"}
    bad = {**_VALID, "candidates": [bad_candidate]}
    with pytest.raises(RecommendationParseError, match="composite_score"):
        extract_recommendation(_wrap_json(bad))


def test_recommendation_pick_missing_selection_raises():
    bad_pick = {"market": "result_3way"}
    bad = {**_VALID, "recommendation_pick": bad_pick}
    with pytest.raises(RecommendationParseError, match="selection"):
        extract_recommendation(_wrap_json(bad))


def test_recommendation_pick_can_be_null():
    """A no_bet call with nothing actionable: recommendation_pick is null,
    candidates can still list what was considered (or be empty)."""
    data = {**_VALID, "overall": "no_bet", "recommendation_pick": None,
             "candidates": [{**_VALID_CANDIDATE, "recommendation_type": "no_bet"}]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] is None


def test_missing_min_odds_no_longer_sinks_the_whole_recommendation():
    """BUG-032: confirmed live -- real DeepSeek output regularly omits
    min_odds on some markets within an otherwise-valid recommendation. Before
    this default, one missing field failed validation for the *entire*
    candidate, discarding every other market's real data along with it.
    min_odds is also effectively vestigial now that A52's target_odds is the
    verified, code-computed field the UI actually shows (W84/W87)."""
    candidate_missing_min_odds = {k: v for k, v in _VALID_CANDIDATE.items() if k != "min_odds"}
    data = {**_VALID, "candidates": [candidate_missing_min_odds]}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["candidates"][0]["recommendation_type"] == "direct_bet"
    assert rec["candidates"][0]["current_odds"] == 2.1
    assert "min_odds" not in rec["candidates"][0]


def test_value_edge_as_string_raises():
    bad_candidate = {**_VALID_CANDIDATE, "value_edge": "high"}
    bad = {**_VALID, "candidates": [bad_candidate]}
    with pytest.raises(RecommendationParseError, match="value_edge"):
        extract_recommendation(_wrap_json(bad))


def test_confidence_empty_string_raises():
    bad = {**_VALID, "confidence": ""}
    with pytest.raises(RecommendationParseError, match="confidence"):
        extract_recommendation(_wrap_json(bad))


def test_arbitrary_recommendation_type_raises():
    bad_candidate = {**_VALID_CANDIDATE, "recommendation_type": "maybe_bet"}
    bad = {**_VALID, "candidates": [bad_candidate]}
    with pytest.raises(RecommendationParseError, match="recommendation_type"):
        extract_recommendation(_wrap_json(bad))


def test_direct_bet_with_null_odds_downgraded_to_no_bet():
    """BUG-013: a market marked direct_bet with current_odds=null must be
    downgraded to no_bet with an explanatory limitations note, not passed
    through as-is and not raised as a parse error."""
    bad_candidate = {**_VALID_CANDIDATE, "current_odds": None}
    bad = {**_VALID, "candidates": [bad_candidate]}

    rec = extract_recommendation(_wrap_json(bad))

    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert rec["candidates"][0]["current_odds"] is None
    assert any("direct_bet" in note and "no_bet" in note for note in rec["limitations"])


def test_conditional_market_with_null_odds_is_not_touched():
    """The downgrade rule is specific to direct_bet -- a conditional market
    with null current_odds (a legitimate state) must be left alone. A54:
    market/selection overridden to an eligible pair (btts/yes) -- result_3way
    (the shared _VALID_CANDIDATE default) is no longer eligible to stay
    conditional at all, tested separately in
    test_agent_conditional_market_eligibility.py."""
    candidate = {**_VALID_CANDIDATE, "market": "btts", "selection": "yes", "recommendation_type": "conditional", "current_odds": None}
    data = {**_VALID, "overall": "conditional", "candidates": [candidate],
            "recommendation_pick": {"market": "btts", "selection": "yes"}}

    rec = extract_recommendation(_wrap_json(data))

    assert rec["candidates"][0]["recommendation_type"] == "conditional"
    assert rec["limitations"] == []


def test_overall_as_unhashable_dict_raises_parse_error_not_type_error():
    """BUG-020: a real qwen2.5-coder:7b response returned `overall` as a dict
    (e.g. {"liverpool_form": ..., "bournemouth_form": ...}) instead of a
    string. `data["overall"] not in _VALID_OVERALL` requires hashing the LHS
    for the set membership test -- a dict/list there raised an unhandled
    `TypeError: unhashable type: 'dict'` that crashed the whole agent-recommend
    call instead of being treated as an invalid `overall` value and falling
    through to RecommendationParseError like any other malformed candidate."""
    bad = {**_VALID, "overall": {"liverpool_form": "mixed", "bournemouth_form": "worse"}}
    with pytest.raises(RecommendationParseError):
        extract_recommendation(_wrap_json(bad))


def test_overall_as_unhashable_list_raises_parse_error_not_type_error():
    bad = {**_VALID, "overall": ["direct_bet", "no_bet"]}
    with pytest.raises(RecommendationParseError):
        extract_recommendation(_wrap_json(bad))


def test_match_mismatch_raises_when_requested_teams_given():
    """BUG-023/024: the agent hallucinated a "Manchester City vs Liverpool"
    analysis for a real Brentford vs Wolverhampton request (confirmed live in
    a sandbox precompute batch). When the real requested teams are passed
    in, a candidate naming a different pair of clubs must be rejected, not
    silently returned."""
    bad = {**_VALID, "match": {"home": "Manchester City", "away": "Liverpool"}}
    with pytest.raises(RecommendationParseError, match="match mismatch"):
        extract_recommendation(_wrap_json(bad), home_team="Brentford", away_team="Wolverhampton")


def test_match_mismatch_check_is_skipped_when_teams_not_supplied():
    """Backward compatible: every pre-existing caller/test omits home_team/
    away_team, so the check must not run (and must not raise) by default."""
    bad = {**_VALID, "match": {"home": "Manchester City", "away": "Liverpool"}}
    rec = extract_recommendation(_wrap_json(bad))
    assert rec["match"]["home"] == "Manchester City"


def test_non_canonical_market_name_raises():
    """The agent has been observed calling result_3way "1X2" for one real
    fixture, and inventing markets entirely outside this schema for others
    ("Asian Handicap", a team name used as the market itself) -- confirmed
    live in the sandbox cache. config/prompts/agent_v1.txt already specifies
    the fixed vocabulary; nothing enforced it until now."""
    bad_candidate = {**_VALID_CANDIDATE, "market": "1X2"}
    bad = {**_VALID, "candidates": [bad_candidate]}
    with pytest.raises(RecommendationParseError, match="market"):
        extract_recommendation(_wrap_json(bad))


def test_non_canonical_selection_raises():
    """Observed live: a result_3way selection reported as a team name
    ("Vasteras SK") or "home_win" instead of the specified home/draw/away."""
    bad_candidate = {**_VALID_CANDIDATE, "selection": "home_win"}
    bad = {**_VALID, "candidates": [bad_candidate]}
    with pytest.raises(RecommendationParseError, match="selection"):
        extract_recommendation(_wrap_json(bad))


def test_home_away_swap_alone_is_not_a_match_mismatch():
    """A plain home/away swap is not the hallucination bug -- only a
    genuinely different pair of clubs should be rejected."""
    data = {**_VALID, "match": {"home": "Chelsea", "away": "Arsenal"}}
    rec = extract_recommendation(_wrap_json(data), home_team="Arsenal", away_team="Chelsea")
    assert rec["overall"] == "direct_bet"


def test_overall_syncs_to_the_resolved_picks_actual_type():
    """A90 (2026-08-31 design): replaces A65's _reconcile_overall_with_markets.
    The LLM claims overall='direct_bet' but the picked candidate's own price
    is outside the configured odds bounds -- Task 3's already-adapted
    _downgrade_direct_bet_outside_odds_bounds downgrades the candidate to
    'conditional'; overall must follow it, not stay stuck at the LLM's
    original, now-stale self-report.

    Market/selection overridden to an eligible pair (total_goals/over_2.5):
    _VALID_CANDIDATE's default (result_3way/home) is never eligible to
    *stay* 'conditional' -- A54's _restrict_conditional_to_eligible_markets
    (Task 3, unmodified here) always downgrades it straight through to
    'no_bet', which would defeat the point of this test (verifying overall
    follows a pick that lands on 'conditional', not 'no_bet')."""
    candidate = {**_VALID_CANDIDATE, "market": "total_goals", "selection": "over_2.5", "current_odds": 15.0}
    pick = {"market": "total_goals", "selection": "over_2.5"}
    data = {**_VALID, "candidates": [candidate], "recommendation_pick": pick}
    rec = extract_recommendation(_wrap_json(data), min_odds_threshold=1.2, max_odds_threshold=11.0)
    assert rec["overall"] == "conditional"
    assert rec["recommendation_pick"] == pick  # still a real pick, just re-typed


def test_pick_downgraded_to_no_bet_is_nulled_and_overall_follows():
    candidate = {**_VALID_CANDIDATE, "value_edge": -0.02}
    data = {**_VALID, "candidates": [candidate], "recommendation_pick": _VALID_PICK}
    rec = extract_recommendation(_wrap_json(data), min_value_edge=0.05)
    assert rec["overall"] == "no_bet"
    assert rec["recommendation_pick"] is None


def test_pick_naming_a_candidate_absent_from_candidates_is_treated_as_no_pick():
    dangling_pick = {"market": "btts", "selection": "yes"}
    data = {**_VALID, "recommendation_pick": dangling_pick}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] is None
    assert any("not present in candidates" in note for note in rec["limitations"])


def test_never_upgrades_insufficient_data():
    """Downgrade-only, same direction A65 already established -- a
    legitimately empty candidates list (the graph's own no-forecast
    short-circuit) with overall already 'insufficient_data' must stay that
    way, not get bumped to 'no_bet' just because there's no resolvable pick."""
    data = {**_VALID, "overall": "insufficient_data", "candidates": [], "recommendation_pick": None}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["overall"] == "insufficient_data"
