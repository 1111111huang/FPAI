"""Regression tests for W13's extracted shared market-resolution utility --
must behave identically to the logic that used to live only in
src/agent/backtest.py, since both backtest scoring and the app's live
settlement job now depend on it."""

from __future__ import annotations

from src.agent.market_resolution import (
    RESOLVABLE_MARKETS,
    build_actual_outcome,
    market_correct,
    resolve_recommendation_pick,
)


def test_resolvable_markets_excludes_per_side_corners_but_includes_total():
    """A101: home_corners/away_corners still have no numeric line field (only
    current_odds/min_odds) and stay unresolvable. total_corners is different
    -- it has a real, fixed line (9.5, matching total_goals' own fixed 2.5
    convention) via the OddsPapi odds pull (A100), so its correctness IS
    programmatically resolvable the same way total_goals already is."""
    assert "home_corners" not in RESOLVABLE_MARKETS
    assert "away_corners" not in RESOLVABLE_MARKETS
    assert RESOLVABLE_MARKETS == {
        "result_3way", "btts", "total_goals", "total_corners", "home_goals", "away_goals",
    }


def test_build_actual_outcome_home_win():
    actual = build_actual_outcome(2, 1)
    assert actual["result"] == "home"
    assert actual["btts"] == "yes"
    assert actual["total_goals"] == 3
    assert actual["total_goals_side"] == "over_2.5"


def test_build_actual_outcome_draw_no_btts():
    actual = build_actual_outcome(0, 0)
    assert actual["result"] == "draw"
    assert actual["btts"] == "no"
    assert actual["total_goals_side"] == "under_2.5"


def test_build_actual_outcome_away_win():
    actual = build_actual_outcome(0, 2)
    assert actual["result"] == "away"


def test_market_correct_result_3way():
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "result_3way", "selection": "home"}, actual) is True
    assert market_correct({"market": "result_3way", "selection": "away"}, actual) is False


def test_market_correct_btts():
    actual = build_actual_outcome(1, 1)
    assert market_correct({"market": "btts", "selection": "yes"}, actual) is True


def test_market_correct_total_goals():
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "total_goals", "selection": "over_2.5"}, actual) is True


def test_market_correct_returns_none_for_corners():
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "home_corners", "selection": "over_4.5"}, actual) is None
    assert market_correct({"market": "away_corners", "selection": "under_4.5"}, actual) is None


def test_build_actual_outcome_includes_total_corners_when_both_sides_given():
    actual = build_actual_outcome(2, 1, home_corners=6, away_corners=5)
    assert actual["total_corners"] == 11
    assert actual["total_corners_side"] == "over_9.5"


def test_build_actual_outcome_under_9_5_corners():
    actual = build_actual_outcome(1, 0, home_corners=4, away_corners=3)
    assert actual["total_corners_side"] == "under_9.5"


def test_build_actual_outcome_omits_total_corners_when_not_given():
    """Every existing caller (app/backend/settlement.py) calls this with just
    home_goals/away_goals -- must stay completely valid, degrading the same
    way missing total_goals_odds already does elsewhere (key simply absent)."""
    actual = build_actual_outcome(2, 1)
    assert "total_corners" not in actual
    assert "total_corners_side" not in actual


def test_market_correct_total_corners():
    actual = build_actual_outcome(2, 1, home_corners=6, away_corners=5)
    assert market_correct({"market": "total_corners", "selection": "over_9.5"}, actual) is True
    assert market_correct({"market": "total_corners", "selection": "under_9.5"}, actual) is False


def test_market_correct_returns_none_for_total_corners_when_actual_lacks_corner_counts():
    """Live settlement (app/backend/settlement.py) doesn't supply corner
    counts yet -- must degrade to 'unknown', never a false loss/win."""
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "total_corners", "selection": "over_9.5"}, actual) is None


def test_resolve_recommendation_pick_finds_the_matching_candidate():
    candidates = [
        {"market": "result_3way", "selection": "home", "value_edge": 0.02},
        {"market": "btts", "selection": "no", "value_edge": 0.08},
    ]
    pick = {"market": "btts", "selection": "no"}
    assert resolve_recommendation_pick(candidates, pick) == candidates[1]


def test_resolve_recommendation_pick_returns_none_for_null_pick():
    candidates = [{"market": "result_3way", "selection": "home", "value_edge": 0.02}]
    assert resolve_recommendation_pick(candidates, None) is None


def test_resolve_recommendation_pick_returns_none_when_pick_not_in_candidates():
    """The LLM pointed recommendation_pick at a market/selection it never
    actually listed in candidates -- a real, new failure mode this schema
    makes detectable for the first time."""
    candidates = [{"market": "result_3way", "selection": "home", "value_edge": 0.02}]
    pick = {"market": "btts", "selection": "yes"}
    assert resolve_recommendation_pick(candidates, pick) is None


def test_resolvable_markets_includes_home_and_away_goals():
    assert RESOLVABLE_MARKETS == {
        "result_3way", "btts", "total_goals", "total_corners", "home_goals", "away_goals",
    }


def test_build_actual_outcome_includes_home_and_away_goals_side_unconditionally():
    """Unlike total_corners (optional -- not every settlement source has
    corner counts), home_goals/away_goals_side is unconditional: home_goals
    and away_goals are this function's own required positional params, so
    there's no missing-data case to guard against."""
    actual = build_actual_outcome(2, 1)
    assert actual["home_goals_side"] == "over_1.5"
    assert actual["away_goals_side"] == "under_1.5"


def test_build_actual_outcome_home_goals_side_under_on_exactly_one():
    actual = build_actual_outcome(1, 0)
    assert actual["home_goals_side"] == "under_1.5"
    assert actual["away_goals_side"] == "under_1.5"


def test_market_correct_home_goals():
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "home_goals", "selection": "over_1.5"}, actual) is True
    assert market_correct({"market": "home_goals", "selection": "under_1.5"}, actual) is False


def test_market_correct_away_goals():
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "away_goals", "selection": "under_1.5"}, actual) is True
    assert market_correct({"market": "away_goals", "selection": "over_1.5"}, actual) is False


def test_resolve_recommendation_pick_tolerates_malformed_raw_data_instead_of_crashing():
    """W193 code-quality followup: this function is now called against
    RAW, not-yet-validated data too (app/backend/recommendations.py's
    validate_and_degrade(), against an LLM response or a cache row before
    any Pydantic check has run) -- unlike its original two callers, which
    only ever pass already-validated dicts. A missing key, a wrong type, or
    a non-dict entry must all degrade to 'no pick found', never raise."""
    # pick missing "selection" entirely
    assert resolve_recommendation_pick(
        [{"market": "result_3way", "selection": "home"}], {"market": "result_3way"}
    ) is None
    # a candidate missing "market" entirely -- skipped, not a crash
    assert resolve_recommendation_pick(
        [{"selection": "home"}], {"market": "result_3way", "selection": "home"}
    ) is None
    # pick is not a dict at all
    assert resolve_recommendation_pick(
        [{"market": "result_3way", "selection": "home"}], "not-a-dict"
    ) is None
    # a candidate is not a dict at all -- skipped, the real one after it still matches
    real_candidate = {"market": "btts", "selection": "no"}
    assert resolve_recommendation_pick(
        ["not-a-dict", real_candidate], {"market": "btts", "selection": "no"}
    ) == real_candidate
