"""Regression tests for W13's extracted shared market-resolution utility --
must behave identically to the logic that used to live only in
src/agent/backtest.py, since both backtest scoring and the app's live
settlement job now depend on it."""

from __future__ import annotations

from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct, pick_recommended_market


def test_resolvable_markets_excludes_corners():
    assert "home_corners" not in RESOLVABLE_MARKETS
    assert "away_corners" not in RESOLVABLE_MARKETS
    assert RESOLVABLE_MARKETS == {"result_3way", "btts", "total_goals"}


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


def test_pick_recommended_market_returns_none_for_empty_list():
    assert pick_recommended_market([]) is None


def test_pick_recommended_market_prefers_non_no_bet():
    markets = [
        {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "value_edge": 0.20},
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "value_edge": 0.05},
    ]
    picked = pick_recommended_market(markets)
    assert picked["market"] == "result_3way"


def test_pick_recommended_market_breaks_ties_by_value_edge():
    markets = [
        {"market": "btts", "selection": "yes", "recommendation_type": "direct_bet", "value_edge": 0.05},
        {"market": "result_3way", "selection": "home", "recommendation_type": "conditional", "value_edge": 0.12},
    ]
    picked = pick_recommended_market(markets)
    assert picked["market"] == "result_3way"


def test_pick_recommended_market_falls_back_to_no_bet_when_nothing_actionable():
    # Mirrors MatchUI.tsx's bestMarket(): when every market is no_bet, still
    # return the highest-value_edge one rather than nothing at all.
    markets = [
        {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "value_edge": -0.02},
        {"market": "result_3way", "selection": "home", "recommendation_type": "no_bet", "value_edge": 0.01},
    ]
    picked = pick_recommended_market(markets)
    assert picked["market"] == "result_3way"


def test_pick_recommended_market_stable_on_tie_picks_first():
    # When two markets have identical value_edge, max() returns the first
    # maximal element -- deterministic tie-break, not arbitrary.
    markets = [
        {"market": "btts", "selection": "yes", "recommendation_type": "direct_bet", "value_edge": 0.10},
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "value_edge": 0.10},
    ]
    picked = pick_recommended_market(markets)
    assert picked["market"] == "btts"
