"""Regression tests for W13's extracted shared market-resolution utility --
must behave identically to the logic that used to live only in
src/agent/backtest.py, since both backtest scoring and the app's live
settlement job now depend on it."""

from __future__ import annotations

from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct


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
