"""Tests for the A123 live-vs-backtest ROI comparison tool.

fetch_live_bets is tested against a fake httpx transport (no real network) so
the join/grading logic (only direct_bet + resolvable + priced picks count,
scored via src.agent.market_resolution) is verified without depending on a
live server. summarize/by_market/compare are pure functions, tested directly.
"""

from __future__ import annotations

from pathlib import Path
import sys

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.live_vs_backtest_roi import by_market, compare, fetch_live_bets, summarize

_FIXTURES = [
    {  # a genuine E0 win to grade
        "match_id": "m1", "utc_date": "2026-09-17T15:00:00Z", "status": "FINISHED",
        "home_team": "Arsenal", "away_team": "Fulham", "home_goals": 3, "away_goals": 0,
        "competition": "E0",
    },
    {  # a genuine E0 loss to grade (btts:yes picked, but away team never scored)
        "match_id": "m2", "utc_date": "2026-09-18T15:00:00Z", "status": "FINISHED",
        "home_team": "Chelsea", "away_team": "Brighton", "home_goals": 2, "away_goals": 0,
        "competition": "E0",
    },
    {  # wrong league -- must be skipped
        "match_id": "m3", "utc_date": "2026-09-18T15:00:00Z", "status": "FINISHED",
        "home_team": "Real Madrid", "away_team": "Barcelona", "home_goals": 2, "away_goals": 1,
        "competition": "SP1",
    },
    {  # not finished yet -- must be skipped
        "match_id": "m4", "utc_date": "2026-09-19T15:00:00Z", "status": "SCHEDULED",
        "home_team": "Liverpool", "away_team": "Everton", "home_goals": None, "away_goals": None,
        "competition": "E0",
    },
    {  # finished, but no cached recommendation at all -- must be skipped
        "match_id": "m5", "utc_date": "2026-09-19T15:00:00Z", "status": "FINISHED",
        "home_team": "Spurs", "away_team": "Wolves", "home_goals": 2, "away_goals": 0,
        "competition": "E0",
    },
    {  # finished, agent said no_bet -- must be skipped
        "match_id": "m6", "utc_date": "2026-09-19T15:00:00Z", "status": "FINISHED",
        "home_team": "Villa", "away_team": "Burnley", "home_goals": 1, "away_goals": 0,
        "competition": "E0",
    },
    {  # finished, direct_bet on an unresolvable market (home_corners) -- must be skipped
        "match_id": "m7", "utc_date": "2026-09-19T15:00:00Z", "status": "FINISHED",
        "home_team": "Newcastle", "away_team": "West Ham", "home_goals": 2, "away_goals": 1,
        "competition": "E0",
    },
]

_RECOMMENDATIONS = {
    "m1": {
        "candidates": [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 1.80},
        ],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
    },
    "m2": {
        "candidates": [
            {"market": "btts", "selection": "yes", "recommendation_type": "direct_bet", "current_odds": 2.10},
        ],
        "recommendation_pick": {"market": "btts", "selection": "yes"},
    },
    "m6": {
        "candidates": [
            {"market": "total_goals", "selection": "over_2.5", "recommendation_type": "no_bet", "current_odds": None},
        ],
        "recommendation_pick": None,
    },
    "m7": {
        "candidates": [
            {"market": "home_corners", "selection": "over_4.5", "recommendation_type": "direct_bet", "current_odds": 1.90},
        ],
        "recommendation_pick": {"market": "home_corners", "selection": "over_4.5"},
    },
}


def _fake_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/fixtures":
            return httpx.Response(200, json=_FIXTURES)
        if request.url.path.startswith("/api/recommendations/"):
            match_id = request.url.path.rsplit("/", 1)[-1]
            if match_id not in _RECOMMENDATIONS:
                return httpx.Response(404, json={"detail": "not found"})
            return httpx.Response(200, json=_RECOMMENDATIONS[match_id])
        raise AssertionError(f"Unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def test_fetch_live_bets_grades_a_win_and_a_loss_and_skips_everything_else(monkeypatch: pytest.MonkeyPatch) -> None:
    real_client_cls = httpx.Client

    def _patched_client(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client_cls(*args, transport=_fake_transport(), **kwargs)

    monkeypatch.setattr("scripts.live_vs_backtest_roi.httpx.Client", _patched_client)

    bets = fetch_live_bets("https://fake", "E0", "2026-09-17", "2026-09-20")

    assert {b["match_id"] for b in bets} == {"m1", "m2"}, (
        "m3 (wrong league), m4 (not finished), m5 (no cached rec), m6 (no_bet), "
        "m7 (unresolvable market) must all be excluded"
    )
    m1 = next(b for b in bets if b["match_id"] == "m1")
    assert m1["won"] is True
    assert m1["payout"] == pytest.approx(0.80)  # odds 1.80, stake 1.0, win
    m2 = next(b for b in bets if b["match_id"] == "m2")
    assert m2["won"] is False
    assert m2["payout"] == pytest.approx(-1.0)  # btts:yes picked, actual was 2-0 (btts:no) -- a loss


def test_summarize_empty() -> None:
    result = summarize([])
    assert result == {"picks": 0, "wins": 0, "hit_rate": None, "roi": None}


def test_summarize_computes_roi_and_hit_rate() -> None:
    bets = [
        {"stake": 1.0, "payout": 0.8, "won": True},
        {"stake": 1.0, "payout": -1.0, "won": False},
    ]
    result = summarize(bets)
    assert result["picks"] == 2
    assert result["wins"] == 1
    assert result["hit_rate"] == pytest.approx(0.5)
    assert result["roi"] == pytest.approx(-0.1)  # (0.8 - 1.0) / 2.0


def test_by_market_groups_by_market_and_selection() -> None:
    bets = [
        {"market": "btts", "selection": "yes", "stake": 1.0, "payout": 1.0, "won": True},
        {"market": "btts", "selection": "no", "stake": 1.0, "payout": -1.0, "won": False},
    ]
    grouped = by_market(bets)
    assert set(grouped) == {"btts/yes", "btts/no"}
    assert grouped["btts/yes"]["roi"] == pytest.approx(1.0)
    assert grouped["btts/no"]["roi"] == pytest.approx(-1.0)


def test_compare_puts_live_and_backtest_side_by_side_with_market_filter() -> None:
    live_bets = [
        {"market": "btts", "selection": "yes", "stake": 1.0, "payout": 1.0, "won": True},
        {"market": "total_goals", "selection": "over_2.5", "stake": 1.0, "payout": -1.0, "won": False},
    ]
    backtest_report = {
        "bets_placed": 40, "bets_won": 22, "hit_rate": 0.55, "roi": 0.08,
        "market_breakdown": {
            "btts/yes": {"picks": 20, "roi": 0.10},
            "total_goals/over_2.5": {"picks": 20, "roi": 0.06},
        },
    }

    result = compare(live_bets, backtest_report, market_filter="btts")

    assert result["backtest"]["overall"]["roi"] == 0.08
    assert result["live"]["overall"]["picks"] == 2
    assert set(result["live"]["by_market"]) == {"btts/yes"}
    assert set(result["backtest"]["by_market"]) == {"btts/yes"}
