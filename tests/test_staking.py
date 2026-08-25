"""Tests for bankroll simulation (A13 flat stake, A15 Kelly)."""
from __future__ import annotations

from src.agent.backtest import BacktestRecord
from src.agent.staking import kelly_fraction, simulate_flat_stake, simulate_kelly_stake


def _record(match_id: str, markets: list[dict]) -> BacktestRecord:
    return BacktestRecord(
        match_id=match_id, home_team="A", away_team="B", date="2025-01-01", league="E0",
        recommendation={"overall": "direct_bet", "markets": markets},
        actual={"result": "home"}, market_results=markets,
    )


def test_flat_stake_winning_bet_increases_bankroll():
    markets = [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "correct": True}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0, stake_pct=0.01)
    assert result.ending_bankroll == 1010.0  # +10 stake * (2.0-1)
    assert len(result.bets) == 1
    assert result.bets[0].won is True


def test_flat_stake_losing_bet_decreases_bankroll():
    markets = [{"market": "result_3way", "selection": "away", "recommendation_type": "direct_bet", "current_odds": 2.0, "correct": False}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0, stake_pct=0.01)
    assert result.ending_bankroll == 990.0


def test_flat_stake_skips_non_direct_bet_markets():
    markets = [{"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 1.8, "correct": True}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.ending_bankroll == 1000.0
    assert result.bets == []


def test_flat_stake_skips_unresolvable_markets():
    markets = [{"market": "home_corners", "selection": "over_4.5", "recommendation_type": "direct_bet", "current_odds": 1.9, "correct": None}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.bets == []


def test_flat_stake_equity_curve_starts_with_initial_bankroll():
    result = simulate_flat_stake([], starting_bankroll=500.0)
    assert result.equity_curve == [500.0]


def test_flat_stake_skips_direct_bet_with_null_odds():
    # Agent can mark a market direct_bet while current_odds is null (no bookmaker
    # odds found for that specific market) — must skip, not crash on float(None).
    markets = [{"market": "btts", "selection": "no", "recommendation_type": "direct_bet", "current_odds": None, "correct": True}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.bets == []
    assert result.ending_bankroll == 1000.0


def test_kelly_stake_skips_direct_bet_with_null_odds():
    markets = [{
        "market": "btts", "selection": "no", "recommendation_type": "direct_bet",
        "current_odds": None, "value_edge": 0.1, "correct": True,
    }]
    result = simulate_kelly_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.bets == []
    assert result.ending_bankroll == 1000.0


def test_kelly_stake_sizes_by_value_edge():
    markets = [{
        "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
        "current_odds": 3.0, "value_edge": 0.10, "correct": True,
    }]
    result = simulate_kelly_stake([_record("m1", markets)], starting_bankroll=1000.0, max_fraction=0.5)
    # fraction = value_edge / (odds - 1) = 0.10 / 2.0 = 0.05 -> stake = 50
    assert result.bets[0].stake == 50.0
    assert result.ending_bankroll == 1000.0 + 50.0 * (3.0 - 1)


def test_kelly_stake_caps_at_max_fraction():
    markets = [{
        "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
        "current_odds": 1.5, "value_edge": 0.9, "correct": True,
    }]
    result = simulate_kelly_stake([_record("m1", markets)], starting_bankroll=1000.0, max_fraction=0.1)
    assert result.bets[0].stake == 100.0  # capped at 10% of 1000


def test_kelly_stake_skips_negative_or_zero_edge():
    markets = [{
        "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
        "current_odds": 2.0, "value_edge": -0.05, "correct": True,
    }]
    result = simulate_kelly_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.bets == []


def test_kelly_fraction_positive_edge():
    # 0.10 / (3.0 - 1) = 0.05
    assert kelly_fraction(0.10, 3.0) == 0.05


def test_kelly_fraction_caps_at_max_fraction():
    assert kelly_fraction(0.9, 1.5, max_fraction=0.1) == 0.1


def test_kelly_fraction_returns_zero_for_non_positive_edge():
    assert kelly_fraction(-0.05, 2.0) == 0.0
    assert kelly_fraction(0.0, 2.0) == 0.0


def test_kelly_fraction_returns_zero_for_odds_at_or_below_one():
    assert kelly_fraction(0.1, 1.0) == 0.0
    assert kelly_fraction(0.1, 0.5) == 0.0
