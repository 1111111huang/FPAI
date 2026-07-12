"""W14: bet tracker summary stats (ROI, hit rate, running bankroll). Adapts
src/agent/evaluation.py's report-building logic (compute_max_drawdown reused
verbatim) for a running single-user ledger of app.backend.bet_tracker.Bet
rows instead of a fixed backtest batch's BankrollResult -- computed only over
settled (won/lost) bets, recalculated fresh from whatever is in the tracker
each call (no persisted running total)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest

from app.backend.bet_stats import DEFAULT_STARTING_BANKROLL, compute_bet_stats
from app.backend.bet_tracker import BetTracker


def test_no_bets_returns_zeroed_stats(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    stats = compute_bet_stats(tracker.list_bets())
    assert stats["bets_settled"] == 0
    assert stats["bets_open"] == 0
    assert stats["roi"] == 0.0
    assert stats["hit_rate"] == 0.0
    assert stats["current_bankroll"] == DEFAULT_STARTING_BANKROLL


def test_open_bets_are_excluded_from_roi_and_hit_rate(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    stats = compute_bet_stats(tracker.list_bets())
    assert stats["bets_settled"] == 0
    assert stats["bets_open"] == 1
    assert stats["roi"] == 0.0


def test_roi_and_hit_rate_over_won_and_lost_bets(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    won = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    lost = tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="C", away_team="D",
        market="btts", selection="yes", odds=1.5, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.settle_bet(won.id, outcome="won")   # profit +10
    tracker.settle_bet(lost.id, outcome="lost")  # profit -10

    stats = compute_bet_stats(tracker.list_bets())
    assert stats["bets_settled"] == 2
    assert stats["bets_won"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["total_staked"] == pytest.approx(20.0)
    assert stats["total_profit"] == pytest.approx(0.0)
    assert stats["roi"] == pytest.approx(0.0)
    assert stats["current_bankroll"] == pytest.approx(DEFAULT_STARTING_BANKROLL)


def test_positive_roi_when_wins_outweigh_losses(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=3.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.settle_bet(bet.id, outcome="won")  # profit +20

    stats = compute_bet_stats(tracker.list_bets())
    assert stats["total_profit"] == pytest.approx(20.0)
    assert stats["roi"] == pytest.approx(2.0)  # 20 profit / 10 staked
    assert stats["current_bankroll"] == pytest.approx(DEFAULT_STARTING_BANKROLL + 20.0)


def test_max_drawdown_reflects_a_losing_streak(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    for i in range(3):
        bet = tracker.create_bet(
            match_id=f"m{i}", date="2026-08-22", home_team="A", away_team="B",
            market="result_3way", selection="home", odds=2.0, stake=100.0,
            source="manual", recommendation_snapshot=None,
        )
        tracker.settle_bet(bet.id, outcome="lost")

    stats = compute_bet_stats(tracker.list_bets(), starting_bankroll=1000.0)
    assert stats["max_drawdown"] == pytest.approx(0.3)  # 300 lost / 1000 peak
    assert stats["current_bankroll"] == pytest.approx(700.0)


def test_custom_starting_bankroll_is_respected(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    stats = compute_bet_stats(tracker.list_bets(), starting_bankroll=500.0)
    assert stats["starting_bankroll"] == 500.0
    assert stats["current_bankroll"] == 500.0
