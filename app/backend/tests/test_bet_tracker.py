"""W12: user_bets storage. SQLite-backed (app/data/user_bets.db), same pattern
as W11's recommendation_cache. Covers CRUD plus the settlement computation
(profit_loss = stake * (odds - 1) if won, -stake if lost, null while open) --
W13 will call settle_bet(); this file only proves the storage/computation
layer itself."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.bet_tracker import BetTracker


def test_create_bet_starts_open_with_null_profit_loss(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=1.8, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    assert bet.outcome == "open"
    assert bet.profit_loss is None
    assert bet.id is not None


def test_create_bet_from_recommendation_stores_the_snapshot(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    snapshot = {"overall": "direct_bet", "markets": [{"market": "btts", "selection": "yes"}]}
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="btts", selection="yes", odds=1.9, stake=5.0,
        source="from_recommendation", recommendation_snapshot=snapshot,
    )
    assert bet.source == "from_recommendation"
    assert bet.recommendation_snapshot == snapshot


def test_delete_bet_removes_it(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )

    deleted = tracker.delete_bet(bet.id)

    assert deleted is True
    assert tracker.get_bet(bet.id) is None


def test_delete_bet_returns_false_for_a_nonexistent_id(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    assert tracker.delete_bet(999) is False


def test_update_bet_changes_only_the_given_fields(tmp_path: Path) -> None:
    """W216: a typo'd stake/odds (or a wrong market/selection) previously
    had no correction path short of delete-and-relog. A partial update --
    only the fields actually provided -- leaves everything else untouched."""
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )

    updated = tracker.update_bet(bet.id, stake=25.0)

    assert updated.stake == 25.0
    assert updated.odds == 2.0  # untouched
    assert updated.market == "result_3way"  # untouched
    assert updated.selection == "home"  # untouched


def test_update_bet_can_change_market_and_selection(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )

    updated = tracker.update_bet(bet.id, market="btts", selection="yes", odds=1.8)

    assert updated.market == "btts"
    assert updated.selection == "yes"
    assert updated.odds == 1.8


def test_update_bet_returns_none_for_a_nonexistent_id(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    assert tracker.update_bet(999, stake=10.0) is None


def test_update_bet_recomputes_profit_loss_for_an_already_won_bet(tmp_path: Path) -> None:
    """Editing odds/stake on a bet that already settled must keep
    profit_loss consistent with its (unchanged) outcome -- not leave a
    stale figure computed against the old terms."""
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.settle_bet(bet.id, outcome="won")

    updated = tracker.update_bet(bet.id, stake=20.0)

    assert updated.outcome == "won"
    assert updated.profit_loss == 20.0  # 20 * (2.0 - 1)


def test_update_bet_recomputes_profit_loss_for_an_already_lost_bet(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.settle_bet(bet.id, outcome="lost")

    updated = tracker.update_bet(bet.id, odds=3.0, stake=15.0)

    assert updated.outcome == "lost"
    assert updated.profit_loss == -15.0


def test_update_bet_leaves_an_open_bets_profit_loss_as_none(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )

    updated = tracker.update_bet(bet.id, stake=20.0)

    assert updated.outcome == "open"
    assert updated.profit_loss is None


def test_list_bets_returns_all_created_bets(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=1.8, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="C", away_team="D",
        market="btts", selection="yes", odds=1.9, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    bets = tracker.list_bets()
    assert len(bets) == 2


def test_settle_bet_won_computes_correct_profit(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=2.5, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    settled = tracker.settle_bet(bet.id, outcome="won")
    assert settled.outcome == "won"
    assert settled.profit_loss == pytest.approx(15.0)  # 10 * (2.5 - 1)


def test_settle_bet_lost_computes_negative_stake(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=2.5, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    settled = tracker.settle_bet(bet.id, outcome="lost")
    assert settled.outcome == "lost"
    assert settled.profit_loss == pytest.approx(-10.0)


def test_settled_bet_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "bets.db"
    bet = BetTracker(db_path=db_path).create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    BetTracker(db_path=db_path).settle_bet(bet.id, outcome="won")

    reloaded = BetTracker(db_path=db_path).list_bets()
    assert reloaded[0].outcome == "won"
    assert reloaded[0].profit_loss == pytest.approx(10.0)


def test_get_bet_returns_none_for_unknown_id(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    assert tracker.get_bet(999) is None


def test_settle_bet_raises_for_unknown_id(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    with pytest.raises(ValueError, match="not found"):
        tracker.settle_bet(999, outcome="won")


def test_open_bets_filter(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    b1 = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=1.8, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="C", away_team="D",
        market="btts", selection="yes", odds=1.9, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.settle_bet(b1.id, outcome="won")

    open_bets = tracker.list_open_bets()
    assert len(open_bets) == 1
    assert open_bets[0].match_id == "m2"


def test_create_bet_stores_user_id_and_list_bets_filters_by_it(tmp_path):
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.1, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=1,
    )
    tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="Chelsea", away_team="Fulham",
        market="result_3way", selection="away", odds=3.0, stake=5.0,
        source="manual", recommendation_snapshot=None, user_id=2,
    )
    user_1_bets = tracker.list_bets(user_id=1)
    assert len(user_1_bets) == 1
    assert user_1_bets[0].match_id == "m1"
    assert user_1_bets[0].user_id == 1
