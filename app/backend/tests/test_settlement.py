"""W13: auto-settlement job. Sourced from an on-demand W05 API call
(FootballDataClient.get_results), never from the batch-refreshed, stale
DuckDB raw_matches table. Only src.agent.market_resolution's pure resolution
logic is reused -- the 'actual' outcome dict is built directly from the live
API result (NormalizedMatch), not a DataFrame row.

Acceptance: a settled E0 fixture, checked via a live W05 API call, auto-
updates any open bet on a scorable market (result_3way/btts/total_goals) to
won/lost and computes profit_loss; a corners bet is never auto-settled,
silently or otherwise."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.bet_tracker import BetTracker
from app.backend.football_data_client import NormalizedMatch
from app.backend.settlement import settle_open_bets


def _match(match_id: str, home_goals: int | None, away_goals: int | None) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date="2026-08-22T15:00:00Z", status="FINISHED",
        home_team="Arsenal", away_team="Everton", home_goals=home_goals, away_goals=away_goals,
    )


def test_settles_result_3way_bet_as_won(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    settled = settle_open_bets(tracker, client)

    assert len(settled) == 1
    assert settled[0].outcome == "won"
    assert settled[0].profit_loss == 10.0
    assert tracker.get_bet(bet.id).outcome == "won"


def test_settles_bet_as_lost(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="away", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    settled = settle_open_bets(tracker, client)

    assert settled[0].outcome == "lost"
    assert settled[0].profit_loss == -10.0


def test_settles_btts_and_total_goals_markets(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="btts", selection="yes", odds=1.9, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="total_goals", selection="over_2.5", odds=1.7, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    settled = settle_open_bets(tracker, client)

    assert len(settled) == 2
    assert all(bet.outcome == "won" for bet in settled)


def test_never_auto_settles_a_corners_bet(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="home_corners", selection="over_4.5", odds=1.9, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    settled = settle_open_bets(tracker, client)

    assert settled == []
    assert tracker.get_bet(bet.id).outcome == "open"
    client.get_results.assert_not_called()


def test_leaves_bet_open_when_match_not_yet_finished(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    client = MagicMock()
    client.get_results.return_value = []  # not finished yet -- not in FINISHED results

    settled = settle_open_bets(tracker, client)

    assert settled == []
    assert tracker.get_bet(bet.id).outcome == "open"


def test_ignores_already_settled_bets(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.settle_bet(bet.id, outcome="won")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    settled = settle_open_bets(tracker, client)

    assert settled == []
    client.get_results.assert_not_called()


def test_groups_api_calls_by_date_not_per_bet(tmp_path: Path) -> None:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.create_bet(
        match_id="m2", date="2026-08-22", home_team="Chelsea", away_team="Fulham",
        market="btts", selection="no", odds=1.8, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1), _match("m2", 1, 0)]

    settled = settle_open_bets(tracker, client)

    assert client.get_results.call_count == 1
    assert len(settled) == 2
