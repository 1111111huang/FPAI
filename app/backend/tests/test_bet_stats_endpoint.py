"""W14: GET /api/bets/stats."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import bets
from app.backend.bet_stats import DEFAULT_STARTING_BANKROLL
from app.backend.bet_tracker import BetTracker
from app.backend.main import app


def _override_tracker(tmp_path: Path) -> BetTracker:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    app.dependency_overrides[bets.get_bet_tracker] = lambda: tracker
    return tracker


def test_stats_endpoint_returns_zeroed_stats_with_no_bets(tmp_path: Path):
    _override_tracker(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/bets/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["bets_settled"] == 0
        assert body["current_bankroll"] == DEFAULT_STARTING_BANKROLL
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_reflects_a_settled_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="A", away_team="B",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None,
    )
    tracker.settle_bet(bet.id, outcome="won")
    try:
        with TestClient(app) as client:
            response = client.get("/api/bets/stats")
        body = response.json()
        assert body["bets_settled"] == 1
        assert body["bets_won"] == 1
        assert body["roi"] == 1.0
        assert body["current_bankroll"] == DEFAULT_STARTING_BANKROLL + 10.0
    finally:
        app.dependency_overrides.clear()
