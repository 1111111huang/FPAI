from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import bets
from app.backend.auth_deps import get_current_user_email
from app.backend.bet_tracker import BetTracker
from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app, get_push_store, get_user_store
from app.backend.push_subscriptions import PushSubscriptionStore
from app.backend.users import UserStore


def _override_tracker(tmp_path: Path) -> BetTracker:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    app.dependency_overrides[bets.get_bet_tracker] = lambda: tracker
    return tracker


def _override_user(tmp_path: Path, email: str = "test-user@example.com"):
    store = UserStore(db_path=tmp_path / "users.db")
    app.dependency_overrides[get_current_user_email] = lambda: email
    app.dependency_overrides[get_user_store] = lambda: store
    return store.get_or_create(email)


def test_settle_open_sends_a_push_notification_for_each_settled_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    user = _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=user.id,
    )
    with patch("app.backend.main.get_sweden_fixtures_client") as mock_sweden:
        mock_sweden.return_value.get_results.return_value = []
        with patch("app.backend.main.get_fixtures_client") as mock_client:
            mock_client.return_value.get_results.return_value = [
                NormalizedMatch(
                    match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED",
                    home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
                ),
            ]
            with patch("app.backend.main.send_push_to_user") as mock_send:
                with TestClient(app) as client:
                    response = client.post("/api/bets/settle-open")
    assert response.status_code == 200
    assert mock_send.call_count == 1
    args, kwargs = mock_send.call_args
    assert args[1] == user.id or kwargs.get("user_id") == user.id
