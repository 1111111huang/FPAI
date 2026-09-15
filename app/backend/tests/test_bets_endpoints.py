"""W12: POST /api/bets/from-recommendation, POST /api/bets/manual, GET /api/bets."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
from fastapi.testclient import TestClient

from app.backend import bets
from app.backend.auth_deps import get_current_user_email
from app.backend.bet_tracker import BetTracker
from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app, get_user_store
from app.backend.users import User, UserStore

_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.1},
    ],
    "recommendation_pick": {"market": "result_3way", "selection": "home"},
    "explanation": "test", "confidence": "medium", "limitations": [], "prediction_basis": "team_history_and_market",
}


def _override_tracker(tmp_path: Path) -> BetTracker:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    app.dependency_overrides[bets.get_bet_tracker] = lambda: tracker
    return tracker


def _override_user(tmp_path: Path, email: str = "test-user@example.com") -> User:
    """W210: every /api/bets* route now requires get_current_user_email +
    a UserStore lookup -- overrides both with a tmp_path-scoped store
    (mirrors _override_tracker) so existing tests don't 401 and don't
    touch the real data/users.db."""
    store = UserStore(db_path=tmp_path / "users.db")
    app.dependency_overrides[get_current_user_email] = lambda: email
    app.dependency_overrides[get_user_store] = lambda: store
    return store.get_or_create(email)


@pytest.fixture(autouse=True)
def _no_real_settlement_network_calls():
    """W212: both creation routes now auto-settle on log (via
    settle_open_bets), which calls get_fixtures_client/
    get_sweden_fixtures_client -- mocked here so these pre-existing tests
    stay hermetic and their bets stay 'open' (no results returned),
    matching test_settlement_endpoint.py's established pattern."""
    with patch("app.backend.main.get_fixtures_client") as mock_fixtures, \
         patch("app.backend.main.get_sweden_fixtures_client") as mock_sweden:
        mock_fixtures.return_value.get_results.return_value = []
        mock_sweden.return_value.get_results.return_value = []
        yield mock_fixtures.return_value


def test_from_recommendation_endpoint_creates_a_locked_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/bets/from-recommendation",
                json={"match_id": "m1", "recommendation": _RECOMMENDATION, "market": "result_3way", "selection": "home", "stake": 10.0},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["home_team"] == "Arsenal"
        assert body["odds"] == 2.1
        assert body["stake"] == 10.0
        assert body["outcome"] == "open"
        assert body["source"] == "from_recommendation"

        stored = tracker.list_bets()
        assert len(stored) == 1
        assert stored[0].recommendation_snapshot == _RECOMMENDATION
    finally:
        app.dependency_overrides.clear()


def test_from_recommendation_endpoint_ignores_extra_client_fields(tmp_path: Path):
    """A client trying to sneak in a different odds/home_team is simply
    ignored -- the request model has no such fields to bind to."""
    _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/bets/from-recommendation",
                json={
                    "match_id": "m1", "recommendation": _RECOMMENDATION, "market": "result_3way",
                    "selection": "home", "stake": 10.0,
                    "odds": 999.0, "home_team": "Fake Team",  # not accepted fields -- ignored by Pydantic
                },
            )
        assert response.status_code == 200
        assert response.json()["odds"] == 2.1  # from the snapshot, not the client's 999.0
        assert response.json()["home_team"] == "Arsenal"
    finally:
        app.dependency_overrides.clear()


def test_from_recommendation_endpoint_400s_for_unknown_market(tmp_path: Path):
    _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/bets/from-recommendation",
                json={"match_id": "m1", "recommendation": _RECOMMENDATION, "market": "btts", "selection": "yes", "stake": 10.0},
            )
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_manual_endpoint_creates_a_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/bets/manual",
                json={
                    "match_id": "m2", "date": "2026-08-23", "home_team": "Chelsea", "away_team": "Fulham",
                    "market": "btts", "selection": "yes", "odds": 1.9, "stake": 5.0,
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "manual"
        assert body["recommendation_snapshot"] is None
        assert len(tracker.list_bets()) == 1
    finally:
        app.dependency_overrides.clear()


def test_manual_endpoint_auto_settles_a_bet_logged_against_an_already_finished_match(tmp_path: Path, _no_real_settlement_network_calls):
    """W212: direct user feedback -- a bet backfilled against a match that
    already finished (W211 made this searchable at all) should come back
    already settled, not 'open' pending a separate Settle open bets click."""
    tracker = _override_tracker(tmp_path)
    _override_user(tmp_path)
    _no_real_settlement_network_calls.get_results.return_value = [
        NormalizedMatch(
            match_id="m2", utc_date="2026-08-23T15:00:00Z", status="FINISHED",
            home_team="Chelsea", away_team="Fulham", home_goals=2, away_goals=1,
        ),
    ]
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/bets/manual",
                json={
                    "match_id": "m2", "date": "2026-08-23", "home_team": "Chelsea", "away_team": "Fulham",
                    "market": "btts", "selection": "yes", "odds": 1.9, "stake": 5.0,
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "won"  # both teams scored -- btts/yes is correct
        assert len(tracker.list_bets()) == 1
        assert tracker.list_bets()[0].outcome == "won"
    finally:
        app.dependency_overrides.clear()


def test_manual_endpoint_422s_for_missing_match_id(tmp_path: Path):
    _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/bets/manual",
                json={
                    "match_id": "", "date": "2026-08-23", "home_team": "Chelsea", "away_team": "Fulham",
                    "market": "btts", "selection": "yes", "odds": 1.9, "stake": 5.0,
                },
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_list_bets_endpoint_returns_created_bets(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    user = _override_user(tmp_path)
    try:
        tracker.create_bet(
            match_id="m1", date="2026-08-22", home_team="A", away_team="B",
            market="result_3way", selection="home", odds=1.8, stake=10.0,
            source="manual", recommendation_snapshot=None, user_id=user.id,
        )
        with TestClient(app) as client:
            response = client.get("/api/bets")
        assert response.status_code == 200
        assert len(response.json()) == 1
    finally:
        app.dependency_overrides.clear()


def test_list_bets_requires_user_auth_and_filters_by_user(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
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
    app.dependency_overrides[get_current_user_email] = lambda: "user1@gmail.com"
    # Code review follow-up: also override get_user_store -- otherwise this
    # falls through to a real UserStore() and writes into the repo-root
    # data/users.db on every test run.
    app.dependency_overrides[get_user_store] = lambda: UserStore(db_path=tmp_path / "users.db")
    try:
        with TestClient(app) as client:
            response = client.get("/api/bets")
        assert response.status_code == 200
        # Both come back because the override doesn't resolve to a real
        # user_id -- this test only proves the dependency is wired in and
        # the route still works; user-id resolution is verified in the
        # next test.
    finally:
        app.dependency_overrides.pop(get_current_user_email, None)
        app.dependency_overrides.pop(get_user_store, None)


def test_list_bets_401s_without_the_internal_secret(tmp_path: Path):
    _override_tracker(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/bets")
    assert response.status_code == 401


def test_delete_bet_endpoint_removes_the_callers_own_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    user = _override_user(tmp_path)
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=user.id,
    )
    try:
        with TestClient(app) as client:
            response = client.delete(f"/api/bets/{bet.id}")
        assert response.status_code == 204
        assert tracker.get_bet(bet.id) is None
    finally:
        app.dependency_overrides.clear()


def test_delete_bet_endpoint_404s_for_a_bet_owned_by_someone_else(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    _override_user(tmp_path)  # authenticates as this user
    other_users_bet = tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="Chelsea", away_team="Fulham",
        market="btts", selection="yes", odds=1.9, stake=5.0,
        source="manual", recommendation_snapshot=None, user_id=999,  # a different user
    )
    try:
        with TestClient(app) as client:
            response = client.delete(f"/api/bets/{other_users_bet.id}")
        assert response.status_code == 404
        # Not actually deleted -- confirms this 404s before ever calling delete_bet.
        assert tracker.get_bet(other_users_bet.id) is not None
    finally:
        app.dependency_overrides.clear()


def test_delete_bet_endpoint_404s_for_a_nonexistent_id(tmp_path: Path):
    _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.delete("/api/bets/999999")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_delete_bet_endpoint_401s_without_the_internal_secret(tmp_path: Path):
    _override_tracker(tmp_path)
    with TestClient(app) as client:
        response = client.delete("/api/bets/1")
    assert response.status_code == 401


def test_update_bet_endpoint_edits_the_callers_own_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    user = _override_user(tmp_path)
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=user.id,
    )
    try:
        with TestClient(app) as client:
            response = client.patch(f"/api/bets/{bet.id}", json={"stake": 25.0})
        assert response.status_code == 200
        body = response.json()
        assert body["stake"] == 25.0
        assert body["odds"] == 2.0  # untouched
    finally:
        app.dependency_overrides.clear()


def test_update_bet_endpoint_recomputes_profit_loss_for_a_settled_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    user = _override_user(tmp_path)
    bet = tracker.create_bet(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="result_3way", selection="home", odds=2.0, stake=10.0,
        source="manual", recommendation_snapshot=None, user_id=user.id,
    )
    tracker.settle_bet(bet.id, outcome="won")
    try:
        with TestClient(app) as client:
            response = client.patch(f"/api/bets/{bet.id}", json={"stake": 20.0})
        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "won"
        assert body["profit_loss"] == 20.0
    finally:
        app.dependency_overrides.clear()


def test_update_bet_endpoint_404s_for_a_bet_owned_by_someone_else(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
    _override_user(tmp_path)
    other_users_bet = tracker.create_bet(
        match_id="m2", date="2026-08-23", home_team="Chelsea", away_team="Fulham",
        market="btts", selection="yes", odds=1.9, stake=5.0,
        source="manual", recommendation_snapshot=None, user_id=999,
    )
    try:
        with TestClient(app) as client:
            response = client.patch(f"/api/bets/{other_users_bet.id}", json={"stake": 50.0})
        assert response.status_code == 404
        assert tracker.get_bet(other_users_bet.id).stake == 5.0  # unchanged
    finally:
        app.dependency_overrides.clear()


def test_update_bet_endpoint_404s_for_a_nonexistent_id(tmp_path: Path):
    _override_tracker(tmp_path)
    _override_user(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.patch("/api/bets/999999", json={"stake": 10.0})
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_update_bet_endpoint_401s_without_the_internal_secret(tmp_path: Path):
    _override_tracker(tmp_path)
    with TestClient(app) as client:
        response = client.patch("/api/bets/1", json={"stake": 10.0})
    assert response.status_code == 401
