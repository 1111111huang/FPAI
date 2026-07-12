"""W12: POST /api/bets/from-recommendation, POST /api/bets/manual, GET /api/bets."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import bets
from app.backend.bet_tracker import BetTracker
from app.backend.main import app

_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.1},
    ],
    "explanation": "test", "confidence": "medium", "limitations": [], "prediction_basis": "team_history_and_market",
}


def _override_tracker(tmp_path: Path) -> BetTracker:
    tracker = BetTracker(db_path=tmp_path / "bets.db")
    app.dependency_overrides[bets.get_bet_tracker] = lambda: tracker
    return tracker


def test_from_recommendation_endpoint_creates_a_locked_bet(tmp_path: Path):
    tracker = _override_tracker(tmp_path)
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


def test_manual_endpoint_422s_for_missing_match_id(tmp_path: Path):
    _override_tracker(tmp_path)
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
    try:
        tracker.create_bet(
            match_id="m1", date="2026-08-22", home_team="A", away_team="B",
            market="result_3way", selection="home", odds=1.8, stake=10.0,
            source="manual", recommendation_snapshot=None,
        )
        with TestClient(app) as client:
            response = client.get("/api/bets")
        assert response.status_code == 200
        assert len(response.json()) == 1
    finally:
        app.dependency_overrides.clear()
