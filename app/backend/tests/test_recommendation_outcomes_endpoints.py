"""W167/W168: POST /api/recommendations/settle-open, GET /api/recommendations/stats.

Mirrors test_settlement_endpoint.py's own patching convention exactly:
settle_open_recommendations() (like settle_open()) calls get_fixtures_client()/
get_sweden_fixtures_client() as plain module-level function calls inside the
endpoint body, not via Depends() -- so tests patch
"app.backend.main.get_fixtures_client" directly rather than using
app.dependency_overrides for those two."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore, get_recommendation_outcome_store


def _override(tmp_path: Path) -> tuple[RecommendationCache, RecommendationOutcomeStore]:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    app.dependency_overrides[recommendations.get_cache] = lambda: cache
    app.dependency_overrides[get_recommendation_outcome_store] = lambda: store
    return cache, store


@pytest.fixture(autouse=True)
def sweden_client_mock():
    """W57 precedent (test_settlement_endpoint.py): defaulted to empty so
    every test here that doesn't care about Sweden keeps working unchanged --
    resolve_pending_recommendations always consults sweden_client when the
    endpoint supplies one, regardless of whether any candidate is Swedish."""
    with patch("app.backend.main.get_sweden_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_results.return_value = []
        yield mock_get_client.return_value


_REC = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1}],
    "confidence": "medium", "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
}


def test_settle_open_endpoint_resolves_and_returns_outcomes(tmp_path: Path):
    cache, store = _override(tmp_path)
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _REC, "scheduled")
    try:
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = [
                NormalizedMatch(match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED", home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1)
            ]
            with TestClient(app) as client:
                response = client.post("/api/recommendations/settle-open")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["correct"] is True
        assert len(store.list_all()) == 1
    finally:
        app.dependency_overrides.clear()


def test_settle_open_endpoint_returns_empty_list_when_nothing_resolves(tmp_path: Path):
    _override(tmp_path)
    try:
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.post("/api/recommendations/settle-open")
        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_returns_zeroed_stats_with_nothing_resolved(tmp_path: Path):
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/stats")
        assert response.status_code == 200
        assert response.json()["overall"]["sample_size"] == 0
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_reflects_a_resolved_outcome(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/stats")
        assert response.status_code == 200
        assert response.json()["overall"]["sample_size"] == 1
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_days_param_filters_by_date(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="old", date="2020-01-01", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2020-01-01T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/stats?days=30")
        assert response.json()["overall"]["sample_size"] == 0
    finally:
        app.dependency_overrides.clear()


def test_stats_endpoint_rejects_out_of_range_days(tmp_path: Path):
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/stats?days=10000000")
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
