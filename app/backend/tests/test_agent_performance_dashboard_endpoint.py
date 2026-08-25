"""W172: GET /api/recommendations/performance-dashboard."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.main import app
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore, get_recommendation_outcome_store


def _override(tmp_path: Path) -> tuple[RecommendationCache, RecommendationOutcomeStore]:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    app.dependency_overrides[recommendations.get_cache] = lambda: cache
    app.dependency_overrides[get_recommendation_outcome_store] = lambda: store
    return cache, store


def test_dashboard_endpoint_returns_empty_state_with_no_data(tmp_path: Path):
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard")
        assert response.status_code == 200
        body = response.json()
        assert body["kelly_roi_simulation"]["bets_placed"] == 0
        assert body["top_winners"] == []
        assert body["top_losers"] == []
        assert body["staked_bets"] == []
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_reflects_a_resolved_outcome(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard")
        assert response.status_code == 200
        body = response.json()
        assert body["kelly_roi_simulation"]["bets_placed"] == 1
        assert len(body["top_winners"]) == 1
        assert "by_market_metrics" in body
        assert "result_3way" in body["by_market_metrics"]
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_respects_top_n_param(tmp_path: Path):
    _, store = _override(tmp_path)
    for i in range(3):
        store.insert(
            match_id=f"m{i}", date="2026-08-22", competition="E0", market="result_3way", selection="home",
            recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1 + i * 0.01, correct=True,
            generated_at="2026-08-22T10:00:00+00:00",
        )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard?top_n=2")
        assert len(response.json()["top_winners"]) == 2
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_rejects_out_of_range_top_n(tmp_path: Path):
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard?top_n=0")
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_dashboard_endpoint_days_param_filters_by_date(tmp_path: Path):
    _, store = _override(tmp_path)
    store.insert(
        match_id="old", date="2020-01-01", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2020-01-01T10:00:00+00:00",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard?days=30")
        assert response.json()["kelly_roi_simulation"]["bets_placed"] == 0
    finally:
        app.dependency_overrides.clear()


def test_dashboard_route_registered_before_match_id_route(tmp_path: Path):
    """Same route-ordering hazard already caught once for /stats (W168) --
    {match_id} is a single-segment pattern that would otherwise swallow the
    literal "performance-dashboard" path segment."""
    _override(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/performance-dashboard")
        # A 422 here (missing required `date` query param) would mean this
        # request got routed to get_cached_recommendation(match_id=...)
        # instead of the dashboard endpoint.
        assert response.status_code != 422
    finally:
        app.dependency_overrides.clear()
