"""W11 acceptance: /api/recommendations reads exclusively from the cache for
already-generated fixtures (GET, never calls run_agent); the manual
"regenerate now" path (POST, existing from W02) is the distinct,
explicitly-triggered action that actually writes into the cache."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.main import app
from app.backend.recommendation_cache import RecommendationCache
from src.agent.agent_config import AgentConfig

_VALID_MARKET = {
    "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
    "current_odds": 2.1, "min_odds": 1.8, "ml_probability": 0.55,
    "implied_probability": 0.48, "value_edge": 0.07,
}
_VALID_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet", "markets": [_VALID_MARKET],
    "explanation": "Value found.", "confidence": "medium", "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _override_cache(tmp_path: Path) -> RecommendationCache:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    app.dependency_overrides[recommendations.get_cache] = lambda: cache
    return cache


def test_get_returns_404_when_nothing_cached_yet(tmp_path: Path):
    _override_cache(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/m1", params={"date": "2026-08-22"})
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_falls_back_to_a_stale_config_hash_row_instead_of_404ing(tmp_path: Path):
    """A65/A66 follow-up: a config change bumps agent_config_hash for every
    match at once -- if regeneration under the new hash then also fails
    (or just hasn't run yet), get_latest() alone finds nothing even though
    a perfectly good prior recommendation, generated under the previous
    config, still exists. Confirmed live: a DeepSeek billing outage failed
    every match in the same batch identically, right after A66's own
    config-field addition had already bumped the hash for all of them."""
    cache = _override_cache(tmp_path)
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="stale-hash-from-before-the-config-change",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0}, recommendation=_VALID_RECOMMENDATION,
        triggered_by="scheduled",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/m1", params={"date": "2026-08-22"})
        assert response.status_code == 200
        assert response.json()["overall"] == "direct_bet"
    finally:
        app.dependency_overrides.clear()


def test_get_never_calls_run_agent(tmp_path: Path):
    _override_cache(tmp_path)
    try:
        with patch("app.backend.recommendations.run_agent") as mock_run:
            with TestClient(app) as client:
                client.get("/api/recommendations/m1", params={"date": "2026-08-22"})
        mock_run.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_post_writes_to_cache_as_manual_regenerate(tmp_path: Path):
    cache = _override_cache(tmp_path)
    try:
        with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION):
            with TestClient(app) as client:
                response = client.post(
                    "/api/recommendations",
                    json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "match_id": "m1"},
                )
        assert response.status_code == 200

        config_hash = compute_agent_config_hash(AgentConfig.default())
        entry = cache.get_latest("m1", "2026-08-22", config_hash)
        assert entry is not None
        assert entry.triggered_by == "manual_regenerate"
        assert entry.recommendation["overall"] == "direct_bet"
    finally:
        app.dependency_overrides.clear()


def test_get_reads_back_what_post_wrote(tmp_path: Path):
    _override_cache(tmp_path)
    try:
        with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION):
            with TestClient(app) as client:
                client.post(
                    "/api/recommendations",
                    json={"home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22", "match_id": "m1"},
                )
                response = client.get("/api/recommendations/m1", params={"date": "2026-08-22"})

        assert response.status_code == 200
        assert response.json()["overall"] == "direct_bet"
    finally:
        app.dependency_overrides.clear()


def test_get_degrades_a_legacy_cached_row_instead_of_500ing(tmp_path: Path):
    """BUG-028: found live -- GET /api/recommendations/{match_id} used to
    call MatchRecommendationOut.model_validate() directly on the raw cached
    dict, which raises an uncaught ValidationError (-> 500) for any
    pre-existing row whose market/selection predates BUG-027's Literal
    constraints. Confirmed live against a real cached row (a local-model
    hallucination with selection="Arsenal" instead of home/draw/away) that
    was crashing this exact endpoint before the fix below."""
    cache = _override_cache(tmp_path)
    legacy_market = {**_VALID_MARKET, "market": "1X2", "selection": "Arsenal"}
    cache.record_generation(
        match_id="m1", date="2026-08-22",
        agent_config_hash=compute_agent_config_hash(AgentConfig.default()),
        odds={}, recommendation={**_VALID_RECOMMENDATION, "markets": [legacy_market]},
        triggered_by="scheduled",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/recommendations/m1", params={"date": "2026-08-22"})
        assert response.status_code == 200
        assert response.json()["markets"] == []
        assert response.json()["invalid_market_count"] == 1
    finally:
        app.dependency_overrides.clear()


def test_cache_entry_exposes_the_odds_it_was_generated_from(tmp_path: Path):
    cache = _override_cache(tmp_path)
    try:
        with patch("app.backend.recommendations.run_agent", return_value=_VALID_RECOMMENDATION):
            with TestClient(app) as client:
                client.post(
                    "/api/recommendations",
                    json={
                        "home_team": "Arsenal", "away_team": "Everton", "date": "2026-08-22",
                        "match_id": "m1", "odds": {"home": 1.5, "draw": 4.0, "away": 6.0},
                    },
                )
        config_hash = compute_agent_config_hash(AgentConfig.default())
        entry = cache.get_latest("m1", "2026-08-22", config_hash)
        assert entry.odds == {"home": 1.5, "draw": 4.0, "away": 6.0}
    finally:
        app.dependency_overrides.clear()
