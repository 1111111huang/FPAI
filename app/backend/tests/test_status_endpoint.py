"""W17: GET /api/status -- a small system status surface for the frontend
footer/badge. Pure reuse of already-exposed src/tools functions
(AGENT_TOOL_CONTRACT.md-documented, isolated src/tools/ package) -- no new
engine work, just wiring their existing dict shapes through the app's API."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.main import app


def test_status_endpoint_returns_data_freshness_and_model_status():
    fake_freshness = {"latest_match_date": "2026-05-24", "days_since_update": 49, "match_count": 3800, "is_stale": True}
    fake_model_status = {
        "league": {"btts": {"model_type": "xgboostclassifier", "primary_metric_value": 0.62, "metric_name": "test_log_loss", "selected_at": "2026-07-01T00:00:00Z"}},
        "international": {},
    }
    with patch("app.backend.main.get_data_freshness", return_value=fake_freshness), \
         patch("app.backend.main.get_model_status", return_value=fake_model_status):
        with TestClient(app) as client:
            response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["data_freshness"] == fake_freshness
    assert body["model_status"] == fake_model_status


def test_status_endpoint_hides_a_display_disabled_league_from_by_league():
    """W104 follow-up: SWE (display_enabled=False) must not appear in the
    sidebar's per-league freshness breakdown, even though the underlying
    engine tool still reports it -- the blended top-level numbers are left
    untouched (engine-wide by contract), only by_league is filtered."""
    fake_freshness = {
        "latest_match_date": "2026-08-10", "days_since_update": 3, "match_count": 5000, "is_stale": False,
        "by_league": {
            "E0": {"latest_match_date": "2026-08-10", "days_since_update": 3, "match_count": 1200, "is_stale": False},
            "SWE": {"latest_match_date": "2026-07-19", "days_since_update": 25, "match_count": 3489, "is_stale": True},
        },
    }
    with patch("app.backend.main.get_data_freshness", return_value=fake_freshness), \
         patch("app.backend.main.get_model_status", return_value={}), \
         patch("app.backend.main.list_display_enabled_competition_ids", return_value=["E0", "SP1"]):
        with TestClient(app) as client:
            response = client.get("/api/status")

    body = response.json()
    assert set(body["data_freshness"]["by_league"].keys()) == {"E0"}
    assert body["data_freshness"]["latest_match_date"] == "2026-08-10"  # top-level untouched
    assert body["data_freshness"]["match_count"] == 5000  # top-level untouched


def test_status_endpoint_reflects_real_tool_output_shape():
    """Not mocked -- confirms the endpoint doesn't crash against whatever
    the real src/tools functions return in this environment today, and that
    the response still has both top-level keys."""
    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert "data_freshness" in body
    assert "model_status" in body
    assert "is_stale" in body["data_freshness"]
