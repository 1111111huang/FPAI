"""W04 support: GET /api/fixtures wraps W05's FootballDataClient directly so
the frontend has a real fixture list to render cards for -- no story yet
covers this narrowly (W09, which would populate fixtures via the cache,
is built last), so this is the minimal connective tissue W04 actually needs."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.football_data_client import NormalizedMatch
from app.backend.main import app


def test_fixtures_endpoint_returns_normalized_matches():
    fake_fixtures = [
        NormalizedMatch(
            match_id="560542", utc_date="2026-08-21T19:00:00Z", status="SCHEDULED",
            home_team="Arsenal", away_team="Coventry City", home_goals=None, away_goals=None,
        ),
    ]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.return_value = fake_fixtures
        with TestClient(app) as client:
            response = client.get("/api/fixtures")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["home_team"] == "Arsenal"
    assert body[0]["away_team"] == "Coventry City"


def test_fixtures_endpoint_passes_date_range_through():
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.get_fixtures.return_value = []
        with TestClient(app) as client:
            client.get("/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"})

    mock_client.get_fixtures.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-28")


def test_fixtures_endpoint_returns_empty_list_gracefully():
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = []
        with TestClient(app) as client:
            response = client.get("/api/fixtures")

    assert response.status_code == 200
    assert response.json() == []
