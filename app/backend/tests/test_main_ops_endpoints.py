"""Tests for the /api/admin/ops/* endpoints -- the four-tab /ops dashboard's
backend. 2026-11-02 design:
docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.main import app
from app.backend.run_isolated import get_failures_store


def test_get_ops_failures_returns_recorded_failures_and_breaker_status(tmp_path: Path) -> None:
    # conftest.py's autouse _default_failures_store fixture already points
    # run_isolated's DEFAULT_DB_PATH at a tmp_path and resets the singleton
    # before this test body runs -- no need to redo that here.
    get_failures_store().record(
        subsystem="t30", unit_id="match_id=558542", exc=ValueError("vendor exploded"),
        alerted=True, league="I1",
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures")

    assert response.status_code == 200
    body = response.json()
    assert len(body["failures"]) == 1
    assert body["failures"][0]["subsystem"] == "t30"
    assert body["failures"][0]["league"] == "I1"
    assert set(body["circuit_breakers"].keys()) == {"the_odds_api", "oddspapi"}


def test_get_ops_failures_filters_by_league(tmp_path: Path) -> None:
    store = get_failures_store()
    store.record(subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0")
    store.record(subsystem="t30", unit_id="match_id=2", exc=ValueError("b"), alerted=False, league="I1")

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures", params={"league": "I1"})

    body = response.json()
    assert len(body["failures"]) == 1
    assert body["failures"][0]["unit_id"] == "match_id=2"


def test_get_ops_failures_filters_by_since(tmp_path: Path) -> None:
    get_failures_store().record(
        subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0",
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures", params={"since": "2099-01-01T00:00:00+00:00"})

    assert response.json()["failures"] == []


def test_get_ops_odds_availability_reports_missing_markets_with_cause(tmp_path: Path) -> None:
    from app.backend.recommendation_cache import CacheEntry

    entry = CacheEntry(
        match_id="558542", date="2026-11-02", agent_config_hash="abc",
        odds={"home": 2.1, "draw": 3.4, "away": 3.6, "total_goals": None, "asian_handicap": None, "corners": None},
        recommendation={"match": {"home": "Udinese", "away": "Roma", "date": "2026-11-02", "league": "I1"}},
        generated_at="2026-11-02T10:00:00+00:00", triggered_by="scheduled",
    )
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = [entry]

    # NOTE on patch target: main.py imports get_circuit_breaker_status via
    # `from app.backend.scheduler_wiring import (..., get_circuit_breaker_status)`,
    # which binds the name directly into main's own module namespace at
    # import time. The new endpoint below calls the bare name
    # `get_circuit_breaker_status()`, which resolves through main's
    # namespace, not scheduler_wiring's -- so patching
    # `app.backend.scheduler_wiring.get_circuit_breaker_status` would leave
    # main's already-bound reference untouched. Patch where it's USED:
    # `app.backend.main.get_circuit_breaker_status`.
    with patch("app.backend.main.get_circuit_breaker_status", return_value={
        "the_odds_api": {"state": "open"}, "oddspapi": {"state": "closed"},
    }):
        with TestClient(app) as client:
            app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
            try:
                response = client.get(
                    "/api/admin/ops/odds-availability",
                    params={"date_from": "2026-11-02", "date_to": "2026-11-02"},
                )
            finally:
                app.dependency_overrides.pop(recommendations.get_cache, None)

    assert response.status_code == 200
    body = response.json()
    assert len(body["fixtures"]) == 1
    row = body["fixtures"][0]
    assert row["match_id"] == "558542"
    assert row["league"] == "I1"
    assert "over/under 2.5" in row["missing_markets"]
    assert row["likely_cause"] == "circuit_open"
