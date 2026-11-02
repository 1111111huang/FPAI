"""Tests for the /api/admin/ops/* endpoints -- the four-tab /ops dashboard's
backend. 2026-11-02 design:
docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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
