"""W27: GET /api/sandbox/status, so the frontend (W30) and test scripts
(W31) can introspect the active sandbox date rather than each needing their
own access to the env vars."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.main import app


def test_sandbox_status_reports_inactive_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/sandbox/status")
    assert response.status_code == 200
    assert response.json() == {"sandbox_mode": False, "as_of": None}


def test_sandbox_status_reports_the_active_override_date(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with TestClient(app) as client:
        response = client.get("/api/sandbox/status")
    assert response.status_code == 200
    assert response.json() == {"sandbox_mode": True, "as_of": "2026-03-01"}
