"""W101: POST /api/admin/trigger-data-refresh -- manually kicks off the
ML-engine's own standalone refresh-data chain (main.py's run_refresh_data)
as a genuinely separate OS process, not an in-process call. Every test here
mocks subprocess.Popen -- this repo's own network-blocking test fixture
(conftest.py, W20) only patches socket.socket in *this* process, which
would NOT stop a real child subprocess from making real network calls if
Popen were ever allowed to actually run un-mocked in a test."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app


def test_triggers_a_subprocess_with_the_correct_command_and_league(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    mock_process = MagicMock()
    mock_process.pid = 12345
    with patch("app.backend.main.subprocess.Popen", return_value=mock_process) as mock_popen:
        with TestClient(app) as client:
            response = client.post("/api/admin/trigger-data-refresh", params={"league": "SWE"})

    assert response.status_code == 200
    assert response.json() == {"league": "SWE", "pid": 12345, "status": "started"}
    args, kwargs = mock_popen.call_args
    command = args[0]
    assert command[-3:] == ["main.py", "refresh-data", "--league"] or command[-2:] == ["--league", "SWE"]
    assert "--league" in command and "SWE" in command
    assert "refresh-data" in command


@pytest.mark.parametrize("league", ["I1", "D1", "F1"])
def test_triggers_a_subprocess_for_the_three_new_leagues(monkeypatch, league):
    """Found live (2026-08-17) while checking what's left for Serie A/
    Bundesliga/Ligue 1 to take effect on the deployed app: the allowlist
    here never got extended alongside match_info.py's COMPETITION_ALLOWLIST
    (W135) -- without this, a deployed instance's own DB (gitignored, not
    synced via git, W100) has no way to be backfilled with these three
    leagues' historical match data remotely, so live forecasts for them
    would stay stuck at cold-start even after the code itself deploys."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    mock_process = MagicMock()
    mock_process.pid = 12345
    with patch("app.backend.main.subprocess.Popen", return_value=mock_process) as mock_popen:
        with TestClient(app) as client:
            response = client.post("/api/admin/trigger-data-refresh", params={"league": league})

    assert response.status_code == 200
    assert response.json() == {"league": league, "pid": 12345, "status": "started"}
    args, kwargs = mock_popen.call_args
    command = args[0]
    assert "--league" in command and league in command
    assert "refresh-data" in command


def test_rejects_a_league_outside_the_fixed_allowlist(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    with patch("app.backend.main.subprocess.Popen") as mock_popen:
        with TestClient(app) as client:
            response = client.post("/api/admin/trigger-data-refresh", params={"league": "PREMIER_LEAGUE"})

    assert response.status_code == 422  # FastAPI's own Literal validation
    mock_popen.assert_not_called()


def test_does_not_redirect_subprocess_output_into_an_unread_pipe(monkeypatch):
    """Found live 2026-08-10: stdout=/stderr=PIPE with nobody ever reading
    it (a) silently hides the refresh's own output from Railway's log
    capture, and (b) risks the classic subprocess deadlock -- the child
    blocks forever once it writes enough output to fill the OS pipe
    buffer. Must inherit this process's own stdout/stderr instead (no
    stdout=/stderr= kwarg at all)."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    mock_process = MagicMock()
    mock_process.pid = 1
    with patch("app.backend.main.subprocess.Popen", return_value=mock_process) as mock_popen:
        with TestClient(app) as client:
            client.post("/api/admin/trigger-data-refresh", params={"league": "E0"})

    _, kwargs = mock_popen.call_args
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs


def test_returns_immediately_without_waiting_for_the_subprocess(monkeypatch):
    """Popen (not run/check_call) must be used -- this must never block on
    the refresh actually finishing, which can take several minutes."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    mock_process = MagicMock()
    mock_process.pid = 1
    with patch("app.backend.main.subprocess.Popen", return_value=mock_process) as mock_popen:
        with TestClient(app) as client:
            client.post("/api/admin/trigger-data-refresh", params={"league": "E0"})

    mock_process.wait.assert_not_called()
    mock_process.communicate.assert_not_called()
    mock_popen.assert_called_once()
