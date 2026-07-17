"""W27: sandbox clock -- an app-wide, overridable "as-of now", driven by two
env vars (SANDBOX_MODE=1, SANDBOX_DATE=YYYY-MM-DD), both absent by default
so normal operation is completely unaffected."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.sandbox_clock import is_sandbox_mode, sandbox_date, sandbox_now, sandbox_status


def test_sandbox_mode_defaults_to_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)
    assert is_sandbox_mode() is False
    assert sandbox_date() is None


def test_sandbox_mode_off_ignores_a_stray_sandbox_date(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert sandbox_date() is None  # gated on SANDBOX_MODE, not just SANDBOX_DATE's presence


def test_sandbox_mode_on_with_date_returns_the_override_date(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert is_sandbox_mode() is True
    assert sandbox_date().isoformat() == "2026-03-01"


def test_sandbox_now_returns_real_clock_when_sandbox_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    before = datetime.now(timezone.utc)
    result = sandbox_now(timezone.utc)
    after = datetime.now(timezone.utc)
    assert before <= result <= after


def test_sandbox_now_returns_the_override_date_at_midnight_when_active(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    result = sandbox_now(timezone.utc)
    assert result == datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_sandbox_status_reflects_active_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert sandbox_status() == {"sandbox_mode": True, "as_of": "2026-03-01"}


def test_sandbox_status_reflects_inactive_sandbox(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("SANDBOX_DATE", raising=False)
    assert sandbox_status() == {"sandbox_mode": False, "as_of": None}
