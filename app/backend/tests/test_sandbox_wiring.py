"""W29: wiring sandbox mode into fixtures/odds/cache/bet-tracker
construction. Each seam is tested in isolation via monkeypatched env vars
and singleton resets -- the full real end-to-end proof (real agent call,
real fixtures/odds, zero effect on real dbs) is W31's runbook, not
re-proven here with a real LLM call."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import bets, recommendations
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.scheduler_wiring import build_odds_client


def _reset_singletons() -> None:
    recommendations._cache_singleton = None
    bets._bet_tracker_singleton = None


def test_get_cache_uses_sandbox_scoped_db_path_when_sandbox_active(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    cache = recommendations.get_cache()

    assert cache._db_path.parent.name == "sandbox"
    _reset_singletons()


def test_get_cache_uses_real_db_path_when_sandbox_inactive(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.delenv("SANDBOX_MODE", raising=False)

    cache = recommendations.get_cache()

    assert cache._db_path.parent.name != "sandbox"
    _reset_singletons()


def test_get_bet_tracker_uses_sandbox_scoped_db_path_when_sandbox_active(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    tracker = bets.get_bet_tracker()

    assert tracker._db_path.parent.name == "sandbox"
    _reset_singletons()


def test_get_bet_tracker_uses_real_db_path_when_sandbox_inactive(monkeypatch) -> None:
    _reset_singletons()
    monkeypatch.delenv("SANDBOX_MODE", raising=False)

    tracker = bets.get_bet_tracker()

    assert tracker._db_path.parent.name != "sandbox"
    _reset_singletons()


def test_build_odds_client_returns_historical_client_when_sandbox_active(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    client = build_odds_client()

    assert isinstance(client, HistoricalOddsClient)


def test_build_odds_client_returns_live_client_when_sandbox_inactive(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "fake-key")

    client = build_odds_client()

    assert not isinstance(client, HistoricalOddsClient)
