"""W29: wiring sandbox mode into fixtures/odds/cache/bet-tracker
construction. Each seam is tested in isolation via monkeypatched env vars
and singleton resets -- the full real end-to-end proof (real agent call,
real fixtures/odds, zero effect on real dbs) is W31's runbook, not
re-proven here with a real LLM call."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest

from app.backend import bets, main, recommendations
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.scheduler_wiring import build_odds_client


@pytest.fixture(autouse=True)
def _reset_singletons():
    recommendations._cache_singleton = None
    bets._bet_tracker_singleton = None
    yield
    recommendations._cache_singleton = None
    bets._bet_tracker_singleton = None


def test_get_cache_uses_sandbox_scoped_db_path_when_sandbox_active(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    cache = recommendations.get_cache()

    assert cache._db_path.parent.name == "sandbox"


def test_get_cache_uses_real_db_path_when_sandbox_inactive(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)

    cache = recommendations.get_cache()

    assert cache._db_path.parent.name != "sandbox"


def test_get_bet_tracker_uses_sandbox_scoped_db_path_when_sandbox_active(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    tracker = bets.get_bet_tracker()

    assert tracker._db_path.parent.name == "sandbox"


def test_get_bet_tracker_uses_real_db_path_when_sandbox_inactive(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)

    tracker = bets.get_bet_tracker()

    assert tracker._db_path.parent.name != "sandbox"


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


def _wired_api_keys(client) -> list[str]:
    """Reaches into FallbackOddsClient's wrapped PersistingOddsClients to
    read back which real ODDS_API_KEY* values actually got wired, in order --
    mirrors this file's own private-attribute-inspection convention
    (cache._db_path etc.) rather than adding a getter used nowhere else."""
    return [c._client._api_key for c in client._clients]


def test_build_odds_client_odds_api_key_3_falls_back_to_primary_when_unset(monkeypatch) -> None:
    """BUG-056/2026-08-30: local/dev setups that only ever configure
    ODDS_API_KEY shouldn't need a second real key just to exercise the
    3-key fallback code path -- ODDS_API_KEY_3 defaults to ODDS_API_KEY's
    own value. Production sets ODDS_API_KEY_3 explicitly (see next test)."""
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "primary-key")
    monkeypatch.delenv("ODDS_API_KEY_2", raising=False)
    monkeypatch.delenv("ODDS_API_KEY_3", raising=False)

    client = build_odds_client()

    assert _wired_api_keys(client) == ["primary-key", "primary-key"]


def test_build_odds_client_honors_explicit_odds_api_key_3(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "primary-key")
    monkeypatch.setenv("ODDS_API_KEY_2", "secondary-key")
    monkeypatch.setenv("ODDS_API_KEY_3", "third-key")

    client = build_odds_client()

    assert _wired_api_keys(client) == ["primary-key", "secondary-key", "third-key"]


def test_sandbox_job_runs_db_path_lives_under_app_data_sandbox_not_app_backend_data() -> None:
    """Regression: main.py lives in app/backend/, so its sandbox path
    constant needs .parent.parent (matching recommendations.py's/bets.py's
    own _SANDBOX_*_DB_PATH constants) to land under app/data/sandbox/ --
    not app/backend/data/sandbox/, which isn't covered by .gitignore and
    isn't where W29 requires JobRunLog to write in sandbox mode."""
    assert main._SANDBOX_JOB_RUNS_DB_PATH.parent.name == "sandbox"
    assert main._SANDBOX_JOB_RUNS_DB_PATH.parent.parent.name == "data"
    assert main._SANDBOX_JOB_RUNS_DB_PATH.parent.parent.parent.name == "app"
