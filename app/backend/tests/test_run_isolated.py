"""Tests for run_isolated.py -- the shared per-unit isolation primitive and
its backing FailuresStore. Mirrors scheduler.py's JobRunLog test pattern
(fresh sqlite3 connection per call, tmp_path-backed instances)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from app.backend.run_isolated import FailuresStore, run_isolated, run_isolated_async


def test_failures_store_records_a_failure(tmp_path: Path) -> None:
    store = FailuresStore(db_path=tmp_path / "failures.db")
    exc = ValueError("bad odds payload")
    store.record(subsystem="t30", unit_id="match_id=558542", exc=exc, alerted=True, league="I1")

    rows = store.list_recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "t30"
    assert row["unit_id"] == "match_id=558542"
    assert row["league"] == "I1"
    assert row["exc_type"] == "ValueError"
    assert row["message"] == "bad odds payload"
    assert "ValueError: bad odds payload" in row["traceback"]
    assert row["retryable"] is None
    assert row["alerted"] == 1


def test_failures_store_league_is_optional(tmp_path: Path) -> None:
    """A shared-builder failure (BUG-101's class) isn't scoped to any one
    league -- league stays NULL rather than guessing at one."""
    store = FailuresStore(db_path=tmp_path / "failures.db")
    store.record(subsystem="pregenerate", unit_id="builder=odds_client", exc=ValueError("x"), alerted=False)

    rows = store.list_recent()
    assert rows[0]["league"] is None


def test_failures_store_filters_by_league(tmp_path: Path) -> None:
    store = FailuresStore(db_path=tmp_path / "failures.db")
    store.record(subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0")
    store.record(subsystem="t30", unit_id="match_id=2", exc=ValueError("b"), alerted=False, league="I1")

    rows = store.list_recent(league="I1")
    assert len(rows) == 1
    assert rows[0]["unit_id"] == "match_id=2"


def test_failures_store_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "failures.db"
    FailuresStore(db_path=db_path).record(
        subsystem="eod_batch", unit_id="league=I1", exc=RuntimeError("boom"), alerted=False,
    )

    reloaded = FailuresStore(db_path=db_path)
    rows = reloaded.list_recent()
    assert len(rows) == 1
    assert rows[0]["subsystem"] == "eod_batch"


def test_run_isolated_returns_the_function_result_on_success(tmp_path: Path) -> None:
    with patch("app.backend.run_isolated.get_failures_store") as mock_get_store:
        result = run_isolated("eod_batch", "match_id=123", lambda: 42)
    assert result == 42
    mock_get_store.assert_not_called()


def test_run_isolated_returns_none_and_records_on_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.backend.run_isolated.DEFAULT_DB_PATH", tmp_path / "failures.db")
    import app.backend.run_isolated as run_isolated_module
    run_isolated_module._singleton_store = None

    def _raise():
        raise ValueError("boom")

    with patch("app.backend.run_isolated.post_slack_alert", return_value=False):
        result = run_isolated("eod_batch", "match_id=123", _raise)

    assert result is None
    rows = run_isolated_module.get_failures_store().list_recent()
    assert len(rows) == 1
    assert rows[0]["exc_type"] == "ValueError"
    assert rows[0]["unit_id"] == "match_id=123"


def test_run_isolated_never_raises_even_if_recording_itself_fails(tmp_path: Path) -> None:
    def _raise():
        raise ValueError("boom")

    with patch("app.backend.run_isolated.get_failures_store", side_effect=RuntimeError("store is down")):
        result = run_isolated("eod_batch", "match_id=123", _raise)

    assert result is None


def test_run_isolated_async_returns_the_function_result_on_success() -> None:
    async def _ok():
        return "fine"

    result = asyncio.run(run_isolated_async("t30", "match_id=558542", _ok))
    assert result == "fine"


def test_run_isolated_async_returns_none_and_records_on_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.backend.run_isolated.DEFAULT_DB_PATH", tmp_path / "failures.db")
    import app.backend.run_isolated as run_isolated_module
    run_isolated_module._singleton_store = None

    async def _raise():
        raise RuntimeError("vendor timeout")

    with patch("app.backend.run_isolated.post_slack_alert", return_value=True):
        result = asyncio.run(run_isolated_async("t30", "match_id=558542", _raise))

    assert result is None
    rows = run_isolated_module.get_failures_store().list_recent()
    assert len(rows) == 1
    assert rows[0]["exc_type"] == "RuntimeError"
    assert rows[0]["alerted"] == 1
