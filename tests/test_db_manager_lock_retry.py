"""W95 (documents/app_user_stories.md Phase 21): DuckDB enforces a real
exclusive file lock (confirmed live, W93/W94's own investigation) -- a
second *process* opening any connection while another already has one open
to the same file fails immediately with duckdb.IOException. Confirmed
separately (not assumed) that this is a per-process lock, not per-connection
-- two connections from the *same* process to the same file coexist fine,
so a genuine collision can only be reproduced with a real second process,
not simulated in-process. That's why this file has both: fast, deterministic
mocked tests for the retry loop's own logic (count, backoff, give-up
behavior), and one real subprocess-based test proving an actual live lock
conflict resolves via retry end-to-end.

Direct user decision: DuckDBManager.connection() should retry on this
specific failure rather than fail immediately -- a collision is expected to
be transient (every real caller's own connections are short-lived, per
US#154/US#155's own investigation), so retrying briefly is expected to
resolve most collisions without ever surfacing to a caller at all.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import duckdb
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.db_manager import DuckDBManager


def _make_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def test_retries_and_succeeds_once_a_transient_lock_conflict_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _make_manager(tmp_path)
    real_connect = duckdb.connect
    calls: list[float] = []

    attempt = {"n": 0}

    def _flaky_connect(path, read_only=False):
        attempt["n"] += 1
        if attempt["n"] < 3:
            raise duckdb.IOException("Conflicting lock is held")
        return real_connect(path, read_only=read_only)

    monkeypatch.setattr(duckdb, "connect", _flaky_connect)

    with manager.connection(sleep_fn=calls.append) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")

    assert attempt["n"] == 3  # 2 failures + 1 success
    assert calls == [1.0, 1.0]  # slept between attempts 1->2 and 2->3, not after the final success


def test_retries_on_connection_exception_too_not_just_io_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W185 code-quality review: duckdb.ConnectionException (raised live
    when two RecoverableScheduler jobs' catch-up threads both open a
    connection to the same file at nearly the same instant -- confirmed
    via the new weekly live-lesson review job racing the existing daily
    one) is a sibling of IOException, not a subclass, so the retry loop
    silently never covered it before this fix."""
    manager = _make_manager(tmp_path)
    real_connect = duckdb.connect
    attempt = {"n": 0}

    def _flaky_connect(path, read_only=False):
        attempt["n"] += 1
        if attempt["n"] < 2:
            raise duckdb.ConnectionException("Can't open a connection to same database file with a different configuration")
        return real_connect(path, read_only=read_only)

    monkeypatch.setattr(duckdb, "connect", _flaky_connect)

    with manager.connection(sleep_fn=lambda _: None) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")

    assert attempt["n"] == 2  # 1 failure + 1 success -- retried instead of raising immediately


def test_raises_the_original_exception_once_retries_are_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _make_manager(tmp_path)
    calls: list[float] = []

    def _always_locked(path, read_only=False):
        raise duckdb.IOException("Conflicting lock is held")

    monkeypatch.setattr(duckdb, "connect", _always_locked)

    with pytest.raises(duckdb.IOException, match="Conflicting lock is held"):
        with manager.connection(max_retries=2, sleep_fn=calls.append):
            pass  # pragma: no cover -- never reached

    # 3 total attempts (max_retries=2 -> initial + 2 retries), slept between
    # each attempt but not after the final, still-failing one.
    assert calls == [1.0, 1.0]


def test_max_retries_zero_preserves_the_original_fail_immediately_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _make_manager(tmp_path)
    call_count = {"n": 0}
    slept: list[float] = []

    def _always_locked(path, read_only=False):
        call_count["n"] += 1
        raise duckdb.IOException("Conflicting lock is held")

    monkeypatch.setattr(duckdb, "connect", _always_locked)

    with pytest.raises(duckdb.IOException):
        with manager.connection(max_retries=0, sleep_fn=slept.append):
            pass  # pragma: no cover -- never reached

    assert call_count["n"] == 1  # exactly one attempt, no retry
    assert slept == []  # never slept at all


def test_bare_construction_preserves_the_original_5_and_1_0_defaults(tmp_path: Path) -> None:
    """US#159: DuckDBManager() with no retry params given must behave
    byte-identically to before this story -- every existing caller that
    doesn't know about default_max_retries/default_retry_delay_seconds."""
    manager = _make_manager(tmp_path)
    assert manager.default_max_retries == 5
    assert manager.default_retry_delay_seconds == 1.0


def test_connection_falls_back_to_the_instance_default_retry_window_when_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """US#159: a manager built with its own default_max_retries/
    default_retry_delay_seconds (e.g. main.py's CLI, given a much longer
    window than the app's live-request default) must actually use that
    window on a plain .connection() call with neither kwarg passed --
    not the hardcoded 5/1.0 from before this story."""
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    manager = DuckDBManager(config_path=str(config_path), default_max_retries=2, default_retry_delay_seconds=0.25)
    calls: list[float] = []

    def _always_locked(path, read_only=False):
        raise duckdb.IOException("Conflicting lock is held")

    monkeypatch.setattr(duckdb, "connect", _always_locked)

    with pytest.raises(duckdb.IOException):
        with manager.connection(sleep_fn=calls.append):  # no max_retries/retry_delay_seconds passed
            pass  # pragma: no cover -- never reached

    assert calls == [0.25, 0.25]  # 2 retries (the instance default), at 0.25s (the instance default)


def test_an_explicit_per_call_value_still_overrides_the_instance_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instance default is only a fallback -- a caller that still wants
    to override it for one specific call (as every existing test in this
    file already does) must still be able to."""
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    manager = DuckDBManager(config_path=str(config_path), default_max_retries=60, default_retry_delay_seconds=10.0)
    calls: list[float] = []

    def _always_locked(path, read_only=False):
        raise duckdb.IOException("Conflicting lock is held")

    monkeypatch.setattr(duckdb, "connect", _always_locked)

    with pytest.raises(duckdb.IOException):
        with manager.connection(max_retries=1, retry_delay_seconds=0.1, sleep_fn=calls.append):
            pass  # pragma: no cover -- never reached

    assert calls == [0.1]  # the per-call override (1 retry, 0.1s), not the instance default (60/10.0)


def test_a_real_cross_process_lock_conflict_resolves_via_retry(tmp_path: Path) -> None:
    """The genuine end-to-end proof: a real second OS process holds a
    read-write connection open for ~2s; DuckDBManager.connection()'s
    default retry (5 attempts, 1s apart) must ride that out and succeed,
    where a bare duckdb.connect() call (no retry) would fail immediately
    -- already proven separately during this story's own live
    investigation, not re-proven here."""
    db_path = tmp_path / "shared.db"
    holder_script = tmp_path / "hold_lock.py"
    holder_script.write_text(
        "import duckdb, time, sys\n"
        "conn = duckdb.connect(sys.argv[1])\n"
        "conn.execute('CREATE TABLE t (x INTEGER)')\n"
        "print('HOLDING', flush=True)\n"
        "time.sleep(2)\n"
        "conn.close()\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(holder_script), str(db_path)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.strip() == "HOLDING"  # confirmed the other process really holds the lock now

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
        manager = DuckDBManager(config_path=str(config_path))

        started = time.monotonic()
        with manager.connection() as conn:  # real sleep_fn/time.sleep defaults -- genuinely waits
            result = conn.execute("SELECT COUNT(*) FROM t").fetchone()
        elapsed = time.monotonic() - started

        assert result == (0,)
        assert elapsed >= 1.0  # proves it genuinely retried at least once, not a lucky immediate success
    finally:
        proc.wait(timeout=10)
