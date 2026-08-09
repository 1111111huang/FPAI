"""W37: wires SnapshotStore record/replay into the sandbox agent-invocation
path. When sandbox mode is active, recommendations.run_agent() replays from
an existing recording if one is already on disk -- either the sandbox's own
prior recording, or (W70) a matching entry in the standalone agent-snapshot
corpus -- and only makes a live call (recording into the sandbox partition)
when neither exists. Otherwise it passes straight through to the real, live
run_agent, unchanged from before this story."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import recommendations
from src.agent.snapshot_store import league_base_dir, SnapshotMissingError

_FIXED_MODEL_SELECTION_HASH = "test-hash-abc123"


_MATCH_INFO = {"home_team": "Arsenal", "away_team": "Everton", "date": "2026-03-01", "league": "E0"}
_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-03-01", "league": "E0"},
    "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
    "limitations": [], "prediction_basis": "market_odds_only",
}
_MATCH_KEY = f"{_MATCH_INFO['home_team']}__{_MATCH_INFO['away_team']}__{_MATCH_INFO['date']}"


def test_lookup_corpus_match_id_resolves_via_team_mapping_and_raw_matches(monkeypatch, tmp_path):
    """A webapp-side team name (potentially non-canonical, e.g. from
    football-data.org) must resolve through TeamNameMapper before matching
    raw_matches' canonical names."""
    class FakeCursor:
        def fetchone(self):
            return ("real-match-id-123",)

    class FakeConnection:
        def execute(self, query, params):
            assert "raw_matches" in query
            # canonical names, not whatever the caller passed in
            assert params == ["E0", "2026-03-01", "Arsenal", "Everton"]
            return FakeCursor()

    class FakeConnectionCtx:
        def __enter__(self):
            return FakeConnection()
        def __exit__(self, *a):
            return False

    with patch("app.backend.recommendations.TeamNameMapper") as mock_mapper_cls, \
         patch("app.backend.recommendations.DuckDBManager") as mock_db_cls:
        mock_mapper = mock_mapper_cls.return_value
        mock_mapper.map_team.side_effect = lambda name: name  # already canonical in this test
        mock_db_cls.return_value.connection.return_value = FakeConnectionCtx()

        result = recommendations._lookup_corpus_match_id("Arsenal", "Everton", "2026-03-01", "E0")

    assert result == "real-match-id-123"


def test_lookup_corpus_match_id_returns_none_for_no_match(monkeypatch):
    class FakeCursor:
        def fetchone(self):
            return None

    class FakeConnectionCtx:
        def __enter__(self):
            return type("C", (), {"execute": lambda self, q, p: FakeCursor()})()
        def __exit__(self, *a):
            return False

    with patch("app.backend.recommendations.TeamNameMapper") as mock_mapper_cls, \
         patch("app.backend.recommendations.DuckDBManager") as mock_db_cls:
        mock_mapper_cls.return_value.map_team.side_effect = lambda name: name
        mock_db_cls.return_value.connection.return_value = FakeConnectionCtx()

        result = recommendations._lookup_corpus_match_id("Nobody FC", "Nowhere United", "2026-03-01", "E0")

    assert result is None


def test_lookup_corpus_match_id_returns_none_when_league_is_missing():
    # No DB call at all when league is unresolved (W03's gate_league can
    # produce a match_info with no league key) -- nothing to look up against.
    assert recommendations._lookup_corpus_match_id("Arsenal", "Everton", "2026-03-01", None) is None


def test_lookup_corpus_match_id_returns_none_instead_of_raising_on_a_db_error(monkeypatch):
    """Covers the try/except added to satisfy this function's own
    documented contract: DuckDBManager()/load_settings() can raise on a
    missing or invalid config.yaml, and conn.execute() can raise a
    duckdb.Error (e.g. no raw_matches table yet in some environment) --
    neither should ever surface as an uncaught exception out of a "just
    tell me if there's a corpus entry" lookup."""
    with patch("app.backend.recommendations.TeamNameMapper") as mock_mapper_cls, \
         patch("app.backend.recommendations.DuckDBManager") as mock_db_cls:
        mock_mapper_cls.return_value.map_team.side_effect = lambda name: name
        mock_db_cls.side_effect = RuntimeError("config.yaml not found")

        result = recommendations._lookup_corpus_match_id("Arsenal", "Everton", "2026-03-01", "E0")

    assert result is None


def test_lookup_corpus_match_id_returns_none_instead_of_raising_on_a_query_error(monkeypatch):
    """Same contract, but the failure happens later -- inside conn.execute()
    itself (e.g. a duckdb.Error because raw_matches doesn't exist yet)."""
    class FakeConnection:
        def execute(self, query, params):
            raise RuntimeError("no such table: raw_matches")

    class FakeConnectionCtx:
        def __enter__(self):
            return FakeConnection()
        def __exit__(self, *a):
            return False

    with patch("app.backend.recommendations.TeamNameMapper") as mock_mapper_cls, \
         patch("app.backend.recommendations.DuckDBManager") as mock_db_cls:
        mock_mapper_cls.return_value.map_team.side_effect = lambda name: name
        mock_db_cls.return_value.connection.return_value = FakeConnectionCtx()

        result = recommendations._lookup_corpus_match_id("Arsenal", "Everton", "2026-03-01", "E0")

    assert result is None


@pytest.fixture()
def _sandbox_snapshot_tmp_dirs(tmp_path, monkeypatch):
    """Points both the sandbox partition and the corpus base dir at a fresh
    tmp_path per test, so no test reads or writes the real
    data/agent_snapshots/ tree. Also stubs out the corpus lookup to return
    None by default (no league-wide DB dependency for tests that don't care
    about corpus replay specifically -- those tests patch it explicitly).

    Not autouse: the three test_lookup_corpus_match_id_* tests above exercise
    _lookup_corpus_match_id itself and must not have it stubbed out from
    under them, so every run_agent test that needs this fixture requests it
    explicitly by name instead."""
    sandbox_dir = tmp_path / "sandbox"
    corpus_dir = tmp_path / "corpus"
    sandbox_dir.mkdir()
    corpus_dir.mkdir()
    monkeypatch.setattr(recommendations, "_SANDBOX_SNAPSHOT_BASE_DIR", sandbox_dir)
    monkeypatch.setattr(recommendations, "_CORPUS_BASE_DIR", corpus_dir)
    # BUG-036: pinned to a fixed value instead of the real config file's
    # content, so these tests don't depend on (or break when someone edits)
    # the real config/model_selection.yaml.
    monkeypatch.setattr(recommendations, "_current_model_selection_hash", lambda: _FIXED_MODEL_SELECTION_HASH)
    with patch("app.backend.recommendations._lookup_corpus_match_id", return_value=None):
        yield sandbox_dir, corpus_dir


def _mark_complete(base_dir: Path, match_id: str, model_selection_hash: str = _FIXED_MODEL_SELECTION_HASH) -> None:
    """Writes a marker in the current (BUG-036-fixed) shape -- fresh under
    _FIXED_MODEL_SELECTION_HASH by default, matching what _sandbox_snapshot_tmp_dirs
    patches _current_model_selection_hash() to return."""
    match_dir = base_dir / match_id
    match_dir.mkdir(parents=True, exist_ok=True)
    (match_dir / "_complete.json").write_text(json.dumps({"model_selection_hash": model_selection_hash}))


def test_passes_through_to_the_real_run_agent_when_sandbox_mode_is_off(monkeypatch):
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION) as mock_run, \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        result = recommendations.run_agent(_MATCH_INFO)

    assert result == _RECOMMENDATION
    mock_run.assert_called_once_with(_MATCH_INFO, config=None)
    mock_configure.assert_not_called()


def test_no_existing_recording_anywhere_uses_record_mode_into_the_sandbox_partition(monkeypatch, _sandbox_snapshot_tmp_dirs):
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs
    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["record", "live"]
    record_call = mock_configure.call_args_list[0]
    assert record_call.kwargs["base_dir"] == sandbox_dir
    assert record_call.kwargs["match_id"] == _MATCH_KEY


def test_a_successful_record_pass_is_replayed_on_the_very_next_call(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """W70: the actual regression test for the restart-persistence fix --
    unlike test_a_prior_sandbox_recording_on_disk_is_replayed_..., which
    fabricates the marker by hand, this one drives a REAL record-mode pass
    through configure_snapshot_store/SnapshotStore.wrap (only _real_run_agent
    itself is mocked, at the LLM/agent boundary) and confirms the write
    actually happens, closing the exact gap that let the marker-never-
    written bug ship undetected."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs

    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION):
        recommendations.run_agent(_MATCH_INFO)  # first call: real record pass, real disk write

    marker = sandbox_dir / _MATCH_KEY / "_complete.json"
    assert marker.exists()

    mode, match_id, base_dir = recommendations._select_sandbox_snapshot_source(_MATCH_INFO)
    assert mode == "replay"
    assert match_id == _MATCH_KEY
    assert base_dir == sandbox_dir


def test_a_prior_sandbox_recording_on_disk_is_replayed_even_from_a_fresh_process(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """W70: the actual regression test for the restart-persistence bug --
    simulates a fresh process by never touching any in-memory state at all,
    only creating the on-disk marker a prior process would have left."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs
    _mark_complete(sandbox_dir, _MATCH_KEY)

    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live"]
    replay_call = mock_configure.call_args_list[0]
    assert replay_call.kwargs["base_dir"] == sandbox_dir
    assert replay_call.kwargs["match_id"] == _MATCH_KEY


def test_a_stale_sandbox_recording_falls_through_to_a_fresh_record_pass(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """BUG-036: a recording made under a *different* model_selection.yaml
    than the one currently on disk (e.g. a model path landed inconsistent,
    then got fixed) must not be trusted for replay -- it's treated the same
    as no recording at all, so this match gets one fresh live re-record
    instead of silently serving whatever the stale recording captured
    forever."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs
    _mark_complete(sandbox_dir, _MATCH_KEY, model_selection_hash="some-older-hash")

    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["record", "live"]  # not replay -- the stale marker didn't count


def test_a_marker_with_no_hash_field_at_all_is_also_treated_as_stale(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """Every marker recorded before this fix landed is shaped exactly like
    this (bare "{}") -- must self-heal via one fresh re-record, not error
    or (worse) silently keep replaying forever, which is the exact
    structural gap BUG-036 documents as unfixed."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs
    match_dir = sandbox_dir / _MATCH_KEY
    match_dir.mkdir(parents=True, exist_ok=True)
    (match_dir / "_complete.json").write_text("{}")  # pre-fix marker shape

    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["record", "live"]


def test_a_fresh_record_pass_writes_the_current_model_selection_hash_into_the_marker(monkeypatch, _sandbox_snapshot_tmp_dirs):
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs

    with patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION):
        recommendations.run_agent(_MATCH_INFO)

    marker = json.loads((sandbox_dir / _MATCH_KEY / "_complete.json").read_text())
    assert marker["model_selection_hash"] == _FIXED_MODEL_SELECTION_HASH
    assert "recorded_at" in marker


def test_a_matching_corpus_entry_is_replayed_when_no_sandbox_recording_exists(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """W70's actual new capability: a fixture with a complete recording in
    the standalone agent-snapshot corpus (not the sandbox's own partition)
    replays from there, making zero live calls."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    _, corpus_dir = _sandbox_snapshot_tmp_dirs
    league_dir = league_base_dir("E0", base_dir=corpus_dir)
    _mark_complete(league_dir, "real-match-id-123")

    with patch("app.backend.recommendations._lookup_corpus_match_id", return_value="real-match-id-123"), \
         patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live"]
    replay_call = mock_configure.call_args_list[0]
    assert replay_call.kwargs["base_dir"] == league_dir
    assert replay_call.kwargs["match_id"] == "real-match-id-123"


def test_sandbox_partition_is_checked_before_the_corpus(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """If both exist, the sandbox's own (possibly more recent / more
    relevant to this exact session) recording wins -- the corpus is a
    fallback, not a replacement."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, corpus_dir = _sandbox_snapshot_tmp_dirs
    _mark_complete(sandbox_dir, _MATCH_KEY)
    league_dir = league_base_dir("E0", base_dir=corpus_dir)
    _mark_complete(league_dir, "real-match-id-123")

    with patch("app.backend.recommendations._lookup_corpus_match_id", return_value="real-match-id-123"), \
         patch("app.backend.recommendations._real_run_agent", return_value=_RECOMMENDATION), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        recommendations.run_agent(_MATCH_INFO)

    replay_call = mock_configure.call_args_list[0]
    assert replay_call.kwargs["base_dir"] == sandbox_dir
    assert replay_call.kwargs["match_id"] == _MATCH_KEY


def test_configure_snapshot_store_resets_to_live_on_agent_exception(monkeypatch, _sandbox_snapshot_tmp_dirs):
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    with patch("app.backend.recommendations._real_run_agent", side_effect=RuntimeError("boom")), \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        with pytest.raises(RuntimeError):
            recommendations.run_agent(_MATCH_INFO)

    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["record", "live"]


def test_run_agent_retries_once_in_record_mode_after_a_replay_snapshot_miss(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """W43 (preserved): a replay-mode SnapshotMissingError (e.g. the LLM
    phrased its optional follow-up web_search query differently than the
    recorded run) must not surface as a raw 500 -- falls back to a fresh
    record-mode pass into the sandbox partition, regardless of whether the
    replay attempt was against the sandbox partition or the corpus."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs
    _mark_complete(sandbox_dir, _MATCH_KEY)

    miss = SnapshotMissingError("web_search", _MATCH_KEY, "deadbeef")
    with patch(
        "app.backend.recommendations._real_run_agent", side_effect=[miss, _RECOMMENDATION],
    ) as mock_run, patch("app.backend.recommendations.agent_tools.configure_snapshot_store") as mock_configure:
        result = recommendations.run_agent(_MATCH_INFO)

    assert result == _RECOMMENDATION
    assert mock_run.call_count == 2
    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live", "record", "live"]
    retry_call = mock_configure.call_args_list[2]
    assert retry_call.kwargs["base_dir"] == sandbox_dir
    assert retry_call.kwargs["match_id"] == _MATCH_KEY


def test_run_agent_does_not_swallow_a_genuinely_different_exception_on_retry(monkeypatch, _sandbox_snapshot_tmp_dirs):
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    sandbox_dir, _ = _sandbox_snapshot_tmp_dirs
    _mark_complete(sandbox_dir, _MATCH_KEY)

    miss = SnapshotMissingError("web_search", _MATCH_KEY, "deadbeef")
    with patch(
        "app.backend.recommendations._real_run_agent", side_effect=[miss, RuntimeError("real failure")],
    ), patch("app.backend.recommendations.agent_tools.configure_snapshot_store"):
        with pytest.raises(RuntimeError, match="real failure"):
            recommendations.run_agent(_MATCH_INFO)


def test_run_agent_does_not_retry_a_snapshot_miss_that_happens_during_record_mode(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """A SnapshotMissingError should only trigger the record-mode fallback
    when it originates from a *replay*-mode call -- if no recording exists
    anywhere (this call starts in record mode) and somehow still raises it,
    it must propagate uncaught rather than retry forever."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    miss = SnapshotMissingError("web_search", _MATCH_KEY, "deadbeef")
    with patch("app.backend.recommendations._real_run_agent", side_effect=miss) as mock_run, \
         patch("app.backend.recommendations.agent_tools.configure_snapshot_store"):
        with pytest.raises(SnapshotMissingError):
            recommendations.run_agent(_MATCH_INFO)

    mock_run.assert_called_once()


def test_two_concurrent_requests_for_the_same_match_do_not_both_record(monkeypatch, _sandbox_snapshot_tmp_dirs):
    """W70 follow-up: two near-simultaneous requests for the identical
    match must not both slip into record mode and both make a live call
    concurrently -- the second must block on the first's per-match lock
    until the first finishes (and writes the completion marker), then
    correctly find that marker on its own disk check and replay instead of
    also recording.

    Proving this needs more than a call-count assertion: call count alone
    can't distinguish "properly serialized" from "raced but both eventually
    ran anyway" -- two unsynchronized calls still add up to 2. Instead this
    pins thread 1 mid-call (blocked on release_thread1), then explicitly
    checks, while thread 1 is still blocked, that thread 2 has NOT yet
    entered its own _real_run_agent call -- i.e. thread 2 is provably still
    waiting on the lock, not racing ahead to its own disk check. This is
    exactly the scenario the old, pre-Task-1 in-memory-set lock was too
    weak to prevent (it only protected the read-check-then-add instant, not
    the multi-second call itself), and exactly what has zero protection at
    all without _lock_for_match."""
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")

    call_count = {"n": 0}
    call_count_guard = threading.Lock()
    thread1_entered_call = threading.Event()
    release_thread1 = threading.Event()

    def slow_real_run_agent(match_info, config=None):
        with call_count_guard:
            call_count["n"] += 1
            my_index = call_count["n"]
        if my_index == 1:
            thread1_entered_call.set()
            assert release_thread1.wait(timeout=5), "test setup deadlocked"
        return _RECOMMENDATION

    with patch("app.backend.recommendations._real_run_agent", side_effect=slow_real_run_agent):
        t1 = threading.Thread(target=recommendations.run_agent, args=(_MATCH_INFO,))
        t1.start()
        assert thread1_entered_call.wait(timeout=5), "thread 1 never reached its agent call"

        t2 = threading.Thread(target=recommendations.run_agent, args=(_MATCH_INFO,))
        t2.start()

        # Thread 1 is deliberately still blocked inside its call right now
        # (holding the per-match lock, on fixed code). Give thread 2 a real
        # window to misbehave: if the lock were missing, thread 2 would sail
        # straight through _select_sandbox_snapshot_source (no marker on
        # disk yet -- thread 1 hasn't written it) into its own
        # _real_run_agent call, bumping call_count to 2 immediately.
        time.sleep(0.3)
        assert call_count["n"] == 1, (
            "thread 2 entered its own agent call while thread 1 was still mid-call -- "
            "the per-match lock did not serialize the two requests"
        )

        release_thread1.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

    assert not t1.is_alive() and not t2.is_alive()
    assert call_count["n"] == 2
