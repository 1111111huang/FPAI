"""W37: wires SnapshotStore record/replay into the sandbox agent-invocation
path. When sandbox mode is active, recommendations.run_agent() replays from
an existing recording if one is already on disk -- either the sandbox's own
prior recording, or (W70) a matching entry in the standalone agent-snapshot
corpus -- and only makes a live call (recording into the sandbox partition)
when neither exists. Otherwise it passes straight through to the real, live
run_agent, unchanged from before this story."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import recommendations
from src.agent.snapshot_store import league_base_dir, SnapshotMissingError


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
    with patch("app.backend.recommendations._lookup_corpus_match_id", return_value=None):
        yield sandbox_dir, corpus_dir


def _mark_complete(base_dir: Path, match_id: str) -> None:
    match_dir = base_dir / match_id
    match_dir.mkdir(parents=True, exist_ok=True)
    (match_dir / "_complete.json").write_text("{}")


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
