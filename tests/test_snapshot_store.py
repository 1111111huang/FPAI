"""Tests for SnapshotStore record/replay/live interception (A09)."""
from __future__ import annotations

import json
import threading

import pytest

from src.agent.snapshot_store import SnapshotMissingError, SnapshotStore


def test_live_mode_passes_through_without_writing(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("live")
    calls = []

    def fn(**kwargs):
        calls.append(kwargs)
        return "live-response"

    result = store.wrap("web_search", fn)(query="man city odds")
    assert result == "live-response"
    assert calls == [{"query": "man city odds"}]
    assert list(tmp_path.rglob("*.json")) == []


def test_record_mode_writes_snapshot_file(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("match-123")

    def fn(**kwargs):
        return "recorded-response"

    result = store.wrap("web_search", fn)(query="man city odds")
    assert result == "recorded-response"

    files = list((tmp_path / "match-123").glob("web_search_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["tool"] == "web_search"
    assert payload["inputs"] == {"query": "man city odds"}
    assert payload["response"] == "recorded-response"
    assert "recorded_at" in payload


def test_replay_mode_reads_recorded_response(tmp_path):
    record_store = SnapshotStore(base_dir=tmp_path)
    record_store.set_mode("record")
    record_store.set_match("match-123")
    record_store.wrap("web_search", lambda **kw: "the-response")(query="q")

    replay_store = SnapshotStore(base_dir=tmp_path)
    replay_store.set_mode("replay")
    replay_store.set_match("match-123")

    def fail_if_called(**kwargs):
        raise AssertionError("live function must not be called during replay")

    result = replay_store.wrap("web_search", fail_if_called)(query="q")
    assert result == "the-response"


def test_replay_missing_snapshot_raises(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("replay")
    store.set_match("match-999")

    with pytest.raises(SnapshotMissingError) as exc_info:
        store.wrap("web_search", lambda **kw: "x")(query="q")

    assert exc_info.value.tool == "web_search"
    assert exc_info.value.match_id == "match-999"


def test_key_is_deterministic_regardless_of_kwarg_order(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    key_a = store.key_for({"a": 1, "b": 2})
    key_b = store.key_for({"b": 2, "a": 1})
    assert key_a == key_b


def test_record_requires_match_id(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    with pytest.raises(ValueError, match="set_match"):
        store.wrap("web_search", lambda **kw: "x")(query="q")


def test_invalid_mode_raises():
    store = SnapshotStore()
    with pytest.raises(ValueError, match="Unknown snapshot mode"):
        store.set_mode("bogus")


def test_mode_and_match_are_thread_local(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("main-thread-match")

    other_thread_mode = []

    def worker():
        # New thread should NOT inherit the main thread's mode/match_id
        other_thread_mode.append(store.mode)
        other_thread_mode.append(store.match_id)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert other_thread_mode == ["live", None]
    # Main thread's state must be unaffected by the other thread
    assert store.mode == "record"
    assert store.match_id == "main-thread-match"
