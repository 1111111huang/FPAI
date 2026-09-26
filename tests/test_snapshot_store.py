"""Tests for SnapshotStore record/replay/live interception (A09)."""
from __future__ import annotations

import json
import threading

import pytest
from langchain_core.runnables.config import ContextThreadPoolExecutor

from src.agent.snapshot_store import SnapshotMissingError, SnapshotRecordingDegraded, SnapshotStore, league_base_dir


def test_league_base_dir_appends_uppercased_league(tmp_path):
    assert league_base_dir("E0", base_dir=tmp_path) == tmp_path / "E0"


def test_league_base_dir_normalizes_case(tmp_path):
    assert league_base_dir("swe", base_dir=tmp_path) == tmp_path / "SWE"


def test_league_base_dir_falls_back_to_unknown_for_none(tmp_path):
    assert league_base_dir(None, base_dir=tmp_path) == tmp_path / "unknown"


def test_league_base_dir_falls_back_to_unknown_for_empty_string(tmp_path):
    assert league_base_dir("", base_dir=tmp_path) == tmp_path / "unknown"
    assert league_base_dir("   ", base_dir=tmp_path) == tmp_path / "unknown"


def test_league_base_dir_different_leagues_are_isolated(tmp_path):
    """BUG-022: E0 and SWE must never resolve to the same directory."""
    assert league_base_dir("E0", base_dir=tmp_path) != league_base_dir("SWE", base_dir=tmp_path)


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


# ---------------------------------------------------------------------------
# Canonicalization: replay tolerant of superficial LLM tool-call argument
# drift (whitespace, case, float precision) without invalidating the
# existing raw-hash-keyed corpus (BUG-072 just re-backfilled it today).
# ---------------------------------------------------------------------------

def test_canonical_key_ignores_whitespace_and_case_differences():
    store = SnapshotStore()
    a = store.canonical_key_for({"query": "Man City  injury news"})
    b = store.canonical_key_for({"query": "man city injury news"})
    assert a == b


def test_canonical_key_rounds_float_precision_noise():
    store = SnapshotStore()
    a = store.canonical_key_for({"odds": 1.8000000123})
    b = store.canonical_key_for({"odds": 1.8})
    assert a == b


def test_canonical_key_still_distinguishes_genuinely_different_inputs():
    store = SnapshotStore()
    a = store.canonical_key_for({"query": "Man City injury news"})
    b = store.canonical_key_for({"query": "Man City lineup news"})
    assert a != b


def test_canonical_key_differs_from_the_raw_key_for_non_canonical_input():
    """Confirms canonicalization is actually doing something -- these two
    keys must NOT collide for typical, non-canonical-form input, or every
    assertion above would be trivially true for the wrong reason."""
    store = SnapshotStore()
    inputs = {"query": "Man City  injury news"}
    assert store.canonical_key_for(inputs) != store.key_for(inputs)


def test_record_writes_under_the_canonical_key(tmp_path):
    """New recordings use the canonical key going forward, not the old raw
    key -- the corpus organically becomes more replay-resilient as it's
    extended, with no bulk migration needed."""
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("match-123")
    inputs = {"query": "Man City  injury news"}

    store.wrap("web_search", lambda **kw: "resp")(**inputs)

    canonical_key = store.canonical_key_for(inputs)
    assert (tmp_path / "match-123" / f"web_search_{canonical_key}.json").exists()


def test_replay_matches_a_recording_despite_whitespace_and_case_drift(tmp_path):
    """The concrete scenario this exists for: the LLM re-generates the same
    semantic tool call with different capitalization/whitespace on a later
    run (temperature=0 does not guarantee byte-identical output) -- replay
    must still find the recording rather than raising SnapshotMissingError."""
    record_store = SnapshotStore(base_dir=tmp_path)
    record_store.set_mode("record")
    record_store.set_match("match-123")
    record_store.wrap("web_search", lambda **kw: "the-response")(query="Man City  injury news")

    replay_store = SnapshotStore(base_dir=tmp_path)
    replay_store.set_mode("replay")
    replay_store.set_match("match-123")

    def fail_if_called(**kwargs):
        raise AssertionError("live function must not be called during replay")

    result = replay_store.wrap("web_search", fail_if_called)(query="man city injury news")
    assert result == "the-response"


def test_replay_still_finds_a_pre_canonicalization_raw_keyed_recording(tmp_path):
    """Backward compatibility: a snapshot file written before canonicalization
    existed (keyed only by the raw, non-canonicalized hash -- exactly what
    every match in the corpus BUG-072 just re-backfilled looks like on disk
    today) must still replay correctly, with no migration or re-recording."""
    store = SnapshotStore(base_dir=tmp_path)
    inputs = {"query": "Man City  injury news"}
    raw_key = store.key_for(inputs)
    match_dir = tmp_path / "match-123"
    match_dir.mkdir(parents=True)
    (match_dir / f"web_search_{raw_key}.json").write_text(
        json.dumps({"tool": "web_search", "inputs": inputs, "response": "old-response", "recorded_at": "x"}),
        encoding="utf-8",
    )

    replay_store = SnapshotStore(base_dir=tmp_path)
    replay_store.set_mode("replay")
    replay_store.set_match("match-123")

    def fail_if_called(**kwargs):
        raise AssertionError("live function must not be called during replay")

    # Same query, exact same phrasing as recorded -- canonical key differs
    # from the raw key on disk, so this only succeeds via the fallback path.
    result = replay_store.wrap("web_search", fail_if_called)(query="Man City  injury news")
    assert result == "old-response"


def test_record_missing_reuses_a_pre_canonicalization_raw_keyed_recording_without_live_fetch(tmp_path):
    """record_missing's own 'do we already have this' check must also
    consult the raw-key fallback -- otherwise every pre-canonicalization
    recording would look 'missing' and get needlessly re-fetched (and
    re-billed) the next time --backfill-missing runs over it."""
    store = SnapshotStore(base_dir=tmp_path)
    inputs = {"query": "Man City  injury news"}
    raw_key = store.key_for(inputs)
    match_dir = tmp_path / "match-123"
    match_dir.mkdir(parents=True)
    (match_dir / f"web_search_{raw_key}.json").write_text(
        json.dumps({"tool": "web_search", "inputs": inputs, "response": "old-response", "recorded_at": "x"}),
        encoding="utf-8",
    )

    backfill_store = SnapshotStore(base_dir=tmp_path)
    backfill_store.set_mode("record_missing")
    backfill_store.set_match("match-123")

    def fail_if_called(**kwargs):
        raise AssertionError("must reuse the existing raw-keyed recording, not live-fetch")

    result = backfill_store.wrap("web_search", fail_if_called)(**inputs)
    assert result == "old-response"


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


def test_tool_mode_overrides_default_empty(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    assert store.tool_mode_overrides == {}


def test_tool_mode_override_takes_precedence_over_global_mode(tmp_path):
    """Built for the 'refresh model without re-fetching Tavily' use case:
    global mode=replay (frozen web_search/resolve_competition), but
    forecast_league overridden to record so it re-invokes the (new) model
    live and overwrites its snapshot file, while an un-overridden tool in
    the same run still replays from the existing recording."""
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("match-123")
    store.wrap("web_search", lambda **kw: "original-search-response")(query="q")
    store.wrap("forecast_league", lambda **kw: "original-forecast-response")(home="A", away="B")

    store.set_mode("replay")
    store.set_tool_mode_overrides({"forecast_league": "record"})

    def new_forecast(**kwargs):
        return "refreshed-forecast-response"

    # forecast_league: overridden to record -- calls the live fn, overwrites the file.
    result = store.wrap("forecast_league", new_forecast)(home="A", away="B")
    assert result == "refreshed-forecast-response"

    # web_search: no override -- still replays the original recording untouched.
    def fail_if_called(**kwargs):
        raise AssertionError("web_search must still replay, not call live fn")

    replayed = store.wrap("web_search", fail_if_called)(query="q")
    assert replayed == "original-search-response"

    saved = json.loads(next((tmp_path / "match-123").glob("forecast_league_*.json")).read_text())
    assert saved["response"] == "refreshed-forecast-response"


def test_tool_mode_override_invalid_mode_raises(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown snapshot mode"):
        store.set_tool_mode_overrides({"forecast_league": "bogus"})


def test_record_missing_mode_replays_an_existing_key_without_calling_fn(tmp_path):
    """BUG-072/A56/A115 gap: research_node gained a new deterministic web_search
    query that pre-A115 snapshot corpora never recorded, and replay mode's
    hard-fail on ANY missing key made the whole match unreplayable even though
    most of its keys were still fine. record_missing lets a partially-stale
    match reuse every key it already has and only live-fetch the ones that
    are new."""
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("match-123")
    store.wrap("web_search", lambda **kw: "original-response")(query="existing query")

    store.set_mode("record_missing")

    def fail_if_called(**kwargs):
        raise AssertionError("an existing key must replay, not call the live fn")

    result = store.wrap("web_search", fail_if_called)(query="existing query")
    assert result == "original-response"


def test_record_missing_mode_records_a_new_key_live(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record_missing")
    store.set_match("match-123")
    calls = []

    def fn(**kwargs):
        calls.append(kwargs)
        return "new-response"

    result = store.wrap("web_search", fn)(query="brand new query")
    assert result == "new-response"
    assert calls == [{"query": "brand new query"}]

    files = list((tmp_path / "match-123").glob("web_search_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["response"] == "new-response"


def test_record_mode_refuses_to_persist_a_degraded_sentinel_response(tmp_path):
    """Real corruption found live (BUG-072, 2026-09-25): _web_search_impl
    degrades to a 'TOOL_PERMANENTLY_UNAVAILABLE: ...' string when every API
    key fails (network timeout, quota) instead of raising -- record mode was
    happily writing that string to disk as if it were a genuine recorded
    answer. Future replays then silently fed the LLM a fake 'tool
    unavailable' message as real evidence forever, with no error anywhere.
    Found 6 matches in D1's original corpus (recorded weeks ago) and 10
    fresh ones from this session's own F1 backfill run already corrupted
    this way."""
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("match-123")

    def degraded_fn(**kwargs):
        return "TOOL_PERMANENTLY_UNAVAILABLE: web_search failed (timeout). Do NOT call again."

    with pytest.raises(SnapshotRecordingDegraded):
        store.wrap("web_search", degraded_fn)(query="q")

    assert list((tmp_path / "match-123").glob("web_search_*.json")) == []


def test_record_missing_mode_refuses_to_persist_a_degraded_sentinel_response(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record_missing")
    store.set_match("match-123")

    def degraded_fn(**kwargs):
        return "TOOL_PERMANENTLY_UNAVAILABLE: web_search has no API key configured. Do NOT call again."

    with pytest.raises(SnapshotRecordingDegraded):
        store.wrap("web_search", degraded_fn)(query="q")

    assert list((tmp_path / "match-123").glob("web_search_*.json")) == []


def test_mode_and_match_propagate_into_context_thread_pool_executor(tmp_path):
    """LangGraph's ToolNode runs every tool call (even a single one) via
    get_executor_for_config(), which returns a ContextThreadPoolExecutor —
    not a plain threading.Thread. That executor explicitly copies the calling
    thread's contextvars.Context into the worker (langchain_core.runnables.config.
    ContextThreadPoolExecutor.submit/map use copy_context().run(...)). configure_snapshot_store()
    is always called on the thread that then invokes graph.invoke(), so the
    mode/match set there must be visible inside this executor's workers, or
    every tool call silently runs in "live" mode regardless of what the
    caller configured (the actual bug: record mode wrote zero snapshot files,
    and replay mode never raised SnapshotMissingError because it never
    replayed anything — see agent_techspec.md Section 18)."""
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("main-thread-match")

    with ContextThreadPoolExecutor() as executor:
        seen_mode, seen_match_id = list(executor.map(lambda _: (store.mode, store.match_id), [None]))[0]

    assert (seen_mode, seen_match_id) == ("record", "main-thread-match")
