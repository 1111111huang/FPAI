# Scenario Testing: Snapshot Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the webapp's sandbox mode replay recommendations from the actively-recording `agent-snapshot` corpus (E0/SWE, Aug 2025–May 2026), fix its own recordings not surviving a backend restart, fix SWE fixture discovery for arbitrary historical dates, and add a repeatable runbook that verifies all of it across the real date range.

**Architecture:** Replace the sandbox's in-memory "have I recorded this match this process" tracking with a disk-based check that looks in two places, in order — the sandbox's own partition first, then the `agent-snapshot` corpus (resolved via the existing `TeamNameMapper` + a `raw_matches` lookup). Add a `raw_matches`-backed historical SWE fixture source, since the Odds API SWE relies on today has no arbitrary-historical-date endpoint at all. Extend the sandbox-launch script's precompute step and add a new date-range-walking runbook script on top of both fixes.

**Tech Stack:** Python (FastAPI backend, DuckDB via `src.utils.db_manager.DuckDBManager`), pytest, existing `src/agent/snapshot_store.py`/`src/ingestion/common/team_mapping.py` infrastructure — no new dependencies.

**Design doc:** `docs/superpowers/specs/2026-07-24-scenario-testing-snapshot-integration-design.md`

---

## Before you start

- `app/backend/recommendations.py` — the file Task 1 rewrites. Read `run_agent()`, `_run_agent_in_mode()`, `_composite_match_key()`, and the `_sandbox_recorded_matches`/`_recorded_matches_lock` module state before touching it.
- `app/backend/tests/test_sandbox_agent_snapshot.py` — the existing test suite for this exact code path. **Task 1 rewrites most of it**, not just adds to it — every existing test manipulates `recommendations._sandbox_recorded_matches` directly (`.clear()`, `.add()`, membership asserts), which this plan removes entirely. Read the whole file before starting; the rewrite must preserve every scenario it currently covers (pass-through when sandbox off, record-then-replay, exception safety, the W43 retry-on-miss fallback, retry-not-triggered-from-record-mode), just expressed against disk state instead of a Python set.
- `src/agent/snapshot_store.py` — `league_base_dir()` (the `<base_dir>/<LEAGUE>/` helper the `agent-snapshot` CLI already uses). **Correction (found during Task 1 code review, not caught before implementation started):** `SnapshotStore` itself has no knowledge of `_complete.json` at all — that marker is written *exclusively* by `main.py`'s standalone `agent-snapshot` CLI (`run_agent_snapshot()`, `main.py:1150-1152`) after a successful record pass, never by `SnapshotStore.wrap()` itself (which only writes per-tool-call `{tool}_{key}.json` files). Any code that wants to check-or-set "is this match fully recorded" against the sandbox's own partition must read *and write* that marker itself — reading it without also writing it (as Task 1 initially did, caught and fixed in code review) silently breaks same-process reuse, not just restart persistence.
- `src/agent/tools.py` — `configure_snapshot_store(mode, match_id=None, match_date=None, base_dir=None)`. `base_dir` is sticky-if-omitted; pass it explicitly whenever you need to guarantee which namespace you're pointing at (as `_run_agent_in_mode` already does today).
- `app/backend/eod_batch.py:64-77` — `odds_lookup()`/`match_odds()`, the existing precedent for "map a webapp-side team name to canonical form via `TeamNameMapper`, then look it up against a canonical-keyed structure." Task 1's corpus lookup follows this exact shape.
- `raw_matches` schema (confirmed live): `match_id VARCHAR`, `league VARCHAR`, `date TIMESTAMP`, `home_team VARCHAR`, `away_team VARCHAR`, `fthg INTEGER`, `ftag INTEGER` (plus odds/stats columns not needed here). Team names in this table are already canonical (the ML engine's own ingestion target) — a webapp-side team name must go through `TeamNameMapper.map_team()` before comparing against it.
- `main.py:1084-1130` (`run_agent_snapshot`, the `agent-snapshot` CLI) — confirms the corpus's on-disk convention precisely: `league_base_dir(row["league"], base_dir=DEFAULT_BASE_DIR) / row["match_id"] / "_complete.json"`.

---

### Task 1: Bridge sandbox replay to the agent-snapshot corpus, fix restart persistence (W70)

**Files:**
- Modify: `app/backend/recommendations.py`
- Modify (substantial rewrite, not just additions): `app/backend/tests/test_sandbox_agent_snapshot.py`

- [ ] **Step 1: Write the new corpus-lookup helper, with a failing test first**

Add to `app/backend/tests/test_sandbox_agent_snapshot.py` (new tests, at the top of the file after the existing imports/fixtures — these test the new helper in isolation before Step 3 wires it into `run_agent`):

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_sandbox_agent_snapshot.py -k "lookup_corpus" -v`
Expected: FAIL — `AttributeError: module 'app.backend.recommendations' has no attribute '_lookup_corpus_match_id'` (and no `TeamNameMapper`/`DuckDBManager` imported yet either).

- [ ] **Step 3: Implement `_lookup_corpus_match_id` and the new imports**

In `app/backend/recommendations.py`, add these imports (alongside the existing ones near the top of the file). The file already has `from src.agent.snapshot_store import SnapshotMissingError` — extend that line to also import `league_base_dir` (the same `<base_dir>/<LEAGUE>/` helper the `agent-snapshot` CLI uses, `main.py:1126`) rather than adding a second import line for the same module:

```python
from src.agent.snapshot_store import league_base_dir, SnapshotMissingError
from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.db_manager import DuckDBManager
```

Add near the existing `_SANDBOX_SNAPSHOT_BASE_DIR` constant:

```python
_CORPUS_BASE_DIR = Path(__file__).parent.parent.parent / "data" / "agent_snapshots"
_TEAM_MAPPING_PATH = Path(__file__).parent.parent.parent / "config" / "team_mapping.json"
```

Add the new function (near `_composite_match_key`):

```python
def _lookup_corpus_match_id(home_team: str, away_team: str, date: str, league: str | None) -> str | None:
    """Resolves a fixture to the real raw_matches.match_id the standalone
    agent-snapshot CLI's corpus is keyed by, so the sandbox can replay from
    it directly. Team names come from whatever the fixtures API returned
    (football-data.org/Odds API), not necessarily the ML engine's canonical
    spelling -- mapped through TeamNameMapper first, same tool/pattern
    eod_batch.py's odds_lookup()/match_odds() already use for the identical
    class of problem (W06/BUG-015). Returns None on any kind of miss
    (unmapped team, no matching row, no league) -- never raises; a miss just
    means "no corpus entry, fall through to record," not an error."""
    if not league:
        return None
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    canonical_home = mapper.map_team(home_team)
    canonical_away = mapper.map_team(away_team)
    db = DuckDBManager()
    with db.connection(read_only=True) as conn:
        row = conn.execute(
            "SELECT match_id FROM raw_matches WHERE league = ? AND date = ? AND home_team = ? AND away_team = ?",
            [league, date, canonical_home, canonical_away],
        ).fetchone()
    return row[0] if row else None
```

- [ ] **Step 4: Run to verify the lookup tests pass**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_sandbox_agent_snapshot.py -k "lookup_corpus" -v`
Expected: all 3 pass.

- [ ] **Step 5: Write the failing tests for the rewritten mode-selection logic**

This step **replaces** the existing tests in `test_sandbox_agent_snapshot.py` that manipulate `_sandbox_recorded_matches` directly, since that module attribute is being removed in Step 7. Replace the entire file's test bodies (keep the module docstring, imports, and `_MATCH_INFO`/`_RECOMMENDATION`/`_MATCH_KEY` constants) with:

```python
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


@pytest.fixture(autouse=True)
def _sandbox_snapshot_tmp_dirs(tmp_path, monkeypatch):
    """Points both the sandbox partition and the corpus base dir at a fresh
    tmp_path per test, so no test reads or writes the real
    data/agent_snapshots/ tree. Also stubs out the corpus lookup to return
    None by default (no league-wide DB dependency for tests that don't care
    about corpus replay specifically -- those tests patch it explicitly)."""
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
```

- [ ] **Step 6: Run to verify the new/changed tests fail**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_sandbox_agent_snapshot.py -v`
Expected: FAIL — `run_agent` still uses the old in-memory-set logic, so `base_dir`/`match_id` won't match the new corpus-aware expectations, and `_sandbox_snapshot_tmp_dirs` patches attributes (`_CORPUS_BASE_DIR`) that don't exist yet.

- [ ] **Step 7: Rewrite `run_agent`'s mode-selection logic**

In `app/backend/recommendations.py`, delete `_sandbox_recorded_matches` and `_recorded_matches_lock` entirely (lines with `_sandbox_recorded_matches: set[str] = set()`, the `_recorded_matches_lock` comment+assignment, and `import threading` if nothing else in the file needs it — check first). Replace `_run_agent_in_mode` and `run_agent` with:

```python
def _run_agent_in_mode(mode: str, match_info: dict, config, match_id: str, base_dir: Path):
    """Configure the snapshot store for `mode` and run the real agent,
    always resetting the store to live mode afterward regardless of
    outcome."""
    agent_tools.configure_snapshot_store(
        mode, match_id=match_id, match_date=match_info.get("date"), base_dir=base_dir,
    )
    try:
        return _real_run_agent(match_info, config=config)
    finally:
        agent_tools.configure_snapshot_store("live")


def _select_sandbox_snapshot_source(match_info: dict) -> tuple[str, str, Path]:
    """W70: decides record vs replay -- and which namespace to use -- by
    checking disk directly, in priority order: (1) the sandbox's own prior
    recording for this exact match (fixes recordings not surviving a
    backend restart -- previously tracked in an in-memory set that started
    empty every process, so a fresh process always re-recorded and silently
    overwrote whatever was already there); (2) a matching complete entry in
    the standalone agent-snapshot corpus, resolved via
    _lookup_corpus_match_id; (3) otherwise, record fresh into the sandbox's
    own partition, unchanged from before this story. Returns
    (mode, match_id, base_dir)."""
    home_team = match_info.get("home_team")
    away_team = match_info.get("away_team")
    date = match_info.get("date")
    sandbox_match_id = _composite_match_key(home_team, away_team, date)

    if (_SANDBOX_SNAPSHOT_BASE_DIR / sandbox_match_id / "_complete.json").exists():
        return "replay", sandbox_match_id, _SANDBOX_SNAPSHOT_BASE_DIR

    league = match_info.get("league")
    corpus_match_id = _lookup_corpus_match_id(home_team, away_team, date, league)
    if corpus_match_id:
        corpus_league_dir = league_base_dir(league, base_dir=_CORPUS_BASE_DIR)
        if (corpus_league_dir / corpus_match_id / "_complete.json").exists():
            return "replay", corpus_match_id, corpus_league_dir

    return "record", sandbox_match_id, _SANDBOX_SNAPSHOT_BASE_DIR


def run_agent(match_info: dict, config=None):
    """W37/W70: routes through SnapshotStore record/replay when sandbox mode
    is active -- replaying from an existing recording (the sandbox's own, or
    W70's agent-snapshot corpus bridge) whenever one already exists on disk,
    recording fresh into the sandbox partition only when neither does.
    Otherwise passes straight through to the real, live run_agent.

    W43: a replay-mode SnapshotMissingError can happen even for a match
    that's genuinely already recorded -- SnapshotStore's replay lookup key
    is a hash of the tool call's exact input arguments (e.g. an LLM-chosen
    optional follow-up web_search query), and that specific text isn't
    reproducible run-to-run (agent_techspec.md Sec 18.6). Rather than let
    that 500 the request, fall back to one fresh record-mode pass into the
    sandbox partition for this request -- matching this codebase's "never
    assume the agent/its own optimizations hold, degrade gracefully"
    philosophy (W02/W15/W16's validate_and_degrade). Any other exception --
    including a second failure from the record-mode retry itself -- is not
    caught here and propagates uncaught, so this isn't a silent catch-all."""
    if not is_sandbox_mode():
        return _real_run_agent(match_info, config=config)

    mode, match_id, base_dir = _select_sandbox_snapshot_source(match_info)
    try:
        return _run_agent_in_mode(mode, match_info, config, match_id, base_dir)
    except SnapshotMissingError:
        if mode != "replay":
            raise
        _LOG.warning(
            "sandbox_agent_replay_miss | match=%s | retrying_in_record_mode", match_id,
        )
        sandbox_match_id = _composite_match_key(
            match_info.get("home_team"), match_info.get("away_team"), match_info.get("date"),
        )
        return _run_agent_in_mode("record", match_info, config, sandbox_match_id, _SANDBOX_SNAPSHOT_BASE_DIR)
```

Note the retry-on-miss always falls back to the sandbox's own partition (never retries into the corpus) — the webapp must never write into the standalone `agent-snapshot` CLI's corpus, which is owned by that separate tool.

- [ ] **Step 8: Run to verify all tests pass**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_sandbox_agent_snapshot.py -v`
Expected: all pass (11 tests: 3 lookup tests from Step 1 + 8 rewritten `run_agent` tests from Step 5).

- [ ] **Step 9: Run the full backend suite to confirm zero regressions elsewhere**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest tests/ app/backend/tests/ -q`
Expected: same pass count as baseline, no failures. (Check nothing else in the codebase references `recommendations._sandbox_recorded_matches` — `grep -rn "_sandbox_recorded_matches" --include="*.py" .` should return nothing after this task.)

- [ ] **Step 10: Commit**

```bash
git add app/backend/recommendations.py app/backend/tests/test_sandbox_agent_snapshot.py
git commit -m "feat(app): bridge sandbox replay to the agent-snapshot corpus, fix restart persistence

Replaces the in-memory 'have I recorded this match this process' set
(which started empty every process start, so a fresh process always
re-recorded and silently overwrote existing recordings) with a
disk-based check: the sandbox's own partition first, then a lookup
against the standalone agent-snapshot corpus via the fixture's real
raw_matches.match_id (resolved through the existing TeamNameMapper).
Lets sandbox scenario testing replay from the actively-recording
Anthropic-API-based corpus (A34) instead of making live calls."
```

---

### Task 2: Fix SWE historical fixture sourcing for arbitrary past dates (W71)

**Files:**
- Modify: `app/backend/sweden_fixtures_client.py`
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_sweden_fixtures_client.py`, `app/backend/tests/test_fixtures_endpoint.py`

- [ ] **Step 1: Write the failing test for the new historical-results function**

Add to `app/backend/tests/test_sweden_fixtures_client.py`:

```python
def test_historical_results_from_raw_matches_queries_and_normalizes(monkeypatch):
    """W71: unlike get_results() (live Odds API, only the last few real
    days), this must be able to return a real historical fixture for any
    date raw_matches has SWE data for -- The Odds API's /scores endpoint
    has no arbitrary-historical-date capability at all (daysFrom<=3 is a
    hard provider limit), so this is a structurally different data source,
    not a parameter change to the existing method."""
    import pandas as pd

    fake_df = pd.DataFrame([
        {
            "match_id": "swe-real-id-1", "date": pd.Timestamp("2025-09-15"),
            "home_team": "Malmo FF", "away_team": "AIK", "fthg": 2, "ftag": 1,
        },
    ])

    class FakeConnectionCtx:
        def __enter__(self):
            return type("C", (), {"execute": lambda self, q, p: type("R", (), {"fetchdf": lambda self: fake_df})()})()
        def __exit__(self, *a):
            return False

    with patch("app.backend.sweden_fixtures_client.DuckDBManager") as mock_db_cls:
        mock_db_cls.return_value.connection.return_value = FakeConnectionCtx()
        results = historical_results_from_raw_matches("2025-09-01", "2025-09-30")

    assert results == [
        NormalizedMatch(
            match_id="swe-real-id-1", utc_date="2025-09-15T00:00:00Z", status="FINISHED",
            home_team="Malmo FF", away_team="AIK", home_goals=2, away_goals=1, competition="SWE",
        )
    ]
```

Add the needed imports at the top of `test_sweden_fixtures_client.py` (check the file first — if `patch`/`NormalizedMatch` aren't already imported, add `from unittest.mock import patch` and `from app.backend.sweden_fixtures_client import historical_results_from_raw_matches`).

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_sweden_fixtures_client.py -k "historical_results" -v`
Expected: FAIL — `ImportError: cannot import name 'historical_results_from_raw_matches'`.

- [ ] **Step 3: Implement `historical_results_from_raw_matches`**

In `app/backend/sweden_fixtures_client.py`, add the import and function:

```python
from app.backend.football_data_client import NormalizedMatch
from src.utils.db_manager import DuckDBManager


def historical_results_from_raw_matches(date_from: str | None, date_to: str | None) -> list[NormalizedMatch]:
    """W71: The Odds API's /scores endpoint (get_results, below) can only
    ever see the last few real days (daysFrom<=3, a hard provider limit) --
    it structurally cannot serve an arbitrary historical date the way
    football-data.org's get_results() does for E0 (W45). raw_matches
    already has real Allsvenskan history back to 2012 (the ML engine's own
    ingestion target, src/ingestion/football_data/sweden_fetcher.py), so
    historical SWE fixtures are sourced from there instead for any
    already-past date range. get_fixtures() (future dates) is unaffected --
    the Odds API's /events endpoint serves that correctly."""
    db = DuckDBManager()
    query = "SELECT match_id, date, home_team, away_team, fthg, ftag FROM raw_matches WHERE league = 'SWE'"
    params: list[str] = []
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
    query += " ORDER BY date"
    with db.connection(read_only=True) as conn:
        rows = conn.execute(query, params).fetchdf()

    return [
        NormalizedMatch(
            match_id=row["match_id"],
            utc_date=row["date"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            status="FINISHED",
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_goals=None if row["fthg"] is None or pd.isna(row["fthg"]) else int(row["fthg"]),
            away_goals=None if row["ftag"] is None or pd.isna(row["ftag"]) else int(row["ftag"]),
            competition="SWE",
        )
        for _, row in rows.iterrows()
    ]
```

Add `import pandas as pd` to the top of the file alongside the existing `import requests`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_sweden_fixtures_client.py -v`
Expected: all pass, including the new test.

- [ ] **Step 5: Wire it into `/api/fixtures`'s past-date branch, with a failing test first**

Add to `app/backend/tests/test_fixtures_endpoint.py`, near the existing Sweden-related tests:

```python
def test_fixtures_endpoint_sources_historical_swe_results_from_raw_matches_not_the_odds_api(sweden_client_mock):
    """W71: the past-date SWE branch must not call the Odds-API-backed
    sweden_client.get_results() at all -- that endpoint structurally can't
    serve an arbitrary historical date. Confirms the real raw_matches-backed
    source is used instead, via a real (not mocked) query against a date
    with known real SWE data (2026-05-24, already confirmed elsewhere in
    this test file to have real SWE rows)."""
    with patch("app.backend.main._current_real_date", return_value=date(2026, 7, 20)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2026-05-24", "date_to": "2026-05-24"}
                )

    assert response.status_code == 200
    body = response.json()
    swe_rows = [m for m in body if m["competition"] == "SWE"]
    assert len(swe_rows) > 0
    sweden_client_mock.get_results.assert_not_called()
```

- [ ] **Step 6: Run to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_fixtures_endpoint.py -k "not_the_odds_api" -v`
Expected: FAIL — `sweden_client_mock.get_results` is still called (the old wiring), so the `assert_not_called()` fails. (If 2026-05-24 turns out to have no real SWE rows in this environment's `raw_matches`, swap the date for one confirmed via `SELECT date FROM raw_matches WHERE league='SWE' ORDER BY date DESC LIMIT 1` — don't guess.)

- [ ] **Step 7: Swap the past-branch call in `get_fixtures()`**

In `app/backend/main.py`, replace the `results_swe` block (currently calling `sweden_client.get_results`) inside `get_fixtures()`:

```python
        # W71: sourced from raw_matches directly, not sweden_client.get_results()
        # (The Odds API's /scores endpoint can only see the last few real
        # days -- it has no arbitrary-historical-date capability at all,
        # unlike football-data.org's get_results() for E0). Still
        # cache-keyed as "results_swe" -- same TTL-cache slot as before,
        # just backed by a different underlying source.
        matches += _tag(
            await _cached_fixture_call(
                ("results_swe", past_from, past_to),
                historical_results_from_raw_matches, date_from=past_from, date_to=past_to,
            ),
            "SWE",
        )
```

Add the import: `from app.backend.sweden_fixtures_client import SwedenFixturesClient, historical_results_from_raw_matches`. The `fixtures_range` (future) block calling `sweden_client.get_fixtures` is unchanged.

- [ ] **Step 8: Run to verify it passes**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_fixtures_endpoint.py -v`
Expected: all pass, including the new test. Check the other existing Sweden-results test (`test_fixtures_endpoint_wholly_past_range_sources_swedish_results_not_football_data`, around line 394 as of this plan) still passes — it currently asserts `sweden_client_mock.get_results.assert_called_once_with(...)` for the past-range case, which this change makes **false** (the past branch no longer calls it at all). This is an intentional behavior change (that's the whole point of this story) — update that specific assertion to confirm `historical_results_from_raw_matches` is used instead (e.g. patch it and assert the call, mirroring the new test above), rather than leaving a contradictory assertion in the suite. Explain this in your self-review, don't silently delete the old assertion.

- [ ] **Step 9: Full backend suite**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest tests/ app/backend/tests/ -q`
Expected: same baseline count (plus the new tests), zero unexplained regressions.

- [ ] **Step 10: Commit**

```bash
git add app/backend/sweden_fixtures_client.py app/backend/main.py app/backend/tests/test_sweden_fixtures_client.py app/backend/tests/test_fixtures_endpoint.py
git commit -m "fix(app): source historical SWE fixtures from raw_matches, mirroring W45's fix for E0

SwedenFixturesClient's get_results() is backed by The Odds API, which
can only ever see the last few real days (daysFrom<=3, a hard provider
limit) -- it structurally cannot serve an arbitrary historical date.
raw_matches already has real Allsvenskan history back to 2012; the
past-date branch of /api/fixtures now sources SWE from there instead,
unblocking SWE scenario testing across arbitrary historical dates."
```

---

### Task 3: Extend `launch_sandbox.py --precompute` to both leagues (W72)

**Files:**
- Modify: `scripts/launch_sandbox.py`

- [ ] **Step 1: Read the current implementation before editing**

`fetch_sandbox_fixtures()` and `precompute_recommendations()` are E0-only today (a single `FootballDataClient`, no Sweden client referenced anywhere). Confirm this matches what's in the file before editing — if it's drifted (e.g. someone else has already started multi-league work here), STOP and report rather than guessing how to reconcile it.

- [ ] **Step 2: Extend `fetch_sandbox_fixtures` to accept a client + competition label generically**

Replace the current signature and body:

```python
def fetch_sandbox_fixtures(
    fixtures_client: FootballDataClient, date_str: str, competition_code: str = "PL",
) -> tuple[list[NormalizedMatch], bool]:
    ...
    exact = fixtures_client.get_results(competition_code=competition_code, date_from=date_str, date_to=date_str)
    if exact:
        return exact, False
    to_date = (date_cls.fromisoformat(date_str) + datetime_mod.timedelta(days=90)).isoformat()
    upcoming = fixtures_client.get_results(competition_code=competition_code, date_from=date_str, date_to=to_date)
    return sorted(upcoming, key=lambda m: m.utc_date)[:10], True
```

with a second function alongside it for SWE (same fallback-window shape, different underlying source — W71's function takes no `competition_code`, it's SWE-only by construction):

```python
def fetch_sandbox_fixtures_swe(date_str: str) -> tuple[list[NormalizedMatch], bool]:
    """W72: SWE analogue of fetch_sandbox_fixtures(), sourced from W71's
    historical_results_from_raw_matches() instead of FootballDataClient --
    SwedenFixturesClient's own get_results() can't serve an arbitrary past
    date at all (see W71), so there's no equivalent single-client call to
    parameterize the way the E0 version does."""
    from app.backend.sweden_fixtures_client import historical_results_from_raw_matches

    exact = historical_results_from_raw_matches(date_str, date_str)
    if exact:
        return exact, False
    to_date = (date_cls.fromisoformat(date_str) + datetime_mod.timedelta(days=90)).isoformat()
    upcoming = historical_results_from_raw_matches(date_str, to_date)
    return sorted(upcoming, key=lambda m: m.utc_date)[:10], True
```

Leave the original `fetch_sandbox_fixtures` function itself unchanged (still E0-specific via its `competition_code` param) — don't rename or change its call signature, since Step 3 calls it explicitly for E0 alongside the new SWE function.

- [ ] **Step 3: Loop `precompute_recommendations` over both leagues**

Replace the body of `precompute_recommendations` (from `fixtures_client = FootballDataClient(...)` through the `run_eod_batch(...)` call) with a per-league loop. Read the current full function first (it prints progress lines and builds a `tally` dict per call) — restructure so each league gets its own `fetch_*` call, its own `run_eod_batch(..., league=...)` call (passing W62's existing `league` parameter — do not reimplement per-competition dispatch, reuse what `run_eod_batch` already accepts), and the progress/summary output clearly labels which league each line belongs to. Something like:

```python
def precompute_recommendations(date_str: str) -> None:
    os.environ["SANDBOX_MODE"] = "1"
    os.environ["SANDBOX_DATE"] = date_str

    from app.backend import recommendations
    from app.backend.eod_batch import run_eod_batch
    from app.backend.scheduler_wiring import build_odds_client
    from src.agent.agent_config import AgentConfig

    odds_client = build_odds_client()
    cache = recommendations.get_cache()
    config = AgentConfig.default()

    fixtures_client = FootballDataClient(api_key=os.environ.get("FOOTBALL_DATA_API_KEY", ""))
    e0_fixtures, e0_fallback = fetch_sandbox_fixtures(fixtures_client, date_str)
    swe_fixtures, swe_fallback = fetch_sandbox_fixtures_swe(date_str)

    for league, fixtures, used_fallback, client_for_batch in (
        ("E0", e0_fixtures, e0_fallback, fixtures_client),
        ("SWE", swe_fixtures, swe_fallback, fixtures_client),  # fixtures_client unused by run_eod_batch when `fixtures` is pre-supplied (W50) -- kept only for signature compatibility
    ):
        if used_fallback:
            print(
                f"Precompute [{league}]: no real fixtures on {date_str} -- falling back to the next "
                f"{len(fixtures)} match(es) in the following 90 days."
            )
        else:
            print(f"Precompute [{league}]: {len(fixtures)} real fixture(s) found for {date_str}.")
        if not fixtures:
            print(f"Precompute [{league}]: nothing to generate.")
            continue

        tally = {"generated": 0, "skipped": 0}

        def _on_progress(fixture: NormalizedMatch, outcome: str, _league=league, _tally=tally, _total=len(fixtures)) -> None:
            _tally[outcome] += 1
            done = _tally["generated"] + _tally["skipped"]
            print(
                f"Precompute [{_league}]: [{done}/{_total}] {fixture.home_team} vs {fixture.away_team}: "
                f"{outcome} (generated={_tally['generated']} skipped={_tally['skipped']})"
            )

        result = asyncio.run(
            run_eod_batch(
                fixtures_client=client_for_batch, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda fixture: None, date_str=date_str, fixtures=fixtures,
                on_progress=_on_progress, league=league,
            )
        )
        print(f"Precompute [{league}] complete: generated={result.generated} skipped={result.skipped} of {len(fixtures)} fixture(s).")
```

(Verify `run_eod_batch`'s actual current parameter order/defaults against `app/backend/eod_batch.py` before finalizing this — the plan's earlier "Before you start" reading covered `league: str = LEAGUE_CODE` as an added kwarg; confirm it's still exactly that shape, not drifted.)

- [ ] **Step 4: Manual verification (no automated test harness exists for this script today)**

This script has no existing pytest coverage (it's a CLI entry point, verified via the sandbox runbook convention, not unit tests) — don't invent a new test file for it out of scope. Instead, run it for real against a date known to have real E0 fixtures (confirm one first: `SELECT date FROM raw_matches WHERE league='E0' ORDER BY date DESC LIMIT 1`), and separately confirm SWE fixtures are attempted (look for `Precompute [SWE]:` lines in the output, even if the specific date has none — confirms the loop runs, not just E0):

```bash
python scripts/launch_sandbox.py <a-real-E0-date> --precompute --dry-run
```
(`--dry-run` first, to avoid a long real-call run while just confirming the script doesn't crash and both `Precompute [E0]:`/`Precompute [SWE]:` lines appear — check `--dry-run`'s actual current behavior in the script first; if it skips precompute output entirely, do a real full run instead and note the real generated/skipped counts for both leagues in your report.)

- [ ] **Step 5: Commit**

```bash
git add scripts/launch_sandbox.py
git commit -m "feat(app): extend launch_sandbox.py --precompute to cover both E0 and SWE

Was E0-only -- a sandbox session precomputed today showed zero Swedish
recommendations regardless of data/corpus availability. Mirrors W62's
existing per-competition loop shape from the live EOD batch, using
W71's new historical SWE source for fixture discovery."
```

---

### Task 4: Repeatable scenario-testing runbook across Aug 2025–May 2026 (W73)

**Files:**
- Create: `scripts/scenario_runbook.py`
- Modify: `documents/app_user_stories.md` (mark W70-W73 completed with real completion notes, per this repo's established convention)

- [ ] **Step 1: Confirm what corpus coverage actually exists before writing the runbook**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python3 -c "
from pathlib import Path
for league in ('E0', 'SWE'):
    d = Path('data/agent_snapshots') / league
    complete = [p.parent.name for p in d.glob('*/_complete.json')] if d.exists() else []
    print(league, len(complete), 'complete recordings')
"`

This tells you what the runbook will actually find when it runs — the corpus may still be mid-recording (it was actively growing as of this plan's design phase). The runbook (Step 2) must handle a mostly-empty or partially-recorded corpus gracefully (report what's there honestly), not assume full Aug 2025–May 2026 coverage exists yet.

- [ ] **Step 2: Write `scripts/scenario_runbook.py`**

Mirror `scripts/sandbox_runbook.py`'s existing style (a single-scenario runbook script, already in this repo) but for a date *range*. Structure:

```python
"""Scenario-testing runbook (W73) -- walks a range of dates, launching the
sandbox and running --precompute for both leagues at each sampled date, and
reports whether each one replayed from the agent-snapshot corpus (W70) or
had to fall back to a live call. Repeatable, evidence-producing check for
"does scenario testing across Aug 2025-May 2026 actually work end-to-end,"
mirroring W31/W44's runbook convention -- not a CI suite (this project has
none), a documented manual/scriptable check.

Usage:
    python scripts/scenario_runbook.py --from-date 2025-08-01 --to-date 2026-05-31 --sample-every-days 14
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sample_dates(from_date: str, to_date: str, every_days: int) -> list[str]:
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    out = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=every_days)
    return out


def run_one_scenario(date_str: str) -> dict:
    """Launches the sandbox for date_str with --precompute, captures
    whether each league found fixtures and whether generation used replay
    (corpus/sandbox) or a live call, then stops the sandbox. Returns a
    result dict for the summary report."""
    result = {"date": date_str, "e0_fixtures": 0, "swe_fixtures": 0, "errors": []}
    launch = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "launch_sandbox.py"), date_str, "--precompute"],
        capture_output=True, text=True, timeout=600,
    )
    output = launch.stdout + launch.stderr
    if launch.returncode != 0:
        result["errors"].append(f"launch failed (exit {launch.returncode}): {output[-2000:]}")
    else:
        for line in output.splitlines():
            if line.startswith("Precompute [E0]:") and "real fixture" in line:
                result["e0_fixtures"] += 1
            if line.startswith("Precompute [SWE]:") and "real fixture" in line:
                result["swe_fixtures"] += 1
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "launch_sandbox.py"), "--stop"],
        capture_output=True, text=True, timeout=60,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument("--sample-every-days", type=int, default=14)
    args = parser.parse_args(argv)

    dates = sample_dates(args.from_date, args.to_date, args.sample_every_days)
    print(f"Scenario runbook: {len(dates)} sampled date(s) from {args.from_date} to {args.to_date}.")
    results = [run_one_scenario(d) for d in dates]

    print("\n=== Summary ===")
    for r in results:
        status = "OK" if not r["errors"] else "ERROR"
        print(f"{r['date']}: E0={r['e0_fixtures']} SWE={r['swe_fixtures']} [{status}]")
        for err in r["errors"]:
            print(f"  {err}")


if __name__ == "__main__":
    main()
```

Treat this as a starting structure, not a rigid spec — while implementing, verify each assumption against the real scripts it shells out to (exact `launch_sandbox.py` CLI flags/output text for the "real fixture(s) found" lines, exact `--stop` behavior) and adjust the string-matching/parsing to what the script actually prints, rather than what this plan guessed it prints.

- [ ] **Step 3: Run it for real, against whatever corpus coverage Step 1 found**

```bash
python scripts/scenario_runbook.py --from-date 2025-08-01 --to-date 2026-05-31 --sample-every-days 14
```

This will make real live LLM/API calls for any sampled date that has no corpus recording yet (expected and correct — the runbook's job is to report reality, not assume full coverage). Record the actual output.

- [ ] **Step 4: Verify replay vs. live-call behavior directly, not just fixture counts**

For at least 2 sampled dates — one likely covered by the corpus (recent, since A34's job was actively recording toward the end of this plan's design phase) and one likely not (e.g. deep in Aug 2025 if the corpus hadn't reached that far back yet) — check the backend log output from `run_one_scenario`'s captured output for `sandbox_agent_replay_miss` (would indicate an unexpected corpus miss) versus clean replay, and note which mode each date actually used. This is the part of the acceptance criteria that fixture-count alone can't prove (a live call and a replay both "find fixtures and generate recommendations" — only the mode tells you whether the corpus bridge is actually doing its job).

- [ ] **Step 5: Write up the real results**

In `documents/app_user_stories.md`, mark **W70**, **W71**, **W72**, **W73** `completed`, each with real completion notes following this doc's established convention (see any `completed` W## row for the expected level of detail: what was built, real test counts, what was found/fixed along the way, and — for W73 specifically — the actual runbook output: how many sampled dates, how many replayed from the corpus vs. fell back to a live call, any errors, and an honest statement of current corpus coverage (from Step 1) rather than a claim of full range coverage if the corpus wasn't finished recording yet.

- [ ] **Step 6: Commit**

```bash
git add scripts/scenario_runbook.py documents/app_user_stories.md
git commit -m "feat(app): add scenario-testing runbook across Aug 2025-May 2026 (W73)

Walks a sampled date range, launching the sandbox + precompute for
both leagues at each date and verifying replay-vs-live-call behavior
directly (not just that fixtures were found) -- confirms the W70
corpus bridge, W71 SWE historical sourcing, and W72 precompute
extension all work together across the real requested range."
```

---

## Summary of files touched

**Backend:** `app/backend/recommendations.py`, `app/backend/sweden_fixtures_client.py`, `app/backend/main.py`, `app/backend/tests/test_sandbox_agent_snapshot.py` (substantial rewrite), `app/backend/tests/test_sweden_fixtures_client.py`, `app/backend/tests/test_fixtures_endpoint.py`

**Scripts:** `scripts/launch_sandbox.py`, `scripts/scenario_runbook.py` (new)

**Docs:** `documents/app_user_stories.md`
