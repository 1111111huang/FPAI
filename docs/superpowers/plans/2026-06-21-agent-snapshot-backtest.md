# Agent Snapshot & Backtest Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement user stories A09–A16 from `documents/agent_user_stories.md`: a record/replay snapshot store, a backtest harness that scores historical agent recommendations against actual outcomes, flat-stake and Kelly bankroll simulation, evaluation reporting, and a config-comparison framework — exposed via three new CLI commands (`agent-snapshot`, `agent-backtest`, `agent-compare`).

**Architecture:** A `SnapshotStore` (thread-local mode/match-id state) wraps each tool function in `src/agent/tools.py` so the same tool code runs in `live`/`record`/`replay` mode with zero call-site changes. `agent-snapshot` drives the agent in `record` mode over historical DuckDB matches. `BacktestHarness` drives it in `replay` mode and scores recommendations against `raw_matches` outcomes via a shared `process_match_row()` function reused by both the synchronous harness and the concurrent `agent-backtest` CLI path (via `asyncio.gather` + `asyncio.to_thread`, since the underlying graph/tools are synchronous). `agent-compare` re-runs `BacktestHarness` for multiple configs over the identical (seeded, stratified) match sample so the only varying factor is the agent config.

**Tech Stack:** Python 3.11, DuckDB, pandas, asyncio (`to_thread` + `Semaphore`), tqdm (already installed), existing `src/agent/*` modules.

**Known scope decisions (read before implementing):**
1. **Thread-local SnapshotStore state.** A14 requires concurrent backtest runs. If `SnapshotStore.mode`/`match_id` were plain instance attributes, concurrent threads (via `asyncio.to_thread`) would race on them. `SnapshotStore` uses `threading.local()` internally from the start (A09), not bolted on later.
2. **Stratified sampling dimension.** A12's `--sample N` is "stratified ... balanced across bet / no-bet outcomes" — but bet/no-bet is the agent's *output*, unknowable before running it. We stratify by actual match result (home/draw/away) instead — the only outcome dimension known in advance from `raw_matches`. Documented inline.
3. **Resolvable markets.** The `MatchRecommendation` schema has no numeric line field for corners markets (only `current_odds`/`min_odds`), so correctness can only be programmatically resolved for `result_3way`, `btts`, and `total_goals` (using the 2.5 goals line). `home_corners`/`away_corners` market correctness is `None` (unresolved) and such bets are skipped in bankroll simulation. Documented inline and in techspec follow-up.
4. **No CLI-dispatch tests.** Existing codebase has no `tests/test_main.py` — CLI argument wiring isn't unit tested anywhere. This plan follows that convention; all new logic is tested at the module level (`snapshot_store.py`, `tools.py`, `backtest.py`, `staking.py`, `evaluation.py`, `comparison.py`).

---

## Task 1: SnapshotStore Core (A09)

**Files:**
- Create: `src/agent/snapshot_store.py`
- Test: `tests/test_snapshot_store.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_snapshot_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.snapshot_store'`

- [ ] **Step 3: Implement SnapshotStore**

```python
"""Record/replay interceptor for agent tool calls (A09).

Lets every tool function in src/agent/tools.py run unmodified in three modes:
  - live:   call the real implementation, no interception
  - record: call the real implementation, save {tool, inputs, response} to disk
  - replay: never call the real implementation — load the saved response or
            raise SnapshotMissingError immediately (no silent fallback)

Mode and match context are stored in thread-local state so concurrent backtest
runs (each on its own thread via asyncio.to_thread) never clobber each other's
snapshot context. This must not be relaxed to plain instance attributes without
re-checking A14 (agent-backtest --concurrency).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

SnapshotMode = Literal["live", "record", "replay"]

_DEFAULT_BASE_DIR = Path("data/agent_snapshots")
_VALID_MODES = {"live", "record", "replay"}


class SnapshotMissingError(Exception):
    """Raised in replay mode when no recorded snapshot exists for a tool call."""

    def __init__(self, tool: str, match_id: str | None, key: str):
        self.tool = tool
        self.match_id = match_id
        self.key = key
        super().__init__(
            f"No snapshot found for tool={tool!r} match_id={match_id!r} key={key} "
            "(run agent-snapshot in record mode for this match first)"
        )


class SnapshotStore:
    """Intercepts tool calls to record live responses or replay recorded ones."""

    def __init__(self, base_dir: str | Path = _DEFAULT_BASE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self._local = threading.local()

    @property
    def mode(self) -> SnapshotMode:
        return getattr(self._local, "mode", "live")

    @property
    def match_id(self) -> str | None:
        return getattr(self._local, "match_id", None)

    @property
    def match_date(self) -> str | None:
        return getattr(self._local, "match_date", None)

    def set_mode(self, mode: SnapshotMode) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown snapshot mode: {mode!r}")
        self._local.mode = mode

    def set_match(self, match_id: str, match_date: str | None = None) -> None:
        self._local.match_id = match_id
        self._local.match_date = match_date

    @staticmethod
    def key_for(inputs: dict[str, Any]) -> str:
        canonical = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, tool: str, key: str) -> Path:
        match_id = self.match_id
        if not match_id:
            raise ValueError("SnapshotStore.set_match() must be called before record/replay use")
        return self.base_dir / match_id / f"{tool}_{key}.json"

    def wrap(self, tool: str, fn: Callable[..., str]) -> Callable[..., str]:
        """Return a callable that records or replays fn's output based on the current mode."""

        def wrapped(**kwargs: Any) -> str:
            mode = self.mode
            if mode == "live":
                return fn(**kwargs)

            key = self.key_for(kwargs)
            path = self._path(tool, key)

            if mode == "replay":
                if not path.exists():
                    raise SnapshotMissingError(tool, self.match_id, key)
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload["response"]

            # record
            response = fn(**kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tool": tool,
                "inputs": kwargs,
                "response": response,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return response

        return wrapped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_snapshot_store.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/snapshot_store.py tests/test_snapshot_store.py
git commit -m "feat(agent): add SnapshotStore for record/replay tool interception (A09)"
```

---

## Task 2: Integrate SnapshotStore into tools.py (A10)

**Files:**
- Modify: `src/agent/tools.py` (entire file — extracting `_impl` functions and wrapping)
- Test: `tests/test_agent_tools_snapshot.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for SnapshotStore integration in agent tools (A10)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent import tools as agent_tools
from src.agent.snapshot_store import SnapshotMissingError


@pytest.fixture(autouse=True)
def reset_snapshot_store(tmp_path):
    """Point the module-level store at a tmp dir and reset to live mode after each test."""
    agent_tools._snapshot_store.base_dir = tmp_path
    agent_tools._snapshot_store.set_mode("live")
    yield
    agent_tools._snapshot_store.set_mode("live")


def test_web_search_record_then_replay(tmp_path):
    with patch("src.agent.tools.os.environ.get", return_value="fake-key"), \
         patch("src.agent.tools.TavilyClient") if False else patch("tavily.TavilyClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.search.return_value = {"results": [{"title": "T", "content": "C", "url": "U"}]}

        agent_tools.configure_snapshot_store("record", match_id="m1")
        first = agent_tools.web_search.invoke({"query": "test odds"})
        assert "T" in first
        assert instance.search.call_count == 1

        # Replay must return the exact same text WITHOUT calling Tavily again
        agent_tools.configure_snapshot_store("replay", match_id="m1")
        second = agent_tools.web_search.invoke({"query": "test odds"})
        assert second == first
        assert instance.search.call_count == 1  # unchanged


def test_web_search_replay_missing_raises():
    agent_tools.configure_snapshot_store("replay", match_id="never-recorded")
    with pytest.raises(SnapshotMissingError):
        agent_tools.web_search.invoke({"query": "anything"})


def test_web_search_unavailable_message_bypasses_snapshot_key_consistently():
    # No TAVILY_API_KEY in this process env by default in CI; record mode without
    # a key returns the fixed unavailable message both times — proves wrap() doesn't
    # choke on the early-return path (no tavily call at all).
    agent_tools.configure_snapshot_store("record", match_id="m2")
    with patch.dict("os.environ", {}, clear=True):
        result = agent_tools.web_search.invoke({"query": "x"})
    assert "TOOL_PERMANENTLY_UNAVAILABLE" in result


def test_web_search_date_filter_applied_during_record_and_replay():
    captured_queries = []

    def fake_impl(query):
        captured_queries.append(query)
        return "ok"

    with patch.object(agent_tools, "_web_search_impl", side_effect=fake_impl):
        agent_tools.configure_snapshot_store("record", match_id="m3", match_date="2025-03-01")
        agent_tools.web_search.invoke({"query": "team news"})

    assert captured_queries == ["team news before:2025-03-01"]


def test_forecast_league_record_then_replay():
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        agent_tools.configure_snapshot_store("record", match_id="m4")
        first = agent_tools.forecast_league.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0",
            "odds_h": 2.0, "odds_d": 3.0, "odds_a": 3.5,
        })
        assert instance.forecast_upcoming.call_count == 1

        agent_tools.configure_snapshot_store("replay", match_id="m4")
        second = agent_tools.forecast_league.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0",
            "odds_h": 2.0, "odds_d": 3.0, "odds_a": 3.5,
        })
        assert second == first
        assert instance.forecast_upcoming.call_count == 1  # not called again


def test_live_mode_is_default_and_unaffected_by_snapshot_machinery():
    """Existing A05/A06 behaviour (no snapshot config called) must be unchanged."""
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result
        result = agent_tools.forecast_league.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0",
            "odds_h": 2.0, "odds_d": 3.0, "odds_a": 3.5,
        })
    assert json.loads(result)["result_3way"]["probabilities"]["home"] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_tools_snapshot.py -v`
Expected: FAIL — `configure_snapshot_store` does not exist yet

- [ ] **Step 3: Rewrite tools.py with extracted impls + SnapshotStore wrapping**

Replace the entire contents of `src/agent/tools.py`:

```python
from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from src.agent.snapshot_store import SnapshotStore, SnapshotMode
from src.utils.logger import get_logger

_LOG = get_logger(__name__)

_snapshot_store = SnapshotStore()


def configure_snapshot_store(mode: SnapshotMode, match_id: str | None = None, match_date: str | None = None) -> None:
    """Configure the module-level SnapshotStore shared by all tool functions.

    Call this before run_agent() to switch between live/record/replay. In record
    and replay mode, match_id is required (raises ValueError otherwise, from
    SnapshotStore._path). match_date, if given, is appended to web_search queries
    as 'before:<match_date>' to reduce post-match result leakage (A10).
    """
    _snapshot_store.set_mode(mode)
    if match_id is not None:
        _snapshot_store.set_match(match_id, match_date)


def get_snapshot_store() -> SnapshotStore:
    return _snapshot_store


def _web_search_impl(query: str) -> str:
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "TOOL_PERMANENTLY_UNAVAILABLE: web_search has no API key configured. "
            "Do NOT call web_search again — it will always return this message. "
            "Output your final JSON recommendation now using only the forecast data already retrieved."
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=5)
    snippets = []
    for r in response.get("results", []):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        snippets.append(f"[{title}]\n{content}\nSource: {url}")
    return "\n\n---\n\n".join(snippets) if snippets else "No results found."


@tool
def web_search(query: str) -> str:
    """Search the web for football match information: odds, team news, injuries, and lineups.

    Use for: finding current bookmaker odds, alternative team name spellings,
    injury/suspension reports, team selection hints, and recent form context.
    Always ignore any result that mentions a final score or match result."""
    effective_query = query
    if _snapshot_store.mode in ("record", "replay") and _snapshot_store.match_date:
        effective_query = f"{query} before:{_snapshot_store.match_date}"
    return _snapshot_store.wrap("web_search", _web_search_impl)(query=effective_query)


def _forecast_league_impl(
    home_team: str,
    away_team: str,
    date: str,
    league: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    try:
        from src.forecast.forecast_service import ForecastService
        svc = ForecastService()
        try:
            result = svc.forecast_upcoming(
                home_team=home_team,
                away_team=away_team,
                date=date,
                league=league,
                odds_h=odds_h,
                odds_d=odds_d,
                odds_a=odds_a,
                match_type="league",
            )
            return json.dumps(result, default=str)
        except FileNotFoundError:
            # League-context models not yet trained — use international (market-odds-only) path
            _LOG.info("forecast_league | league_models_absent | falling_back_to_international | home=%s away=%s", home_team, away_team)
            result = svc.forecast_upcoming(
                home_team=home_team,
                away_team=away_team,
                date=date,
                league=league,
                odds_h=odds_h,
                odds_d=odds_d,
                odds_a=odds_a,
                match_type="international",
            )
            result.setdefault("data_quality", {})["prediction_basis"] = "market_odds_only_league_fallback"
            return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "status": "tool_error"})


@tool
def forecast_league(
    home_team: str,
    away_team: str,
    date: str,
    league: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    """Get ML probability forecast for a domestic league match.

    Uses full team history + market odds features (114 features).
    Use when both teams play in a known domestic league with historical data.

    Args:
        home_team: Home team name — fuzzy-matched against known teams in the database.
                   Use web_search first to find the correct name variant if unsure.
        away_team: Away team name.
        date: Match date in YYYY-MM-DD format.
        league: League code, e.g. 'E0' (Premier League), 'SP1' (La Liga), 'D1' (Bundesliga).
        odds_h: Home win decimal odds from bookmaker.
        odds_d: Draw decimal odds.
        odds_a: Away win decimal odds.

    Returns JSON with probabilities for result_3way, btts, goals, and corners targets.
    Includes data_quality.prediction_basis to indicate which features were used."""
    return _snapshot_store.wrap("forecast_league", _forecast_league_impl)(
        home_team=home_team, away_team=away_team, date=date, league=league,
        odds_h=odds_h, odds_d=odds_d, odds_a=odds_a,
    )


def _forecast_international_impl(
    home_team: str,
    away_team: str,
    date: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    try:
        from src.forecast.forecast_service import ForecastService
        svc = ForecastService()
        result = svc.forecast_upcoming(
            home_team=home_team,
            away_team=away_team,
            date=date,
            league="",
            odds_h=odds_h,
            odds_d=odds_d,
            odds_a=odds_a,
            match_type="international",
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "status": "tool_error"})


@tool
def forecast_international(
    home_team: str,
    away_team: str,
    date: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    """Get ML probability forecast using market odds only — no team history required.

    Use for: international fixtures, cup matches between teams from different leagues,
    or when team historical data is unavailable in the database.

    Args:
        home_team: Home team name.
        away_team: Away team name.
        date: Match date in YYYY-MM-DD format.
        odds_h: Home win decimal odds.
        odds_d: Draw decimal odds.
        odds_a: Away win decimal odds.

    Returns JSON with market-implied probability forecasts.
    data_quality.prediction_basis will be 'market_odds_only'."""
    return _snapshot_store.wrap("forecast_international", _forecast_international_impl)(
        home_team=home_team, away_team=away_team, date=date,
        odds_h=odds_h, odds_d=odds_d, odds_a=odds_a,
    )


def get_default_tools() -> list:
    return [web_search, forecast_league, forecast_international]
```

- [ ] **Step 4: Run new tests, then the full existing agent suite**

Run: `python -m pytest tests/test_agent_tools_snapshot.py -v`
Expected: 6 passed

Run: `python -m pytest tests/test_agent_config.py tests/test_agent_schema.py tests/test_agent_graph.py -v`
Expected: all previously-passing tests still pass (live-mode default behaviour is unchanged — `forecast_league`'s fallback test from A07/BUG-010 work must still pass untouched)

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools.py tests/test_agent_tools_snapshot.py
git commit -m "feat(agent): route all tools through SnapshotStore, add date-filtered web_search for snapshot collection (A10)"
```

---

## Task 3: Add extra_system_instructions to run_agent (needed by A11)

**Files:**
- Modify: `src/agent/graph.py:115-160` (the `run_agent` function)
- Test: `tests/test_agent_graph.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_graph.py`:

```python
def test_run_agent_appends_extra_system_instructions():
    from unittest.mock import patch, MagicMock
    from src.agent.graph import run_agent

    captured = {}

    def fake_build_graph(config, tools):
        mock_compiled = MagicMock()

        def fake_invoke(initial_state):
            captured["system_content"] = initial_state["messages"][0].content
            return {"recommendation": {"overall": "no_bet"}}

        mock_compiled.invoke.side_effect = fake_invoke
        return mock_compiled

    cfg = _make_config()
    with patch("src.agent.graph._build_llm"), \
         patch("src.agent.graph._load_system_prompt", return_value="BASE PROMPT"), \
         patch("src.agent.graph.build_graph", side_effect=fake_build_graph):
        run_agent(
            match_info={"home_team": "A", "away_team": "B", "date": "2025-01-01"},
            config=cfg,
            tools=[],
            extra_system_instructions="EXTRA INSTRUCTIONS HERE",
        )

    assert "BASE PROMPT" in captured["system_content"]
    assert "EXTRA INSTRUCTIONS HERE" in captured["system_content"]


def test_run_agent_without_extra_instructions_unchanged():
    from unittest.mock import patch, MagicMock
    from src.agent.graph import run_agent

    captured = {}

    def fake_build_graph(config, tools):
        mock_compiled = MagicMock()

        def fake_invoke(initial_state):
            captured["system_content"] = initial_state["messages"][0].content
            return {"recommendation": {"overall": "no_bet"}}

        mock_compiled.invoke.side_effect = fake_invoke
        return mock_compiled

    cfg = _make_config()
    with patch("src.agent.graph._build_llm"), \
         patch("src.agent.graph._load_system_prompt", return_value="BASE PROMPT"), \
         patch("src.agent.graph.build_graph", side_effect=fake_build_graph):
        run_agent(
            match_info={"home_team": "A", "away_team": "B", "date": "2025-01-01"},
            config=cfg,
            tools=[],
        )

    assert captured["system_content"] == "BASE PROMPT"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_graph.py -k extra_system_instructions -v`
Expected: FAIL — `run_agent() got an unexpected keyword argument 'extra_system_instructions'`

- [ ] **Step 3: Modify run_agent**

In `src/agent/graph.py`, change the signature and body of `run_agent` (replace lines 115–160):

```python
def run_agent(
    match_info: dict,
    config: AgentConfig | None = None,
    tools: list | None = None,
    extra_system_instructions: str | None = None,
) -> MatchRecommendation:
    """Run the betting agent for a single match and return a structured recommendation.

    Args:
        match_info: Dict with keys: home_team, away_team, date, and optionally league.
        config: AgentConfig instance. Loads from config/agent_config.yaml if None.
        tools: List of LangChain tools. Loads default tools if None.
        extra_system_instructions: Appended to the loaded system prompt. Used by
            agent-snapshot (A11) to inject snapshot-collection-only rules (e.g.
            "ignore any result mentioning a final score") without forking the
            whole prompt file.
    """
    if config is None:
        config = AgentConfig.default()
    if tools is None:
        from src.agent.tools import get_default_tools
        tools = get_default_tools()

    system_prompt = _load_system_prompt(config)
    if extra_system_instructions:
        system_prompt = f"{system_prompt}\n\n{extra_system_instructions}"

    prompt = (
        f"Analyse the upcoming match: {match_info['home_team']} vs {match_info['away_team']}"
        f" on {match_info['date']}"
    )
    if match_info.get("league"):
        prompt += f" in league {match_info['league']}"
    odds = match_info.get("odds")
    if odds:
        prompt += (
            f". Bookmaker odds: home={odds['home']}, draw={odds['draw']}, away={odds['away']}. "
            "Use these exact odds_h/odds_d/odds_a values when calling the forecast tool."
        )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ],
        "match_info": match_info,
        "recommendation": None,
        "tool_call_count": 0,
    }

    compiled = build_graph(config, tools)
    result = compiled.invoke(initial_state)
    return result["recommendation"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_graph.py -v`
Expected: all pass (previous tests + 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_agent_graph.py
git commit -m "feat(agent): support extra_system_instructions in run_agent for snapshot-mode prompting"
```

---

## Task 4: agent-snapshot CLI Command (A11)

**Files:**
- Modify: `main.py` (add parser + `run_agent_snapshot` function + dispatch)
- No new test file — CLI dispatch is untested elsewhere in this codebase (see plan-level scope decision #4); `run_agent_snapshot`'s DB query logic is simple enough to verify manually per Step 4 below.

- [ ] **Step 1: Add the `agent-snapshot` subparser**

In `main.py`, immediately after the existing `agent-recommend` parser block (find `agent_recommend_parser.add_argument("--config", ...)` and insert after it, before `return parser`):

```python
    # agent-snapshot
    agent_snapshot_parser = subparsers.add_parser(
        "agent-snapshot",
        help="Collect tool-call snapshots for historical matches (record mode) for later backtesting",
    )
    agent_snapshot_parser.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD (inclusive)")
    agent_snapshot_parser.add_argument("--to-date", required=True, help="End date YYYY-MM-DD (inclusive)")
    agent_snapshot_parser.add_argument("--league", default=None, help="League code (e.g. E0). Omit for all leagues.")
    agent_snapshot_parser.add_argument("--config", default=None, help="Path to agent_config.yaml (default: config/agent_config.yaml)")
    agent_snapshot_parser.add_argument("--dry-run", action="store_true", help="List matches that would be processed without running the agent")
```

- [ ] **Step 2: Add the `run_agent_snapshot` function**

In `main.py`, immediately after `run_agent_recommend` (after its closing `print(json.dumps(recommendation, indent=2))` line), add:

```python
def run_agent_snapshot(
    from_date: str,
    to_date: str,
    league: str | None,
    config_path: str | None,
    dry_run: bool,
) -> None:
    """Drive the agent in record mode over historical matches to build a snapshot corpus (A11)."""
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    from src.agent.agent_config import AgentConfig
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools
    from src.utils.db_manager import DuckDBManager

    snapshot_addendum = (
        "## SNAPSHOT COLLECTION MODE\n\n"
        "You are collecting training data from a historical match. Discard and ignore any "
        "web_search result that mentions a final score, match result, or post-match analysis — "
        "treat this match as still upcoming."
    )

    cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    db = DuckDBManager()
    query = (
        "SELECT match_id, league, date, home_team, away_team, odds_h, odds_d, odds_a "
        "FROM raw_matches WHERE date >= ? AND date <= ?"
    )
    params: list = [from_date, to_date]
    if league:
        query += " AND UPPER(league) = ?"
        params.append(league.upper())
    query += " ORDER BY date"
    with db.connection() as conn:
        matches = conn.execute(query, params).fetchdf()

    base_dir = Path("data/agent_snapshots")
    to_process = []
    skipped = 0
    for _, row in matches.iterrows():
        marker = base_dir / row["match_id"] / "_complete.json"
        if marker.exists():
            skipped += 1
            continue
        to_process.append(row)

    print(f"Matches in range: {len(matches)} | already complete: {skipped} | to process: {len(to_process)}")
    if dry_run:
        for row in to_process:
            date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])
            print(f"  {date_str} {row['home_team']} vs {row['away_team']} [{row['league']}]")
        return

    errors = 0
    for i, row in enumerate(to_process, 1):
        match_id = row["match_id"]
        date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])
        match_info = {"home_team": row["home_team"], "away_team": row["away_team"], "date": date_str, "league": row["league"]}
        if row["odds_h"] and row["odds_d"] and row["odds_a"]:
            match_info["odds"] = {"home": row["odds_h"], "draw": row["odds_d"], "away": row["odds_a"]}

        agent_tools.configure_snapshot_store("record", match_id=match_id, match_date=date_str)
        try:
            run_agent(match_info=match_info, config=cfg, extra_system_instructions=snapshot_addendum)
            marker_path = base_dir / match_id / "_complete.json"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(json.dumps({"completed_at": datetime.now(timezone.utc).isoformat()}))
            print(f"[{i}/{len(to_process)}] OK {match_info['home_team']} vs {match_info['away_team']}")
        except Exception as exc:
            errors += 1
            print(f"[{i}/{len(to_process)}] ERROR {match_info['home_team']} vs {match_info['away_team']}: {exc}", file=sys.stderr)
        finally:
            agent_tools.configure_snapshot_store("live")

    print(f"\nDone. Processed: {len(to_process) - errors} | Errors: {errors} | Skipped: {skipped}")
```

- [ ] **Step 3: Add dispatch in `main()`**

In `main.py`, find:
```python
    elif args.command == "agent-recommend":
        run_agent_recommend(
            ...
        )
```
Add immediately after it:
```python
    elif args.command == "agent-snapshot":
        run_agent_snapshot(
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            config_path=args.config,
            dry_run=args.dry_run,
        )
```

- [ ] **Step 4: Manual verification**

Run: `python main.py agent-snapshot --from-date 2025-01-01 --to-date 2025-01-31 --league E0 --dry-run`
Expected: prints a count of matches in range and a list of fixtures, exits 0, writes nothing to `data/agent_snapshots/`

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(agent): add agent-snapshot CLI command (A11)"
```

---

## Task 5: Outcome Loader + BacktestRecord + BacktestHarness (A12)

**Files:**
- Create: `src/agent/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for outcome loading and BacktestHarness (A12)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.agent_config import AgentConfig
from src.agent.backtest import (
    BacktestHarness,
    BacktestRecord,
    load_outcome,
    process_match_row,
)
from src.agent.snapshot_store import SnapshotMissingError


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        model="stub-model", provider="ollama", temperature=0.0, max_tool_calls=5,
        min_odds_threshold=2.0, min_value_edge=0.05, markets=["result_3way"],
        system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _row(**overrides) -> pd.Series:
    base = dict(
        match_id="m1", league="E0", date=pd.Timestamp("2025-03-01"),
        home_team="City", away_team="Arsenal",
        odds_h=1.9, odds_d=3.5, odds_a=4.0,
        fthg=2, ftag=1, hc=5.0, ac=4.0,
    )
    base.update(overrides)
    return pd.Series(base)


def test_load_outcome_home_win():
    outcome = load_outcome(_row(fthg=2, ftag=1))
    assert outcome["result"] == "home"
    assert outcome["btts"] == "yes"
    assert outcome["total_goals"] == 3
    assert outcome["total_goals_side"] == "over_2.5"


def test_load_outcome_draw_and_no_btts():
    outcome = load_outcome(_row(fthg=0, ftag=0))
    assert outcome["result"] == "draw"
    assert outcome["btts"] == "no"
    assert outcome["total_goals_side"] == "under_2.5"


def test_load_outcome_away_win():
    outcome = load_outcome(_row(fthg=0, ftag=2))
    assert outcome["result"] == "away"


def test_process_match_row_scores_markets_correctly():
    recommendation = {
        "match": {}, "overall": "direct_bet",
        "markets": [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 1.9, "min_odds": 1.9, "ml_probability": 0.6, "implied_probability": 0.52, "value_edge": 0.08},
            {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 1.8, "min_odds": 2.0, "ml_probability": 0.5, "implied_probability": 0.55, "value_edge": -0.05},
            {"market": "home_corners", "selection": "over_4.5", "recommendation_type": "direct_bet", "current_odds": 1.9, "min_odds": 1.9, "ml_probability": 0.5, "implied_probability": 0.52, "value_edge": -0.02},
        ],
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure:
        record = process_match_row(_row(fthg=2, ftag=1), _make_config())

    assert isinstance(record, BacktestRecord)
    assert record.actual["result"] == "home"
    by_market = {m["market"]: m for m in record.market_results}
    assert by_market["result_3way"]["correct"] is True
    assert by_market["btts"]["correct"] is True  # actual btts is "yes" (2-1, both scored); selection was "yes"
    assert by_market["home_corners"]["correct"] is None  # unresolvable, documented limitation

    # configure_snapshot_store called with replay then live (record_calls captures the mode transitions)
    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live"]
    mock_run.assert_called_once()


def test_process_match_row_propagates_snapshot_missing_error():
    with patch("src.agent.graph.run_agent", side_effect=SnapshotMissingError("web_search", "m1", "abc")), \
         patch("src.agent.tools.configure_snapshot_store"):
        with pytest.raises(SnapshotMissingError):
            process_match_row(_row(), _make_config())


def test_backtest_harness_load_matches_filters_by_date_and_league():
    harness = BacktestHarness(config=_make_config())
    fake_df = pd.DataFrame([
        _row(match_id="a", date=pd.Timestamp("2025-01-15")),
        _row(match_id="b", date=pd.Timestamp("2025-02-15")),
    ])
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df
    with patch.object(harness.db, "connection") as mock_connection:
        mock_connection.return_value.__enter__.return_value = mock_conn
        result = harness.load_matches("2025-01-01", "2025-03-01", league="E0")

    assert len(result) == 2
    sql_used = mock_conn.execute.call_args[0][0]
    assert "raw_matches" in sql_used
    assert "league" in sql_used.lower()


def test_backtest_harness_stratified_sample_balances_result_categories():
    harness = BacktestHarness(config=_make_config())
    rows = (
        [_row(match_id=f"h{i}", fthg=2, ftag=0, date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)) for i in range(6)]
        + [_row(match_id=f"d{i}", fthg=1, ftag=1, date=pd.Timestamp("2025-02-01") + pd.Timedelta(days=i)) for i in range(6)]
        + [_row(match_id=f"a{i}", fthg=0, ftag=2, date=pd.Timestamp("2025-03-01") + pd.Timedelta(days=i)) for i in range(6)]
    )
    df = pd.DataFrame(rows)
    sampled = harness._stratified_sample(df, sample=9)
    assert len(sampled) <= 9

    def _result(r):
        return "home" if r["fthg"] > r["ftag"] else ("away" if r["fthg"] < r["ftag"] else "draw")

    counts = sampled.apply(_result, axis=1).value_counts()
    # Roughly balanced: no category should be completely absent given 9 from 3 equal groups of 6
    assert set(counts.index) == {"home", "draw", "away"}


def test_backtest_harness_run_uses_process_match_row():
    harness = BacktestHarness(config=_make_config())
    fake_df = pd.DataFrame([_row(match_id="only-one")])
    with patch.object(harness, "load_matches", return_value=fake_df), \
         patch("src.agent.backtest.process_match_row") as mock_process:
        mock_process.return_value = "RECORD-SENTINEL"
        records = harness.run("2025-01-01", "2025-12-31")

    assert records == ["RECORD-SENTINEL"]
    mock_process.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.backtest'`

- [ ] **Step 3: Implement backtest.py**

```python
"""Outcome loading and backtest replay (A12).

process_match_row() is the single source of truth for "run one historical
match through the agent in replay mode and score it" — used by both the
synchronous BacktestHarness.run() and the concurrent agent-backtest CLI path
(main.py), so the two never drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.agent.agent_config import AgentConfig
from src.utils.db_manager import DuckDBManager

# Markets whose correctness can be programmatically resolved from raw_matches.
# home_corners/away_corners are excluded: the MatchRecommendation schema has no
# numeric line field for them (only current_odds/min_odds), so we cannot tell
# what threshold the agent's "selection" refers to. Documented limitation —
# see documents/agent_techspec.md Known Limitations.
_RESOLVABLE_MARKETS = {"result_3way", "btts", "total_goals"}


@dataclass
class BacktestRecord:
    match_id: str
    home_team: str
    away_team: str
    date: str
    league: str
    recommendation: dict[str, Any]
    actual: dict[str, Any]
    market_results: list[dict[str, Any]] = field(default_factory=list)


def load_outcome(row: pd.Series) -> dict[str, Any]:
    """Derive the resolvable outcome categories for a finished match."""
    fthg, ftag = int(row["fthg"]), int(row["ftag"])
    if fthg > ftag:
        result = "home"
    elif fthg < ftag:
        result = "away"
    else:
        result = "draw"
    total_goals = fthg + ftag
    return {
        "fthg": fthg,
        "ftag": ftag,
        "result": result,
        "btts": "yes" if (fthg > 0 and ftag > 0) else "no",
        "total_goals": total_goals,
        "total_goals_side": "over_2.5" if total_goals > 2 else "under_2.5",
    }


def _market_correct(market_rec: dict[str, Any], actual: dict[str, Any]) -> bool | None:
    market = market_rec.get("market")
    selection = market_rec.get("selection")
    if market == "result_3way":
        return selection == actual["result"]
    if market == "btts":
        return selection == actual["btts"]
    if market == "total_goals":
        return selection == actual["total_goals_side"]
    return None


def _date_str(row: pd.Series) -> str:
    value = row["date"]
    return str(value.date()) if hasattr(value, "date") else str(value)


def _build_match_info(row: pd.Series) -> dict[str, Any]:
    match_info: dict[str, Any] = {
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "date": _date_str(row),
        "league": row["league"],
    }
    if row.get("odds_h") and row.get("odds_d") and row.get("odds_a"):
        match_info["odds"] = {"home": row["odds_h"], "draw": row["odds_d"], "away": row["odds_a"]}
    return match_info


def process_match_row(row: pd.Series, config: AgentConfig) -> BacktestRecord:
    """Replay one historical match through the agent and score its recommendation.

    Sets the module-level SnapshotStore to replay mode for this match_id before
    calling run_agent, and always resets it to live mode afterward (even on
    error) so a failed match doesn't leave a later, unrelated call in replay
    mode by accident.
    """
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools

    match_id = row["match_id"]
    match_info = _build_match_info(row)

    agent_tools.configure_snapshot_store("replay", match_id=match_id)
    try:
        recommendation = run_agent(match_info=match_info, config=config)
    finally:
        agent_tools.configure_snapshot_store("live")

    actual = load_outcome(row)
    market_results = [
        {**m, "correct": _market_correct(m, actual)}
        for m in recommendation.get("markets", [])
    ]
    return BacktestRecord(
        match_id=match_id,
        home_team=row["home_team"],
        away_team=row["away_team"],
        date=match_info["date"],
        league=row["league"],
        recommendation=recommendation,
        actual=actual,
        market_results=market_results,
    )


class BacktestHarness:
    """Loads historical matches and replays them through the agent via process_match_row()."""

    def __init__(self, config: AgentConfig | None = None, db_path: str = "config.yaml") -> None:
        self.config = config or AgentConfig.default()
        self.db = DuckDBManager(config_path=db_path)

    def load_matches(
        self,
        from_date: str,
        to_date: str,
        league: str | None = None,
        sample: int | None = None,
    ) -> pd.DataFrame:
        query = (
            "SELECT match_id, league, date, home_team, away_team, "
            "odds_h, odds_d, odds_a, fthg, ftag, hc, ac "
            "FROM raw_matches WHERE date >= ? AND date <= ? AND fthg IS NOT NULL AND ftag IS NOT NULL"
        )
        params: list[Any] = [from_date, to_date]
        if league:
            query += " AND UPPER(league) = ?"
            params.append(league.upper())
        query += " ORDER BY date"
        with self.db.connection() as conn:
            matches = conn.execute(query, params).fetchdf()

        if sample is not None and len(matches) > sample:
            matches = self._stratified_sample(matches, sample)
        return matches

    @staticmethod
    def _stratified_sample(matches: pd.DataFrame, sample: int) -> pd.DataFrame:
        """Stratify by actual result (home/draw/away) — the only outcome dimension
        known before running the agent. ('bet/no-bet' is the agent's own output,
        so it can't be used to pre-stratify the input sample.) Seeded for
        reproducibility so agent-compare (A16) can re-run different configs over
        the identical sample.
        """

        def _result(row: pd.Series) -> str:
            if row["fthg"] > row["ftag"]:
                return "home"
            if row["fthg"] < row["ftag"]:
                return "away"
            return "draw"

        matches = matches.copy()
        matches["_stratum"] = matches.apply(_result, axis=1)
        n_strata = matches["_stratum"].nunique()
        per_stratum = max(1, sample // n_strata)
        sampled = (
            matches.groupby("_stratum", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=42))
        )
        return sampled.drop(columns="_stratum").sort_values("date").reset_index(drop=True)

    def run(
        self,
        from_date: str,
        to_date: str,
        league: str | None = None,
        sample: int | None = None,
    ) -> list[BacktestRecord]:
        matches = self.load_matches(from_date, to_date, league=league, sample=sample)
        return [process_match_row(row, self.config) for _, row in matches.iterrows()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/backtest.py tests/test_backtest.py
git commit -m "feat(agent): add BacktestHarness, outcome loader, process_match_row (A12)"
```

---

## Task 6: Flat-Stake and Kelly Bankroll Simulation (A13 + A15)

**Files:**
- Create: `src/agent/staking.py`
- Test: `tests/test_staking.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for bankroll simulation (A13 flat stake, A15 Kelly)."""
from __future__ import annotations

from src.agent.backtest import BacktestRecord
from src.agent.staking import simulate_flat_stake, simulate_kelly_stake


def _record(match_id: str, markets: list[dict]) -> BacktestRecord:
    return BacktestRecord(
        match_id=match_id, home_team="A", away_team="B", date="2025-01-01", league="E0",
        recommendation={"overall": "direct_bet", "markets": markets},
        actual={"result": "home"}, market_results=markets,
    )


def test_flat_stake_winning_bet_increases_bankroll():
    markets = [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "correct": True}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0, stake_pct=0.01)
    assert result.ending_bankroll == 1010.0  # +10 stake * (2.0-1)
    assert len(result.bets) == 1
    assert result.bets[0].won is True


def test_flat_stake_losing_bet_decreases_bankroll():
    markets = [{"market": "result_3way", "selection": "away", "recommendation_type": "direct_bet", "current_odds": 2.0, "correct": False}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0, stake_pct=0.01)
    assert result.ending_bankroll == 990.0


def test_flat_stake_skips_non_direct_bet_markets():
    markets = [{"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 1.8, "correct": True}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.ending_bankroll == 1000.0
    assert result.bets == []


def test_flat_stake_skips_unresolvable_markets():
    markets = [{"market": "home_corners", "selection": "over_4.5", "recommendation_type": "direct_bet", "current_odds": 1.9, "correct": None}]
    result = simulate_flat_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.bets == []


def test_flat_stake_equity_curve_starts_with_initial_bankroll():
    result = simulate_flat_stake([], starting_bankroll=500.0)
    assert result.equity_curve == [500.0]


def test_kelly_stake_sizes_by_value_edge():
    markets = [{
        "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
        "current_odds": 3.0, "value_edge": 0.10, "correct": True,
    }]
    result = simulate_kelly_stake([_record("m1", markets)], starting_bankroll=1000.0, max_fraction=0.5)
    # fraction = value_edge / (odds - 1) = 0.10 / 2.0 = 0.05 -> stake = 50
    assert result.bets[0].stake == 50.0
    assert result.ending_bankroll == 1000.0 + 50.0 * (3.0 - 1)


def test_kelly_stake_caps_at_max_fraction():
    markets = [{
        "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
        "current_odds": 1.5, "value_edge": 0.9, "correct": True,
    }]
    result = simulate_kelly_stake([_record("m1", markets)], starting_bankroll=1000.0, max_fraction=0.1)
    assert result.bets[0].stake == 100.0  # capped at 10% of 1000


def test_kelly_stake_skips_negative_or_zero_edge():
    markets = [{
        "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
        "current_odds": 2.0, "value_edge": -0.05, "correct": True,
    }]
    result = simulate_kelly_stake([_record("m1", markets)], starting_bankroll=1000.0)
    assert result.bets == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_staking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.staking'`

- [ ] **Step 3: Implement staking.py**

```python
"""Bankroll simulation: flat-stake (A13) and Kelly criterion (A15) modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.backtest import BacktestRecord


@dataclass
class BetOutcome:
    match_id: str
    market: str
    selection: str
    odds: float
    stake: float
    won: bool
    payout: float  # net profit (positive) or loss (negative) — already includes stake direction


@dataclass
class BankrollResult:
    starting_bankroll: float
    ending_bankroll: float
    equity_curve: list[float] = field(default_factory=list)
    bets: list[BetOutcome] = field(default_factory=list)


def simulate_flat_stake(
    records: list["BacktestRecord"],
    starting_bankroll: float = 1000.0,
    stake_pct: float = 0.01,
) -> BankrollResult:
    """Fixed stake = stake_pct * starting_bankroll on every direct_bet recommendation
    with a resolvable outcome. Win: bankroll += stake * (odds - 1). Loss: bankroll -= stake."""
    bankroll = starting_bankroll
    equity_curve = [bankroll]
    bets: list[BetOutcome] = []
    flat_stake = starting_bankroll * stake_pct

    for record in records:
        for m in record.market_results:
            if m.get("recommendation_type") != "direct_bet":
                continue
            if m.get("correct") is None:
                continue  # unresolvable market (e.g. corners) — cannot settle, skip
            odds = float(m["current_odds"])
            won = bool(m["correct"])
            payout = flat_stake * (odds - 1) if won else -flat_stake
            bankroll += payout
            equity_curve.append(bankroll)
            bets.append(BetOutcome(
                match_id=record.match_id, market=m["market"], selection=m["selection"],
                odds=odds, stake=flat_stake, won=won, payout=payout,
            ))

    return BankrollResult(starting_bankroll=starting_bankroll, ending_bankroll=bankroll, equity_curve=equity_curve, bets=bets)


def simulate_kelly_stake(
    records: list["BacktestRecord"],
    starting_bankroll: float = 1000.0,
    max_fraction: float = 0.10,
) -> BankrollResult:
    """Kelly stake = value_edge / (odds - 1) * current bankroll, capped at max_fraction.
    Bets with non-positive edge are skipped (Kelly fraction would be <= 0)."""
    bankroll = starting_bankroll
    equity_curve = [bankroll]
    bets: list[BetOutcome] = []

    for record in records:
        for m in record.market_results:
            if m.get("recommendation_type") != "direct_bet":
                continue
            if m.get("correct") is None:
                continue
            odds = float(m["current_odds"])
            value_edge = float(m.get("value_edge", 0.0))
            if odds <= 1.0 or value_edge <= 0:
                continue
            fraction = min(value_edge / (odds - 1.0), max_fraction)
            stake = bankroll * fraction
            won = bool(m["correct"])
            payout = stake * (odds - 1) if won else -stake
            bankroll += payout
            equity_curve.append(bankroll)
            bets.append(BetOutcome(
                match_id=record.match_id, market=m["market"], selection=m["selection"],
                odds=odds, stake=stake, won=won, payout=payout,
            ))

    return BankrollResult(starting_bankroll=starting_bankroll, ending_bankroll=bankroll, equity_curve=equity_curve, bets=bets)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_staking.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/staking.py tests/test_staking.py
git commit -m "feat(agent): add flat-stake and Kelly bankroll simulation (A13, A15)"
```

---

## Task 7: Evaluation Report (A13 part 2)

**Files:**
- Create: `src/agent/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for evaluation report computation (A13)."""
from __future__ import annotations

import json

from src.agent.agent_config import AgentConfig
from src.agent.evaluation import (
    build_evaluation_report,
    compute_max_drawdown,
    config_hash,
    save_report,
)
from src.agent.staking import BankrollResult, BetOutcome


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        model="llama3.1:8b", provider="ollama", temperature=0.1, max_tool_calls=10,
        min_odds_threshold=2.0, min_value_edge=0.05, markets=["result_3way", "btts"],
        system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_compute_max_drawdown_simple_peak_to_trough():
    curve = [1000, 1100, 900, 950, 1200]
    # peak 1100 -> trough 900 = (1100-900)/1100 = 0.1818...
    assert round(compute_max_drawdown(curve), 4) == 0.1818


def test_compute_max_drawdown_no_drawdown_when_monotonic_increase():
    assert compute_max_drawdown([1000, 1100, 1200]) == 0.0


def test_build_evaluation_report_computes_roi_and_hit_rate():
    bankroll = BankrollResult(
        starting_bankroll=1000.0, ending_bankroll=1010.0,
        equity_curve=[1000.0, 1010.0],
        bets=[BetOutcome(match_id="m1", market="result_3way", selection="home", odds=2.0, stake=10.0, won=True, payout=10.0)],
    )

    class _Rec:
        def __init__(self, overall):
            self.recommendation = {"overall": overall}

    records = [_Rec("direct_bet"), _Rec("insufficient_data")]
    report = build_evaluation_report(records, bankroll)

    assert report["bets_placed"] == 1
    assert report["bets_won"] == 1
    assert report["hit_rate"] == 1.0
    assert report["roi"] == 1.0  # 10 profit / 10 staked
    assert report["bet_frequency"] == 0.5  # 1 bet / 2 matches
    assert report["insufficient_data_rate"] == 0.5
    assert report["matches_evaluated"] == 2


def test_build_evaluation_report_handles_zero_bets():
    bankroll = BankrollResult(starting_bankroll=1000.0, ending_bankroll=1000.0, equity_curve=[1000.0], bets=[])

    class _Rec:
        recommendation = {"overall": "no_bet"}

    report = build_evaluation_report([_Rec()], bankroll)
    assert report["roi"] == 0.0
    assert report["hit_rate"] == 0.0
    assert report["bets_placed"] == 0


def test_config_hash_deterministic_and_order_independent():
    cfg_a = _make_config(markets=["btts", "result_3way"])
    cfg_b = _make_config(markets=["result_3way", "btts"])
    assert config_hash(cfg_a) == config_hash(cfg_b)
    assert len(config_hash(cfg_a)) == 8


def test_config_hash_differs_for_different_model():
    cfg_a = _make_config(model="llama3.1:8b")
    cfg_b = _make_config(model="llama3.2:3b")
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_save_report_writes_json_file(tmp_path):
    report = {"roi": 0.05, "hit_rate": 0.5}
    path = save_report(report, _make_config(), base_dir=str(tmp_path))
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded == report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.evaluation'`

- [ ] **Step 3: Implement evaluation.py**

```python
"""Backtest evaluation report computation (A13)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.agent_config import AgentConfig

if TYPE_CHECKING:
    from src.agent.staking import BankrollResult


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Largest peak-to-trough fractional decline observed in the equity curve."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd


def build_evaluation_report(records: list[Any], bankroll_result: "BankrollResult") -> dict[str, Any]:
    total_staked = sum(bet.stake for bet in bankroll_result.bets)
    total_profit = sum(bet.payout for bet in bankroll_result.bets)
    bets_won = sum(1 for bet in bankroll_result.bets if bet.won)
    bets_placed = len(bankroll_result.bets)
    insufficient = sum(1 for r in records if r.recommendation.get("overall") == "insufficient_data")

    roi = total_profit / total_staked if total_staked > 0 else 0.0
    hit_rate = bets_won / bets_placed if bets_placed > 0 else 0.0
    bet_frequency = bets_placed / len(records) if records else 0.0
    insufficient_data_rate = insufficient / len(records) if records else 0.0

    return {
        "matches_evaluated": len(records),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "roi": round(roi, 6),
        "hit_rate": round(hit_rate, 6),
        "bet_frequency": round(bet_frequency, 6),
        "max_drawdown": round(compute_max_drawdown(bankroll_result.equity_curve), 6),
        "insufficient_data_rate": round(insufficient_data_rate, 6),
        "starting_bankroll": bankroll_result.starting_bankroll,
        "ending_bankroll": round(bankroll_result.ending_bankroll, 2),
    }


def config_hash(config: AgentConfig) -> str:
    """Stable 8-char hash identifying a config's relevant tuning fields (order-independent on markets)."""
    canonical = json.dumps(
        {
            "model": config.model,
            "provider": config.provider,
            "temperature": config.temperature,
            "max_tool_calls": config.max_tool_calls,
            "min_odds_threshold": config.min_odds_threshold,
            "min_value_edge": config.min_value_edge,
            "markets": sorted(config.markets),
            "system_prompt_version": config.system_prompt_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def save_report(report: dict[str, Any], config: AgentConfig, base_dir: str = "reports/agent_backtest") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{timestamp}_{config_hash(config)}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("Agent Backtest Evaluation Report")
    print("=" * 50)
    for key, value in report.items():
        print(f"  {key:<22}: {value}")
    print("=" * 50)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluation.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/evaluation.py tests/test_evaluation.py
git commit -m "feat(agent): add backtest evaluation report computation (A13)"
```

---

## Task 8: agent-backtest CLI with Parallelism (A14)

**Files:**
- Modify: `main.py` (add parser, async helper, `run_agent_backtest`, dispatch)

- [ ] **Step 1: Add the `agent-backtest` subparser**

In `main.py`, after the `agent-snapshot` parser block added in Task 4:

```python
    # agent-backtest
    agent_backtest_parser = subparsers.add_parser(
        "agent-backtest",
        help="Replay recorded snapshots through the agent and report bankroll performance",
    )
    agent_backtest_parser.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD (inclusive)")
    agent_backtest_parser.add_argument("--to-date", required=True, help="End date YYYY-MM-DD (inclusive)")
    agent_backtest_parser.add_argument("--league", default=None, help="League code (e.g. E0). Omit for all leagues.")
    agent_backtest_parser.add_argument("--stake-mode", choices=["flat", "kelly"], default="flat")
    agent_backtest_parser.add_argument("--sample", type=int, default=None, help="Stratified sample size before running the full set")
    agent_backtest_parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent agent runs")
    agent_backtest_parser.add_argument("--config", default=None, help="Path to agent_config.yaml (default: config/agent_config.yaml)")
```

- [ ] **Step 2: Add the async concurrency helper and `run_agent_backtest`**

In `main.py`, after `run_agent_snapshot` (added in Task 4), add:

```python
async def _run_backtest_concurrent(matches, config, concurrency: int) -> list:
    """Run process_match_row for every match concurrently, bounded by a semaphore.
    Each call runs in its own thread via asyncio.to_thread since the agent graph
    and tools are synchronous; SnapshotStore's thread-local state (A09) keeps
    concurrent replay contexts from clobbering each other."""
    import asyncio

    from tqdm import tqdm

    from src.agent.backtest import process_match_row

    semaphore = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(matches), desc="Backtesting")
    rows = [row for _, row in matches.iterrows()]

    async def _run_one(row):
        async with semaphore:
            record = await asyncio.to_thread(process_match_row, row, config)
            progress.update(1)
            return record

    try:
        records = await asyncio.gather(*[_run_one(row) for row in rows])
    finally:
        progress.close()
    return list(records)


def run_agent_backtest(
    from_date: str,
    to_date: str,
    league: str | None,
    stake_mode: str,
    sample: int | None,
    concurrency: int,
    config_path: str | None,
) -> None:
    """Replay agent recommendations over historical snapshots and report bankroll performance (A14)."""
    import asyncio

    from src.agent.agent_config import AgentConfig
    from src.agent.backtest import BacktestHarness
    from src.agent.evaluation import build_evaluation_report, print_report, save_report
    from src.agent.staking import simulate_flat_stake, simulate_kelly_stake

    cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    harness = BacktestHarness(config=cfg)
    matches = harness.load_matches(from_date, to_date, league=league, sample=sample)
    print(f"Running backtest over {len(matches)} matches (concurrency={concurrency})...")

    records = asyncio.run(_run_backtest_concurrent(matches, cfg, concurrency))

    stake_fn = simulate_kelly_stake if stake_mode == "kelly" else simulate_flat_stake
    bankroll_result = stake_fn(records)
    report = build_evaluation_report(records, bankroll_result)
    print_report(report)
    path = save_report(report, cfg)
    print(f"\nReport saved to {path}")
```

- [ ] **Step 3: Add dispatch in `main()`**

After the `agent-snapshot` dispatch block (added in Task 4):

```python
    elif args.command == "agent-backtest":
        run_agent_backtest(
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            stake_mode=args.stake_mode,
            sample=args.sample,
            concurrency=args.concurrency,
            config_path=args.config,
        )
```

- [ ] **Step 4: Manual verification (requires existing snapshots — skip the live run if none exist yet, just verify parsing)**

Run: `python main.py agent-backtest --help`
Expected: prints usage with `--from-date`, `--to-date`, `--stake-mode`, `--sample`, `--concurrency`, `--config` listed, exits 0

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(agent): add agent-backtest CLI with asyncio-based concurrency (A14)"
```

---

## Task 9: Config Comparison Framework (A16)

**Files:**
- Create: `src/agent/comparison.py`
- Modify: `main.py` (add `agent-compare` parser + function + dispatch)
- Test: `tests/test_comparison.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for config comparison framework (A16)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.agent.comparison import compare_configs, print_comparison_table, save_comparison


def test_compare_configs_runs_each_config_and_collects_reports():
    fake_report_a = {"roi": 0.05, "hit_rate": 0.5, "bet_frequency": 0.2, "max_drawdown": 0.1, "insufficient_data_rate": 0.0}
    fake_report_b = {"roi": -0.02, "hit_rate": 0.4, "bet_frequency": 0.3, "max_drawdown": 0.2, "insufficient_data_rate": 0.1}

    with patch("src.agent.comparison.AgentConfig") as MockCfg, \
         patch("src.agent.comparison.BacktestHarness") as MockHarness, \
         patch("src.agent.comparison.simulate_flat_stake") as mock_stake, \
         patch("src.agent.comparison.build_evaluation_report", side_effect=[fake_report_a, fake_report_b]):
        MockCfg.from_yaml.side_effect = lambda p: MagicMock(name=p)
        instance_a = MagicMock()
        instance_b = MagicMock()
        MockHarness.side_effect = [instance_a, instance_b]
        instance_a.run.return_value = ["rec-a"]
        instance_b.run.return_value = ["rec-b"]
        mock_stake.return_value = MagicMock()

        results = compare_configs(
            ["config/a.yaml", "config/b.yaml"],
            from_date="2025-01-01", to_date="2025-06-01", league="E0", sample=20,
        )

    assert results == {"config/a.yaml": fake_report_a, "config/b.yaml": fake_report_b}
    instance_a.run.assert_called_once_with("2025-01-01", "2025-06-01", league="E0", sample=20)


def test_print_comparison_table_runs_without_error(capsys):
    results = {
        "config/a.yaml": {"roi": 0.05, "hit_rate": 0.5, "bet_frequency": 0.2, "max_drawdown": 0.1, "insufficient_data_rate": 0.0},
    }
    print_comparison_table(results)
    captured = capsys.readouterr()
    assert "config/a.yaml" in captured.out
    assert "roi" in captured.out


def test_save_comparison_writes_json(tmp_path):
    results = {"config/a.yaml": {"roi": 0.05}}
    path = save_comparison(results, base_dir=str(tmp_path))
    assert path.exists()
    assert json.loads(path.read_text()) == results
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.comparison'`

- [ ] **Step 3: Implement comparison.py**

```python
"""Config comparison framework: re-run BacktestHarness for multiple configs
over the identical (seeded) match sample so the only varying factor is the
agent config itself (A16)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.agent_config import AgentConfig
from src.agent.backtest import BacktestHarness
from src.agent.evaluation import build_evaluation_report
from src.agent.staking import simulate_flat_stake, simulate_kelly_stake


def compare_configs(
    config_paths: list[str],
    from_date: str,
    to_date: str,
    league: str | None = None,
    sample: int | None = None,
    stake_mode: str = "flat",
) -> dict[str, dict[str, Any]]:
    """Run each config's agent over the same match set (same from_date/to_date/league/
    sample -> BacktestHarness._stratified_sample's fixed random_state=42 guarantees an
    identical sample across configs) and return {config_path: evaluation_report}."""
    stake_fn = simulate_kelly_stake if stake_mode == "kelly" else simulate_flat_stake
    results: dict[str, dict[str, Any]] = {}
    for path in config_paths:
        cfg = AgentConfig.from_yaml(path)
        harness = BacktestHarness(config=cfg)
        records = harness.run(from_date, to_date, league=league, sample=sample)
        bankroll_result = stake_fn(records)
        results[path] = build_evaluation_report(records, bankroll_result)
    return results


def print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    metrics = ["roi", "hit_rate", "bet_frequency", "max_drawdown", "insufficient_data_rate"]
    header = f"{'config':<40}" + "".join(f"{m:>16}" for m in metrics)
    print(header)
    print("-" * len(header))
    for path, report in results.items():
        row = f"{path:<40}" + "".join(f"{report.get(m, ''):>16}" for m in metrics)
        print(row)


def save_comparison(results: dict[str, dict[str, Any]], base_dir: str = "reports/agent_backtest") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"comparison_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_comparison.py -v`
Expected: 3 passed

- [ ] **Step 5: Add the `agent-compare` CLI command**

In `main.py`, after the `agent-backtest` parser block (Task 8):

```python
    # agent-compare
    agent_compare_parser = subparsers.add_parser(
        "agent-compare",
        help="Compare multiple agent configs over the same backtest snapshot set",
    )
    agent_compare_parser.add_argument("--configs", nargs="+", required=True, help="Paths to two or more agent_config.yaml files")
    agent_compare_parser.add_argument("--from-date", required=True)
    agent_compare_parser.add_argument("--to-date", required=True)
    agent_compare_parser.add_argument("--league", default=None)
    agent_compare_parser.add_argument("--sample", type=int, default=None)
    agent_compare_parser.add_argument("--stake-mode", choices=["flat", "kelly"], default="flat")
```

After `run_agent_backtest` in `main.py`, add:

```python
def run_agent_compare(
    config_paths: list[str],
    from_date: str,
    to_date: str,
    league: str | None,
    sample: int | None,
    stake_mode: str,
) -> None:
    """Compare agent configs over the same backtest snapshot set (A16)."""
    from src.agent.comparison import compare_configs, print_comparison_table, save_comparison

    results = compare_configs(config_paths, from_date, to_date, league=league, sample=sample, stake_mode=stake_mode)
    print_comparison_table(results)
    path = save_comparison(results)
    print(f"\nComparison saved to {path}")
```

And dispatch, after the `agent-backtest` dispatch block (Task 8):

```python
    elif args.command == "agent-compare":
        run_agent_compare(
            config_paths=args.configs,
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            sample=args.sample,
            stake_mode=args.stake_mode,
        )
```

- [ ] **Step 6: Manual verification**

Run: `python main.py agent-compare --help`
Expected: prints usage with `--configs`, `--from-date`, `--to-date`, `--league`, `--sample`, `--stake-mode` listed, exits 0

- [ ] **Step 7: Commit**

```bash
git add src/agent/comparison.py tests/test_comparison.py main.py
git commit -m "feat(agent): add config comparison framework and agent-compare CLI (A16)"
```

---

## Task 10: Full Regression Pass + Update agent_user_stories.md

**Files:**
- Modify: `documents/agent_user_stories.md`

- [ ] **Step 1: Run the complete agent test suite**

Run:
```bash
python -m pytest tests/test_agent_config.py tests/test_agent_schema.py tests/test_agent_graph.py \
  tests/test_snapshot_store.py tests/test_agent_tools_snapshot.py tests/test_backtest.py \
  tests/test_staking.py tests/test_evaluation.py tests/test_comparison.py -v
```
Expected: all pass, zero failures. If any test fails, return to the relevant Task above and fix before proceeding — do not edit `agent_user_stories.md` until this is green.

- [ ] **Step 2: Update story statuses**

In `documents/agent_user_stories.md`, change `**Status:** active` to `**Status:** completed` for stories A09 through A16 (8 edits — one per story header line). Example for A09:

```diff
-### A09 — Implement SnapshotStore
-**Size:** M | **Status:** active | **Milestone:** M3 | **Depends on:** A01
+### A09 — Implement SnapshotStore
+**Size:** M | **Status:** completed | **Milestone:** M3 | **Depends on:** A01
```

Repeat the same `active` → `completed` edit for A10, A11, A12, A13, A14, A15, A16.

- [ ] **Step 3: Commit**

```bash
git add documents/agent_user_stories.md
git commit -m "docs(agent): mark A09-A16 completed after snapshot/backtest implementation"
```

---

## Self-Review Notes (already applied above, kept for reviewer reference)

- **Spec coverage:** A09 (Task 1), A10 (Task 2), A11 (Task 4), A12 (Task 5), A13 (Tasks 6+7), A14 (Task 8), A15 (Task 6), A16 (Task 9) — every story has a task. `agent_prd.md`'s CLI command examples (`agent-snapshot`, `agent-backtest`) match the flags implemented.
- **Thread-safety fix folded into A09**, not deferred — see plan-level scope decision #1. Without this, A14's `--concurrency` flag would silently corrupt replay state across threads.
- **DRY:** `process_match_row()` (Task 5) is the single implementation used by both `BacktestHarness.run()` (sync, used by A12 and A16) and `_run_backtest_concurrent()` (async, used by A14) — no duplicated match-processing logic.
- **Type consistency check:** `BacktestRecord.market_results` entries are plain dicts with `correct: bool | None` added by `process_match_row`/`_market_correct` — `staking.py` and `evaluation.py` both consume this same shape (`m.get("recommendation_type")`, `m.get("correct")`, `m["current_odds"]`, `m.get("value_edge", 0.0)`) consistently across Tasks 5, 6, 7.
- **No placeholders:** every step has complete, runnable code; no "TODO" or "add error handling" left unstated.
