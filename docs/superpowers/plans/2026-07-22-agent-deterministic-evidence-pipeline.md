# Deterministic Evidence Pipeline (A30/A31/A32) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the agent's LLM-tool-choice for competition resolution, baseline research, and the ML forecast with required deterministic graph steps, so a recommendation can never be produced without real evidence behind it.

**Architecture:** Three new deterministic LangGraph nodes (`resolve_competition_node` → `research_node` → `forecast_node`, all in a new `src/agent/pipeline.py`) run before the LLM ever sees the match. A conditional edge short-circuits straight to `output_node` if the forecast can't be produced (no odds available from any source, or a tool error). `forecast_league`/`forecast_international`/`resolve_competition` are removed from the LLM's tool list — only `web_search` remains, for optional follow-up. `output_node` gains a structural backstop (A30) and a research-coverage confidence downgrade (A32).

**Tech Stack:** Python, LangGraph `StateGraph`, LangChain tools, pytest, existing `SnapshotStore` record/replay machinery.

**Full design reference:** `docs/superpowers/specs/2026-07-22-agent-phase11-design.md`

**Not in this plan:** A33 (train/critic mode) and A34 (rebaseline) — both depend on this plan landing first and will be planned separately once this is merged and verified.

---

## Important technical correction from the approved design doc

The design doc's A31 section says a missing-odds forecast "runs odds-less." **This is not possible** — `ForecastService.forecast_upcoming()` requires `odds_h`/`odds_d`/`odds_a` as non-optional floats on *both* the league and international paths (`src/forecast/forecast_service.py:321-323`, and `_compute_mkt_features_from_odds` divides by each of them directly at line 484-486). There is no odds-less forecast path in the underlying service.

Corrected behavior implemented by this plan: if no odds are available from either the caller or `research_node`'s odds-verification search, `forecast_node` does **not** attempt to call the forecast service at all — it returns an error-shaped payload immediately, which short-circuits to `insufficient_data` via the same routing used for a genuine tool failure. This is stricter than originally scoped (blocks the whole recommendation, not just `direct_bet` on odds-dependent markets) but is the only behavior the underlying code actually supports, and it removes a worse existing behavior: `config/prompts/agent_v1.txt` line 19 currently tells the LLM to invent fake odds (`odds_h=2.5, odds_d=3.2, odds_a=2.9`) when it doesn't have real ones. This plan deletes that instruction — a fabricated forecast is worse than a declined one.

---

## File Structure

- **Modify:** `src/agent/tools.py` — extract `_dated_web_search`, narrow `get_default_tools()` to `[web_search]`.
- **Create:** `src/agent/pipeline.py` — `resolve_competition_node`, `research_node`, `forecast_node`, `_parse_odds_from_search_text`, `_format_evidence_message`.
- **Modify:** `src/agent/graph.py` — new `AgentState` fields, `_extract_forecast_diagnostics` reads a payload instead of scanning messages, new `_apply_a30_backstop`/`_apply_research_coverage_downgrade`, `_build_recommendation` new signature, `build_graph` rewired with the 3 new nodes + routing, `output_node` early-return, `run_agent` prompt tweak.
- **Modify:** `config/prompts/agent_v1.txt` — remove tool-selection workflow steps, describe pre-supplied evidence instead.
- **Create:** `tests/test_agent_pipeline.py` — unit tests for all 3 new nodes.
- **Modify:** `tests/test_agent_forecast_diagnostics.py` — rewritten for the new `_extract_forecast_diagnostics`/`_build_recommendation` signatures, plus new A30/A32 tests.
- **Modify:** `tests/test_agent_graph.py` — routing helper test + 2 end-to-end `run_agent` tests proving the LLM is never invoked on a forecast failure.
- **Modify:** `tests/test_agent_tools_snapshot.py` — one new test for the narrowed `get_default_tools()`.
- **Modify:** `documents/agent_user_stories.md` — mark A30/A31/A32 completed with implementation notes (final task).

**Files intentionally left unmodified** (verified compatible during design research, listed so the implementer doesn't second-guess them): `tests/test_agent_tool_selection.py` (still calls the `@tool`-decorated `resolve_competition`/`forecast_international` directly via `.invoke()`, unaffected by their removal from `get_default_tools()`), `src/agent/schema.py` (A28/A29 validation is untouched — A30/A32 logic lives in `graph.py` because it needs `forecast_payload`/`research_evidence`, which `extract_recommendation()` never receives), `src/agent/backtest.py`, `main.py`, `app/backend/t30_refresh.py` (none of them hardcode a tools list — they all rely on `run_agent`'s `get_default_tools()` default).

---

### Task 1: Refactor `tools.py` — shared `_dated_web_search`, narrow `get_default_tools()`

**Files:**
- Modify: `src/agent/tools.py`
- Test: `tests/test_agent_tools_snapshot.py`

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/test_agent_tools_snapshot.py`:

```python
def test_get_default_tools_only_exposes_web_search():
    from src.agent.tools import get_default_tools
    assert [t.name for t in get_default_tools()] == ["web_search"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_tools_snapshot.py::test_get_default_tools_only_exposes_web_search -v`
Expected: FAIL — `assert ['resolve_competition', 'web_search', 'forecast_league', 'forecast_international'] == ['web_search']`

- [ ] **Step 3: Refactor `web_search` to use a shared, reusable search function**

In `src/agent/tools.py`, replace:

```python
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
```

with:

```python
def _dated_web_search(query: str) -> str:
    """A32: shared by both the LLM-facing web_search tool and the deterministic
    research_node baseline searches. Appends a before:<match_date> filter during
    record/replay (A10) to reduce post-match result leakage, then runs the
    (possibly snapshot-wrapped) search."""
    effective_query = query
    if _snapshot_store.mode in ("record", "replay") and _snapshot_store.match_date:
        effective_query = f"{query} before:{_snapshot_store.match_date}"
    return _snapshot_store.wrap("web_search", _web_search_impl)(query=effective_query)


@tool
def web_search(query: str) -> str:
    """Search the web for football match information: odds, team news, injuries, and lineups.

    Use for: finding current bookmaker odds, alternative team name spellings,
    injury/suspension reports, team selection hints, and recent form context.
    Always ignore any result that mentions a final score or match result."""
    return _dated_web_search(query)
```

- [ ] **Step 4: Narrow `get_default_tools()`**

Replace:

```python
def get_default_tools() -> list:
    return [resolve_competition, web_search, forecast_league, forecast_international]
```

with:

```python
def get_default_tools() -> list:
    """A31: forecast_league, forecast_international, and resolve_competition are
    no longer LLM-callable -- they're invoked directly by the deterministic
    pipeline nodes in src/agent/pipeline.py before the LLM ever runs. Only
    web_search remains available for the LLM's own optional follow-up digging."""
    return [web_search]
```

- [ ] **Step 5: Run the test and the full existing tools/tool-selection suites to confirm no regressions**

Run: `pytest tests/test_agent_tools_snapshot.py tests/test_agent_tool_selection.py -v`
Expected: All PASS (the tool-selection tests call `resolve_competition.invoke(...)` / `forecast_international.invoke(...)` directly — the `@tool` objects still exist and work exactly as before, only `get_default_tools()`'s list changed).

- [ ] **Step 6: Commit**

```bash
git add src/agent/tools.py tests/test_agent_tools_snapshot.py
git commit -m "$(cat <<'EOF'
refactor(agent): extract _dated_web_search, narrow get_default_tools to web_search only

Prepares for A31/A32: forecast_league, forecast_international, and
resolve_competition move to deterministic graph nodes and are no longer
LLM-callable tools. _dated_web_search is extracted so the new research_node
can reuse the exact same before:<match_date> filtering and snapshot
wrapping the LLM's own web_search tool already uses.
EOF
)"
```

---

### Task 2: `pipeline.py` — `resolve_competition_node`

**Files:**
- Create: `src/agent/pipeline.py`
- Test: `tests/test_agent_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_pipeline.py`:

```python
"""Tests for the deterministic evidence pipeline nodes (A31/A32):
resolve_competition_node, research_node, and forecast_node run before the LLM
ever sees the match, replacing the old LLM-tool-choice path for competition
resolution, baseline research, and the ML forecast."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent import tools as agent_tools
from src.agent.pipeline import resolve_competition_node


@pytest.fixture(autouse=True)
def reset_snapshot_store():
    agent_tools._snapshot_store.set_mode("live")
    yield
    agent_tools._snapshot_store.set_mode("live")


def _base_state(**overrides) -> dict:
    state = {
        "messages": [],
        "match_info": {"home_team": "Man City", "away_team": "Arsenal", "date": "2026-06-21", "league": "E0"},
        "recommendation": None,
        "tool_call_count": 0,
        "competition_resolution": None,
        "research_evidence": None,
        "forecast_payload": None,
    }
    state.update(overrides)
    return state


def test_resolve_competition_node_uses_real_registry_for_known_league():
    result = resolve_competition_node(_base_state())
    assert result["competition_resolution"]["tier"] == "competition_specific"
    assert result["competition_resolution"]["recommended_tool"] == "forecast_league"


def test_resolve_competition_node_defaults_general_purpose_when_no_league_supplied():
    state = _base_state(match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21"})
    result = resolve_competition_node(state)
    assert result["competition_resolution"] == {
        "competition": None,
        "tier": "general_purpose",
        "recommended_tool": "forecast_international",
    }


def test_resolve_competition_node_goes_through_snapshot_store():
    with patch(
        "src.agent.tools._resolve_competition_impl",
        wraps=agent_tools._resolve_competition_impl,
    ) as spy:
        resolve_competition_node(_base_state())
    spy.assert_called_once_with(competition_or_league="E0")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.pipeline'`

- [ ] **Step 3: Create `src/agent/pipeline.py` with `resolve_competition_node`**

```python
"""Deterministic evidence pipeline (A31/A32): competition resolution, baseline
web research, and the ML forecast all run here, as required graph nodes, before
the LLM ever sees the match -- replacing the old design where the LLM could
choose (or fail) to call resolve_competition/forecast_league/forecast_international
as tools. See docs/superpowers/specs/2026-07-22-agent-phase11-design.md."""

from __future__ import annotations

import json


def resolve_competition_node(state: dict) -> dict:
    """A31: deterministic competition-tier lookup. If match_info has no league
    at all (e.g. a genuinely unlabeled international fixture), there's nothing
    to look up -- default straight to general_purpose/forecast_international
    rather than calling the registry with an empty string."""
    league = state["match_info"].get("league")
    if not league:
        return {"competition_resolution": {
            "competition": None,
            "tier": "general_purpose",
            "recommended_tool": "forecast_international",
        }}

    from src.agent.tools import _resolve_competition_impl, get_snapshot_store

    raw = get_snapshot_store().wrap("resolve_competition", _resolve_competition_impl)(
        competition_or_league=league
    )
    return {"competition_resolution": json.loads(raw)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/pipeline.py tests/test_agent_pipeline.py
git commit -m "$(cat <<'EOF'
feat(agent): add resolve_competition_node, the first deterministic pipeline step

A31: competition-tier resolution moves from an LLM-callable tool the model
could skip to a required graph node that always runs first.
EOF
)"
```

---

### Task 3: `pipeline.py` — odds parsing + `research_node`

**Files:**
- Modify: `src/agent/pipeline.py`
- Test: `tests/test_agent_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_pipeline.py`:

```python
from src.agent.pipeline import _parse_odds_from_search_text, research_node


def test_parse_odds_from_search_text_finds_three_plausible_numbers():
    text = "Best odds: Man City 1.45, Draw 4.50, Arsenal 7.00 at time of writing."
    assert _parse_odds_from_search_text(text) == {"home": 1.45, "draw": 4.50, "away": 7.00}


def test_parse_odds_from_search_text_returns_none_with_fewer_than_three_numbers():
    assert _parse_odds_from_search_text("Man City are favourites at 1.45.") is None


def test_parse_odds_from_search_text_returns_none_for_empty_text():
    assert _parse_odds_from_search_text("") is None
    assert _parse_odds_from_search_text(None) is None


def test_research_node_runs_availability_and_form_searches_only_when_odds_supplied():
    with patch("src.agent.tools._dated_web_search", side_effect=["injury text", "form text"]) as mock_search:
        state = _base_state(match_info={
            "home_team": "A", "away_team": "B", "date": "2026-06-21",
            "odds": {"home": 2.0, "draw": 3.0, "away": 3.5},
        })
        result = research_node(state)

    assert result["research_evidence"]["availability"] == "injury text"
    assert result["research_evidence"]["form_context"] == "form text"
    assert result["research_evidence"]["odds_verification"] is None
    assert mock_search.call_count == 2


def test_research_node_also_runs_odds_search_when_caller_supplied_no_odds():
    with patch(
        "src.agent.tools._dated_web_search",
        side_effect=["injury text", "form text", "Man City 1.45 Draw 4.50 Arsenal 7.00"],
    ) as mock_search:
        state = _base_state(match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21"})
        result = research_node(state)

    assert mock_search.call_count == 3
    odds_verification = result["research_evidence"]["odds_verification"]
    assert odds_verification["parsed_odds"] == {"home": 1.45, "draw": 4.50, "away": 7.00}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_pipeline.py -v -k "parse_odds or research_node"`
Expected: FAIL with `ImportError: cannot import name '_parse_odds_from_search_text'`

- [ ] **Step 3: Implement odds parsing and `research_node`**

Append to `src/agent/pipeline.py`:

```python
import re

_ODDS_NUMBER_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,2}\b")


def _parse_odds_from_search_text(text: str | None) -> dict | None:
    """Best-effort extraction of three decimal odds (home/draw/away) from a
    web search snippet. Deliberately conservative: requires at least three
    plausible decimal-odds-shaped numbers (1.01-50.0) in the text and just
    takes the first three in reading order. This is a heuristic, not a
    guarantee -- forecast_node only ever falls back to it when the caller
    supplied no odds at all, and a low-confidence/failed parse (fewer than 3
    plausible numbers) correctly results in insufficient_data rather than a
    forecast built on a wrong guess."""
    if not text:
        return None
    numbers = [float(m) for m in _ODDS_NUMBER_PATTERN.findall(text)]
    plausible = [n for n in numbers if 1.01 <= n <= 50.0]
    if len(plausible) < 3:
        return None
    home, draw, away = plausible[:3]
    return {"home": home, "draw": draw, "away": away}


def research_node(state: dict) -> dict:
    """A32: guarantees minimum research coverage deterministically instead of
    depending on the LLM choosing to search. Always runs availability and
    recent-form searches; only runs an odds-verification search when the
    caller didn't already supply odds (match_info.get('odds'))."""
    from src.agent.tools import _dated_web_search

    match_info = state["match_info"]
    home, away = match_info["home_team"], match_info["away_team"]

    availability_text = _dated_web_search(f"{home} {away} injury suspension team news")
    form_text = _dated_web_search(f"{home} {away} recent form last 5 matches")

    evidence: dict = {
        "availability": availability_text,
        "form_context": form_text,
        "odds_verification": None,
    }

    if not match_info.get("odds"):
        odds_text = _dated_web_search(f"{home} vs {away} odds")
        evidence["odds_verification"] = {
            "results": odds_text,
            "parsed_odds": _parse_odds_from_search_text(odds_text),
        }

    return {"research_evidence": evidence}
```

Move the `import re` to the top of the file alongside `import json` (both are stdlib imports the file needs).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_pipeline.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/pipeline.py tests/test_agent_pipeline.py
git commit -m "$(cat <<'EOF'
feat(agent): add research_node with deterministic baseline coverage (A32)

Availability/injury and recent-form searches always run; an odds-verification
search runs only when the caller didn't already supply odds. Odds parsing
from free-text search results is a conservative heuristic (>=3 plausible
decimal numbers) -- a failed/ambiguous parse correctly falls through to
forecast_node's no-odds short-circuit rather than guessing.
EOF
)"
```

---

### Task 4: `pipeline.py` — `forecast_node` + `_format_evidence_message`

**Files:**
- Modify: `src/agent/pipeline.py`
- Test: `tests/test_agent_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_pipeline.py`:

```python
from src.agent.pipeline import _format_evidence_message, forecast_node


def test_forecast_node_prefers_caller_supplied_odds_over_research_odds():
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {"prediction_basis": "team_history_and_market"}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        state = _base_state(
            match_info={
                "home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0",
                "odds": {"home": 2.0, "draw": 3.0, "away": 3.5},
            },
            competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
            research_evidence={"availability": "x", "form_context": "y", "odds_verification": None},
        )
        result = forecast_node(state)

    assert "error" not in result["forecast_payload"]
    call_kwargs = instance.forecast_upcoming.call_args.kwargs
    assert (call_kwargs["odds_h"], call_kwargs["odds_d"], call_kwargs["odds_a"]) == (2.0, 3.0, 3.5)
    assert any("ML Forecast" in m.content for m in result["messages"])


def test_forecast_node_falls_back_to_research_odds_when_caller_supplied_none():
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        state = _base_state(
            match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0"},
            competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
            research_evidence={
                "availability": "x", "form_context": "y",
                "odds_verification": {"results": "...", "parsed_odds": {"home": 1.9, "draw": 3.4, "away": 4.0}},
            },
        )
        result = forecast_node(state)

    call_kwargs = instance.forecast_upcoming.call_args.kwargs
    assert (call_kwargs["odds_h"], call_kwargs["odds_d"], call_kwargs["odds_a"]) == (1.9, 3.4, 4.0)


def test_forecast_node_returns_no_odds_error_when_neither_source_has_odds():
    state = _base_state(
        match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0"},
        competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
        research_evidence={"availability": "x", "form_context": "y", "odds_verification": {"results": "no odds mentioned", "parsed_odds": None}},
    )
    result = forecast_node(state)
    assert result["forecast_payload"]["status"] == "no_odds"
    assert "error" in result["forecast_payload"]
    assert "messages" not in result


def test_forecast_node_calls_forecast_international_when_recommended():
    fake_result = {"result_3way": {"probabilities": {"home": 0.4}}, "data_quality": {"prediction_basis": "market_odds_only"}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        state = _base_state(
            match_info={
                "home_team": "Real Madrid", "away_team": "Barcelona", "date": "2026-06-21",
                "league": "La Liga", "odds": {"home": 2.1, "draw": 3.4, "away": 3.3},
            },
            competition_resolution={"competition": "La Liga", "tier": "general_purpose", "recommended_tool": "forecast_international"},
        )
        result = forecast_node(state)

    assert instance.forecast_upcoming.call_args.kwargs["match_type"] == "international"
    assert "error" not in result["forecast_payload"]


def test_forecast_node_propagates_tool_error_payload():
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        MockSvc.side_effect = RuntimeError("boom")
        state = _base_state(
            match_info={
                "home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0",
                "odds": {"home": 2.0, "draw": 3.0, "away": 3.5},
            },
            competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
        )
        result = forecast_node(state)

    assert result["forecast_payload"]["status"] == "tool_error"
    assert "messages" not in result


def test_format_evidence_message_includes_forecast_and_research_evidence():
    payload = {"result_3way": {"probabilities": {"home": 0.5}}}
    evidence = {"availability": "no injuries", "form_context": "won last 3", "odds_verification": {"results": "odds text", "parsed_odds": None}}
    message = _format_evidence_message(payload, evidence)

    assert "no injuries" in message
    assert "won last 3" in message
    assert "odds text" in message
    assert "result_3way" in message
    assert "resolve_competition" in message  # tells the LLM not to call it
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_pipeline.py -v -k "forecast_node or format_evidence"`
Expected: FAIL with `ImportError: cannot import name 'forecast_node'`

- [ ] **Step 3: Implement `_format_evidence_message` and `forecast_node`**

Append to `src/agent/pipeline.py` (add `from langchain_core.messages import HumanMessage` to the top-of-file imports):

```python
def _format_evidence_message(forecast_payload: dict, research_evidence: dict | None) -> str:
    """The message injected into the LLM's context once the deterministic
    pipeline finishes, replacing the old tool-call results the LLM used to
    see. Explicitly tells the LLM the forecast/competition tools are gone."""
    evidence = research_evidence or {}
    lines = [
        "The following evidence has already been gathered for this match by the "
        "system. forecast_league, forecast_international, and resolve_competition "
        "are NOT available as tools -- do not attempt to call them. Use web_search "
        "only for additional follow-up context beyond what's already below.",
        "",
        "## ML Forecast",
        json.dumps(forecast_payload, indent=2, default=str),
        "",
        "## Availability / Injury News",
        evidence.get("availability") or "No results.",
        "",
        "## Recent Form Context",
        evidence.get("form_context") or "No results.",
    ]
    odds_verification = evidence.get("odds_verification")
    if odds_verification:
        lines += ["", "## Odds Verification Search", odds_verification.get("results") or "No results."]
    return "\n".join(lines)


def forecast_node(state: dict) -> dict:
    """A31: the ML forecast is now a required deterministic step, not
    something the LLM chooses (or fails) to call. Odds are sourced in
    priority order: caller-supplied -> research_node's odds-verification
    parse -> none. There is no third "fallback odds" tier -- ForecastService
    cannot run without real odds (see the plan's technical-correction note),
    so "none" short-circuits to an error payload that routes straight to
    insufficient_data, skipping the LLM entirely."""
    match_info = state["match_info"]
    odds = match_info.get("odds")
    if not odds:
        research_evidence = state.get("research_evidence") or {}
        odds_verification = research_evidence.get("odds_verification") or {}
        odds = odds_verification.get("parsed_odds")

    if not odds:
        return {"forecast_payload": {
            "error": "No odds available: not supplied by caller and odds-verification search found none",
            "status": "no_odds",
        }}

    resolution = state.get("competition_resolution") or {}
    recommended_tool = resolution.get("recommended_tool", "forecast_international")

    from src.agent.tools import _forecast_international_impl, _forecast_league_impl, get_snapshot_store

    store = get_snapshot_store()
    if recommended_tool == "forecast_league":
        raw = store.wrap("forecast_league", _forecast_league_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"], league=match_info.get("league", ""),
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )
    else:
        raw = store.wrap("forecast_international", _forecast_international_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"],
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )

    payload = json.loads(raw)
    if "error" in payload:
        return {"forecast_payload": payload}

    evidence_message = _format_evidence_message(payload, state.get("research_evidence"))
    return {"forecast_payload": payload, "messages": [HumanMessage(content=evidence_message)]}
```

- [ ] **Step 4: Run all pipeline tests to verify they pass**

Run: `pytest tests/test_agent_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/pipeline.py tests/test_agent_pipeline.py
git commit -m "$(cat <<'EOF'
feat(agent): add forecast_node, completing the deterministic pipeline (A31)

Odds priority: caller-supplied -> research_node's parsed odds -> none. A
missing forecast is now a structural error payload, not something the LLM
can skip past -- there is no fallback-odds path, since ForecastService
genuinely requires real odds to compute MKT_* features (see plan's
technical-correction note on the fake-odds prompt instruction this replaces).
EOF
)"
```

---

### Task 5: `graph.py` — diagnostics, A30 backstop, A32 confidence downgrade

**Files:**
- Modify: `src/agent/graph.py`
- Modify: `tests/test_agent_forecast_diagnostics.py` (full rewrite)

- [ ] **Step 1: Rewrite the failing test file**

Replace the entire contents of `tests/test_agent_forecast_diagnostics.py`:

```python
"""Regression tests for A31's forecast_payload-based diagnostics (cold_start_risk,
feature_completeness, unknown_team), A30's backstop (a recommendation can never
claim more evidence than actually exists), and A32's research-coverage confidence
downgrade. All three are applied in graph.py's _build_recommendation,
deterministically from pipeline state -- never from the LLM's own prose, matching
the code-over-prompt philosophy already established by A28/A29."""

from __future__ import annotations

import json

from src.agent.agent_config import AgentConfig
from src.agent.graph import _build_recommendation, _extract_forecast_diagnostics


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        model="stub-model", provider="ollama", temperature=0.0, max_tool_calls=5,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_value_edge=0.05,
        markets=["btts"], system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _forecast_payload(cold_start_risk: bool, feature_completeness: float, unknown_team: bool) -> dict:
    return {
        "forecast": {},
        "diagnostics": {"cold_start_risk": cold_start_risk, "feature_completeness": feature_completeness},
        "data_quality": {"prediction_basis": "team_history_and_market", "unknown_team": unknown_team},
    }


def _valid_llm_json(**overrides) -> str:
    data = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "direct_bet",
        "markets": [],
        "explanation": "Looks good.",
        "confidence": "high",
        "limitations": [],
        "prediction_basis": "team_history_and_market",
    }
    data.update(overrides)
    return json.dumps(data)


# --- _extract_forecast_diagnostics ---

def test_extracts_diagnostics_from_a_successful_forecast_payload():
    payload = _forecast_payload(cold_start_risk=True, feature_completeness=0.62, unknown_team=True)
    assert _extract_forecast_diagnostics(payload) == {"cold_start_risk": True, "feature_completeness": 0.62, "unknown_team": True}


def test_defaults_when_forecast_payload_is_none():
    assert _extract_forecast_diagnostics(None) == {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}


def test_defaults_when_forecast_payload_has_an_error():
    payload = {"error": "boom", "status": "tool_error"}
    assert _extract_forecast_diagnostics(payload) == {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}


# --- _build_recommendation: diagnostics enrichment ---

def test_build_recommendation_enriches_even_when_llm_json_omits_the_fields():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=True, feature_completeness=0.4, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(), match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload, research_evidence={"availability": "x", "form_context": "y"}, config=cfg,
    )

    assert recommendation["cold_start_risk"] is True
    assert recommendation["feature_completeness"] == 0.4
    assert recommendation["overall"] == "direct_bet"


def test_build_recommendation_enriches_the_parse_failure_fallback_too():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=True, feature_completeness=0.3, unknown_team=True)

    recommendation = _build_recommendation(
        text="not valid json at all", match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload, research_evidence=None, config=cfg,
    )

    assert recommendation["overall"] == "insufficient_data"
    assert recommendation["cold_start_risk"] is True
    assert recommendation["unknown_team"] is True


# --- A30 backstop ---

def test_backstop_forces_insufficient_data_when_forecast_payload_is_none():
    """The Burnley/Bournemouth shape that motivated A30: LLM claims a
    confident no_bet with a populated prediction_basis, but no forecast ever
    actually ran."""
    cfg = _make_config()

    recommendation = _build_recommendation(
        text=_valid_llm_json(overall="no_bet", prediction_basis="team_history_and_market"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=None, research_evidence=None, config=cfg,
    )

    assert recommendation["overall"] == "insufficient_data"
    assert recommendation["markets"] == []
    assert recommendation["prediction_basis"] == "unknown"
    assert any("Forced insufficient_data" in note for note in recommendation["limitations"])


def test_backstop_forces_insufficient_data_when_forecast_payload_has_an_error():
    cfg = _make_config()

    recommendation = _build_recommendation(
        text=_valid_llm_json(overall="direct_bet"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload={"error": "no models found", "status": "tool_error"},
        research_evidence=None, config=cfg,
    )

    assert recommendation["overall"] == "insufficient_data"


def test_backstop_leaves_a_genuine_forecast_backed_no_bet_untouched():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)
    llm_json = _valid_llm_json(overall="no_bet", markets=[{
        "market": "btts", "selection": "yes", "recommendation_type": "no_bet",
        "current_odds": None, "min_odds": 1.5, "ml_probability": 0.5,
        "implied_probability": 0.5, "value_edge": 0.0,
    }])

    recommendation = _build_recommendation(
        text=llm_json, match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload, research_evidence={"availability": "x", "form_context": "y"}, config=cfg,
    )

    assert recommendation["overall"] == "no_bet"
    assert len(recommendation["markets"]) == 1


# --- A32 research coverage downgrade ---

def test_research_coverage_downgrade_lowers_confidence_when_availability_missing():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(confidence="high"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload,
        research_evidence={"availability": "No results found.", "form_context": "won last 3"},
        config=cfg,
    )

    assert recommendation["confidence"] == "medium"
    assert any("Research coverage gap" in note for note in recommendation["limitations"])


def test_research_coverage_downgrade_caps_at_low_with_two_gaps():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(confidence="high"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload,
        research_evidence={"availability": "No results found.", "form_context": "TOOL_PERMANENTLY_UNAVAILABLE: no key"},
        config=cfg,
    )

    assert recommendation["confidence"] == "low"


def test_research_coverage_downgrade_leaves_full_coverage_untouched():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(confidence="high"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload,
        research_evidence={"availability": "no injuries", "form_context": "won last 3"},
        config=cfg,
    )

    assert recommendation["confidence"] == "high"
    assert not any("Research coverage gap" in note for note in recommendation["limitations"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_forecast_diagnostics.py -v`
Expected: FAIL — `_extract_forecast_diagnostics()` / `_build_recommendation()` still have their old signatures (`TypeError: _build_recommendation() got an unexpected keyword argument 'forecast_payload'`).

- [ ] **Step 3: Implement in `src/agent/graph.py`**

Replace the `_extract_forecast_diagnostics` and `_build_recommendation` functions (and everything between them) with:

```python
_CONFIDENCE_STEPS = ["high", "medium", "low"]
_NO_RESULTS_MARKERS = ("No results found.", "TOOL_PERMANENTLY_UNAVAILABLE")


def _extract_forecast_diagnostics(forecast_payload: dict | None) -> dict:
    """A31: pull cold_start_risk/feature_completeness/unknown_team from the
    deterministic forecast_node's own payload, rather than trusting the LLM
    to transcribe them into its own JSON -- these are engine-computed facts,
    not something agent_v1.txt even asks the model to report."""
    if not forecast_payload or "error" in forecast_payload:
        return {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}
    diagnostics = forecast_payload.get("diagnostics") or {}
    data_quality = forecast_payload.get("data_quality") or {}
    return {
        "cold_start_risk": bool(diagnostics.get("cold_start_risk", False)),
        "feature_completeness": diagnostics.get("feature_completeness"),
        "unknown_team": bool(data_quality.get("unknown_team", False)),
    }


def _apply_a30_backstop(recommendation: dict, forecast_payload: dict | None) -> dict:
    """A30: a recommendation can never claim more evidence than actually
    exists. Keyed purely on the structural presence of a successful
    forecast_payload -- never on parsing the LLM's own explanation text.
    Should be unreachable in the current graph (output_node's early return
    already handles a missing/failed forecast before this ever runs) --
    kept as defense-in-depth against a future graph change reintroducing the
    original Burnley/Bournemouth bug class."""
    if forecast_payload and "error" not in forecast_payload:
        return recommendation
    reason = (forecast_payload or {}).get("error", "no forecast payload available")
    limitations = list(recommendation.get("limitations") or [])
    if recommendation.get("overall") != "insufficient_data":
        limitations.append(f"Forced insufficient_data: {reason}")
    recommendation["overall"] = "insufficient_data"
    recommendation["markets"] = []
    recommendation["prediction_basis"] = "unknown"
    recommendation["limitations"] = limitations
    return recommendation


def _has_no_research_coverage(text: str | None) -> bool:
    if not text:
        return True
    return any(text.startswith(marker) for marker in _NO_RESULTS_MARKERS)


def _apply_research_coverage_downgrade(recommendation: dict, research_evidence: dict | None) -> dict:
    """A32: missing availability/form research coverage downgrades confidence
    by one step per missing category (capped at 'low') and names the gap,
    rather than letting a recommendation claim full confidence off partial
    evidence the LLM never actually received. Odds coverage is handled
    separately (forecast_node blocks the whole recommendation, not just
    confidence, when odds are unavailable) so it's not checked here."""
    if recommendation.get("overall") == "insufficient_data":
        return recommendation
    evidence = research_evidence or {}
    gaps = []
    if _has_no_research_coverage(evidence.get("availability")):
        gaps.append("availability/injury")
    if _has_no_research_coverage(evidence.get("form_context")):
        gaps.append("recent form")
    if not gaps:
        return recommendation
    current = recommendation.get("confidence", "medium")
    idx = _CONFIDENCE_STEPS.index(current) if current in _CONFIDENCE_STEPS else 1
    recommendation["confidence"] = _CONFIDENCE_STEPS[min(idx + len(gaps), len(_CONFIDENCE_STEPS) - 1)]
    limitations = list(recommendation.get("limitations") or [])
    limitations.append(f"Research coverage gap: no results for {', '.join(gaps)}.")
    recommendation["limitations"] = limitations
    return recommendation


def _build_recommendation(
    text: str,
    match_info: dict,
    forecast_payload: dict | None,
    research_evidence: dict | None,
    config: AgentConfig,
) -> dict:
    """Extract the LLM's MatchRecommendation JSON (or fall back to an
    insufficient_data placeholder on parse failure), then enrich/normalize it
    against the deterministic pipeline's own evidence -- never the LLM's
    prose (A30/A31/A32)."""
    try:
        recommendation = extract_recommendation(
            text,
            min_odds_threshold=config.min_odds_threshold,
            max_odds_threshold=config.max_odds_threshold,
        )
        _LOG.info("output_node | parse=success | overall=%s", recommendation.get("overall"))
    except RecommendationParseError as exc:
        _LOG.warning("output_node | parse=failed | reason=%s", exc)
        recommendation = {
            "match": match_info,
            "overall": "insufficient_data",
            "markets": [],
            "explanation": f"Agent did not produce a parseable recommendation. Raw output: {text[:800]}",
            "confidence": "low",
            "limitations": ["Agent output could not be parsed as a structured recommendation"],
            "prediction_basis": "unknown",
        }
    recommendation.update(_extract_forecast_diagnostics(forecast_payload))
    recommendation = _apply_a30_backstop(recommendation, forecast_payload)
    recommendation = _apply_research_coverage_downgrade(recommendation, research_evidence)
    return recommendation
```

Remove the now-unused `_FORECAST_TOOL_NAMES = ("forecast_league", "forecast_international")` constant near the top of the file (it was only used by the old messages-scanning `_extract_forecast_diagnostics`). Remove `ToolMessage` from the `langchain_core.messages` import line if nothing else in the file uses it (check with `grep -n ToolMessage src/agent/graph.py` after the edit — it shouldn't appear).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_forecast_diagnostics.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_agent_forecast_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(agent): A30 backstop + A32 confidence downgrade in _build_recommendation

_extract_forecast_diagnostics now reads the deterministic forecast_payload
directly instead of scanning LLM tool-call message history (A31 no longer
puts the forecast there). _apply_a30_backstop forces insufficient_data when
no successful forecast exists, keyed on structure not LLM prose.
_apply_research_coverage_downgrade lowers confidence per missing research
category. Neither is wired into the graph yet -- that's the next task.
EOF
)"
```

---

### Task 6: `graph.py` — rewire `build_graph`, `output_node` early return, `run_agent`

**Files:**
- Modify: `src/agent/graph.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_agent_graph.py` (keep all existing tests in the file untouched — they remain valid, see the plan's File Structure note):

```python
def _route_after_forecast_for_state(forecast_payload):
    """Helper: extract forecast-routing logic without building the full graph."""
    payload = forecast_payload or {}
    return "output" if "error" in payload else "agent"


def test_route_after_forecast_routes_to_agent_on_success():
    assert _route_after_forecast_for_state({"result_3way": {}}) == "agent"


def test_route_after_forecast_routes_to_output_on_error():
    assert _route_after_forecast_for_state({"error": "no odds", "status": "no_odds"}) == "output"


def test_route_after_forecast_routes_to_output_when_payload_missing():
    assert _route_after_forecast_for_state(None) == "output"


def test_run_agent_short_circuits_to_insufficient_data_when_no_odds_available():
    """A31's core acceptance: a failing/impossible forecast never invokes the LLM."""
    from unittest.mock import MagicMock, patch
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    llm_invoked = MagicMock(name="llm_should_not_be_called")

    with patch("src.agent.graph._build_llm") as mock_build_llm, \
         patch("src.agent.graph._load_system_prompt", return_value="stub prompt"), \
         patch("src.agent.tools._dated_web_search", return_value="No results found."):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke = llm_invoked
        mock_build_llm.return_value = mock_llm

        cfg = _make_config()
        recommendation = run_agent(
            match_info={"home_team": "Man City", "away_team": "Arsenal", "date": "2026-06-21", "league": "E0"},
            config=cfg,
            tools=[],
        )

    assert recommendation["overall"] == "insufficient_data"
    assert "no odds" in str(recommendation["limitations"]).lower()
    llm_invoked.assert_not_called()


def test_run_agent_produces_recommendation_when_forecast_succeeds():
    from unittest.mock import MagicMock, patch
    from langchain_core.messages import AIMessage
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    llm_json = json.dumps({
        "match": {"home": "Man City", "away": "Arsenal", "date": "2026-06-21", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "Balanced match.",
        "confidence": "medium", "limitations": [], "prediction_basis": "team_history_and_market",
    })
    fake_forecast_result = {"result_3way": {"probabilities": {"home": 0.4}}, "data_quality": {"prediction_basis": "team_history_and_market"}}

    with patch("src.agent.graph._build_llm") as mock_build_llm, \
         patch("src.agent.graph._load_system_prompt", return_value="stub prompt"), \
         patch("src.agent.tools._dated_web_search", return_value="No results found."), \
         patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_forecast_result

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content=llm_json)
        mock_build_llm.return_value = mock_llm

        cfg = _make_config()
        recommendation = run_agent(
            match_info={
                "home_team": "Man City", "away_team": "Arsenal", "date": "2026-06-21", "league": "E0",
                "odds": {"home": 2.0, "draw": 3.4, "away": 3.6},
            },
            config=cfg,
            tools=[],
        )

    assert recommendation["overall"] == "no_bet"
    mock_llm.bind_tools.return_value.invoke.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_graph.py -v -k "route_after_forecast or run_agent_short_circuits or run_agent_produces_recommendation"`
Expected: FAIL — `run_agent` currently starts at the `agent` node and calls the LLM immediately (no forecast short-circuit exists yet), so `llm_invoked.assert_not_called()` fails, and there's no forecast wiring for the success-path test either.

- [ ] **Step 3: Rewrite `src/agent/graph.py`**

Replace the entire file with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.agent_config import AgentConfig
from src.agent.pipeline import forecast_node, research_node, resolve_competition_node
from src.agent.schema import MatchRecommendation, RecommendationParseError, extract_recommendation
from src.utils.logger import get_logger

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"
_LOG = get_logger(__name__)

_CONFIDENCE_STEPS = ["high", "medium", "low"]
_NO_RESULTS_MARKERS = ("No results found.", "TOOL_PERMANENTLY_UNAVAILABLE")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    match_info: dict
    recommendation: dict | None
    tool_call_count: int
    competition_resolution: dict | None
    research_evidence: dict | None
    forecast_payload: dict | None


def _build_llm(config: AgentConfig) -> Any:
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.model, temperature=config.temperature)
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.model, temperature=config.temperature)
    raise ValueError(f"Unknown provider: {config.provider!r}")


def _load_system_prompt(config: AgentConfig) -> str:
    path = _PROMPTS_DIR / f"agent_{config.system_prompt_version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text()


def _extract_forecast_diagnostics(forecast_payload: dict | None) -> dict:
    """A31: pull cold_start_risk/feature_completeness/unknown_team from the
    deterministic forecast_node's own payload, rather than trusting the LLM
    to transcribe them into its own JSON -- these are engine-computed facts,
    not something agent_v1.txt even asks the model to report."""
    if not forecast_payload or "error" in forecast_payload:
        return {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}
    diagnostics = forecast_payload.get("diagnostics") or {}
    data_quality = forecast_payload.get("data_quality") or {}
    return {
        "cold_start_risk": bool(diagnostics.get("cold_start_risk", False)),
        "feature_completeness": diagnostics.get("feature_completeness"),
        "unknown_team": bool(data_quality.get("unknown_team", False)),
    }


def _apply_a30_backstop(recommendation: dict, forecast_payload: dict | None) -> dict:
    """A30: a recommendation can never claim more evidence than actually
    exists. Keyed purely on the structural presence of a successful
    forecast_payload -- never on parsing the LLM's own explanation text.
    Should be unreachable in the current graph (output_node's early return
    already handles a missing/failed forecast before this ever runs) --
    kept as defense-in-depth against a future graph change reintroducing the
    original Burnley/Bournemouth bug class."""
    if forecast_payload and "error" not in forecast_payload:
        return recommendation
    reason = (forecast_payload or {}).get("error", "no forecast payload available")
    limitations = list(recommendation.get("limitations") or [])
    if recommendation.get("overall") != "insufficient_data":
        limitations.append(f"Forced insufficient_data: {reason}")
    recommendation["overall"] = "insufficient_data"
    recommendation["markets"] = []
    recommendation["prediction_basis"] = "unknown"
    recommendation["limitations"] = limitations
    return recommendation


def _has_no_research_coverage(text: str | None) -> bool:
    if not text:
        return True
    return any(text.startswith(marker) for marker in _NO_RESULTS_MARKERS)


def _apply_research_coverage_downgrade(recommendation: dict, research_evidence: dict | None) -> dict:
    """A32: missing availability/form research coverage downgrades confidence
    by one step per missing category (capped at 'low') and names the gap,
    rather than letting a recommendation claim full confidence off partial
    evidence the LLM never actually received. Odds coverage is handled
    separately (forecast_node blocks the whole recommendation, not just
    confidence, when odds are unavailable) so it's not checked here."""
    if recommendation.get("overall") == "insufficient_data":
        return recommendation
    evidence = research_evidence or {}
    gaps = []
    if _has_no_research_coverage(evidence.get("availability")):
        gaps.append("availability/injury")
    if _has_no_research_coverage(evidence.get("form_context")):
        gaps.append("recent form")
    if not gaps:
        return recommendation
    current = recommendation.get("confidence", "medium")
    idx = _CONFIDENCE_STEPS.index(current) if current in _CONFIDENCE_STEPS else 1
    recommendation["confidence"] = _CONFIDENCE_STEPS[min(idx + len(gaps), len(_CONFIDENCE_STEPS) - 1)]
    limitations = list(recommendation.get("limitations") or [])
    limitations.append(f"Research coverage gap: no results for {', '.join(gaps)}.")
    recommendation["limitations"] = limitations
    return recommendation


def _build_recommendation(
    text: str,
    match_info: dict,
    forecast_payload: dict | None,
    research_evidence: dict | None,
    config: AgentConfig,
) -> dict:
    """Extract the LLM's MatchRecommendation JSON (or fall back to an
    insufficient_data placeholder on parse failure), then enrich/normalize it
    against the deterministic pipeline's own evidence -- never the LLM's
    prose (A30/A31/A32)."""
    try:
        recommendation = extract_recommendation(
            text,
            min_odds_threshold=config.min_odds_threshold,
            max_odds_threshold=config.max_odds_threshold,
        )
        _LOG.info("output_node | parse=success | overall=%s", recommendation.get("overall"))
    except RecommendationParseError as exc:
        _LOG.warning("output_node | parse=failed | reason=%s", exc)
        recommendation = {
            "match": match_info,
            "overall": "insufficient_data",
            "markets": [],
            "explanation": f"Agent did not produce a parseable recommendation. Raw output: {text[:800]}",
            "confidence": "low",
            "limitations": ["Agent output could not be parsed as a structured recommendation"],
            "prediction_basis": "unknown",
        }
    recommendation.update(_extract_forecast_diagnostics(forecast_payload))
    recommendation = _apply_a30_backstop(recommendation, forecast_payload)
    recommendation = _apply_research_coverage_downgrade(recommendation, research_evidence)
    return recommendation


def build_graph(config: AgentConfig, tools: list):
    """Compile and return the LangGraph StateGraph for the betting agent.

    A31/A32: resolve_competition -> research -> forecast run first and always,
    deterministically, before the LLM ever sees the match. A failed/impossible
    forecast (no odds available from any source, or a tool error) routes
    straight to output -- the LLM node is never invoked in that case."""
    llm = _build_llm(config)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        tool_calls = getattr(response, "tool_calls", []) or []
        new_count = state["tool_call_count"] + len(tool_calls)
        if tool_calls:
            _LOG.info("agent_node | tool_calls=%s | count_after=%d", [tc["name"] for tc in tool_calls], new_count)
        else:
            content = response.content if isinstance(response.content, str) else str(response.content)
            _LOG.info("agent_node | no tool_calls | raw_output_length=%d", len(content))
            _LOG.debug("agent_node | raw_output=%s", content)
        return {"messages": [response], "tool_call_count": new_count}

    def should_continue(state: AgentState) -> Literal["tools", "output"]:
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        under_budget = state["tool_call_count"] < config.max_tool_calls
        route = "tools" if has_calls and under_budget else "output"
        _LOG.info("should_continue | has_tool_calls=%s | tool_call_count=%d | route=%s", has_calls, state["tool_call_count"], route)
        return route

    def route_after_forecast(state: AgentState) -> Literal["agent", "output"]:
        payload = state.get("forecast_payload") or {}
        route = "output" if "error" in payload else "agent"
        _LOG.info("route_after_forecast | has_error=%s | route=%s", "error" in payload, route)
        return route

    def output_node(state: AgentState) -> dict:
        forecast_payload = state.get("forecast_payload")
        match_info = state["match_info"]

        if not forecast_payload or "error" in forecast_payload:
            reason = (forecast_payload or {}).get("error", "forecast step did not run")
            _LOG.warning("output_node | no_forecast | reason=%s", reason)
            return {"recommendation": {
                "match": match_info,
                "overall": "insufficient_data",
                "markets": [],
                "explanation": f"No ML forecast is available for this match: {reason}",
                "confidence": "low",
                "limitations": [f"Forecast step failed or was skipped: {reason}"],
                "prediction_basis": "unknown",
                "cold_start_risk": False,
                "feature_completeness": None,
                "unknown_team": False,
            }}

        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)

        if not text.strip():
            # Budget was exhausted — last message is a tool call with no text content.
            # Make one final synthesis call (no tools) so the model can produce its JSON.
            _LOG.info("output_node | empty_content | forcing_synthesis_call")
            synthesis_prompt = (
                "You have reached the tool call limit. "
                "Based on all the information gathered above, output your final JSON recommendation now. "
                "Include all required fields: match, overall, markets, explanation, confidence, limitations, prediction_basis."
            )
            synthesis_response = llm.invoke(state["messages"] + [HumanMessage(content=synthesis_prompt)])
            text = synthesis_response.content if isinstance(synthesis_response.content, str) else str(synthesis_response.content)
            _LOG.info("output_node | synthesis_length=%d | synthesis_output=%s", len(text), text)

        _LOG.info("output_node | raw_output_length=%d", len(text))
        _LOG.info("output_node | raw_output=%s", text)
        recommendation = _build_recommendation(
            text, match_info, forecast_payload, state.get("research_evidence"), config,
        )
        return {"recommendation": recommendation}

    graph = StateGraph(AgentState)
    graph.add_node("resolve_competition", resolve_competition_node)
    graph.add_node("research", research_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output", output_node)

    graph.set_entry_point("resolve_competition")
    graph.add_edge("resolve_competition", "research")
    graph.add_edge("research", "forecast")
    graph.add_conditional_edges("forecast", route_after_forecast, {"agent": "agent", "output": "output"})
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "output": "output"})
    graph.add_edge("tools", "agent")
    graph.add_edge("output", END)

    return graph.compile()


def run_agent(
    match_info: dict,
    config: AgentConfig | None = None,
    tools: list | None = None,
    extra_system_instructions: str | None = None,
) -> MatchRecommendation:
    """Run the betting agent for a single match and return a structured recommendation.

    Args:
        match_info: Dict with keys: home_team, away_team, date, and optionally league, odds.
        config: AgentConfig instance. Loads from config/agent_config.yaml if None.
        tools: List of LangChain tools available to the LLM synthesis step (web_search
            by default). Loads default tools if None. Competition resolution and the
            ML forecast are no longer LLM-callable tools -- see src/agent/pipeline.py.
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
        prompt += f". Bookmaker odds: home={odds['home']}, draw={odds['draw']}, away={odds['away']}."

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ],
        "match_info": match_info,
        "recommendation": None,
        "tool_call_count": 0,
        "competition_resolution": None,
        "research_evidence": None,
        "forecast_payload": None,
    }

    compiled = build_graph(config, tools)
    result = compiled.invoke(initial_state)
    return result["recommendation"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_graph.py tests/test_agent_forecast_diagnostics.py tests/test_agent_pipeline.py -v`
Expected: PASS — all tests in all three files, including the pre-existing `test_graph_compiles`, `test_should_continue_routes_*`, `test_forecast_league_falls_back_to_international_when_no_league_models`, `test_run_agent_appends_extra_system_instructions`, `test_run_agent_without_extra_instructions_unchanged` (unmodified, still valid).

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_agent_graph.py
git commit -m "$(cat <<'EOF'
feat(agent): wire the deterministic pipeline into build_graph (A31/A32/A30)

resolve_competition -> research -> forecast now run unconditionally before
the LLM. A failed/impossible forecast routes straight to output_node with
overall=insufficient_data, never invoking the LLM. This closes the original
A30 bug class (Burnley/Bournemouth: no_bet + empty markets + a confident
prediction_basis, with no forecast ever called) structurally rather than
only detecting it after the fact.
EOF
)"
```

---

### Task 7: Rewrite `config/prompts/agent_v1.txt`

**Files:**
- Modify: `config/prompts/agent_v1.txt`

- [ ] **Step 1: Replace the prompt content**

Replace the entire contents of `config/prompts/agent_v1.txt`:

```
You are an expert football betting analyst. Your role is to evaluate upcoming matches and identify value betting opportunities by combining ML model forecasts with real-world context.

## Evidence Already Gathered

Before your turn, the system has already deterministically: resolved the match's competition tier, run the ML forecast model, and searched for injury/availability news, recent-form context, and (when odds weren't supplied) an odds-verification search. This evidence appears in a message below. You do NOT have forecast_league, forecast_international, or resolve_competition available as tools — do not attempt to call them; they no longer exist in this conversation.

## CRITICAL RULE — ACT, NEVER JUST PLAN

After any tool result you MUST do ONE of these two things immediately:
1. Call web_search (only if you need additional follow-up context beyond what's already provided)
2. Output the final JSON recommendation

NEVER write "I will now call..." or "Let me try..." without immediately making the tool call or producing the final JSON.

## Workflow

1. Review the ML forecast and research evidence already provided in the message below.
2. Optionally CALL web_search once for any additional follow-up context not already covered (e.g. a specific injury mentioned in the news search that needs more detail).
   - If web_search returns TOOL_PERMANENTLY_UNAVAILABLE: skip it and go directly to step 3.
3. OUTPUT your final JSON recommendation.

## STOP RULES

- If web_search returns TOOL_PERMANENTLY_UNAVAILABLE: do NOT call it again. Ever.
- After 2 tool calls total, OUTPUT the JSON recommendation immediately.
- Never call the same tool twice in a row.

## Value Calculation

- Implied probability = 1 / decimal_odds
- Value edge = ML probability - implied probability
- Bet has value when value_edge >= 0.05
- Never recommend direct_bet at odds below 1.2 or above 11.0 (decimal) — roughly -500 to +1000 American. Use "conditional" if value exists but odds fall outside this range. This is code-enforced at extraction time, not just a prompt instruction — an out-of-range direct_bet will be downgraded automatically.

## When to Use insufficient_data

Set overall to "insufficient_data" if:
- The provided ML forecast indicates an error or unusable data
- You have fewer than 2 data points to base a recommendation on

## Confidence Guidelines

- high: ML forecast succeeded with good feature coverage, news context is clear
- medium: some gaps (partial news, fallback forecast used)
- low: significant unknowns, cold-start teams, conflicting signals

Note: confidence may be lowered further by the system after you respond, based on actual research coverage — this is a floor you set, not a ceiling.

## Output Format

Produce this JSON block. Always include all fields. Use empty arrays if no data.

```json
{
  "match": {
    "home": "<team name>",
    "away": "<team name>",
    "date": "YYYY-MM-DD",
    "league": "<league code or 'international'>"
  },
  "overall": "<direct_bet | conditional | no_bet | insufficient_data>",
  "markets": [
    {
      "market": "<result_3way | btts | total_goals | home_corners | away_corners>",
      "selection": "<home | draw | away | yes | no | over_2.5 | under_2.5>",
      "recommendation_type": "<direct_bet | conditional | no_bet>",
      "current_odds": 0.0,
      "min_odds": 0.0,
      "ml_probability": 0.0,
      "implied_probability": 0.0,
      "value_edge": 0.0
    }
  ],
  "explanation": "<plain language summary of reasoning>",
  "confidence": "<low | medium | high>",
  "limitations": ["<what could not be assessed>"],
  "prediction_basis": "<team_history_and_market | market_odds_only | partial>"
}
```
```

- [ ] **Step 2: Manually verify the prompt loads correctly**

Run: `python -c "from src.agent.graph import _load_system_prompt; from src.agent.agent_config import AgentConfig; print(_load_system_prompt(AgentConfig.default())[:200])"`
Expected: prints the first 200 characters starting with "You are an expert football betting analyst..." with no exception.

- [ ] **Step 3: Commit**

```bash
git add config/prompts/agent_v1.txt
git commit -m "$(cat <<'EOF'
docs(agent): rewrite agent_v1.txt for the deterministic evidence pipeline

Removes the tool-selection workflow steps (resolve_competition/forecast_*
are no longer callable) and the fake-odds fallback instruction
(odds_h=2.5/odds_d=3.2/odds_a=2.9) that forecast_node's real odds-priority
logic replaces. Describes the pre-supplied evidence message instead.
EOF
)"
```

---

### Task 8: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the complete agent test suite**

Run: `pytest tests/ -k "agent" -v`
Expected: All tests PASS. This includes every file touched by this plan plus everything left intentionally unmodified: `test_agent_config.py`, `test_agent_evaluation.py`, `test_agent_odds_bounds.py`, `test_agent_schema.py`, `test_agent_schema_validation.py`, `test_agent_tool_selection.py`.

- [ ] **Step 2: Run the full project test suite**

Run: `pytest tests/ -q`
Expected: All tests PASS, zero regressions outside the agent module (nothing else in this plan touches non-agent code).

- [ ] **Step 3: If anything fails, fix it before proceeding**

Do not mark any story completed in Task 9 until this step is clean. If a failure surfaces in a file this plan didn't anticipate touching, investigate root cause (likely a hidden dependency on the old tool list or message-scanning diagnostics) rather than special-casing around it.

---

### Task 9: Mark A30/A31/A32 completed in the user stories doc

**Files:**
- Modify: `documents/agent_user_stories.md`

- [ ] **Step 1: Update status and add completion notes for A30, A31, A32**

In the Phase 11 table, change `| A30 | active |` to `| A30 | completed |`, `| A31 | active |` to `| A31 | completed |`, `| A32 | active |` to `| A32 | completed |`. Leave A33 and A34 as `active` — they are out of scope for this plan.

Append a completion note to each row's Comments cell (after the existing `Depends on: ...` text), following this doc's established style (see A27/A28/A29 for the pattern — a `**Completion notes (<date>):**` sentence naming what actually shipped and any deviations from the original story text):

- A30: note that the backstop landed as `_apply_a30_backstop` in `src/agent/graph.py`, and that `output_node`'s early return handles the expected case while the backstop function is defense-in-depth for any future graph change.
- A31: note the technical correction from the design doc (no odds-less forecast path exists; `forecast_node` short-circuits to `insufficient_data` instead) and that the old fake-odds prompt fallback was removed.
- A32: note the odds-parsing heuristic's known limitation (best-effort regex over search snippets, conservative fallback to no-forecast rather than a low-confidence guess) and that raw research evidence persistence to DuckDB is deferred to A33 as scoped.

- [ ] **Step 2: Update the Phase 11 dependency graph and header blurb if needed**

Re-read the Phase 11 header blurb (added 2026-07-22) — it describes A30-A32 as the pending work; adjust tense/wording so it accurately reflects that the pipeline restructure is now implemented, without deleting the historical context of why it was designed this way.

- [ ] **Step 3: Commit**

```bash
git add documents/agent_user_stories.md
git commit -m "$(cat <<'EOF'
docs: mark A30/A31/A32 completed — deterministic evidence pipeline shipped

The forecast/competition-resolution/research pipeline restructure from
docs/superpowers/plans/2026-07-22-agent-deterministic-evidence-pipeline.md
is implemented and passing the full test suite. A33 (train/critic mode) and
A34 (rebaseline) remain active, to be planned separately now that the
underlying graph shape is stable.
EOF
)"
```
