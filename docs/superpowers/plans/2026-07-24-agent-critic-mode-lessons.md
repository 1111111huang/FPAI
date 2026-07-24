# A33 Critic/Train Mode + Competition-Scoped Lessons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `agent-train` CLI (critic mode) that replays completed matches, scores them like `agent-backtest`, and writes one competition/tier-tagged lesson candidate plus a raw-evidence telemetry row per match to DuckDB; add an `agent-lessons approve/reject` CLI for human review; and make live agent runs load only reviewer-approved, competition/tier-matching lessons into the LLM's context.

**Architecture:** Reuses the existing deterministic-node pattern from A31/A32 (`src/agent/pipeline.py`) — a new `lessons_node` runs right after `forecast_node` succeeds and injects approved lessons as a `HumanMessage`, gated on `SnapshotStore.mode == "live"` so backtest/train replay never sees future-approved lessons. Reuses `BacktestHarness`/`process_match_row` (`src/agent/backtest.py`) for match replay — `agent-train` is `agent-backtest` plus DB writes, not a parallel implementation. All persistence lives in one new module, `src/agent/lessons.py`, so `graph.py`'s live path only ever imports the one read-only, structurally-can't-see-outcomes function (`load_approved_lessons`).

**Tech Stack:** Python, LangGraph (`StateGraph`), DuckDB (`duckdb` Python package, via `src/utils/db_manager.py`), pytest, argparse (`main.py`).

**Full design:** `docs/superpowers/specs/2026-07-22-agent-phase11-design.md` (A33 section, revised 2026-07-24).

---

## File Structure

- Create: `src/agent/lessons.py` — DuckDB schema (`agent_lessons`, `agent_telemetry`), CRUD functions, `generate_lesson_text()`, `extract_competition_scope()`.
- Create: `tests/test_agent_lessons.py`
- Create: `tests/test_main_agent_train.py`
- Create: `tests/test_main_agent_lessons.py`
- Modify: `src/agent/graph.py` — `run_agent(..., return_full_state=False)`; new `lessons` node wired into the graph between `forecast` and `agent`.
- Modify: `src/agent/pipeline.py` — new `lessons_node()`.
- Modify: `src/agent/backtest.py` — `BacktestRecord.full_state`; `process_match_row(..., capture_state=False)`.
- Modify: `main.py` — `_run_backtest_concurrent(..., capture_state=False)`; new `_write_train_artifacts()`, `run_agent_train()`, `run_agent_lessons_approve()`, `run_agent_lessons_reject()`; new `agent-train`/`agent-lessons` subparsers and dispatch.
- Modify: `config/prompts/agent_v1.txt` — one sentence noting an optional "Lessons from past evaluated matches" message may appear.
- Modify: `tests/test_agent_graph.py` — `return_full_state` tests; patch the one existing test that now reaches `lessons_node`; new lessons-injection integration test.
- Modify: `tests/test_agent_pipeline.py` — `lessons_node` tests, including the structural "can't see outcomes" test.
- Modify: `tests/test_backtest.py` — `capture_state` tests.
- Modify: `documents/agent_techspec.md` — new section documenting the CLI/schema.
- Modify: `documents/agent_user_stories.md` — mark A33 completed with completion notes (final task only).

---

### Task 1: Lessons persistence module

**Files:**
- Create: `src/agent/lessons.py`
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for A33's lesson/telemetry persistence (src/agent/lessons.py)."""
from __future__ import annotations

import inspect
import json

import duckdb
import pytest

from src.agent.lessons import (
    approve_lesson,
    create_lessons_tables,
    extract_competition_scope,
    generate_lesson_text,
    insert_lesson_candidate,
    insert_telemetry,
    load_approved_lessons,
    reject_lesson,
)


def _conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    return conn


def test_create_lessons_tables_creates_both_tables():
    conn = _conn()
    lesson_cols = {row[1] for row in conn.execute("PRAGMA table_info('agent_lessons')").fetchall()}
    telemetry_cols = {row[1] for row in conn.execute("PRAGMA table_info('agent_telemetry')").fetchall()}
    assert lesson_cols == {
        "id", "lesson_text", "status", "competition_id", "tier", "scope",
        "source_match_id", "created_at", "reviewed_at", "reviewer",
    }
    assert telemetry_cols == {
        "match_id", "run_id", "competition_resolution", "research_evidence",
        "forecast_payload", "recommendation", "created_at",
    }


def test_create_lessons_tables_is_idempotent():
    conn = _conn()
    create_lessons_tables(conn)  # second call must not raise
    assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 0


def test_insert_lesson_candidate_defaults_to_pending_with_null_scope():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0 matches...", "E0", "competition_specific", "m1")
    row = conn.execute(
        "SELECT status, scope, competition_id, tier, source_match_id FROM agent_lessons WHERE id = ?", [lesson_id]
    ).fetchone()
    assert row == ("pending", None, "E0", "competition_specific", "m1")


def test_insert_lesson_candidate_allows_null_competition_id():
    """Leagueless internationals (resolve_competition returns competition=None)."""
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating international matches...", None, "general_purpose", "m2")
    row = conn.execute("SELECT competition_id, tier FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == (None, "general_purpose")


def test_insert_telemetry_round_trips_json_fields():
    conn = _conn()
    insert_telemetry(
        conn,
        match_id="m1",
        run_id="run-1",
        competition_resolution={"competition": "E0", "tier": "competition_specific"},
        research_evidence={"availability": "ok"},
        forecast_payload={"result_3way": {"probabilities": {"home": 0.5}}},
        recommendation={"overall": "no_bet"},
    )
    row = conn.execute(
        "SELECT match_id, run_id, competition_resolution, research_evidence, forecast_payload, recommendation "
        "FROM agent_telemetry WHERE match_id = 'm1'"
    ).fetchone()
    assert row[0] == "m1"
    assert row[1] == "run-1"
    assert json.loads(row[2]) == {"competition": "E0", "tier": "competition_specific"}
    assert json.loads(row[3]) == {"availability": "ok"}
    assert json.loads(row[4]) == {"result_3way": {"probabilities": {"home": 0.5}}}
    assert json.loads(row[5]) == {"overall": "no_bet"}


def test_approve_lesson_sets_status_scope_reviewed_at_reviewer():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    approve_lesson(conn, lesson_id, "competition", "alice")
    row = conn.execute("SELECT status, scope, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == ("approved", "competition", "alice")
    reviewed_at = conn.execute("SELECT reviewed_at FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert reviewed_at is not None


def test_approve_lesson_rejects_invalid_scope():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    with pytest.raises(ValueError, match="scope"):
        approve_lesson(conn, lesson_id, "league", "alice")


def test_approve_lesson_raises_for_unknown_id():
    conn = _conn()
    with pytest.raises(ValueError, match="999"):
        approve_lesson(conn, 999, "competition", "alice")


def test_reject_lesson_sets_status_rejected():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    reject_lesson(conn, lesson_id, "bob")
    row = conn.execute("SELECT status, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == ("rejected", "bob")


def test_reject_lesson_raises_for_unknown_id():
    conn = _conn()
    with pytest.raises(ValueError, match="999"):
        reject_lesson(conn, 999, "bob")


def test_load_approved_lessons_matches_competition_scope_only_for_same_competition():
    conn = _conn()
    e0_id = insert_lesson_candidate(conn, "E0 lesson", "E0", "competition_specific", "m1")
    sp1_id = insert_lesson_candidate(conn, "SP1 lesson", "SP1", "competition_specific", "m2")
    approve_lesson(conn, e0_id, "competition", "alice")
    approve_lesson(conn, sp1_id, "competition", "alice")

    result = load_approved_lessons(conn, "E0", "competition_specific")
    assert result == ["E0 lesson"]


def test_load_approved_lessons_matches_tier_scope_regardless_of_competition():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "tier lesson", "SWE_ALLS", "general_purpose", "m1")
    approve_lesson(conn, lesson_id, "tier", "alice")

    result = load_approved_lessons(conn, "SOME_OTHER_LEAGUE", "general_purpose")
    assert result == ["tier lesson"]


def test_load_approved_lessons_excludes_pending_and_rejected():
    conn = _conn()
    pending_id = insert_lesson_candidate(conn, "pending lesson", "E0", "competition_specific", "m1")
    rejected_id = insert_lesson_candidate(conn, "rejected lesson", "E0", "competition_specific", "m2")
    reject_lesson(conn, rejected_id, "alice")
    # pending_id stays pending -- never approved

    result = load_approved_lessons(conn, "E0", "competition_specific")
    assert result == []


def test_load_approved_lessons_returns_empty_list_when_table_missing():
    conn = duckdb.connect(":memory:")  # create_lessons_tables() never called
    assert load_approved_lessons(conn, "E0", "competition_specific") == []


def test_load_approved_lessons_signature_has_no_status_override_parameter():
    """A33 acceptance: live mode must structurally be unable to fetch
    pending/rejected lessons -- proven here by the function itself having no
    parameter that could select anything but the hardcoded status='approved'."""
    params = set(inspect.signature(load_approved_lessons).parameters)
    assert params == {"conn", "competition_id", "tier"}


def test_extract_competition_scope_reads_competition_and_tier():
    full_state = {"competition_resolution": {"competition": "E0", "tier": "competition_specific"}}
    assert extract_competition_scope(full_state) == ("E0", "competition_specific")


def test_extract_competition_scope_defaults_tier_general_purpose_when_missing():
    assert extract_competition_scope({}) == (None, "general_purpose")
    full_state = {"competition_resolution": {"competition": None, "tier": None}}
    assert extract_competition_scope(full_state) == (None, "general_purpose")


class _FakeRecord:
    def __init__(self, league, recommendation, market_results, actual):
        self.league = league
        self.recommendation = recommendation
        self.market_results = market_results
        self.actual = actual


def test_generate_lesson_text_includes_context_overall_and_market_outcomes():
    record = _FakeRecord(
        league="E0",
        recommendation={
            "overall": "direct_bet", "confidence": "high",
            "prediction_basis": "team_history_and_market", "limitations": [],
        },
        market_results=[{"market": "result_3way", "selection": "home", "correct": True}],
        actual={"result": "home"},
    )
    text = generate_lesson_text(record)
    assert text.startswith("WHEN evaluating E0 matches")
    assert "direct_bet" in text
    assert "result_3way=home (correct)" in text


def test_generate_lesson_text_handles_no_markets_and_limitations():
    record = _FakeRecord(
        league=None,
        recommendation={
            "overall": "insufficient_data", "confidence": "low",
            "prediction_basis": "unknown", "limitations": ["no odds available"],
        },
        market_results=[],
        actual={"result": "draw"},
    )
    text = generate_lesson_text(record)
    assert "an unlabeled competition" in text
    assert "no markets recommended" in text
    assert "no odds available" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.lessons'`

- [ ] **Step 3: Implement `src/agent/lessons.py`**

```python
"""A33: DuckDB persistence for critic/train mode -- reviewed, competition/
tier-scoped lesson candidates and per-run evidence telemetry. See
docs/superpowers/specs/2026-07-22-agent-phase11-design.md (A33 section,
revised 2026-07-24 for competition scoping).

load_approved_lessons() is the ONLY function this module exposes that the
live agent path (src/agent/pipeline.py's lessons_node) imports -- its SQL
hardcodes status='approved' and never touches an outcome-bearing table, so
live mode is structurally unable to read match outcomes or pending/rejected
lessons, not just conventionally forbidden from it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import duckdb

_VALID_SCOPES = ("competition", "tier")


def create_lessons_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create agent_lessons and agent_telemetry if they don't already exist."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS agent_lessons_id_seq START 1")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_lessons (
            id INTEGER PRIMARY KEY DEFAULT nextval('agent_lessons_id_seq'),
            lesson_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            competition_id TEXT,
            tier TEXT NOT NULL,
            scope TEXT,
            source_match_id TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            reviewed_at TIMESTAMP,
            reviewer TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_telemetry (
            match_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            competition_resolution TEXT,
            research_evidence TEXT,
            forecast_payload TEXT,
            recommendation TEXT,
            created_at TIMESTAMP NOT NULL,
            PRIMARY KEY (match_id, run_id)
        )
        """
    )


def insert_lesson_candidate(
    conn: duckdb.DuckDBPyConnection,
    lesson_text: str,
    competition_id: str | None,
    tier: str,
    source_match_id: str,
) -> int:
    """Insert a pending, unscoped lesson candidate. Returns its id."""
    row = conn.execute(
        """
        INSERT INTO agent_lessons (lesson_text, status, competition_id, tier, source_match_id, created_at)
        VALUES (?, 'pending', ?, ?, ?, ?)
        RETURNING id
        """,
        [lesson_text, competition_id, tier, source_match_id, datetime.now(timezone.utc)],
    ).fetchone()
    return int(row[0])


def insert_telemetry(
    conn: duckdb.DuckDBPyConnection,
    match_id: str,
    run_id: str,
    competition_resolution: dict[str, Any] | None,
    research_evidence: dict[str, Any] | None,
    forecast_payload: dict[str, Any] | None,
    recommendation: dict[str, Any] | None,
) -> None:
    conn.execute(
        """
        INSERT INTO agent_telemetry
            (match_id, run_id, competition_resolution, research_evidence, forecast_payload, recommendation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            match_id,
            run_id,
            json.dumps(competition_resolution),
            json.dumps(research_evidence),
            json.dumps(forecast_payload),
            json.dumps(recommendation),
            datetime.now(timezone.utc),
        ],
    )


def _require_lesson_exists(conn: duckdb.DuckDBPyConnection, lesson_id: int) -> None:
    count = conn.execute("SELECT COUNT(*) FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    if not count:
        raise ValueError(f"No lesson with id={lesson_id}")


def approve_lesson(conn: duckdb.DuckDBPyConnection, lesson_id: int, scope: str, reviewer: str) -> None:
    """Approve a lesson, requiring the reviewer to pick a scope explicitly.

    scope='competition' pins the lesson to its recorded competition_id;
    scope='tier' widens it to every match resolving to its recorded tier.
    """
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
    _require_lesson_exists(conn, lesson_id)
    conn.execute(
        "UPDATE agent_lessons SET status = 'approved', scope = ?, reviewed_at = ?, reviewer = ? WHERE id = ?",
        [scope, datetime.now(timezone.utc), reviewer, lesson_id],
    )


def reject_lesson(conn: duckdb.DuckDBPyConnection, lesson_id: int, reviewer: str) -> None:
    _require_lesson_exists(conn, lesson_id)
    conn.execute(
        "UPDATE agent_lessons SET status = 'rejected', reviewed_at = ?, reviewer = ? WHERE id = ?",
        [datetime.now(timezone.utc), reviewer, lesson_id],
    )


def load_approved_lessons(conn: duckdb.DuckDBPyConnection, competition_id: str | None, tier: str) -> list[str]:
    """Approved lesson text for one match's competition_id/tier. Tolerates a
    missing agent_lessons table (e.g. agent-train has never been run yet) by
    returning no lessons rather than raising -- live recommendation runs must
    never fail just because train mode hasn't produced anything yet."""
    try:
        rows = conn.execute(
            """
            SELECT lesson_text FROM agent_lessons
            WHERE status = 'approved'
              AND ((scope = 'competition' AND competition_id = ?)
                OR (scope = 'tier' AND tier = ?))
            ORDER BY created_at
            """,
            [competition_id, tier],
        ).fetchall()
    except duckdb.CatalogException:
        return []
    return [row[0] for row in rows]


def extract_competition_scope(full_state: dict[str, Any]) -> tuple[str | None, str]:
    """(competition_id, tier) from an AgentState-shaped dict's
    competition_resolution block, defaulting tier to general_purpose when
    absent -- mirrors resolve_competition_node's own leagueless-international
    default so a lesson from that path still records a real tier."""
    resolution = full_state.get("competition_resolution") or {}
    return resolution.get("competition"), resolution.get("tier") or "general_purpose"


def generate_lesson_text(record: Any) -> str:
    """Deterministic lesson-candidate template from a BacktestRecord-shaped
    object (duck-typed: .league, .recommendation, .market_results, .actual --
    see src/agent/backtest.py). Not an attempt at insightful NLG -- the
    reviewer judges usefulness at approval time; this just surfaces a
    structured summary of what happened for them to judge."""
    context_label = record.league or "an unlabeled competition"
    overall = record.recommendation.get("overall", "unknown")
    confidence = record.recommendation.get("confidence", "unknown")
    basis = record.recommendation.get("prediction_basis", "unknown")
    limitations = record.recommendation.get("limitations") or []

    market_lines = []
    for market in record.market_results:
        correct = market.get("correct")
        outcome = "correct" if correct is True else "incorrect" if correct is False else "unresolved"
        market_lines.append(f"{market.get('market')}={market.get('selection')} ({outcome})")
    markets_summary = "; ".join(market_lines) if market_lines else "no markets recommended"
    limitations_summary = "; ".join(limitations) if limitations else "none noted"

    return (
        f"WHEN evaluating {context_label} matches: a recommendation of '{overall}' "
        f"(confidence={confidence}, basis={basis}) had actual result={record.actual.get('result')}. "
        f"Markets: {markets_summary}. Limitations noted at the time: {limitations_summary}."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: PASS (18 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "$(cat <<'EOF'
feat(agent): add A33 lesson/telemetry persistence module

New src/agent/lessons.py: agent_lessons + agent_telemetry DuckDB tables,
competition/tier-scoped approve/reject/load, and a deterministic lesson-
text template. load_approved_lessons() is the only function the live path
will import -- its SQL hardcodes status='approved' and touches no
outcome-bearing table.
EOF
)"
```

---

### Task 2: `run_agent(return_full_state=...)`

**Files:**
- Modify: `src/agent/graph.py:284-339` (`run_agent`)
- Test: `tests/test_agent_graph.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_graph.py`:

```python
def test_run_agent_returns_full_state_when_requested():
    from unittest.mock import patch, MagicMock
    from src.agent.graph import run_agent

    full_state = {
        "recommendation": {"overall": "no_bet"},
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
    }

    def fake_build_graph(config, tools):
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = full_state
        return mock_compiled

    cfg = _make_config()
    with patch("src.agent.graph._build_llm"), \
         patch("src.agent.graph._load_system_prompt", return_value="BASE PROMPT"), \
         patch("src.agent.graph.build_graph", side_effect=fake_build_graph):
        result = run_agent(
            match_info={"home_team": "A", "away_team": "B", "date": "2025-01-01"},
            config=cfg,
            tools=[],
            return_full_state=True,
        )

    assert result == full_state


def test_run_agent_returns_recommendation_only_by_default():
    from unittest.mock import patch, MagicMock
    from src.agent.graph import run_agent

    full_state = {
        "recommendation": {"overall": "no_bet"},
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
    }

    def fake_build_graph(config, tools):
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = full_state
        return mock_compiled

    cfg = _make_config()
    with patch("src.agent.graph._build_llm"), \
         patch("src.agent.graph._load_system_prompt", return_value="BASE PROMPT"), \
         patch("src.agent.graph.build_graph", side_effect=fake_build_graph):
        result = run_agent(
            match_info={"home_team": "A", "away_team": "B", "date": "2025-01-01"},
            config=cfg,
            tools=[],
        )

    assert result == {"overall": "no_bet"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_graph.py -k return_full_state -v`
Expected: FAIL with `TypeError: run_agent() got an unexpected keyword argument 'return_full_state'`

- [ ] **Step 3: Implement**

In `src/agent/graph.py`, change the `run_agent` signature and final return (lines 284-338):

```python
def run_agent(
    match_info: dict,
    config: AgentConfig | None = None,
    tools: list | None = None,
    extra_system_instructions: str | None = None,
    return_full_state: bool = False,
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
        return_full_state: A33 -- when True, return the full graph state dict
            (recommendation, competition_resolution, research_evidence,
            forecast_payload) instead of just the recommendation. Used by
            agent-train to persist raw evidence to DuckDB.
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
    if return_full_state:
        return result
    return result["recommendation"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_graph.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_agent_graph.py
git commit -m "$(cat <<'EOF'
feat(agent): add return_full_state option to run_agent

A33 needs competition_resolution/research_evidence/forecast_payload
persisted to DuckDB telemetry, not just the final recommendation.
Backward compatible: defaults to today's recommendation-only return.
EOF
)"
```

---

### Task 3: `BacktestRecord.full_state` + `process_match_row(capture_state=...)`

**Files:**
- Modify: `src/agent/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backtest.py`:

```python
def test_process_match_row_captures_full_state_when_requested():
    full_state = {
        "recommendation": {
            "match": {}, "overall": "no_bet", "markets": [],
            "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
        },
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
    }
    with patch("src.agent.graph.run_agent", return_value=full_state) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"):
        record = process_match_row(_row(fthg=1, ftag=1), _make_config(), capture_state=True)

    assert record.full_state == full_state
    assert record.recommendation == full_state["recommendation"]
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["return_full_state"] is True


def test_process_match_row_full_state_none_by_default():
    recommendation = {
        "match": {}, "overall": "no_bet", "markets": [],
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"):
        record = process_match_row(_row(fthg=1, ftag=1), _make_config())

    assert record.full_state is None
    assert "return_full_state" not in mock_run.call_args.kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest.py -k capture_state -v`
Expected: FAIL with `TypeError: process_match_row() got an unexpected keyword argument 'capture_state'`

- [ ] **Step 3: Implement**

In `src/agent/backtest.py`, add the field to `BacktestRecord` (after `market_results`):

```python
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
    full_state: dict[str, Any] | None = None
```

Replace `process_match_row`:

```python
def process_match_row(row: pd.Series, config: AgentConfig, capture_state: bool = False) -> BacktestRecord:
    """Replay one historical match through the agent and score its recommendation.

    Sets the module-level SnapshotStore to replay mode for this match_id before
    calling run_agent, and always resets it to live mode afterward (even on
    error) so a failed match doesn't leave a later, unrelated call in replay
    mode by accident.

    capture_state (A33): when True, also captures the full graph state
    (competition_resolution/research_evidence/forecast_payload) on the
    returned record's full_state, for agent-train's telemetry persistence.
    """
    # Local imports: keep these inside the function — tests patch
    # src.agent.graph.run_agent and src.agent.tools.configure_snapshot_store,
    # which only works if these names are resolved at call time, not import time.
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools
    from src.agent.snapshot_store import league_base_dir

    match_id = row["match_id"]
    match_info = _build_match_info(row)

    agent_tools.configure_snapshot_store(
        "replay", match_id=match_id, base_dir=league_base_dir(row["league"]),
    )
    try:
        if capture_state:
            full_state = run_agent(match_info=match_info, config=config, return_full_state=True)
            recommendation = full_state["recommendation"]
        else:
            recommendation = run_agent(match_info=match_info, config=config)
            full_state = None
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
        full_state=full_state,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/agent/backtest.py tests/test_backtest.py
git commit -m "$(cat <<'EOF'
feat(agent): add capture_state option to process_match_row

A33's agent-train needs the raw competition_resolution/research_evidence/
forecast_payload per match to persist as telemetry, not just the final
recommendation agent-backtest already scores. Off by default, so
agent-backtest's behavior and existing tests are unchanged.
EOF
)"
```

---

### Task 4: `lessons_node` + graph wiring + prompt note

**Files:**
- Modify: `src/agent/pipeline.py`
- Modify: `src/agent/graph.py:1-20, 194-282` (imports, `AgentState`, `build_graph`)
- Modify: `config/prompts/agent_v1.txt`
- Test: `tests/test_agent_pipeline.py`, `tests/test_agent_graph.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_pipeline.py`:

```python
def test_lessons_node_returns_empty_dict_when_not_live_mode():
    from unittest.mock import patch
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("replay")
    try:
        with patch("src.agent.lessons.load_approved_lessons") as mock_load:
            result = lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}})
        mock_load.assert_not_called()
        assert result == {}
    finally:
        agent_tools._snapshot_store.set_mode("live")


def test_lessons_node_returns_empty_dict_when_no_approved_lessons():
    from unittest.mock import patch, MagicMock
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    with patch("src.agent.lessons.load_approved_lessons", return_value=[]), \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        result = lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}})
    assert result == {}


def test_lessons_node_appends_human_message_with_approved_lessons():
    from unittest.mock import patch, MagicMock
    from langchain_core.messages import HumanMessage
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    with patch("src.agent.lessons.load_approved_lessons", return_value=["Lesson A", "Lesson B"]), \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        result = lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}})

    assert len(result["messages"]) == 1
    message = result["messages"][0]
    assert isinstance(message, HumanMessage)
    assert "Lesson A" in message.content
    assert "Lesson B" in message.content


def test_lessons_node_uses_competition_resolution_from_state():
    from unittest.mock import patch, MagicMock
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    with patch("src.agent.lessons.load_approved_lessons", return_value=[]) as mock_load, \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        lessons_node({"competition_resolution": {"competition": "SP1", "tier": "competition_specific"}})

    args = mock_load.call_args.args
    assert args[1] == "SP1"
    assert args[2] == "competition_specific"


def test_pipeline_module_never_imports_lesson_write_or_review_functions():
    """A33 acceptance: the live code path (this module) must have no
    function available to it that can write, approve, or reject lessons --
    only load_approved_lessons, which itself can't read outcomes (see
    tests/test_agent_lessons.py)."""
    import pathlib
    source = pathlib.Path("src/agent/pipeline.py").read_text()
    assert "load_approved_lessons" in source
    for forbidden in ("insert_lesson_candidate", "insert_telemetry", "approve_lesson", "reject_lesson"):
        assert forbidden not in source
```

Add to `tests/test_agent_graph.py` (a new lessons-injection integration test, plus a required patch on the existing `test_run_agent_produces_recommendation_when_forecast_succeeds` test since it now reaches the new node):

```python
def test_run_agent_injects_lessons_message_before_llm_call_in_live_mode():
    from unittest.mock import MagicMock, patch
    from langchain_core.messages import AIMessage, HumanMessage
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
         patch("src.forecast.forecast_service.ForecastService") as MockSvc, \
         patch("src.agent.lessons.load_approved_lessons", return_value=["Historical lesson text"]), \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_forecast_result

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content=llm_json)
        mock_build_llm.return_value = mock_llm

        cfg = _make_config()
        run_agent(
            match_info={"home_team": "Man City", "away_team": "Arsenal", "date": "2026-06-21", "league": "E0"},
            config=cfg,
            tools=[],
        )

    call_messages = mock_llm.bind_tools.return_value.invoke.call_args.args[0]
    lesson_messages = [
        m for m in call_messages
        if isinstance(m, HumanMessage) and "Historical lesson text" in m.content
    ]
    assert len(lesson_messages) == 1
```

In the *existing* `test_run_agent_produces_recommendation_when_forecast_succeeds` test, add two lines to the `with patch(...)` block so it stays hermetic now that the real graph reaches `lessons_node`:

```python
    with patch("src.agent.graph._build_llm") as mock_build_llm, \
         patch("src.agent.graph._load_system_prompt", return_value="stub prompt"), \
         patch("src.agent.tools._dated_web_search", return_value="No results found."), \
         patch("src.forecast.forecast_service.ForecastService") as MockSvc, \
         patch("src.agent.lessons.load_approved_lessons", return_value=[]), \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        instance = MagicMock()
```

(keep the rest of that test body unchanged — only the `with` header and the one added `MockDB` line change).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_pipeline.py -k lessons -v tests/test_agent_graph.py -k "lessons or produces_recommendation"`
Expected: FAIL — `ImportError: cannot import name 'lessons_node'` (pipeline tests) and the graph test fails/errors because `lessons_node`/wiring don't exist yet and the patched-but-unused `MockDB`/`load_approved_lessons` mean nothing until the node exists.

- [ ] **Step 3: Implement**

In `src/agent/pipeline.py`, add (near the other node functions, after `forecast_node`):

```python
def lessons_node(state: dict) -> dict:
    """A33: inject reviewer-approved lessons scoped to this match's
    competition/tier as a HumanMessage before the LLM's turn -- same
    injection pattern forecast_node uses for evidence (a node-returned
    "messages" list is appended via AgentState's add_messages reducer).

    Gated on SnapshotStore mode == "live": outside genuine live runs
    (agent-backtest/agent-train replay, or agent-snapshot record), lessons
    are skipped entirely. Injecting lessons approved *after* a historical
    match would leak future information into backtest/train scoring,
    corrupting the A13/A21/A34 baseline methodology agent-backtest and
    agent-train share. Gating here (rather than a config flag) means the
    same compiled graph is correct for every CLI entry point.

    Only imports load_approved_lessons from src.agent.lessons -- see that
    module's docstring and tests/test_agent_lessons.py for why that function
    alone can't read match outcomes or pending/rejected lessons.
    """
    from src.agent.tools import get_snapshot_store

    if get_snapshot_store().mode != "live":
        return {}

    from src.agent.lessons import extract_competition_scope, load_approved_lessons
    from src.utils.db_manager import DuckDBManager

    competition_id, tier = extract_competition_scope(state)
    with DuckDBManager().connection(read_only=True) as conn:
        lessons = load_approved_lessons(conn, competition_id, tier)
    if not lessons:
        return {}
    lessons_text = "Lessons from past evaluated matches:\n" + "\n".join(f"- {lesson}" for lesson in lessons)
    return {"messages": [HumanMessage(content=lessons_text)]}
```

In `src/agent/graph.py`, update the import (line 13):

```python
from src.agent.pipeline import forecast_node, lessons_node, research_node, resolve_competition_node
```

Update `route_after_forecast` (around line 214) to route to the new node on success:

```python
    def route_after_forecast(state: AgentState) -> Literal["lessons", "output"]:
        payload = state.get("forecast_payload")
        succeeded = bool(payload) and "error" not in payload
        route = "lessons" if succeeded else "output"
        _LOG.info("route_after_forecast | succeeded=%s | route=%s", succeeded, route)
        return route
```

Update graph wiring (around lines 265-279):

```python
    graph = StateGraph(AgentState)
    graph.add_node("resolve_competition", resolve_competition_node)
    graph.add_node("research", research_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("lessons", lessons_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output", output_node)

    graph.set_entry_point("resolve_competition")
    graph.add_edge("resolve_competition", "research")
    graph.add_edge("research", "forecast")
    graph.add_conditional_edges("forecast", route_after_forecast, {"lessons": "lessons", "output": "output"})
    graph.add_edge("lessons", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "output": "output"})
    graph.add_edge("tools", "agent")
    graph.add_edge("output", END)

    return graph.compile()
```

In `config/prompts/agent_v1.txt`, extend the "Evidence Already Gathered" section (around line 9) with one sentence:

```
## Evidence Already Gathered

Before your turn, the system has already deterministically: resolved the match's competition tier, run the ML forecast model, and searched for injury/availability news, recent-form context, and (when odds weren't supplied) an odds-verification search. This evidence appears in a message below. A "Lessons from past evaluated matches" message may also appear below with reviewer-approved notes from prior matches in this competition or tier -- treat it as advisory context, not as fact about this specific match; it will not always be present. You do NOT have forecast_league, forecast_international, or resolve_competition available as tools — do not attempt to call them; they no longer exist in this conversation.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_pipeline.py tests/test_agent_graph.py -v`
Expected: PASS (all tests, including the new ones)

- [ ] **Step 5: Run the full agent test suite to check for regressions**

Run: `python -m pytest tests/ -k agent -v`
Expected: PASS, zero regressions

- [ ] **Step 6: Commit**

```bash
git add src/agent/pipeline.py src/agent/graph.py config/prompts/agent_v1.txt tests/test_agent_pipeline.py tests/test_agent_graph.py
git commit -m "$(cat <<'EOF'
feat(agent): wire competition-scoped lessons into the live graph (A33)

New lessons_node runs after forecast succeeds, loading reviewer-approved
lessons matching the match's competition_id/tier and injecting them as a
HumanMessage before the LLM's turn -- same pattern forecast_node uses for
evidence. Gated on SnapshotStore.mode == "live" so agent-backtest/
agent-train replay never sees lessons approved after the fact, which
would otherwise leak future information into baseline scoring.
EOF
)"
```

---

### Task 5: `agent-train` CLI

**Files:**
- Modify: `main.py`
- Test: `tests/test_main_agent_train.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for main.py's run_agent_train CLI entry point (A33)."""
from __future__ import annotations

import duckdb

from main import _write_train_artifacts
from src.agent.backtest import BacktestRecord


def _record(match_id="m1", league="E0", full_state=None) -> BacktestRecord:
    return BacktestRecord(
        match_id=match_id,
        home_team="City",
        away_team="Arsenal",
        date="2025-03-01",
        league=league,
        recommendation={
            "overall": "no_bet", "confidence": "medium",
            "prediction_basis": "team_history_and_market", "limitations": [],
        },
        actual={"result": "home", "btts": "yes", "total_goals": 3, "total_goals_side": "over_2.5"},
        market_results=[],
        full_state=full_state,
    )


def test_write_train_artifacts_writes_one_lesson_and_telemetry_row_per_record():
    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
    })

    written = _write_train_artifacts(conn, [record], run_id="run-1")

    assert written == 1
    lesson_row = conn.execute(
        "SELECT competition_id, tier, status, source_match_id FROM agent_lessons"
    ).fetchone()
    assert lesson_row == ("E0", "competition_specific", "pending", "m1")

    telemetry_row = conn.execute("SELECT match_id, run_id FROM agent_telemetry").fetchone()
    assert telemetry_row == ("m1", "run-1")


def test_write_train_artifacts_skips_records_without_full_state():
    conn = duckdb.connect(":memory:")
    record = _record(full_state=None)

    written = _write_train_artifacts(conn, [record], run_id="run-1")

    assert written == 0
    assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM agent_telemetry").fetchone()[0] == 0


def test_write_train_artifacts_handles_multiple_records():
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
        _record(match_id="m2", league="SP1", full_state={
            "competition_resolution": {"competition": "SP1", "tier": "competition_specific"},
        }),
    ]

    written = _write_train_artifacts(conn, records, run_id="run-2")

    assert written == 2
    assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM agent_telemetry").fetchone()[0] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main_agent_train.py -v`
Expected: FAIL with `ImportError: cannot import name '_write_train_artifacts' from 'main'`

- [ ] **Step 3: Implement**

In `main.py`, update `_run_backtest_concurrent` (around line 1132) to thread through `capture_state`:

```python
async def _run_backtest_concurrent(matches, config, concurrency: int, capture_state: bool = False) -> list:
    """Run process_match_row for every match concurrently, bounded by a semaphore.
    Each call runs in its own thread via asyncio.to_thread since the agent graph
    and tools are synchronous; SnapshotStore's thread-local state (A09) keeps
    concurrent replay contexts from clobbering each other. Per-match failures
    (e.g. SnapshotMissingError for an unrecorded match) are caught and skipped
    so one bad match doesn't abort the whole batch — mirrors run_agent_snapshot's
    error-tolerance pattern.

    capture_state (A33): threaded through to process_match_row so agent-train
    can persist each match's raw evidence to DuckDB telemetry."""
    import asyncio
    import sys

    from tqdm import tqdm

    from src.agent.backtest import process_match_row

    semaphore = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(matches), desc="Backtesting")
    rows = [row for _, row in matches.iterrows()]

    async def _run_one(row):
        async with semaphore:
            try:
                record = await asyncio.to_thread(process_match_row, row, config, capture_state=capture_state)
            except Exception as exc:
                match_id = row.get("match_id", "?") if hasattr(row, "get") else "?"
                print(f"  SKIP {match_id}: {exc}", file=sys.stderr)
                record = None
            finally:
                progress.update(1)
            return record

    try:
        results = await asyncio.gather(*[_run_one(row) for row in rows])
    finally:
        progress.close()
    records = [r for r in results if r is not None]
    skipped = len(results) - len(records)
    if skipped:
        print(f"Skipped {skipped}/{len(results)} matches (see stderr for details)")
    return records
```

Add `_write_train_artifacts` and `run_agent_train` after `run_agent_backtest` (around line 1207):

```python
def _write_train_artifacts(conn, records: list, run_id: str) -> int:
    """Write one telemetry row and one pending lesson candidate per scored
    record that captured full graph state. Records without full_state (e.g.
    a per-match failure that skipped capture) are silently skipped -- there's
    nothing to persist. Returns the number of lessons written."""
    from src.agent.lessons import (
        create_lessons_tables,
        extract_competition_scope,
        generate_lesson_text,
        insert_lesson_candidate,
        insert_telemetry,
    )

    create_lessons_tables(conn)
    written = 0
    for record in records:
        if not record.full_state:
            continue
        competition_id, tier = extract_competition_scope(record.full_state)
        insert_telemetry(
            conn,
            match_id=record.match_id,
            run_id=run_id,
            competition_resolution=record.full_state.get("competition_resolution"),
            research_evidence=record.full_state.get("research_evidence"),
            forecast_payload=record.full_state.get("forecast_payload"),
            recommendation=record.recommendation,
        )
        lesson_text = generate_lesson_text(record)
        insert_lesson_candidate(conn, lesson_text, competition_id, tier, record.match_id)
        written += 1
    return written


def run_agent_train(
    from_date: str,
    to_date: str,
    league: str | None,
    stake_mode: str,
    sample: int | None,
    concurrency: int,
    config_path: str | None,
) -> None:
    """Critic/train mode (A33): replay completed matches, score them the same
    way agent-backtest does, and additionally write one competition/tier-
    tagged lesson candidate plus a raw-evidence telemetry row per match."""
    import asyncio
    import uuid

    from src.agent.agent_config import AgentConfig
    from src.agent.backtest import BacktestHarness
    from src.agent.evaluation import build_evaluation_report, print_report, save_report
    from src.agent.staking import simulate_flat_stake, simulate_kelly_stake

    if concurrency < 1:
        raise ValueError(f"--concurrency must be >= 1, got {concurrency}")

    cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    harness = BacktestHarness(config=cfg)
    matches = harness.load_matches(from_date, to_date, league=league, sample=sample)
    print(f"Running train mode over {len(matches)} matches (concurrency={concurrency})...")

    records = asyncio.run(_run_backtest_concurrent(matches, cfg, concurrency, capture_state=True))

    stake_fn = simulate_kelly_stake if stake_mode == "kelly" else simulate_flat_stake
    bankroll_result = stake_fn(records)
    report = build_evaluation_report(records, bankroll_result)
    print_report(report)
    path = save_report(report, cfg, base_dir="reports/agent_train")
    print(f"\nReport saved to {path}")

    run_id = uuid.uuid4().hex
    with harness.db.connection() as conn:
        lessons_written = _write_train_artifacts(conn, records, run_id)
    print(f"Wrote {lessons_written} lesson candidates and telemetry rows (run_id={run_id})")
```

Add the `agent-train` subparser after `agent-backtest` (around line 322, before `# agent-compare`):

```python
    # agent-train (A33)
    agent_train_parser = subparsers.add_parser(
        "agent-train",
        help="Critic/train mode: score completed matches and record reviewed lesson candidates in DuckDB",
    )
    agent_train_parser.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD (inclusive)")
    agent_train_parser.add_argument("--to-date", required=True, help="End date YYYY-MM-DD (inclusive)")
    agent_train_parser.add_argument("--league", default=None, help="League code (e.g. E0). Omit for all leagues.")
    agent_train_parser.add_argument("--stake-mode", choices=["flat", "kelly"], default="flat")
    agent_train_parser.add_argument("--sample", type=int, default=None, help="Stratified sample size before running the full set")
    agent_train_parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent agent runs")
    agent_train_parser.add_argument("--config", default=None, help="Path to agent_config.yaml (default: config/agent_config.yaml)")
```

Add dispatch in `main()` after the `agent-backtest` branch (around line 1446):

```python
    elif args.command == "agent-train":
        run_agent_train(
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            stake_mode=args.stake_mode,
            sample=args.sample,
            concurrency=args.concurrency,
            config_path=args.config,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main_agent_train.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Smoke-test the CLI wiring**

Run: `python main.py agent-train --help`
Expected: argparse help text listing `--from-date`, `--to-date`, `--league`, `--stake-mode`, `--sample`, `--concurrency`, `--config`, no traceback.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main_agent_train.py
git commit -m "$(cat <<'EOF'
feat(agent): add agent-train CLI (A33 critic/train mode)

Structurally parallel to agent-backtest: same replay/scoring path, plus
persists one competition/tier-tagged lesson candidate and a raw-evidence
telemetry row per match to DuckDB via the new src/agent/lessons.py module.
EOF
)"
```

---

### Task 6: `agent-lessons approve/reject` CLI

**Files:**
- Modify: `main.py`
- Test: `tests/test_main_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for main.py's agent-lessons CLI entry points (A33)."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from main import run_agent_lessons_approve, run_agent_lessons_reject
from src.agent.lessons import create_lessons_tables, insert_lesson_candidate


def _fake_db_manager(conn):
    manager = MagicMock()

    @contextmanager
    def _connection(read_only=False):
        yield conn

    manager.connection.side_effect = _connection
    return manager


def test_run_agent_lessons_approve_sets_status_and_scope():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        run_agent_lessons_approve(lesson_id=lesson_id, scope="competition", reviewer="alice")

    row = conn.execute("SELECT status, scope, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == ("approved", "competition", "alice")


def test_run_agent_lessons_approve_defaults_reviewer_to_current_user():
    import getpass

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        run_agent_lessons_approve(lesson_id=lesson_id, scope="tier", reviewer=None)

    row = conn.execute("SELECT reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row[0] == getpass.getuser()


def test_run_agent_lessons_approve_raises_for_unknown_id():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        with pytest.raises(ValueError, match="999"):
            run_agent_lessons_approve(lesson_id=999, scope="competition", reviewer="alice")


def test_run_agent_lessons_reject_sets_status_rejected():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        run_agent_lessons_reject(lesson_id=lesson_id, reviewer="bob")

    row = conn.execute("SELECT status, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == ("rejected", "bob")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main_agent_lessons.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_agent_lessons_approve' from 'main'`

- [ ] **Step 3: Implement**

In `main.py`, add after `run_agent_train` (from Task 5):

```python
def run_agent_lessons_approve(lesson_id: int, scope: str, reviewer: str | None) -> None:
    """Approve a pending lesson candidate (A33). scope='competition' pins it
    to its source competition; scope='tier' widens it to the whole tier."""
    import getpass

    from src.agent.lessons import approve_lesson, create_lessons_tables
    from src.utils.db_manager import DuckDBManager

    db = DuckDBManager()
    with db.connection() as conn:
        create_lessons_tables(conn)
        approve_lesson(conn, lesson_id, scope, reviewer or getpass.getuser())
    print(f"Approved lesson {lesson_id} (scope={scope})")


def run_agent_lessons_reject(lesson_id: int, reviewer: str | None) -> None:
    """Reject a pending lesson candidate (A33)."""
    import getpass

    from src.agent.lessons import create_lessons_tables, reject_lesson
    from src.utils.db_manager import DuckDBManager

    db = DuckDBManager()
    with db.connection() as conn:
        create_lessons_tables(conn)
        reject_lesson(conn, lesson_id, reviewer or getpass.getuser())
    print(f"Rejected lesson {lesson_id}")
```

Add the `agent-lessons` subparser with nested `approve`/`reject` subcommands, after the `agent-train` subparser added in Task 5:

```python
    # agent-lessons (A33)
    agent_lessons_parser = subparsers.add_parser(
        "agent-lessons",
        help="Review pending lesson candidates written by agent-train",
    )
    agent_lessons_subparsers = agent_lessons_parser.add_subparsers(dest="lessons_action", required=True)

    agent_lessons_approve_parser = agent_lessons_subparsers.add_parser("approve", help="Approve a pending lesson")
    agent_lessons_approve_parser.add_argument("id", type=int, help="Lesson id")
    agent_lessons_approve_parser.add_argument(
        "--scope", required=True, choices=["competition", "tier"],
        help="competition: applies only to the lesson's source competition. tier: applies to every match in the lesson's tier.",
    )
    agent_lessons_approve_parser.add_argument("--reviewer", default=None, help="Reviewer name (default: current OS user)")

    agent_lessons_reject_parser = agent_lessons_subparsers.add_parser("reject", help="Reject a pending lesson")
    agent_lessons_reject_parser.add_argument("id", type=int, help="Lesson id")
    agent_lessons_reject_parser.add_argument("--reviewer", default=None, help="Reviewer name (default: current OS user)")
```

Add dispatch in `main()` after the `agent-train` branch (from Task 5):

```python
    elif args.command == "agent-lessons":
        if args.lessons_action == "approve":
            run_agent_lessons_approve(lesson_id=args.id, scope=args.scope, reviewer=args.reviewer)
        elif args.lessons_action == "reject":
            run_agent_lessons_reject(lesson_id=args.id, reviewer=args.reviewer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main_agent_lessons.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Smoke-test the CLI wiring**

Run: `python main.py agent-lessons approve --help` and `python main.py agent-lessons reject --help`
Expected: argparse help text for each, no traceback. Also run `python main.py agent-lessons --help` and confirm `approve`/`reject` are listed as subcommands.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main_agent_lessons.py
git commit -m "$(cat <<'EOF'
feat(agent): add agent-lessons approve/reject CLI (A33)

Minimal review CLI, no UI. approve requires --scope (competition|tier)
explicitly -- no default -- since that's the one place a human judges
whether a lesson generalizes across a whole tier or stays pinned to its
source competition.
EOF
)"
```

---

### Task 7: Full regression, docs, and completion

**Files:**
- Modify: `documents/agent_techspec.md`
- Modify: `documents/agent_user_stories.md`

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS, zero regressions outside intentional changes (previous full-suite baseline was 501 passed / 1 skipped per A31's completion notes; expect that count plus this plan's new tests, e.g. 18 + 2 + 2 + 5 + 3 + 4 = 34 new tests, minus none removed).

- [ ] **Step 2: End-to-end smoke test against a real (small) date range**

Run (adjust dates to a range with completed matches already in `raw_matches`, e.g. one already used by `agent-backtest` in prior runs):

```bash
python main.py agent-train --from-date 2026-01-01 --to-date 2026-01-07 --league E0 --sample 2 --concurrency 1
```

Expected: prints a report identical in shape to `agent-backtest`'s, then "Wrote N lesson candidates and telemetry rows (run_id=...)". Then inspect what was written:

```bash
python3 -c "
import duckdb
conn = duckdb.connect('data/fpai_core.db', read_only=True)
print(conn.execute('SELECT id, status, competition_id, tier, scope, source_match_id FROM agent_lessons ORDER BY id DESC LIMIT 5').fetchall())
print(conn.execute('SELECT match_id, run_id FROM agent_telemetry ORDER BY created_at DESC LIMIT 5').fetchall())
"
```

Expected: rows present with `status='pending'`, `scope=NULL`, real `competition_id`/`tier` values. Then approve one:

```bash
python main.py agent-lessons approve <id-from-above> --scope competition
```

Expected: "Approved lesson <id> (scope=competition)"; re-running the inspect query shows `status='approved'`, `scope='competition'`, `reviewed_at` populated.

- [ ] **Step 3: Document in `documents/agent_techspec.md`**

Add a new numbered section after the last existing section (`## 19. League-Aware Routing...`, ending around line 764 — check the actual last line number first and continue the sequence, e.g. `## 20.`):

```markdown
## 20. Critic/Train Mode and Competition-Scoped Lessons (A33)

Design: `docs/superpowers/specs/2026-07-22-agent-phase11-design.md` (A33 section, revised 2026-07-24).

### 20.1 `agent-train` CLI

Structurally parallel to `agent-backtest` (Section 13): same `BacktestHarness.load_matches()` + `process_match_row()` replay path, same `src/agent/evaluation.py` ROI/hit-rate/drawdown scoring, same report shape (saved under `reports/agent_train/` instead of `reports/agent_backtest/` to keep the two apart). Additionally, for every match that captured full graph state (`process_match_row(..., capture_state=True)`), writes:

- One row to `agent_telemetry` (`match_id`, `run_id`, `competition_resolution`, `research_evidence`, `forecast_payload`, `recommendation` — JSON-serialized TEXT columns — `created_at`). `run_id` is a single `uuid4().hex` shared by every match in one `agent-train` invocation.
- One `status='pending'` row to `agent_lessons` (`lesson_text` from a deterministic template — see `generate_lesson_text()` in `src/agent/lessons.py` — plus `competition_id`/`tier` recorded automatically from that match's `competition_resolution`).

```bash
python main.py agent-train --from-date 2026-01-01 --to-date 2026-01-31 --league E0 --stake-mode flat
```

### 20.2 `agent-lessons approve/reject` CLI

```bash
python main.py agent-lessons approve <id> --scope competition   # or --scope tier
python main.py agent-lessons reject <id>
```

`--scope` is required on `approve`, no default: `competition` pins the lesson to its recorded `competition_id`; `tier` widens it to every match resolving to its recorded `tier` (`general_purpose` / `competition_specific`), regardless of competition. This is the only point a human judges whether a lesson generalizes — `agent-train` itself makes no such judgment.

### 20.3 Live-mode injection (`lessons_node`, `src/agent/pipeline.py`)

A new required graph node runs after `forecast_node` succeeds (`resolve_competition → research → forecast → lessons → agent`). It loads `agent_lessons` rows where `status='approved'` AND (`scope='competition'` AND `competition_id` matches this match) OR (`scope='tier'` AND `tier` matches this match's tier), and injects them as a `HumanMessage` ahead of the LLM's turn — the same mechanism `forecast_node` uses for forecast/research evidence.

Gated on `SnapshotStore.mode == "live"` (`src/agent/tools.get_snapshot_store()`): `agent-backtest`/`agent-train` replay and `agent-snapshot` record never see lessons, since injecting anything approved after a historical match ran would leak future information into the A13/A21/A34 baseline scoring methodology `agent-backtest` and `agent-train` share.

`load_approved_lessons()` (`src/agent/lessons.py`) is the only function `lessons_node` imports from the lessons module — its SQL hardcodes `status='approved'` and never touches an outcome-bearing table, so live mode is structurally, not just conventionally, unable to read match outcomes or pending/rejected lessons.

### 20.4 Schema

```sql
CREATE SEQUENCE agent_lessons_id_seq START 1;
CREATE TABLE agent_lessons (
    id INTEGER PRIMARY KEY DEFAULT nextval('agent_lessons_id_seq'),
    lesson_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
    competition_id TEXT,                       -- NULL for leagueless internationals
    tier TEXT NOT NULL,                         -- general_purpose | competition_specific
    scope TEXT,                                 -- NULL until approved; competition | tier
    source_match_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    reviewed_at TIMESTAMP,
    reviewer TEXT
);

CREATE TABLE agent_telemetry (
    match_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    competition_resolution TEXT,   -- JSON
    research_evidence TEXT,        -- JSON
    forecast_payload TEXT,         -- JSON
    recommendation TEXT,           -- JSON
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, run_id)
);
```

### 20.5 Known limitation, accepted as designed

Approved lessons accumulate indefinitely within each `competition_id`/`tier` bucket, with no cap or conflict resolution. Acceptable for initial land (lesson volume will be low early on); revisit if it becomes a real prompt-bloat problem.
```

- [ ] **Step 4: Mark A33 completed in `documents/agent_user_stories.md`**

In the A33 row (already revised with competition-scoping language earlier), change `active` to `completed` and append completion notes matching the style of A30/A31/A32's entries — after the existing description and before the trailing ` | Size L · Milestone M12 · Depends on: A31...`, add:

```
**Completion notes (<today's date>):** Implemented as `src/agent/lessons.py` (schema + CRUD), `lessons_node` in `src/agent/pipeline.py` (wired into `src/agent/graph.py` between `forecast` and `agent`, gated on `SnapshotStore.mode == "live"` so backtest/train replay never sees lessons — this gating wasn't explicit in the original design doc language and was added during implementation to protect A21/A34 baseline integrity), `agent-train`/`agent-lessons` CLI in `main.py`. TDD throughout: `tests/test_agent_lessons.py`, `tests/test_agent_pipeline.py`, `tests/test_agent_graph.py`, `tests/test_backtest.py`, `tests/test_main_agent_train.py`, `tests/test_main_agent_lessons.py`. Full suite: <paste actual pass/skip counts from Step 1>. Implementation plan: `docs/superpowers/plans/2026-07-24-agent-critic-mode-lessons.md`.
```

(Replace `<today's date>` and `<paste actual pass/skip counts...>` with the real values observed when this step is executed — do not leave the placeholders in the committed text.)

- [ ] **Step 5: Commit**

```bash
git add documents/agent_techspec.md documents/agent_user_stories.md
git commit -m "$(cat <<'EOF'
docs(agent): document A33 critic/train mode and mark story completed

Adds Section 20 to agent_techspec.md (CLI, schema, live-mode injection
mechanism, the SnapshotStore-mode gating decision made during
implementation) and marks A33 completed in agent_user_stories.md.
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** every A33 bullet from the design doc (lessons table + new columns, telemetry table, `agent-train`, `agent-lessons approve/reject --scope`, live-mode scoped loading, structural outcome-isolation) maps to a task above. The one deliberate deviation from the literal design text — lessons injected as a `HumanMessage` rather than concatenated into the system prompt string, and gated on `SnapshotStore.mode == "live"` (not mentioned explicitly in the design doc) — is called out in Task 4's node docstring, Section 20.3, and A33's completion notes, since it's a real implementation decision, not a silent shortcut. It follows the codebase's own established pattern (`forecast_node`'s evidence injection) and is required to avoid corrupting A21/A34 baseline scoring, which the brainstorming session did not explicitly discuss.
- **Type/name consistency checked:** `run_agent(return_full_state=...)` (Task 2) → `process_match_row(capture_state=...)` calls it with that exact kwarg (Task 3) → `_run_backtest_concurrent(capture_state=...)` threads it through (Task 5) → `BacktestRecord.full_state` is read the same way by both `_write_train_artifacts` (Task 5) and `lessons_node`'s counterpart `extract_competition_scope` (Task 1/4). `agent_lessons`/`agent_telemetry` column names are identical across `src/agent/lessons.py` (Task 1), the SQL in Section 20.4, and every test file.
- **No placeholders:** all code blocks are complete; the only bracketed placeholders left (`<today's date>`, `<paste actual...>`, `<id-from-above>`) are in Task 7's manual doc-writing and smoke-test steps, which are inherently run-time-dependent, not implementation gaps.
