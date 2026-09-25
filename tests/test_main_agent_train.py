"""Tests for main.py's run_agent_train CLI entry point (A33)."""
from __future__ import annotations

import json
from unittest.mock import patch

import duckdb
from langchain_core.messages import AIMessage, SystemMessage

from main import _write_telemetry_rows, _write_train_artifacts
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

    lessons_written, telemetry_written = _write_train_artifacts(conn, [record], run_id="run-1")

    assert (lessons_written, telemetry_written) == (1, 1)
    lesson_row = conn.execute(
        "SELECT competition_id, tier, status, source_match_id FROM agent_lessons"
    ).fetchone()
    assert lesson_row == ("E0", "competition_specific", "pending", "m1")

    telemetry_row = conn.execute("SELECT match_id, run_id FROM agent_telemetry").fetchone()
    assert telemetry_row == ("m1", "run-1")


def test_write_train_artifacts_threads_run_id_into_lesson_candidates():
    """A127: agent_lessons.run_id groups one agent-train run's lesson
    candidates for reviewer convenience."""
    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
    })

    _write_train_artifacts(conn, [record], run_id="run-xyz")

    stored_run_id = conn.execute("SELECT run_id FROM agent_lessons").fetchone()[0]
    assert stored_run_id == "run-xyz"


def test_write_train_artifacts_threads_run_id_for_batched_lesson_candidates():
    """Same, for the batch_size>1 code path (a separate insert_lesson_candidate call site)."""
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
            "research_evidence": {"availability": "ok"}, "forecast_payload": {"result_3way": {}},
        }),
        _record(match_id="m2", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
            "research_evidence": {"availability": "ok"}, "forecast_payload": {"result_3way": {}},
        }),
    ]

    _write_train_artifacts(conn, records, run_id="run-batch-1", batch_size=2)

    stored_run_id = conn.execute("SELECT run_id FROM agent_lessons").fetchone()[0]
    assert stored_run_id == "run-batch-1"


def test_write_telemetry_rows_shared_helper_writes_one_row_per_full_state_record():
    """A107: the extracted helper both _write_train_artifacts and
    run_agent_backtest now call -- exercised directly here."""
    conn = duckdb.connect(":memory:")
    from src.agent.lessons import create_lessons_tables
    create_lessons_tables(conn)

    records = [
        _record(match_id="m1", full_state={"competition_resolution": {"competition": "E0", "tier": "competition_specific"}}),
        _record(match_id="m2", full_state=None),  # skipped -- no full_state
    ]

    scoped = _write_telemetry_rows(conn, records, run_id="run-1")

    assert len(scoped) == 1
    assert scoped[0][0].match_id == "m1"
    assert scoped[0][1:] == ("E0", "competition_specific")
    rows = conn.execute("SELECT match_id FROM agent_telemetry").fetchall()
    assert rows == [("m1",)]


def test_write_train_artifacts_persists_reasoning_trace_from_full_state_messages():
    """A106: full_state["messages"] (the LLM's own reasoning/tool-call
    trace) is serialized and persisted, not just the final recommendation."""
    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
        "messages": [
            SystemMessage(content="system prompt"),
            AIMessage(content="Away win looks like value here."),
        ],
    })

    _write_train_artifacts(conn, [record], run_id="run-1")

    reasoning_trace = conn.execute("SELECT reasoning_trace FROM agent_telemetry WHERE match_id = 'm1'").fetchone()[0]
    assert json.loads(reasoning_trace) == [
        {"role": "system", "content": "system prompt"},
        {"role": "ai", "content": "Away win looks like value here."},
    ]


def test_write_train_artifacts_uses_llm_reflection_for_batch_size_1_when_config_given():
    """A109: batch_size<=1 now reflects via generate_match_reflection when a
    config (and therefore an llm_invoke) is given -- previously config was
    accepted but silently unused on this path."""
    from src.agent.agent_config import AgentConfig

    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
        "messages": [AIMessage(content="Picked no_bet, no clear edge.")],
    })
    config = AgentConfig(
        model="stub-model", provider="ollama", temperature=0.0, max_tool_calls=5,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_conditional_odds_threshold=1.5,
        min_value_edge=0.05, markets=["result_3way"], system_prompt_version="v1",
    )

    with patch("main._build_llm_invoke", return_value=lambda p: "The agent correctly found no edge and stood aside."):
        _write_train_artifacts(conn, [record], run_id="run-7", batch_size=1, config=config)

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert lesson_text == "The agent correctly found no edge and stood aside."


def test_write_train_artifacts_batch_size_1_falls_back_without_config():
    """Unchanged-behavior guard: every pre-existing caller of this path
    passes no config (the default), so it must keep producing exactly
    generate_lesson_text's template, not attempt any LLM call."""
    from src.agent.lessons import generate_lesson_text

    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
        "messages": [AIMessage(content="Picked no_bet, no clear edge.")],
    })

    _write_train_artifacts(conn, [record], run_id="run-8", batch_size=1)

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert lesson_text == generate_lesson_text(record)


def test_write_train_artifacts_skips_records_without_full_state():
    conn = duckdb.connect(":memory:")
    record = _record(full_state=None)

    lessons_written, telemetry_written = _write_train_artifacts(conn, [record], run_id="run-1")

    assert (lessons_written, telemetry_written) == (0, 0)
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

    lessons_written, telemetry_written = _write_train_artifacts(conn, records, run_id="run-2")

    assert (lessons_written, telemetry_written) == (2, 2)
    assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM agent_telemetry").fetchone()[0] == 2


def test_write_train_artifacts_batches_same_scope_records():
    """A39: batch_size=2 aggregates same-(competition_id, tier) records into
    one lesson row instead of one per match, comma-joining source_match_id."""
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
        _record(match_id="m2", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
        _record(match_id="m3", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
    ]

    lessons_written, telemetry_written = _write_train_artifacts(conn, records, run_id="run-3", batch_size=2)

    assert telemetry_written == 3  # still one telemetry row per match
    assert lessons_written == 2  # ceil(3/2): one batch of 2, one batch of 1
    rows = conn.execute("SELECT source_match_id FROM agent_lessons ORDER BY id").fetchall()
    assert rows == [("m1,m2",), ("m3",)]


def test_write_train_artifacts_uses_batch_match_comparisons_when_config_given():
    """2026-09-12 redesign: passing config replaces the batch's lesson text
    entirely with generate_batch_match_comparisons()'s per-match-vs-actual
    comparison (not appended on top of the deterministic stats anymore --
    see generate_batch_lesson_text()'s own docstring update)."""
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
        _record(match_id="m2", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
    ]

    with patch("main._build_llm_invoke", return_value=lambda prompt: "Per-match comparison text here.") as mock_builder:
        lessons_written, telemetry_written = _write_train_artifacts(
            conn, records, run_id="run-5", batch_size=5, config="fake-config",
        )

    mock_builder.assert_called_once_with("fake-config")
    assert (lessons_written, telemetry_written) == (1, 2)
    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert lesson_text == "Per-match comparison text here."


def test_write_train_artifacts_skips_llm_reflection_when_config_is_none():
    """Regression: batching without config (the default) never touches the
    LLM at all -- stats-only lesson text, same as before this feature."""
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
    ]

    with patch("main._build_llm_invoke") as mock_builder:
        _write_train_artifacts(conn, records, run_id="run-6", batch_size=5)

    mock_builder.assert_not_called()
    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert "Reflection:" not in lesson_text


def test_write_train_artifacts_falls_back_to_stats_only_when_reflection_fails():
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
    ]

    def _raising_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    with patch("main._build_llm_invoke", return_value=_raising_invoke):
        lessons_written, telemetry_written = _write_train_artifacts(
            conn, records, run_id="run-7", batch_size=5, config="fake-config",
        )

    assert (lessons_written, telemetry_written) == (1, 1)
    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert "Reflection:" not in lesson_text
    assert lesson_text.startswith("WHEN evaluating a batch of 1 matches")


def test_write_train_artifacts_batches_do_not_span_scope_boundary():
    """A39: a batch never mixes two different (competition_id, tier) pairs,
    even if batch_size would otherwise allow it."""
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
        _record(match_id="m2", league="SP1", full_state={
            "competition_resolution": {"competition": "SP1", "tier": "competition_specific"},
        }),
    ]

    lessons_written, telemetry_written = _write_train_artifacts(conn, records, run_id="run-4", batch_size=5)

    assert lessons_written == 2  # never merged despite batch_size=5
    rows = conn.execute("SELECT competition_id, source_match_id FROM agent_lessons ORDER BY id").fetchall()
    assert rows == [("E0", "m1"), ("SP1", "m2")]
