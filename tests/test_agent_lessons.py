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
