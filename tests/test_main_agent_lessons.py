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


def test_run_agent_lessons_approve_sets_status_scope_and_rule_text():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        run_agent_lessons_approve(lesson_id=lesson_id, scope="competition", reviewer="alice", rule="NEVER do X.")

    row = conn.execute(
        "SELECT status, scope, rule_text, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]
    ).fetchone()
    assert row == ("approved", "competition", "NEVER do X.", "alice")


def test_run_agent_lessons_approve_defaults_reviewer_to_current_user():
    import getpass

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        run_agent_lessons_approve(lesson_id=lesson_id, scope="tier", reviewer=None, rule="NEVER do X.")

    row = conn.execute("SELECT reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row[0] == getpass.getuser()


def test_run_agent_lessons_approve_raises_for_unknown_id():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        with pytest.raises(ValueError, match="999"):
            run_agent_lessons_approve(lesson_id=999, scope="competition", reviewer="alice")


def test_run_agent_lessons_approve_auto_distills_rule_when_not_given():
    """A44: omitting --rule triggers auto-distillation via the configured LLM."""
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=lambda prompt: "NEVER auto-distilled rule.") as mock_builder:
        run_agent_lessons_approve(lesson_id=lesson_id, scope="competition", reviewer="alice")

    mock_builder.assert_called_once()
    row = conn.execute("SELECT rule_text FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row[0] == "NEVER auto-distilled rule."


def test_run_agent_lessons_approve_raises_when_auto_distillation_fails():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    def _raising_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=_raising_invoke):
        with pytest.raises(ValueError, match="Could not auto-distill"):
            run_agent_lessons_approve(lesson_id=lesson_id, scope="competition", reviewer="alice")

    row = conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row[0] == "pending"  # never approved -- the failed distillation must not leave a half-approved row


def test_run_agent_lessons_approve_raises_on_detected_conflict_without_force():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    existing_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")
    new_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m2")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=lambda p: "Approved existing rule."):
        run_agent_lessons_approve(lesson_id=existing_id, scope="competition", reviewer="alice", rule="EXISTING RULE.")

    def conflicting_invoke(prompt: str) -> str:
        return "Conflicts with rule 1: EXISTING RULE. -- says the opposite."

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=conflicting_invoke):
        with pytest.raises(ValueError, match="conflicts with an existing approved rule"):
            run_agent_lessons_approve(lesson_id=new_id, scope="competition", reviewer="bob", rule="NEW RULE.")

    row = conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [new_id]).fetchone()
    assert row[0] == "pending"  # blocked, never approved


def test_run_agent_lessons_approve_force_overrides_detected_conflict():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    existing_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")
    new_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m2")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=lambda p: "NONE"):
        run_agent_lessons_approve(lesson_id=existing_id, scope="competition", reviewer="alice", rule="EXISTING RULE.")

    def conflicting_invoke(prompt: str) -> str:
        return "Conflicts with rule 1."

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=conflicting_invoke):
        run_agent_lessons_approve(
            lesson_id=new_id, scope="competition", reviewer="bob", rule="NEW RULE.", force=True,
        )

    row = conn.execute("SELECT status, rule_text FROM agent_lessons WHERE id = ?", [new_id]).fetchone()
    assert row == ("approved", "NEW RULE.")


def test_run_agent_lessons_approve_fails_open_when_conflict_check_raises():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    existing_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")
    new_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m2")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=lambda p: "NONE"):
        run_agent_lessons_approve(lesson_id=existing_id, scope="competition", reviewer="alice", rule="EXISTING RULE.")

    def raising_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=raising_invoke):
        run_agent_lessons_approve(lesson_id=new_id, scope="competition", reviewer="bob", rule="NEW RULE.")

    row = conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [new_id]).fetchone()
    assert row[0] == "approved"  # fail-open: check couldn't run, approval still proceeds


def test_run_agent_lessons_approve_only_compares_against_cooccurring_scopes():
    """A rule for a different competition_id and different tier is never
    even sent to the conflict-check LLM call."""
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    other_id = insert_lesson_candidate(conn, "WHEN evaluating SP1...", "SP1", "competition_specific", "m1")
    new_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m2")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=lambda p: "NONE"):
        run_agent_lessons_approve(lesson_id=other_id, scope="competition", reviewer="alice", rule="SP1 RULE.")

    calls = []

    def spy_invoke(prompt: str) -> str:
        calls.append(prompt)
        return "NONE"

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)), \
         patch("main._build_llm_invoke", return_value=spy_invoke):
        run_agent_lessons_approve(lesson_id=new_id, scope="competition", reviewer="bob", rule="E0 RULE.")

    assert calls == []  # find_conflicting_rule never invoked -- no co-occurring existing rules


def test_run_agent_lessons_reject_sets_status_rejected():
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "WHEN evaluating E0...", "E0", "competition_specific", "m1")

    with patch("src.utils.db_manager.DuckDBManager", return_value=_fake_db_manager(conn)):
        run_agent_lessons_reject(lesson_id=lesson_id, reviewer="bob")

    row = conn.execute("SELECT status, reviewer FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row == ("rejected", "bob")
