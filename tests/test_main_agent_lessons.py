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
