"""GET /api/admin/lessons -- read-only browse of agent_lessons, newest first.
No prior way to see what the daily live-lessons job has actually written to
a deployed instance short of a raw DB query. Already covered by
RequireAppTokenMiddleware like every other non-health route -- this file
covers the endpoint's own logic only."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient
from unittest.mock import patch

from app.backend.main import app
from src.agent.lessons import create_lessons_tables, insert_lesson_candidate
from src.utils.db_manager import DuckDBManager


def _db_manager_for(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "fpai_core.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _seed_lesson(manager: DuckDBManager, *, source: str = "live", status: str = "pending", **overrides) -> int:
    with manager.connection() as conn:
        create_lessons_tables(conn)
        lesson_id = insert_lesson_candidate(
            conn,
            overrides.get("lesson_text", "Draw-framing fallacy observed again."),
            overrides.get("competition_id", "E0"),
            overrides.get("tier", "competition_specific"),
            overrides.get("source_match_id", "m1"),
        )
        conn.execute("UPDATE agent_lessons SET source = ?, status = ? WHERE id = ?", [source, status, lesson_id])
    return lesson_id


def test_list_lessons_empty_db_returns_empty_list(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons")
    assert response.status_code == 200
    assert response.json() == {"lessons": []}


def test_list_lessons_returns_newest_first(tmp_path):
    manager = _db_manager_for(tmp_path)
    first_id = _seed_lesson(manager, source_match_id="m1")
    with manager.connection() as conn:
        conn.execute("UPDATE agent_lessons SET created_at = TIMESTAMP '2026-08-01 00:00:00' WHERE id = ?", [first_id])
    second_id = _seed_lesson(manager, source_match_id="m2")
    with manager.connection() as conn:
        conn.execute("UPDATE agent_lessons SET created_at = TIMESTAMP '2026-08-29 00:00:00' WHERE id = ?", [second_id])

    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons")

    ids = [row["id"] for row in response.json()["lessons"]]
    assert ids == [second_id, first_id]


def test_list_lessons_filters_by_status(tmp_path):
    manager = _db_manager_for(tmp_path)
    _seed_lesson(manager, status="pending", source_match_id="m1")
    _seed_lesson(manager, status="rejected", source_match_id="m2")

    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons", params={"status": "rejected"})

    lessons = response.json()["lessons"]
    assert len(lessons) == 1
    assert lessons[0]["status"] == "rejected"


def test_list_lessons_filters_by_source(tmp_path):
    manager = _db_manager_for(tmp_path)
    _seed_lesson(manager, source="live", source_match_id="m1")
    _seed_lesson(manager, source="train", source_match_id="m2")

    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons", params={"source": "train"})

    lessons = response.json()["lessons"]
    assert len(lessons) == 1
    assert lessons[0]["source"] == "train"


def test_list_lessons_respects_limit(tmp_path):
    manager = _db_manager_for(tmp_path)
    for i in range(3):
        _seed_lesson(manager, source_match_id=f"m{i}")

    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons", params={"limit": 2})

    assert len(response.json()["lessons"]) == 2


def test_list_lessons_rejects_invalid_status(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons", params={"status": "bogus"})
    assert response.status_code == 422


def test_list_lessons_creates_table_if_missing(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons")
    assert response.status_code == 200
