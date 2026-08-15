"""POST /api/admin/sync-lessons -- lets an approved lesson reviewed against
a local/dev database (agent_lessons is NOT part of git, and is NOT safely
syncable via /api/admin/restore-database's full-file overwrite, which would
also clobber production's own fresher raw_matches) reach a deployed
instance's live serving path without touching anything else in the DB.
Already covered by RequireAppTokenMiddleware like every other non-health
route -- this file covers the endpoint's own logic only."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import yaml

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.main import app
from src.utils.db_manager import DuckDBManager


def _payload(**overrides) -> dict:
    base = dict(
        competition_id="SP1", tier="competition_specific", scope="competition",
        rule_text="Do not default to no_bet purely on high entropy when the edge is positive.",
    )
    base.update(overrides)
    return base


def _db_manager_for(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "fpai_core.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def test_sync_lessons_inserts_a_new_approved_lesson(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.post("/api/admin/sync-lessons", json=[_payload()])

    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "skipped_duplicates": 0}

    with manager.connection(read_only=True) as conn:
        rows = conn.execute(
            "SELECT status, scope, competition_id, tier, rule_text FROM agent_lessons"
        ).fetchall()
    assert rows == [("approved", "competition", "SP1", "competition_specific", _payload()["rule_text"])]


def test_sync_lessons_is_idempotent_skips_exact_duplicate(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            first = client.post("/api/admin/sync-lessons", json=[_payload()])
            second = client.post("/api/admin/sync-lessons", json=[_payload()])

    assert first.json() == {"inserted": 1, "skipped_duplicates": 0}
    assert second.json() == {"inserted": 0, "skipped_duplicates": 1}

    with manager.connection(read_only=True) as conn:
        count = conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0]
    assert count == 1


def test_sync_lessons_creates_table_if_missing(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.post("/api/admin/sync-lessons", json=[_payload()])
    assert response.status_code == 200


def test_sync_lessons_accepts_multiple_and_reports_mixed_result(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            client.post("/api/admin/sync-lessons", json=[_payload()])
            response = client.post(
                "/api/admin/sync-lessons",
                json=[_payload(), _payload(rule_text="A genuinely different rule.")],
            )

    assert response.json() == {"inserted": 1, "skipped_duplicates": 1}


def test_sync_lessons_rejects_invalid_scope(tmp_path):
    manager = _db_manager_for(tmp_path)
    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.post("/api/admin/sync-lessons", json=[_payload(scope="bogus")])
    assert response.status_code == 422
