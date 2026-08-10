"""W100: POST /api/admin/restore-database -- lets an already-deployed
instance pull a real database file from a URL (e.g. a GitHub Release asset)
onto its own persistent volume, since there's no reliable way to push a
large local file directly onto a hosting platform's remote volume. Already
covered by RequireAppTokenMiddleware (app_token_middleware tests) like
every other non-health route -- this file covers the endpoint's own logic
only: URL scheme validation, the fixed target allowlist, and that the
downloaded bytes actually land at the right path."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.main import app


def _fake_requests_get(content_chunks: list[bytes]):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = iter(content_chunks)
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    return MagicMock(return_value=mock_response)


def test_rejects_a_non_https_source_url():
    with TestClient(app) as client:
        response = client.post(
            "/api/admin/restore-database",
            params={"target": "fpai_core", "source_url": "http://example.com/db"},
        )
    assert response.status_code == 400


def test_downloads_and_writes_the_recommendation_cache_target(tmp_path):
    dest = tmp_path / "nested" / "recommendation_cache.db"
    with patch("app.backend.main.RECOMMENDATION_CACHE_DB_PATH", dest), \
         patch("app.backend.main.requests.get", _fake_requests_get([b"hello ", b"world"])):
        with TestClient(app) as client:
            response = client.post(
                "/api/admin/restore-database",
                params={"target": "recommendation_cache", "source_url": "https://example.com/db.sqlite"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["bytes_written"] == len(b"hello world")
    assert dest.read_bytes() == b"hello world"


def test_downloads_and_writes_the_fpai_core_target(tmp_path):
    dest = tmp_path / "fpai_core.db"
    fake_manager = MagicMock()
    fake_manager.db_path = dest
    with patch("app.backend.main.DuckDBManager", return_value=fake_manager), \
         patch("app.backend.main.requests.get", _fake_requests_get([b"duckdb-bytes"])):
        with TestClient(app) as client:
            response = client.post(
                "/api/admin/restore-database",
                params={"target": "fpai_core", "source_url": "https://example.com/fpai_core.db"},
            )

    assert response.status_code == 200
    assert dest.read_bytes() == b"duckdb-bytes"


def test_rejects_a_target_outside_the_fixed_allowlist():
    with TestClient(app) as client:
        response = client.post(
            "/api/admin/restore-database",
            params={"target": "not_a_real_target", "source_url": "https://example.com/db"},
        )
    assert response.status_code == 422  # FastAPI's own Literal validation
