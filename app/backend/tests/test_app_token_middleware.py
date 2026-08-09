"""W97: gates every request behind a shared-secret header once the app is
reachable from the public internet -- see RequireAppTokenMiddleware's own
docstring (app/backend/main.py) for the CORS-ordering rationale. Off by
default (APP_ACCESS_TOKEN unset), matching every other opt-in-only
production-readiness flag in this app (ENABLE_SCHEDULER)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.main import _parse_cors_origins, app


def test_no_token_configured_leaves_every_request_unaffected(monkeypatch):
    """Default (local dev, every existing test): APP_ACCESS_TOKEN unset --
    behavior must be byte-identical to before this middleware existed."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200


def test_a_protected_request_with_no_header_is_rejected_once_a_token_is_configured(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_TOKEN", "secret-123")
    with TestClient(app) as client:
        response = client.get("/api/status")
    assert response.status_code == 401


def test_a_protected_request_with_the_wrong_header_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_TOKEN", "secret-123")
    with TestClient(app) as client:
        response = client.get("/api/status", headers={"X-App-Token": "wrong-guess"})
    assert response.status_code == 401


def test_a_protected_request_with_the_correct_header_succeeds(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_TOKEN", "secret-123")
    with TestClient(app) as client:
        response = client.get("/api/status", headers={"X-App-Token": "secret-123"})
    assert response.status_code != 401


def test_health_is_exempt_even_when_a_token_is_configured(monkeypatch):
    """A hosting platform's own health check never sends this header --
    must not be mistaken for a deploy failure."""
    monkeypatch.setenv("APP_ACCESS_TOKEN", "secret-123")
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200


def test_a_401_response_still_carries_cors_headers(monkeypatch):
    """CORSMiddleware must be outermost (added after this middleware, see
    RequireAppTokenMiddleware's docstring) -- otherwise a browser would
    report a confusing header-less CORS error instead of the real 401.
    Uses whatever origin the already-imported app was actually configured
    with (CORS_ALLOWED_ORIGINS is read once at import time, like any other
    env-driven app config -- see _parse_cors_origins's own tests below for
    coverage of the parsing itself, which doesn't require reimporting the
    app to exercise)."""
    monkeypatch.setenv("APP_ACCESS_TOKEN", "secret-123")
    configured_origin = app.user_middleware[0].kwargs["allow_origins"][0]
    with TestClient(app) as client:
        response = client.get("/api/status", headers={"Origin": configured_origin})
    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") == configured_origin


def test_parse_cors_origins_splits_trims_and_drops_blanks():
    assert _parse_cors_origins("https://a.com, https://b.com") == ["https://a.com", "https://b.com"]


def test_parse_cors_origins_drops_a_trailing_empty_entry():
    assert _parse_cors_origins("https://a.com,") == ["https://a.com"]


def test_parse_cors_origins_single_value_no_comma():
    assert _parse_cors_origins("http://localhost:3000") == ["http://localhost:3000"]
