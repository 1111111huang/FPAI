from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend.auth_deps import get_current_user_email
from app.backend.main import app, get_push_store, get_user_store
from app.backend.push_subscriptions import PushSubscriptionStore
from app.backend.users import UserStore


def _override_user(tmp_path: Path, email: str = "test-user@example.com"):
    store = UserStore(db_path=tmp_path / "users.db")
    app.dependency_overrides[get_current_user_email] = lambda: email
    app.dependency_overrides[get_user_store] = lambda: store
    return store.get_or_create(email)


def test_push_subscribe_stores_the_subscription(tmp_path: Path):
    user = _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    with TestClient(app) as client:
        response = client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key", "auth": "auth"},
        )
    assert response.status_code == 204
    subs = push_store.list_for_user(user.id)
    assert len(subs) == 1
    assert subs[0].endpoint == "https://push.example/abc"


def test_push_unsubscribe_removes_the_subscription(tmp_path: Path):
    _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    with TestClient(app) as client:
        client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key", "auth": "auth"},
        )
        response = client.request(
            "DELETE", "/api/push/subscribe", json={"endpoint": "https://push.example/abc"}
        )
    assert response.status_code == 204
    assert push_store.list_subscribed_user_ids() == []
