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


def test_push_unsubscribe_404s_for_someone_elses_subscription(tmp_path: Path):
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store

    user_a = _override_user(tmp_path, email="user-a@example.com")
    with TestClient(app) as client:
        client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key", "auth": "auth"},
        )

        _override_user(tmp_path, email="user-b@example.com")
        response = client.request(
            "DELETE", "/api/push/subscribe", json={"endpoint": "https://push.example/abc"}
        )

    assert response.status_code == 404
    subs = push_store.list_for_user(user_a.id)
    assert len(subs) == 1
    assert subs[0].endpoint == "https://push.example/abc"


def test_push_unsubscribe_404s_for_unknown_endpoint(tmp_path: Path):
    _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    with TestClient(app) as client:
        response = client.request(
            "DELETE", "/api/push/subscribe", json={"endpoint": "https://push.example/missing"}
        )
    assert response.status_code == 404


def test_push_subscribe_409s_when_endpoint_owned_by_a_different_user(tmp_path: Path):
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store

    _override_user(tmp_path, email="user-a@example.com")
    with TestClient(app) as client:
        client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key", "auth": "auth"},
        )

        _override_user(tmp_path, email="user-b@example.com")
        response = client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key2", "auth": "auth2"},
        )

    assert response.status_code == 409


def test_push_subscribe_renewal_by_the_same_user_still_succeeds(tmp_path: Path):
    user = _override_user(tmp_path)
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    app.dependency_overrides[get_push_store] = lambda: push_store
    with TestClient(app) as client:
        client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key", "auth": "auth"},
        )
        response = client.post(
            "/api/push/subscribe",
            json={"endpoint": "https://push.example/abc", "p256dh": "key2", "auth": "auth2"},
        )
    assert response.status_code == 204
    subs = push_store.list_for_user(user.id)
    assert len(subs) == 1
    assert subs[0].p256dh == "key2"
    assert subs[0].auth == "auth2"
