from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from pywebpush import WebPushException

from app.backend.push_notifications import send_to_all_subscribers, send_to_user
from app.backend.push_subscriptions import PushSubscriptionStore


def _store_with_one_subscription(tmp_path: Path, user_id: int = 1) -> PushSubscriptionStore:
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=user_id, endpoint="https://push.example/abc", p256dh="key", auth="auth")
    return store


def test_send_to_user_calls_webpush_for_each_subscription(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "test-private-key")
    store = _store_with_one_subscription(tmp_path)
    with patch("app.backend.push_notifications.webpush") as mock_webpush:
        send_to_user(store, user_id=1, title="Hi", body="Body", url="/bets")
    assert mock_webpush.call_count == 1
    _, kwargs = mock_webpush.call_args
    assert kwargs["subscription_info"]["endpoint"] == "https://push.example/abc"


def test_send_to_user_skips_silently_when_vapid_key_unset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
    store = _store_with_one_subscription(tmp_path)
    with patch("app.backend.push_notifications.webpush") as mock_webpush:
        send_to_user(store, user_id=1, title="Hi", body="Body")
    mock_webpush.assert_not_called()


def test_send_to_user_prunes_an_expired_subscription(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "test-private-key")
    store = _store_with_one_subscription(tmp_path)

    class _Response:
        status_code = 410

    with patch(
        "app.backend.push_notifications.webpush",
        side_effect=WebPushException("gone", response=_Response()),
    ):
        send_to_user(store, user_id=1, title="Hi", body="Body")
    assert store.list_for_user(1) == []


def test_send_to_all_subscribers_sends_to_every_subscribed_user(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "test-private-key")
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")
    store.add(user_id=2, endpoint="https://push.example/b", p256dh="k", auth="a")
    with patch("app.backend.push_notifications.webpush") as mock_webpush:
        send_to_all_subscribers(store, title="Hi", body="Body")
    assert mock_webpush.call_count == 2
