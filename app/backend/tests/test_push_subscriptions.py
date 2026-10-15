from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.push_subscriptions import PushSubscriptionStore


def test_add_creates_a_new_subscription(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    sub = store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key1", auth="auth1")
    assert sub.user_id == 1
    assert sub.endpoint == "https://push.example/abc"


def test_add_is_idempotent_for_the_same_endpoint(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key1", auth="auth1")
    store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key2", auth="auth2")
    subs = store.list_for_user(1)
    assert len(subs) == 1
    assert subs[0].p256dh == "key2"


def test_remove_deletes_the_subscription(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key1", auth="auth1")
    store.remove("https://push.example/abc")
    assert store.list_for_user(1) == []


def test_get_by_endpoint_returns_the_subscription(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/abc", p256dh="key1", auth="auth1")
    sub = store.get_by_endpoint("https://push.example/abc")
    assert sub is not None
    assert sub.user_id == 1
    assert sub.endpoint == "https://push.example/abc"


def test_get_by_endpoint_returns_none_when_not_found(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    assert store.get_by_endpoint("https://push.example/missing") is None


def test_list_subscribed_user_ids_returns_distinct_users(tmp_path: Path):
    store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")
    store.add(user_id=1, endpoint="https://push.example/b", p256dh="k", auth="a")
    store.add(user_id=2, endpoint="https://push.example/c", p256dh="k", auth="a")
    assert sorted(store.list_subscribed_user_ids()) == [1, 2]
