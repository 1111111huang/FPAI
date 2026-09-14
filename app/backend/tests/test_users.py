from __future__ import annotations

from pathlib import Path

from app.backend.users import UserStore


def test_get_or_create_creates_a_new_user_on_first_sight(tmp_path: Path):
    store = UserStore(db_path=tmp_path / "users.db")
    user = store.get_or_create("jane@gmail.com")
    assert user.email == "jane@gmail.com"
    assert user.id is not None


def test_get_or_create_returns_the_same_user_on_second_call(tmp_path: Path):
    store = UserStore(db_path=tmp_path / "users.db")
    first = store.get_or_create("jane@gmail.com")
    second = store.get_or_create("jane@gmail.com")
    assert first.id == second.id


def test_get_or_create_is_case_insensitive_on_email(tmp_path: Path):
    store = UserStore(db_path=tmp_path / "users.db")
    first = store.get_or_create("Jane@Gmail.com")
    second = store.get_or_create("jane@gmail.com")
    assert first.id == second.id
