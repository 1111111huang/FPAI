from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.fixture_cache_store import FixtureCacheStore
from app.backend.football_data_client import NormalizedMatch


def _match(match_id: str, utc_date: str, status: str = "SCHEDULED") -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date=utc_date, status=status,
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )


def test_store_then_load_round_trips_matches(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [_match("m1", "2026-08-21T15:00:00Z")])

    loaded = store.load_all()

    assert loaded[("fixtures", "2026-08-21")][0].match_id == "m1"


def test_an_empty_day_round_trips_as_an_empty_list_not_a_missing_key(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [])

    loaded = store.load_all()

    assert ("fixtures", "2026-08-21") in loaded
    assert loaded[("fixtures", "2026-08-21")] == []


def test_a_day_never_stored_has_no_key_at_all(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [])

    loaded = store.load_all()

    assert ("fixtures", "2026-08-22") not in loaded


def test_storing_the_same_day_twice_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [_match("m1", "2026-08-21T15:00:00Z")])
    store.store_day("fixtures", "2026-08-21", [_match("m2", "2026-08-21T18:00:00Z")])

    loaded = store.load_all()

    match_ids = {m.match_id for m in loaded[("fixtures", "2026-08-21")]}
    assert match_ids == {"m2"}


def test_prune_older_than_removes_old_entries_and_keeps_newer_ones(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-10", [_match("old", "2026-08-10T15:00:00Z")])
    store.store_day("fixtures", "2026-08-25", [_match("new", "2026-08-25T15:00:00Z")])

    store.prune_older_than("2026-08-20")
    loaded = store.load_all()

    assert ("fixtures", "2026-08-10") not in loaded
    assert ("fixtures", "2026-08-25") in loaded


def test_different_call_types_on_the_same_day_are_independent(tmp_path: Path) -> None:
    store = FixtureCacheStore(db_path=tmp_path / "test.db")
    store.store_day("fixtures", "2026-08-21", [_match("m1", "2026-08-21T15:00:00Z")])
    store.store_day("results_sp1", "2026-08-21", [_match("m2", "2026-08-21T15:00:00Z")])

    loaded = store.load_all()

    assert {m.match_id for m in loaded[("fixtures", "2026-08-21")]} == {"m1"}
    assert {m.match_id for m in loaded[("results_sp1", "2026-08-21")]} == {"m2"}
