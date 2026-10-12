"""W213: ResultsCache -- persistent (competition_code, date)-keyed cache of
FINISHED match results, wrapped by FootballDataClient.get_results() (see
test_football_data_client.py for that integration)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
import app.backend.results_cache as results_cache_module
from app.backend.results_cache import ResultsCache


def _match(match_id: str) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date="2026-08-22T15:00:00Z", status="FINISHED",
        home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
    )


def test_get_is_a_miss_before_anything_is_stored(tmp_path: Path) -> None:
    cache = ResultsCache(db_path=tmp_path / "results.db")
    assert cache.get("PL", "2026-08-22") is None


def test_store_then_get_returns_the_same_matches(tmp_path: Path) -> None:
    cache = ResultsCache(db_path=tmp_path / "results.db")
    cache.store("PL", "2026-08-22", [_match("m1")])

    assert cache.get("PL", "2026-08-22") == [_match("m1")]


def test_a_genuine_zero_results_day_is_still_a_cache_hit(tmp_path: Path) -> None:
    """A rest day/midweek gap must be distinguishable from 'never queried' --
    otherwise it would be re-fetched live forever."""
    cache = ResultsCache(db_path=tmp_path / "results.db")
    cache.store("PL", "2026-08-23", [])

    assert cache.get("PL", "2026-08-23") == []


def test_different_competition_codes_are_cached_independently(tmp_path: Path) -> None:
    cache = ResultsCache(db_path=tmp_path / "results.db")
    cache.store("PL", "2026-08-22", [_match("m1")])

    assert cache.get("PD", "2026-08-22") is None


def test_dates_before_supported_since_are_never_cached(tmp_path: Path) -> None:
    cache = ResultsCache(db_path=tmp_path / "results.db")
    cache.store("PL", "2026-07-01", [_match("m1")])

    assert cache.get("PL", "2026-07-01") is None


def test_an_empty_result_fetched_too_soon_after_the_match_day_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found live, 2026-10-11: an empty result got cached forever before
    football-data.org had actually posted that day's results -- the
    original "FINISHED is immutable once set" premise only holds once the
    day's results are actually in, not just whenever someone happened to
    first ask. Fetched only 1 day after the match day (inside
    EMPTY_RESULT_TRUST_DELAY_DAYS), this empty answer isn't trusted yet --
    a repeat get() is a miss, forcing the caller to re-fetch live."""
    monkeypatch.setattr(results_cache_module, "_utc_now", lambda: datetime(2026, 8, 24, tzinfo=timezone.utc))
    cache = ResultsCache(db_path=tmp_path / "results.db")
    cache.store("PL", "2026-08-23", [])

    assert cache.get("PL", "2026-08-23") is None


def test_an_empty_result_fetched_safely_after_the_match_day_is_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same scenario, but fetched 2 days after the match day -- old enough
    to trust as a genuine rest day, matching the pre-existing cache-forever
    behavior for the case it was actually built for."""
    monkeypatch.setattr(results_cache_module, "_utc_now", lambda: datetime(2026, 8, 25, tzinfo=timezone.utc))
    cache = ResultsCache(db_path=tmp_path / "results.db")
    cache.store("PL", "2026-08-23", [])

    assert cache.get("PL", "2026-08-23") == []


def test_a_non_empty_result_is_trusted_even_fetched_the_same_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real FINISHED match is unambiguous proof the day is done -- unlike
    an empty answer, it needs no trust delay at all."""
    monkeypatch.setattr(results_cache_module, "_utc_now", lambda: datetime(2026, 8, 23, tzinfo=timezone.utc))
    cache = ResultsCache(db_path=tmp_path / "results.db")
    cache.store("PL", "2026-08-23", [_match("m1")])

    assert cache.get("PL", "2026-08-23") == [_match("m1")]


def test_re_storing_after_a_distrusted_empty_miss_updates_fetched_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The self-healing path end to end: a too-soon empty result is a
    miss, the caller re-fetches and stores again -- once that second
    store happens late enough, it's trusted without any manual purge of
    the original bad row."""
    cache = ResultsCache(db_path=tmp_path / "results.db")
    monkeypatch.setattr(results_cache_module, "_utc_now", lambda: datetime(2026, 8, 24, tzinfo=timezone.utc))
    cache.store("PL", "2026-08-23", [])
    assert cache.get("PL", "2026-08-23") is None

    monkeypatch.setattr(results_cache_module, "_utc_now", lambda: datetime(2026, 8, 26, tzinfo=timezone.utc))
    cache.store("PL", "2026-08-23", [_match("m1")])

    assert cache.get("PL", "2026-08-23") == [_match("m1")]


def test_a_second_cache_instance_over_the_same_db_path_sees_prior_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    ResultsCache(db_path=db_path).store("PL", "2026-08-22", [_match("m1")])

    assert ResultsCache(db_path=db_path).get("PL", "2026-08-22") == [_match("m1")]
