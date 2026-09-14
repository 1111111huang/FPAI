"""W213: ResultsCache -- persistent (competition_code, date)-keyed cache of
FINISHED match results, wrapped by FootballDataClient.get_results() (see
test_football_data_client.py for that integration)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
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


def test_a_second_cache_instance_over_the_same_db_path_sees_prior_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    ResultsCache(db_path=db_path).store("PL", "2026-08-22", [_match("m1")])

    assert ResultsCache(db_path=db_path).get("PL", "2026-08-22") == [_match("m1")]
