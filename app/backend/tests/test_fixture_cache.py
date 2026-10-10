from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend import fixture_cache
from app.backend.fixture_cache import (
    TTL_CEILING_SECONDS, TTL_DEFAULT_SECONDS, TTL_FLOOR_SECONDS, clear, compute_ttl_seconds,
    fixtures_call_type, force_refresh_range, get_range, results_call_type,
)
from app.backend.football_data_client import NormalizedMatch

_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear()
    yield
    clear()


def _match(status: str, hours_from_now: float) -> NormalizedMatch:
    kickoff = _NOW + timedelta(hours=hours_from_now)
    return NormalizedMatch(
        match_id="m1", utc_date=kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"), status=status,
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )


def test_empty_day_gets_the_plain_default():
    assert compute_ttl_seconds([], _NOW) == TTL_DEFAULT_SECONDS


def test_a_live_match_gets_the_floor():
    assert compute_ttl_seconds([_match("LIVE", -0.5)], _NOW) == TTL_FLOOR_SECONDS


def test_in_play_and_paused_also_get_the_floor():
    assert compute_ttl_seconds([_match("IN_PLAY", -0.2)], _NOW) == TTL_FLOOR_SECONDS
    assert compute_ttl_seconds([_match("PAUSED", -0.2)], _NOW) == TTL_FLOOR_SECONDS


def test_a_live_match_wins_over_a_coexisting_finished_match():
    matches = [_match("FINISHED", -5), _match("LIVE", -0.2)]
    assert compute_ttl_seconds(matches, _NOW) == TTL_FLOOR_SECONDS


def test_only_finished_matches_get_the_ceiling():
    matches = [_match("FINISHED", -5), _match("FINISHED", -3)]
    assert compute_ttl_seconds(matches, _NOW) == TTL_CEILING_SECONDS


def test_scheduled_match_expires_at_its_own_kickoff():
    # 2 hours out -> 7200s, comfortably inside [floor, ceiling].
    assert compute_ttl_seconds([_match("SCHEDULED", 2)], _NOW) == 7200.0


def test_scheduled_match_kickoff_ttl_is_clamped_to_the_floor():
    # 10 seconds out would compute to 10s -- clamped up to the 60s floor so
    # the cache doesn't thrash on a match seconds from kickoff.
    assert compute_ttl_seconds([_match("SCHEDULED", 10 / 3600)], _NOW) == TTL_FLOOR_SECONDS


def test_scheduled_match_kickoff_ttl_is_clamped_to_the_ceiling():
    # 30 days out would compute to a huge number -- clamped down to the
    # 4h ceiling so the reconciliation job's own cadence is the real backstop.
    assert compute_ttl_seconds([_match("SCHEDULED", 30 * 24)], _NOW) == TTL_CEILING_SECONDS


def test_earliest_upcoming_kickoff_wins_among_several_scheduled_matches():
    matches = [_match("SCHEDULED", 3), _match("SCHEDULED", 1), _match("SCHEDULED", 5)]
    assert compute_ttl_seconds(matches, _NOW) == 3600.0


def test_a_scheduled_match_whose_kickoff_already_passed_is_ignored_like_finished():
    # Found live risk: a SCHEDULED fixture snapshot can be stale by the time
    # this runs (mirrors has_kicked_off's own docstring reasoning) -- a
    # "kickoff" in the past must not count as "upcoming" and drag the TTL
    # down to near-zero.
    matches = [_match("SCHEDULED", -1), _match("FINISHED", -2)]
    assert compute_ttl_seconds(matches, _NOW) == TTL_CEILING_SECONDS


def test_an_unparseable_utc_date_is_ignored_rather_than_raising():
    bad = NormalizedMatch(
        match_id="m1", utc_date="not-a-date", status="SCHEDULED",
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )
    assert compute_ttl_seconds([bad], _NOW) == TTL_CEILING_SECONDS


def _future_match(day: str, hour: int = 15) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=f"m-{day}-{hour}", utc_date=f"{day}T{hour:02d}:00:00Z", status="SCHEDULED",
        home_team="A", away_team="B", home_goals=None, away_goals=None,
    )


def test_get_range_fetches_once_on_a_cold_cache():
    fetch = MagicMock(return_value=[_future_match("2026-08-21")])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        result = asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
    assert len(result) == 1
    fetch.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-21")


def test_get_range_second_call_within_ttl_does_not_refetch():
    fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=20)])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
    fetch.assert_called_once()


def test_get_range_shares_overlapping_days_across_two_different_outer_ranges():
    """The original motivating bug: Dashboard's today..+90 and Match
    Explorer's today-30..+90 must share every day they have in common, not
    just the lucky case where their ranges split identically."""
    fetch = MagicMock(return_value=[_future_match("2026-08-22")])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-22", date_to="2026-08-24"))
        fetch.reset_mock()
        fetch.return_value = []
        # A second, differently-shaped range whose every day is already
        # covered by the first one's cached days -- so no new call at all.
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-23", date_to="2026-08-24"))
    fetch.assert_not_called()


def test_get_range_only_fetches_the_missing_days_as_one_contiguous_call():
    fetch = MagicMock(return_value=[])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-22", date_to="2026-08-22"))
        fetch.reset_mock()
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-20", date_to="2026-08-24"))
    # 08-22 already cached -- the two missing spans are 08-20..08-21 and 08-23..08-24.
    assert fetch.call_count == 2
    fetch.assert_any_call(date_from="2026-08-20", date_to="2026-08-21")
    fetch.assert_any_call(date_from="2026-08-23", date_to="2026-08-24")


def test_get_range_passes_through_extra_kwargs_like_competition_code():
    fetch = MagicMock(return_value=[])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range(
            "fixtures_sp1", fetch, date_from="2026-08-21", date_to="2026-08-21", competition_code="PD",
        ))
    fetch.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-21", competition_code="PD")


def test_get_range_unbounded_range_bypasses_the_cache_entirely():
    fetch = MagicMock(return_value=[])
    asyncio.run(get_range("fixtures", fetch, date_from=None, date_to=None))
    asyncio.run(get_range("fixtures", fetch, date_from=None, date_to=None))
    assert fetch.call_count == 2


def test_get_range_concurrent_identical_calls_dedupe_to_one_fetch():
    entered = threading.Event()
    release = threading.Event()

    call_count = {"n": 0}

    def counting_fetch(date_from=None, date_to=None):
        call_count["n"] += 1
        entered.set()
        assert release.wait(timeout=5)
        return []

    async def _runner():
        task = asyncio.ensure_future(asyncio.gather(
            get_range("fixtures", counting_fetch, date_from="2026-08-21", date_to="2026-08-21"),
            get_range("fixtures", counting_fetch, date_from="2026-08-21", date_to="2026-08-21"),
        ))
        await asyncio.sleep(0.05)
        release.set()
        await task

    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(_runner())
    assert call_count["n"] == 1


def test_get_range_reraises_and_cleans_up_pending_when_fetch_raises():
    def failing_fetch(date_from=None, date_to=None):
        raise RuntimeError("upstream down")

    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        with pytest.raises(RuntimeError):
            asyncio.run(get_range("fixtures", failing_fetch, date_from="2026-08-21", date_to="2026-08-21"))

        # _pending must not be left with a stuck entry -- a second call
        # must retry (not hang waiting on a dead task).
        assert fixture_cache._pending == {}

        ok_fetch = MagicMock(return_value=[])
        asyncio.run(get_range("fixtures", ok_fetch, date_from="2026-08-21", date_to="2026-08-21"))
        ok_fetch.assert_called_once()


def test_force_refresh_range_always_fetches_even_if_cached():
    fetch = MagicMock(return_value=[])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
        fetch.reset_mock()
        asyncio.run(force_refresh_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))
    fetch.assert_called_once_with(date_from="2026-08-21", date_to="2026-08-21")


def test_force_refresh_range_overwrites_the_cached_day_and_warms_it_for_get_range():
    stale_fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=10)])
    fresh_fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=18)])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", stale_fetch, date_from="2026-08-21", date_to="2026-08-21"))
        asyncio.run(force_refresh_range("fixtures", fresh_fetch, date_from="2026-08-21", date_to="2026-08-21"))
        # A plain get_range() afterward must see the force-refreshed data,
        # not re-fetch (the warming half of W249).
        never_called = MagicMock()
        result = asyncio.run(get_range("fixtures", never_called, date_from="2026-08-21", date_to="2026-08-21"))
    never_called.assert_not_called()
    assert result[0].match_id == "m-2026-08-21-18"


def test_results_call_type_and_fixtures_call_type_naming():
    assert results_call_type("E0") == "results"
    assert fixtures_call_type("E0") == "fixtures"
    assert results_call_type("SP1") == "results_sp1"
    assert fixtures_call_type("SWE") == "fixtures_swe"
