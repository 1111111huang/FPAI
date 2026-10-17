from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
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
from app.backend.fixture_cache_store import FixtureCacheStore
from app.backend.football_data_client import NormalizedMatch

_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # BUG-086: fixture_cache now write-through persists to a module-level
    # _store -- point it at a throwaway tmp_path file for every test in
    # this file, fully isolating these tests from the real
    # data/fixture_cache.db (same tmp_path-scoped-db convention already
    # used elsewhere in this codebase, e.g. JobRunLog/RecommendationCache
    # in other test files).
    monkeypatch.setattr(fixture_cache, "_store", FixtureCacheStore(db_path=tmp_path / "fixture_cache_test.db"))
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


def test_todays_finished_only_day_does_not_get_the_ceiling():
    # BUG-089 (found live 2026-10-17): a results_call_type fetch is FINISHED-
    # only by construction (football-data.org's own status=FINISHED filter),
    # so every match in it always looks "immutable" to this function -- the
    # ceiling branch above. Fine for a genuinely past day, wrong for today:
    # later kickoffs the same day keep finishing and need to show up, the
    # exact "immutable once set" trap FootballDataClient.get_results()'s own
    # is_today bypass already exists to avoid at its lower cache layer.
    # Confirmed live: AS Roma v Genoa (18:45 UTC kickoff) stayed missing from
    # the dashboard for hours after finishing, while earlier same-day results
    # (Bologna v Inter, 16:00 UTC) rendered fine -- the day's cache entry had
    # already been fetched (and ceiling-TTL'd) before Roma's match finished.
    matches = [_match("FINISHED", -5), _match("FINISHED", -3)]
    today = _NOW.date().isoformat()
    assert compute_ttl_seconds(matches, _NOW, day=today) == TTL_DEFAULT_SECONDS


def test_a_genuinely_past_days_finished_matches_still_get_the_ceiling():
    # The `day` param only changes behavior for *today* -- a real past day's
    # results really are immutable, same as before this function took `day`
    # at all (every other test above omits it entirely).
    matches = [_match("FINISHED", -5), _match("FINISHED", -3)]
    yesterday = (_NOW.date() - timedelta(days=1)).isoformat()
    assert compute_ttl_seconds(matches, _NOW, day=yesterday) == TTL_CEILING_SECONDS


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


def test_get_range_serves_stale_data_when_a_refetch_fails():
    """BUG-084 (found live 2026-10-17): a match that was already cached
    successfully must not vanish from the response just because a LATER
    refetch (TTL expired, upstream rate-limited) failed -- the user
    reported the dashboard's match count flapping between reloads, a day
    sometimes missing entirely and sometimes not. Serving the stale list
    instead of nothing is strictly better than the match disappearing.
    The pre-existing cold-cache-raises contract
    (test_get_range_reraises_and_cleans_up_pending_when_fetch_raises,
    above) still holds -- this only changes behavior when there's a stale
    entry to fall back on."""
    ok_fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=10)])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        first = asyncio.run(get_range("fixtures", ok_fetch, date_from="2026-08-21", date_to="2026-08-21"))
        assert len(first) == 1

        # Expire the cached entry, then make the next refetch fail.
        with fixture_cache._cache_lock:
            key = ("fixtures", "2026-08-21")
            _, matches = fixture_cache._cache[key]
            fixture_cache._cache[key] = (fixture_cache.now_monotonic() - 1, matches)

        failing_fetch = MagicMock(side_effect=RuntimeError("upstream rate-limited"))
        second = asyncio.run(get_range("fixtures", failing_fetch, date_from="2026-08-21", date_to="2026-08-21"))

    failing_fetch.assert_called_once()
    assert len(second) == 1
    assert second[0].match_id == first[0].match_id


def test_get_range_a_cold_failing_span_does_not_discard_a_sibling_spans_cached_data():
    """BUG-085 (found live 2026-10-17): a wide range (e.g. Dashboard's own
    90-day-forward window) can split into several missing spans in one
    get_range() call -- a near one this cache already has data for, a far
    one still genuinely cold. Previously, the far cold span's failure
    raised immediately and aborted the whole function, discarding the
    near span's perfectly good data along with it (confirmed live: a
    league with today's matches already cached came back completely
    empty on a fresh page load, because the rest of the 90-day window's
    own fetch happened to fail). The near span's stale data must survive
    even when a *different*, unrelated span in the same call hard-fails."""
    # 2026-08-21: cached once, then expired (BUG-084's "stale fallback"
    # case). 2026-08-22: cached and still fresh (untouched by the missing-
    # day computation at all). 2026-08-23: never cached (genuinely cold).
    warm_then_stale_fetch = MagicMock(return_value=[_future_match("2026-08-21", hour=9)])
    fresh_fetch = MagicMock(return_value=[_future_match("2026-08-22", hour=9)])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", warm_then_stale_fetch, date_from="2026-08-21", date_to="2026-08-21"))
        asyncio.run(get_range("fixtures", fresh_fetch, date_from="2026-08-22", date_to="2026-08-22"))

        with fixture_cache._cache_lock:
            key = ("fixtures", "2026-08-21")
            _, matches = fixture_cache._cache[key]
            fixture_cache._cache[key] = (fixture_cache.now_monotonic() - 1, matches)

        failing_fetch = MagicMock(side_effect=RuntimeError("upstream rate-limited"))
        result = asyncio.run(get_range("fixtures", failing_fetch, date_from="2026-08-21", date_to="2026-08-23"))

    # Both the 08-21 span and the 08-23 span hit the same failing fetch --
    # confirms this is genuinely a two-span call, not a fluke of caching.
    assert failing_fetch.call_count == 2
    match_ids = {m.match_id for m in result}
    assert match_ids == {"m-2026-08-21-9", "m-2026-08-22-9"}


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


def test_store_day_persists_to_the_store():
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        fixture_cache.store_day("fixtures", "2026-08-21", [_future_match("2026-08-21")])

    persisted = fixture_cache._store.load_all()
    assert persisted[("fixtures", "2026-08-21")][0].match_id == "m-2026-08-21-15"


def test_load_from_disk_seeds_the_in_memory_cache_already_expired():
    fixture_cache._store.store_day("fixtures", "2026-08-21", [_future_match("2026-08-21")])

    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        fixture_cache.load_from_disk()

    # Already expired -- _missing_days() wants to refetch it on next use...
    assert fixture_cache._missing_days("fixtures", ["2026-08-21"]) == ["2026-08-21"]
    # ...but _read_days() (BUG-084) still serves it immediately regardless.
    assert fixture_cache._read_days("fixtures", ["2026-08-21"])[0].match_id == "m-2026-08-21-15"


def test_load_from_disk_prunes_entries_older_than_retention_before_loading():
    old_day = (date.today() - timedelta(days=fixture_cache.RETENTION_DAYS + 1)).isoformat()
    fixture_cache._store.store_day("fixtures", old_day, [_future_match(old_day)])

    fixture_cache.load_from_disk()

    assert fixture_cache._read_days("fixtures", [old_day]) == []
    assert fixture_cache._store.load_all() == {}


def test_a_real_get_range_call_also_persists_to_the_store():
    """End-to-end: the existing get_range() path (not a direct store_day()
    call) must also reach the persisted store -- proves the write-through
    is wired into the actual fetch flow, not just the low-level function."""
    fetch = MagicMock(return_value=[_future_match("2026-08-21")])
    with patch("app.backend.fixture_cache.wall_clock_now", return_value=_NOW):
        asyncio.run(get_range("fixtures", fetch, date_from="2026-08-21", date_to="2026-08-21"))

    persisted = fixture_cache._store.load_all()
    assert persisted[("fixtures", "2026-08-21")][0].match_id == "m-2026-08-21-15"
