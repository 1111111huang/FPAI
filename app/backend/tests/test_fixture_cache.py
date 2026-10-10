from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.fixture_cache import (
    TTL_CEILING_SECONDS, TTL_DEFAULT_SECONDS, TTL_FLOOR_SECONDS, compute_ttl_seconds,
)
from app.backend.football_data_client import NormalizedMatch

_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


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
