"""Tests for app/backend/ops_trends.py's week-bucketing helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.ops_trends import recent_week_starts, thursday_week_start


def test_thursday_maps_to_itself():
    assert thursday_week_start("2026-10-08") == "2026-10-08"  # a Thursday


def test_friday_maps_to_the_preceding_thursday():
    assert thursday_week_start("2026-10-09") == "2026-10-08"


def test_wednesday_maps_to_the_thursday_six_days_earlier():
    assert thursday_week_start("2026-10-14") == "2026-10-08"  # the Wednesday ending that week


def test_recent_week_starts_returns_oldest_to_newest():
    # 2026-11-03 is a Tuesday; its week started Thursday 2026-10-29.
    starts = recent_week_starts(date(2026, 11, 3), weeks=4)
    assert starts == ["2026-10-08", "2026-10-15", "2026-10-22", "2026-10-29"]


def test_recent_week_starts_length_matches_weeks_param():
    starts = recent_week_starts(date(2026, 11, 3), weeks=8)
    assert len(starts) == 8
    assert starts[-1] == "2026-10-29"  # newest bucket is always "today"'s own week


@pytest.mark.parametrize("date_str,expected", [
    ("2026-10-08", "2026-10-08"),  # Thursday
    ("2026-10-09", "2026-10-08"),  # Friday
    ("2026-10-10", "2026-10-08"),  # Saturday
    ("2026-10-11", "2026-10-08"),  # Sunday
    ("2026-10-12", "2026-10-08"),  # Monday
    ("2026-10-13", "2026-10-08"),  # Tuesday
    ("2026-10-14", "2026-10-08"),  # Wednesday
])
def test_thursday_week_start_covers_every_day_of_the_week(date_str, expected):
    assert thursday_week_start(date_str) == expected
