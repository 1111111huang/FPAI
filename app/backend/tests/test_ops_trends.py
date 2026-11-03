"""Tests for app/backend/ops_trends.py's week-bucketing helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

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
