"""Tests for app/backend/ops_trends.py's week-bucketing helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.ops_trends import (
    FeatureWeekStats,
    aggregate_feature_weeks,
    recent_week_starts,
    thursday_week_start,
)


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


def test_aggregate_feature_weeks_computes_mean_and_null_rate_per_feature_per_week():
    # Two matches in the same week: XOC_HOME appears both times (values 2.0, 4.0);
    # DEF_ANCHOR_AWAY appears once, null that time.
    entries = [
        {"week_start": "2026-10-08", "top_features": [
            {"name": "XOC_HOME", "value": 2.0, "importance": 0.5},
            {"name": "DEF_ANCHOR_AWAY", "value": None, "importance": 0.3},
        ]},
        {"week_start": "2026-10-08", "top_features": [
            {"name": "XOC_HOME", "value": 4.0, "importance": 0.4},
        ]},
    ]
    result = aggregate_feature_weeks(entries, week_starts=["2026-10-08"])
    assert result["XOC_HOME"]["2026-10-08"] == FeatureWeekStats(appearances=2, mean=3.0, null_rate=0.0)
    assert result["DEF_ANCHOR_AWAY"]["2026-10-08"] == FeatureWeekStats(appearances=1, mean=None, null_rate=1.0)


def test_flag_drifting_features_requires_minimum_sample_size_in_both_halves():
    from app.backend.ops_trends import flag_drifting_features

    # XOC_HOME: baseline mean 2.0 (2 weeks), recent mean 8.0 (2 weeks) -- a real 4x jump, enough samples.
    weekly = {
        "XOC_HOME": {
            "2026-10-08": FeatureWeekStats(appearances=5, mean=2.0, null_rate=0.0),
            "2026-10-15": FeatureWeekStats(appearances=5, mean=2.0, null_rate=0.0),
            "2026-10-22": FeatureWeekStats(appearances=5, mean=8.0, null_rate=0.0),
            "2026-10-29": FeatureWeekStats(appearances=5, mean=8.0, null_rate=0.0),
        },
        # LOW_SAMPLE: real-looking drift ratio but only 1 appearance total -- must NOT be flagged.
        "LOW_SAMPLE": {
            "2026-10-08": FeatureWeekStats(appearances=1, mean=1.0, null_rate=0.0),
            "2026-10-15": FeatureWeekStats(appearances=0, mean=None, null_rate=0.0),
            "2026-10-22": FeatureWeekStats(appearances=0, mean=None, null_rate=0.0),
            "2026-10-29": FeatureWeekStats(appearances=1, mean=9.0, null_rate=0.0),
        },
    }
    flagged = flag_drifting_features(weekly, week_starts=["2026-10-08", "2026-10-15", "2026-10-22", "2026-10-29"])
    names = {f["name"] for f in flagged}
    assert "XOC_HOME" in names
    assert "LOW_SAMPLE" not in names
