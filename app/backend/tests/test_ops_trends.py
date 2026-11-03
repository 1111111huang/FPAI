"""Tests for app/backend/ops_trends.py's week-bucketing helpers."""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.ops_trends import (
    FeatureWeekStats,
    aggregate_feature_weeks,
    build_week_records,
    recent_week_starts,
    thursday_week_start,
)
from app.backend.recommendation_outcomes import RecommendationOutcome


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


def test_flag_drifting_features_excludes_a_clearly_low_sample_feature():
    # Not a boundary test -- LOW_SAMPLE has only 1 appearance per half,
    # nowhere near MIN_APPEARANCES_PER_HALF=3. See
    # test_flag_drifting_features_minimum_appearances_boundary below for the
    # actual <3 vs >=3 edge (this test alone wouldn't catch a `<` vs `<=`
    # off-by-one in the gate).
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


def test_flag_drifting_features_minimum_appearances_boundary():
    # Real edge of MIN_APPEARANCES_PER_HALF=3: a feature with exactly 2
    # appearances per half (below the gate) must be excluded; one with
    # exactly 3 (the gate itself) must be included, given an otherwise
    # clear drift. Catches a `<` vs `<=` off-by-one that a 10-vs-1 sample
    # mismatch would never reach.
    from app.backend.ops_trends import flag_drifting_features

    weekly = {
        "BELOW_GATE": {
            "2026-10-08": FeatureWeekStats(appearances=2, mean=2.0, null_rate=0.0),
            "2026-10-15": FeatureWeekStats(appearances=2, mean=8.0, null_rate=0.0),
        },
        "AT_GATE": {
            "2026-10-08": FeatureWeekStats(appearances=3, mean=2.0, null_rate=0.0),
            "2026-10-15": FeatureWeekStats(appearances=3, mean=8.0, null_rate=0.0),
        },
    }
    flagged = flag_drifting_features(weekly, week_starts=["2026-10-08", "2026-10-15"])
    names = {f["name"] for f in flagged}
    assert "BELOW_GATE" not in names
    assert "AT_GATE" in names


def test_feature_category_prefers_the_more_specific_def_anchor_prefix():
    # DEF_ANCHOR_HOME/AWAY are real computed features (src/features/
    # lineup_features.py) that also start with "DEF_" -- a naively-ordered
    # prefix list matches "DEF_" first and misclassifies them. Mirrors the
    # same shadowing precedent src/logic/feature_groups.py's
    # resolve_feature_group_tag() already guards against.
    from app.backend.ops_trends import _feature_category

    assert _feature_category("DEF_ANCHOR_HOME") == "def_anchor"
    assert _feature_category("DEF_ANCHOR_AWAY") == "def_anchor"
    assert _feature_category("DEF_GOALS_CONCEDED_R5") == "def"


def _outcome(match_id, date_, market, selection, rec_type, odds, edge, correct, home_goals=2, away_goals=1):
    return RecommendationOutcome(
        id=0, match_id=match_id, date=date_, competition="E0", market=market, selection=selection,
        recommendation_type=rec_type, confidence="medium", odds=odds, value_edge=edge, correct=correct,
        generated_at=f"{date_}T12:00:00Z", resolved_at=f"{date_}T13:00:00Z",
        home_goals=home_goals, away_goals=away_goals,
    )


def test_one_per_match_uses_the_single_stored_outcome_directly():
    outcomes = [_outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True)]
    records = build_week_records(outcomes, {}, pick_scope="one_per_match", include_conditional=False)
    assert len(records) == 1
    assert records[0].market_results == [
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1, "correct": True},
    ]


def test_all_direct_grades_every_direct_bet_candidate_from_the_cached_recommendation():
    outcome = _outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True, home_goals=1, away_goals=0)
    cached_candidates_by_key = {
        ("m1", "2026-10-09"): [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1},
            {"market": "btts", "selection": "no", "recommendation_type": "direct_bet", "current_odds": 1.8, "value_edge": 0.05},
            {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 2.1, "value_edge": -0.1},
        ],
    }
    records = build_week_records([outcome], cached_candidates_by_key, pick_scope="all_direct", include_conditional=False)
    assert len(records) == 1
    results = {(m["market"], m["selection"]): m for m in records[0].market_results}
    assert len(results) == 2  # only the two direct_bet candidates, not the no_bet one
    assert results[("result_3way", "home")]["correct"] is True  # home won 1-0
    assert results[("btts", "no")]["correct"] is True  # 1-0 is indeed btts=no


def test_include_conditional_adds_conditional_candidates_staked_at_current_odds():
    outcome = _outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True, home_goals=1, away_goals=0)
    cached_candidates_by_key = {
        ("m1", "2026-10-09"): [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1},
            {"market": "total_goals", "selection": "under_2.5", "recommendation_type": "conditional", "current_odds": 1.9, "value_edge": 0.03},
        ],
    }
    without = build_week_records([outcome], cached_candidates_by_key, pick_scope="all_direct", include_conditional=False)
    with_cond = build_week_records([outcome], cached_candidates_by_key, pick_scope="all_direct", include_conditional=True)
    assert len(without[0].market_results) == 1
    assert len(with_cond[0].market_results) == 2
    conditional_result = next(m for m in with_cond[0].market_results if m["market"] == "total_goals")
    assert conditional_result["recommendation_type"] == "direct_bet"  # staked AS IF direct, per the user's chosen semantics
    assert conditional_result["correct"] is True  # 1-0 total=1, under_2.5 is correct


def test_one_per_match_ignores_include_conditional_the_single_outcome_passes_through_unchanged():
    outcome = _outcome("m1", "2026-10-09", "total_goals", "under_2.5", "conditional", 1.9, 0.03, True)
    without = build_week_records([outcome], {}, pick_scope="one_per_match", include_conditional=False)
    with_cond = build_week_records([outcome], {}, pick_scope="one_per_match", include_conditional=True)
    assert without[0].market_results == with_cond[0].market_results
    assert without[0].market_results[0]["recommendation_type"] == "conditional"  # never retagged in this mode


def test_all_direct_skips_matches_with_no_known_final_score():
    outcome = _outcome("m1", "2026-10-09", "result_3way", "home", "direct_bet", 2.0, 0.1, True)
    outcome = dataclasses.replace(outcome, home_goals=None, away_goals=None)  # pre-W175 row, no stored score
    records = build_week_records(
        [outcome],
        {("m1", "2026-10-09"): [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1}]},
        pick_scope="all_direct", include_conditional=False,
    )
    assert records == []
