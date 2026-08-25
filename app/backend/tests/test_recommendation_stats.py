"""W168: hit-rate breakdown + Kelly ROI simulation over resolved
recommendation_outcomes."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.recommendation_outcomes import RecommendationOutcome
from app.backend.recommendation_stats import compute_recommendation_stats


def _outcome(
    match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
    recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
) -> RecommendationOutcome:
    return RecommendationOutcome(
        id=1, match_id=match_id, date=date, competition=competition, market=market, selection=selection,
        recommendation_type=recommendation_type, confidence=confidence, odds=odds, value_edge=value_edge,
        correct=correct, generated_at="2026-08-22T10:00:00+00:00", resolved_at="2026-08-23T00:00:00+00:00",
    )


def test_empty_outcomes_returns_zeroed_stats():
    stats = compute_recommendation_stats([])
    assert stats["overall"]["sample_size"] == 0
    assert stats["overall"]["hit_rate"] == 0.0
    assert stats["kelly_roi_simulation"]["bets_placed"] == 0


def test_overall_hit_rate_across_mixed_outcomes():
    outcomes = [_outcome(correct=True), _outcome(match_id="m2", correct=False)]
    stats = compute_recommendation_stats(outcomes)
    assert stats["overall"]["sample_size"] == 2
    assert stats["overall"]["correct"] == 1
    assert stats["overall"]["hit_rate"] == 0.5


def test_breakdown_by_market_and_competition_and_confidence():
    outcomes = [
        _outcome(market="result_3way", competition="E0", confidence="high", correct=True),
        _outcome(match_id="m2", market="btts", competition="SP1", confidence="low", correct=False),
    ]
    stats = compute_recommendation_stats(outcomes)
    assert stats["by_market"]["result_3way"]["sample_size"] == 1
    assert stats["by_market"]["btts"]["sample_size"] == 1
    assert stats["by_competition"]["E0"]["hit_rate"] == 1.0
    assert stats["by_competition"]["SP1"]["hit_rate"] == 0.0
    assert stats["by_confidence"]["high"]["correct"] == 1
    assert stats["by_confidence"]["low"]["correct"] == 0


def test_kelly_roi_simulation_only_includes_direct_bet_picks():
    # A conditional pick was never actually staked -- same convention
    # src/agent/staking.py's own simulators already use.
    outcomes = [
        _outcome(recommendation_type="direct_bet", odds=3.0, value_edge=0.10, correct=True),
        _outcome(match_id="m2", recommendation_type="conditional", odds=1.6, value_edge=-0.02, correct=False),
    ]
    stats = compute_recommendation_stats(outcomes)
    assert stats["kelly_roi_simulation"]["bets_placed"] == 1


def test_kelly_roi_simulation_skips_null_odds():
    outcomes = [_outcome(recommendation_type="direct_bet", odds=None, correct=True)]
    stats = compute_recommendation_stats(outcomes)
    assert stats["kelly_roi_simulation"]["bets_placed"] == 0
