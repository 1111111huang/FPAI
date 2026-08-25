"""W171: top/bottom staked-bet examples enriched with match/team context
for the agent performance dashboard. The only piece of this feature that
needs RecommendationCache (DB I/O) -- recommendation_stats.py stays pure
aggregation on purpose."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.agent_performance_dashboard import compute_agent_performance_dashboard
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcome


def _outcome(match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
             recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True) -> RecommendationOutcome:
    return RecommendationOutcome(
        id=1, match_id=match_id, date=date, competition=competition, market=market, selection=selection,
        recommendation_type=recommendation_type, confidence=confidence, odds=odds, value_edge=value_edge,
        correct=correct, generated_at="2026-08-22T10:00:00+00:00", resolved_at="2026-08-23T00:00:00+00:00",
    )


_REC = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.0, "value_edge": 0.1}],
    "confidence": "medium", "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
}


def test_top_winners_enriched_with_team_names(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _REC, "scheduled")
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]

    result = compute_agent_performance_dashboard(outcomes, cache)

    assert len(result["top_winners"]) == 1
    winner = result["top_winners"][0]
    assert winner["home_team"] == "Arsenal"
    assert winner["away_team"] == "Everton"
    assert winner["payout"] > 0
    assert winner["date"] == "2026-08-22"
    assert winner["competition"] == "E0"


def test_top_losers_enriched_and_sorted_most_negative_first(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _REC, "scheduled")
    cache.record_generation("m2", "2026-08-23", "hash1", {}, {**_REC, "match": {**_REC["match"], "home": "Chelsea", "away": "Brighton"}}, "scheduled")
    outcomes = [
        _outcome(match_id="m1", date="2026-08-22", odds=3.0, value_edge=0.2, correct=False),
        _outcome(match_id="m2", date="2026-08-23", odds=2.0, value_edge=0.1, correct=False),
    ]

    result = compute_agent_performance_dashboard(outcomes, cache)

    assert len(result["top_losers"]) == 2
    # Most negative payout first (bigger stake loses more since value_edge is higher).
    assert result["top_losers"][0]["payout"] < result["top_losers"][1]["payout"]
    assert result["top_losers"][0]["home_team"] == "Arsenal"


def test_cache_miss_degrades_team_names_to_none_not_a_crash(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")  # nothing recorded
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]

    result = compute_agent_performance_dashboard(outcomes, cache)

    assert len(result["top_winners"]) == 1
    assert result["top_winners"][0]["home_team"] is None
    assert result["top_winners"][0]["away_team"] is None
    assert result["top_winners"][0]["match_id"] == "m1"


def test_respects_top_n(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    outcomes = [
        _outcome(match_id=f"m{i}", date="2026-08-22", odds=2.0, value_edge=0.1 + i * 0.01, correct=True)
        for i in range(3)
    ]

    result = compute_agent_performance_dashboard(outcomes, cache, top_n=2)

    assert len(result["top_winners"]) == 2


def test_empty_outcomes_returns_empty_top_lists(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    result = compute_agent_performance_dashboard([], cache)
    assert result["top_winners"] == []
    assert result["top_losers"] == []


def test_no_losers_when_every_staked_bet_won(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]
    result = compute_agent_performance_dashboard(outcomes, cache)
    assert result["top_losers"] == []
    assert len(result["top_winners"]) == 1


def test_result_still_includes_everything_compute_recommendation_stats_returns(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    outcomes = [_outcome(match_id="m1", odds=2.0, value_edge=0.1, correct=True)]
    result = compute_agent_performance_dashboard(outcomes, cache)
    assert "overall" in result
    assert "by_market_metrics" in result
    assert "kelly_roi_simulation" in result
    assert "staked_bets" in result
