"""W168: diagnostics aggregation over resolved recommendation_outcomes --
hit-rate breakdown by market/competition/confidence, plus a Kelly-sized ROI
simulation reusing src/agent/staking.py's own simulate_kelly_stake (A80's
kelly_fraction) via a thin BacktestRecord adapter. Mirrors bet_stats.py's
own separation from its storage class (bet_tracker.py) -- this file is pure
aggregation, no DB I/O of its own.

Denominated in UB (an abstract Unit Bet), not dollars -- starting_bankroll
is just a plain number, same as src/agent/staking.py's own bankroll
parameter always was; see docs/superpowers/specs/2026-08-25-live-recommendation-tracking-design.md."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.backend.recommendation_outcomes import RecommendationOutcome
from src.agent.backtest import BacktestRecord
from src.agent.evaluation import build_evaluation_report
from src.agent.staking import simulate_kelly_stake

DEFAULT_STARTING_BANKROLL = 1000.0


def _hit_rate(outcomes: list[RecommendationOutcome]) -> dict[str, Any]:
    correct = sum(1 for o in outcomes if o.correct)
    return {
        "sample_size": len(outcomes),
        "correct": correct,
        "hit_rate": round(correct / len(outcomes), 6) if outcomes else 0.0,
    }


def _breakdown_by(outcomes: list[RecommendationOutcome], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[RecommendationOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[getattr(outcome, key) or "unknown"].append(outcome)
    return {group_key: _hit_rate(group) for group_key, group in groups.items()}


def _to_backtest_records(outcomes: list[RecommendationOutcome]) -> list[BacktestRecord]:
    return [
        BacktestRecord(
            match_id=o.match_id, home_team="", away_team="", date=o.date, league=o.competition or "",
            recommendation={}, actual={},
            market_results=[{
                "market": o.market, "selection": o.selection, "recommendation_type": o.recommendation_type,
                "current_odds": o.odds, "value_edge": o.value_edge, "correct": o.correct,
            }],
        )
        for o in outcomes
    ]


def compute_recommendation_stats(
    outcomes: list[RecommendationOutcome], starting_bankroll: float = DEFAULT_STARTING_BANKROLL
) -> dict[str, Any]:
    records = _to_backtest_records(outcomes)
    bankroll_result = simulate_kelly_stake(records, starting_bankroll=starting_bankroll)

    return {
        "overall": _hit_rate(outcomes),
        "by_market": _breakdown_by(outcomes, "market"),
        "by_competition": _breakdown_by(outcomes, "competition"),
        "by_confidence": _breakdown_by(outcomes, "confidence"),
        "kelly_roi_simulation": build_evaluation_report(records, bankroll_result),
    }
