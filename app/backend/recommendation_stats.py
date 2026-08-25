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
import dataclasses
from typing import Any, Callable

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


def _segment_kelly_report(
    outcomes: list[RecommendationOutcome],
    key_fn: Callable[[RecommendationOutcome], str | None],
    starting_bankroll: float = DEFAULT_STARTING_BANKROLL,
) -> dict[str, dict[str, Any]]:
    """Same hit-rate-plus-Kelly-ROI report compute_recommendation_stats's
    own kelly_roi_simulation produces, run once per group instead of once
    overall -- powers the dashboard's Market / Market+Direction / League
    breakdown tables (W170). None-valued keys (competition can be null)
    group under "unknown", matching _breakdown_by's own convention."""
    groups: dict[str, list[RecommendationOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[key_fn(outcome) or "unknown"].append(outcome)
    result: dict[str, dict[str, Any]] = {}
    for group_key, group_outcomes in groups.items():
        records = _to_backtest_records(group_outcomes)
        bankroll_result = simulate_kelly_stake(records, starting_bankroll=starting_bankroll)
        result[group_key] = build_evaluation_report(records, bankroll_result)
    return result


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
        # W170: full (hit-rate + Kelly ROI) metrics per segment, for the
        # dashboard's three breakdown tables -- distinct from by_market/
        # by_competition above, which only report hit-rate across every
        # resolved pick (conditional included). These three are scoped to
        # the same staked (direct_bet-only) population as kelly_roi_simulation.
        "by_market_metrics": _segment_kelly_report(outcomes, lambda o: o.market, starting_bankroll),
        "by_market_selection_metrics": _segment_kelly_report(
            outcomes, lambda o: f"{o.market}:{o.selection}", starting_bankroll
        ),
        "by_league_metrics": _segment_kelly_report(outcomes, lambda o: o.competition, starting_bankroll),
        # Raw per-bet list (plain dicts, not BetOutcome instances -- always
        # JSON-safe without relying on FastAPI's implicit dataclass
        # handling) -- feeds the dashboard's odds/stake histograms (bucketed
        # client-side) and is the source list compute_agent_performance_dashboard
        # sorts for its top/bottom examples.
        "staked_bets": [dataclasses.asdict(bet) for bet in bankroll_result.bets],
    }
