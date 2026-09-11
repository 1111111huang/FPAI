"""Backtest evaluation report computation (A13)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.agent_config import AgentConfig

if TYPE_CHECKING:
    from src.agent.staking import BankrollResult, BetOutcome


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Largest peak-to-trough fractional decline observed in the equity curve."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd


def _summarize_bets(bets: list["BetOutcome"]) -> dict[str, Any]:
    """Shared pick-counts/hit-rate/ROI summary for one group of bets --
    used by both build_market_breakdown and build_confidence_breakdown so
    the two never compute this differently."""
    staked = sum(bet.stake for bet in bets)
    profit = sum(bet.payout for bet in bets)
    wins = sum(1 for bet in bets if bet.won)
    return {
        "picks": len(bets),
        "wins": wins,
        "hit_rate": round(wins / len(bets), 6),
        "roi": round(profit / staked, 6) if staked > 0 else 0.0,
        "total_staked": round(staked, 2),
        "total_profit": round(profit, 2),
    }


def build_market_breakdown(bets: list["BetOutcome"]) -> dict[str, dict[str, Any]]:
    """A105: per-(market, selection) pick counts/hit-rate/ROI, keyed
    "<market>/<selection>". Grouped straight from the BetOutcome records
    simulate_flat_stake/simulate_kelly_stake already build (each one already
    carries market/selection/odds/stake/won/payout) -- no new resolution
    logic, this is the exact per-market breakdown A104 had to hand-build
    from agent_telemetry because agent-backtest's saved report never kept it."""
    groups: dict[tuple[str, str], list["BetOutcome"]] = defaultdict(list)
    for bet in bets:
        groups[(bet.market, bet.selection)].append(bet)

    return {
        f"{market}/{selection}": {"market": market, "selection": selection, **_summarize_bets(group)}
        for (market, selection), group in sorted(groups.items())
    }


_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2, "unknown": 3}


def build_confidence_breakdown(bets: list["BetOutcome"]) -> dict[str, dict[str, Any]]:
    """A107: is the LLM's own self-reported confidence (low/medium/high,
    MatchRecommendation.confidence) actually predictive of hit rate? Not
    checked anywhere before this -- grouped the same way build_market_breakdown
    groups by (market, selection), just keyed on BetOutcome.confidence instead."""
    groups: dict[str, list["BetOutcome"]] = defaultdict(list)
    for bet in bets:
        groups[bet.confidence].append(bet)

    return {
        confidence: {"confidence": confidence, **_summarize_bets(group)}
        for confidence, group in sorted(groups.items(), key=lambda item: _CONFIDENCE_ORDER.get(item[0], 99))
    }


def build_no_bet_breakdown(records: list[Any]) -> dict[str, int]:
    """A107: why a match ended up with no actionable bet -- previously only
    visible as one pooled insufficient_data_rate, with no way to tell
    "the LLM itself declined" from "a guardrail downgraded its own pick" or
    "no pick was ever resolved" without a full reasoning-trace read (the
    exact gap BUG-054's investigation hit).

    - insufficient_data: no ML forecast was available at all.
    - model_declined: the picked candidate's own initial_recommendation_type
      (A107, src/agent/schema.py) was already 'no_bet' -- the LLM's own
      call, no guardrail involved.
    - guardrail_downgraded: initial_recommendation_type differs from the
      final 'no_bet' -- a downgrade pass changed it (see the candidate's
      own `limitations` entries for which one and why).
    - no_pick_resolved: overall is 'no_bet' but market_results is empty --
      no recommendation_pick was offered, or it named a candidate
      resolve_recommendation_pick couldn't find.
    - unknown: a 'no_bet' record whose market_results predates A107 (no
      initial_recommendation_type recorded) -- can't classify further."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        overall = record.recommendation.get("overall")
        if overall == "insufficient_data":
            counts["insufficient_data"] += 1
            continue
        if overall != "no_bet":
            continue
        market_results = getattr(record, "market_results", None) or []
        if not market_results:
            counts["no_pick_resolved"] += 1
            continue
        initial_type = market_results[0].get("initial_recommendation_type")
        if initial_type is None:
            counts["unknown"] += 1
        elif initial_type == "no_bet":
            counts["model_declined"] += 1
        else:
            counts["guardrail_downgraded"] += 1
    return dict(counts)


def build_evaluation_report(records: list[Any], bankroll_result: "BankrollResult") -> dict[str, Any]:
    total_staked = sum(bet.stake for bet in bankroll_result.bets)
    total_profit = sum(bet.payout for bet in bankroll_result.bets)
    bets_won = sum(1 for bet in bankroll_result.bets if bet.won)
    bets_placed = len(bankroll_result.bets)
    insufficient = sum(1 for r in records if r.recommendation.get("overall") == "insufficient_data")

    roi = total_profit / total_staked if total_staked > 0 else 0.0
    hit_rate = bets_won / bets_placed if bets_placed > 0 else 0.0
    bet_frequency = bets_placed / len(records) if records else 0.0
    insufficient_data_rate = insufficient / len(records) if records else 0.0

    return {
        "matches_evaluated": len(records),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "roi": round(roi, 6),
        "hit_rate": round(hit_rate, 6),
        "bet_frequency": round(bet_frequency, 6),
        "max_drawdown": round(compute_max_drawdown(bankroll_result.equity_curve), 6),
        "insufficient_data_rate": round(insufficient_data_rate, 6),
        "starting_bankroll": bankroll_result.starting_bankroll,
        "ending_bankroll": round(bankroll_result.ending_bankroll, 2),
        # A83: already computed above (total_staked/total_profit locals) --
        # just never returned. Needed by the agent performance dashboard's
        # Main Metrics row (Total Stake, Money Won). Purely additive: every
        # existing caller (main.py's agent-backtest/agent-train reporting,
        # src/agent/comparison.py, recommendation_stats.py) reads specific
        # keys or dumps the dict generically (print_report/save_report both
        # iterate report.items()) -- nothing breaks from two new keys.
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        # A105: per-(market, selection) breakdown -- see build_market_breakdown.
        "market_breakdown": build_market_breakdown(bankroll_result.bets),
        # A107: is self-reported confidence predictive? -- see build_confidence_breakdown.
        "confidence_breakdown": build_confidence_breakdown(bankroll_result.bets),
        # A107: why matches ended up with no bet -- see build_no_bet_breakdown.
        "no_bet_breakdown": build_no_bet_breakdown(records),
    }


def config_hash(config: AgentConfig) -> str:
    """Stable 8-char hash identifying a config's relevant tuning fields (order-independent on markets)."""
    canonical = json.dumps(
        {
            "model": config.model,
            "provider": config.provider,
            "temperature": config.temperature,
            "max_tool_calls": config.max_tool_calls,
            "min_odds_threshold": config.min_odds_threshold,
            "min_value_edge": config.min_value_edge,
            "markets": sorted(config.markets),
            "system_prompt_version": config.system_prompt_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def save_report(report: dict[str, Any], config: AgentConfig, base_dir: str = "reports/agent_backtest") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{timestamp}_{config_hash(config)}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("Agent Backtest Evaluation Report")
    print("=" * 50)
    for key, value in report.items():
        print(f"  {key:<22}: {value}")
    print("=" * 50)
