"""Backtest evaluation report computation (A13)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.agent_config import AgentConfig

if TYPE_CHECKING:
    from src.agent.staking import BankrollResult


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
