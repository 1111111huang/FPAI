"""Bankroll simulation: flat-stake (A13) and Kelly criterion (A15) modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.backtest import BacktestRecord


@dataclass
class BetOutcome:
    match_id: str
    market: str
    selection: str
    odds: float
    stake: float
    won: bool
    payout: float  # net profit (positive) or loss (negative) — already includes stake direction


@dataclass
class BankrollResult:
    starting_bankroll: float
    ending_bankroll: float
    equity_curve: list[float] = field(default_factory=list)
    bets: list[BetOutcome] = field(default_factory=list)


def simulate_flat_stake(
    records: list["BacktestRecord"],
    starting_bankroll: float = 1000.0,
    stake_pct: float = 0.01,
) -> BankrollResult:
    """Fixed stake = stake_pct * starting_bankroll on every direct_bet recommendation
    with a resolvable outcome. Win: bankroll += stake * (odds - 1). Loss: bankroll -= stake."""
    bankroll = starting_bankroll
    equity_curve = [bankroll]
    bets: list[BetOutcome] = []
    flat_stake = starting_bankroll * stake_pct

    for record in records:
        for m in record.market_results:
            if m.get("recommendation_type") != "direct_bet":
                continue
            if m.get("correct") is None:
                continue  # unresolvable market (e.g. corners) — cannot settle, skip
            odds = float(m["current_odds"])
            won = bool(m["correct"])
            payout = flat_stake * (odds - 1) if won else -flat_stake
            bankroll += payout
            equity_curve.append(bankroll)
            bets.append(BetOutcome(
                match_id=record.match_id, market=m["market"], selection=m["selection"],
                odds=odds, stake=flat_stake, won=won, payout=payout,
            ))

    return BankrollResult(starting_bankroll=starting_bankroll, ending_bankroll=bankroll, equity_curve=equity_curve, bets=bets)


def simulate_kelly_stake(
    records: list["BacktestRecord"],
    starting_bankroll: float = 1000.0,
    max_fraction: float = 0.10,
) -> BankrollResult:
    """Kelly stake = value_edge / (odds - 1) * current bankroll, capped at max_fraction.
    Bets with non-positive edge are skipped (Kelly fraction would be <= 0)."""
    bankroll = starting_bankroll
    equity_curve = [bankroll]
    bets: list[BetOutcome] = []

    for record in records:
        for m in record.market_results:
            if m.get("recommendation_type") != "direct_bet":
                continue
            if m.get("correct") is None:
                continue
            odds = float(m["current_odds"])
            value_edge = float(m.get("value_edge", 0.0))
            if odds <= 1.0 or value_edge <= 0:
                continue
            fraction = min(value_edge / (odds - 1.0), max_fraction)
            stake = bankroll * fraction
            won = bool(m["correct"])
            payout = stake * (odds - 1) if won else -stake
            bankroll += payout
            equity_curve.append(bankroll)
            bets.append(BetOutcome(
                match_id=record.match_id, market=m["market"], selection=m["selection"],
                odds=odds, stake=stake, won=won, payout=payout,
            ))

    return BankrollResult(starting_bankroll=starting_bankroll, ending_bankroll=bankroll, equity_curve=equity_curve, bets=bets)
