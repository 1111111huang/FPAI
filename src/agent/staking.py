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
            if m.get("current_odds") is None:
                continue  # agent marked direct_bet with no odds found — cannot stake
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


def kelly_fraction(value_edge: float, odds: float, max_fraction: float = 0.10) -> float:
    """Kelly stake as a fraction of bankroll: value_edge / (odds - 1), capped
    at max_fraction. Returns 0.0 for non-positive edge or odds <= 1.0 -- no
    Kelly fraction is defined/worth taking there; callers must treat 0.0 as
    "no stake", not a computation error.

    Extracted (A80) from simulate_kelly_stake's own inline formula so
    schema.py's unit_bet_multiplier enrichment (A82) and the app's
    outcome-based ROI simulation (app/backend/recommendation_stats.py, W168)
    reuse the exact same math backtesting already relies on, instead of a
    second, potentially-drifting copy."""
    if odds <= 1.0 or value_edge <= 0:
        return 0.0
    return min(value_edge / (odds - 1.0), max_fraction)


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
            if m.get("current_odds") is None:
                continue  # agent marked direct_bet with no odds found — cannot stake
            odds = float(m["current_odds"])
            value_edge = float(m.get("value_edge", 0.0))
            fraction = kelly_fraction(value_edge, odds, max_fraction)
            if fraction <= 0:
                continue
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
