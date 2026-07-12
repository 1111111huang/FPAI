"""W14: bet tracker summary stats (ROI, hit rate, running bankroll). Adapts
src/agent/evaluation.py's report-building logic for a running single-user
ledger of Bet rows instead of a fixed backtest batch's BankrollResult --
compute_max_drawdown is reused verbatim (it's already a pure function over
any equity curve); ROI/hit-rate/bankroll are recomputed here to match the
Bet dataclass's shape (profit_loss already resolved at settlement time,
BUG-014/W13's market_correct() has no part in this -- that's settlement,
this is reporting on what's already settled).

Computed only over settled (won/lost) bets, fresh from whatever is in the
tracker on each call -- no persisted running total, so it's always
consistent with the ledger.
"""

from __future__ import annotations

from typing import Any

from app.backend.bet_tracker import Bet
from src.agent.evaluation import compute_max_drawdown

# Matches src/agent/staking.py's existing default starting_bankroll --
# a notional bankroll for computing ROI/drawdown against, not a claim about
# the user's real available funds.
DEFAULT_STARTING_BANKROLL = 1000.0


def compute_bet_stats(bets: list[Bet], starting_bankroll: float = DEFAULT_STARTING_BANKROLL) -> dict[str, Any]:
    settled = [b for b in bets if b.outcome in ("won", "lost")]
    bets_won = sum(1 for b in settled if b.outcome == "won")
    total_staked = sum(b.stake for b in settled)
    total_profit = sum(b.profit_loss or 0.0 for b in settled)

    equity_curve = [starting_bankroll]
    running = starting_bankroll
    for bet in settled:
        running += bet.profit_loss or 0.0
        equity_curve.append(running)

    return {
        "bets_settled": len(settled),
        "bets_open": len(bets) - len(settled),
        "bets_won": bets_won,
        "roi": round(total_profit / total_staked, 6) if total_staked > 0 else 0.0,
        "hit_rate": round(bets_won / len(settled), 6) if settled else 0.0,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "max_drawdown": round(compute_max_drawdown(equity_curve), 6),
        "starting_bankroll": starting_bankroll,
        "current_bankroll": round(running, 2),
    }
