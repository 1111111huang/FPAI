"""W12: bet-logging request models and CRUD glue (D3a).

Two logging paths:
1. from-recommendation -- home_team/away_team/date/odds are always derived
   from the recommendation snapshot itself (BetFromRecommendationRequest has
   no fields for them at all), so stake is structurally the only thing a
   client can supply besides which market/selection they're betting on and
   the snapshot to lock in. This is what "rejects an attempt to edit any
   field but stake" means in practice -- there's nothing to submit.
2. manual -- user provides everything, but match_id must be a real, resolved
   fixture reference (non-empty), not free-typed team names.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.backend.bet_tracker import Bet, BetTracker
from app.backend.sandbox_clock import is_sandbox_mode


class BetOut(BaseModel):
    id: int
    match_id: str
    date: str
    home_team: str
    away_team: str
    market: str
    selection: str
    odds: float
    stake: float
    outcome: str
    profit_loss: float | None
    source: str
    recommendation_snapshot: dict | None
    created_at: str

    @classmethod
    def from_bet(cls, bet: Bet) -> "BetOut":
        return cls(**bet.__dict__)


class BetFromRecommendationRequest(BaseModel):
    match_id: str
    recommendation: dict
    market: str
    selection: str
    stake: float


class BetManualRequest(BaseModel):
    match_id: str = Field(min_length=1)
    date: str
    home_team: str
    away_team: str
    market: str
    selection: str
    odds: float
    stake: float


def resolve_from_recommendation(request: BetFromRecommendationRequest) -> dict:
    """Derive home_team/away_team/date/odds from the recommendation snapshot
    itself -- raises if the chosen market/selection isn't actually in it, or
    has no current_odds to bet against."""
    match = request.recommendation.get("match") or {}
    home_team = match.get("home") or match.get("home_team")
    away_team = match.get("away") or match.get("away_team")
    date = match.get("date")

    matching_market = next(
        (
            m for m in request.recommendation.get("markets") or []
            if m.get("market") == request.market and m.get("selection") == request.selection
        ),
        None,
    )
    if matching_market is None:
        raise ValueError(
            f"Market {request.market!r}/selection {request.selection!r} not found in the given recommendation."
        )
    odds = matching_market.get("current_odds")
    if odds is None:
        raise ValueError(f"Market {request.market!r}/selection {request.selection!r} has no current_odds to bet against.")

    return {
        "match_id": request.match_id,
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "market": request.market,
        "selection": request.selection,
        "odds": odds,
        "stake": request.stake,
    }


_bet_tracker_singleton: BetTracker | None = None
_SANDBOX_BET_TRACKER_DB_PATH = Path(__file__).parent.parent / "data" / "sandbox" / "user_bets.db"


def get_bet_tracker() -> BetTracker:
    """FastAPI dependency -- overridden in tests via app.dependency_overrides.
    Sandbox mode (W29) points this at a scratch db path so sandbox runs
    never touch real dev data."""
    global _bet_tracker_singleton
    if _bet_tracker_singleton is None:
        _bet_tracker_singleton = (
            BetTracker(db_path=_SANDBOX_BET_TRACKER_DB_PATH) if is_sandbox_mode() else BetTracker()
        )
    return _bet_tracker_singleton
