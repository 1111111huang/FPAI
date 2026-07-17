"""W02: POST /api/recommendations -- request/response models and an
app-owned Pydantic validation layer, wholly independent of the agent's own
extract_recommendation (A28/agent_techspec.md Gaps). A malformed market is
flagged/omitted, not a 500 for the whole request -- the app should never
blindly trust the agent's schema guarantees hold, including for recommendations
that predate A28/A29 (e.g. from a future cache)."""

from __future__ import annotations

from pathlib import Path
import threading

from pydantic import BaseModel, ValidationError

from app.backend.match_info import gate_league
from app.backend.recommendation_cache import RecommendationCache
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_scoped_path
from src.agent import tools as agent_tools
from src.agent.graph import run_agent as _real_run_agent

_cache_singleton: RecommendationCache | None = None
_SANDBOX_CACHE_DB_PATH = sandbox_scoped_path("recommendation_cache.db")
_SANDBOX_SNAPSHOT_BASE_DIR = Path(__file__).parent.parent.parent / "data" / "agent_snapshots" / "sandbox"
_sandbox_recorded_matches: set[str] = set()
# Guards the read-check-then-add sequence around _sandbox_recorded_matches in
# run_agent() below. Without this, two concurrent requests for the *same*
# sandboxed match could both see match_key not yet recorded and both run in
# "record" mode simultaneously -- wasteful duplicate live calls that defeat
# the "first run records, rest replay" guarantee.
_recorded_matches_lock = threading.Lock()


def _composite_match_key(home_team: str, away_team: str, date: str) -> str:
    return f"{home_team}__{away_team}__{date}".replace(" ", "_")


def run_agent(match_info: dict, config=None):
    """W37: routes through SnapshotStore record/replay when sandbox mode is
    active, so a sandboxed match's real web_search calls are date-filtered
    (record) and every subsequent run of the same match makes zero live
    calls at all (replay) -- otherwise passes straight through to the real,
    live run_agent, unchanged from before this story."""
    if not is_sandbox_mode():
        return _real_run_agent(match_info, config=config)

    match_key = _composite_match_key(
        match_info.get("home_team"), match_info.get("away_team"), match_info.get("date"),
    )
    with _recorded_matches_lock:
        mode = "replay" if match_key in _sandbox_recorded_matches else "record"
    agent_tools.configure_snapshot_store(
        mode, match_id=match_key, match_date=match_info.get("date"), base_dir=_SANDBOX_SNAPSHOT_BASE_DIR,
    )
    try:
        result = _real_run_agent(match_info, config=config)
    finally:
        agent_tools.configure_snapshot_store("live")
    if mode == "record":
        with _recorded_matches_lock:
            _sandbox_recorded_matches.add(match_key)
    return result


def get_cache() -> RecommendationCache:
    """FastAPI dependency -- overridden in tests via app.dependency_overrides.
    Sandbox mode (W29) points this at a scratch db path so sandbox runs
    never touch real dev data."""
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = (
            RecommendationCache(db_path=_SANDBOX_CACHE_DB_PATH) if is_sandbox_mode() else RecommendationCache()
        )
    return _cache_singleton


class OddsInput(BaseModel):
    home: float
    draw: float
    away: float


class RecommendationRequest(BaseModel):
    home_team: str
    away_team: str
    date: str
    league: str | None = None
    odds: OddsInput | None = None
    match_id: str | None = None

    def to_match_info(self) -> dict:
        # W03: the app decides whether 'league' is set, via a hardcoded
        # allowlist independent of the engine/agent's own routing -- not a
        # direct pass-through of whatever the caller supplied.
        match_info = {"home_team": self.home_team, "away_team": self.away_team, "date": self.date}
        gated_league = gate_league(self.league)
        if gated_league is not None:
            match_info["league"] = gated_league
        if self.odds is not None:
            match_info["odds"] = self.odds.model_dump()
        return match_info

    def effective_match_id(self) -> str:
        if self.match_id:
            return self.match_id
        return _composite_match_key(self.home_team, self.away_team, self.date)


class MarketRecommendationOut(BaseModel):
    market: str
    selection: str
    recommendation_type: str
    current_odds: float | None
    min_odds: float
    ml_probability: float
    implied_probability: float
    value_edge: float


class MatchRecommendationOut(BaseModel):
    match: dict
    overall: str
    markets: list[MarketRecommendationOut]
    explanation: str
    confidence: str
    limitations: list[str]
    prediction_basis: str
    invalid_market_count: int = 0
    # W15: surfaced so the UI can treat cold_start_risk as a first-class
    # trust signal regardless of what prediction_basis claims. Default safely
    # for recommendations cached before W15 shipped (no such keys at all).
    cold_start_risk: bool = False
    feature_completeness: float | None = None
    unknown_team: bool = False


def validate_and_degrade(raw: dict) -> MatchRecommendationOut:
    """Validate a raw MatchRecommendation dict (from run_agent, a cache, or
    anywhere else), dropping any market that fails validation rather than
    raising for the whole request. Top-level fields default safely too, so
    even a badly malformed payload can't crash the endpoint."""
    valid_markets: list[MarketRecommendationOut] = []
    invalid_count = 0
    for market in raw.get("markets") or []:
        try:
            valid_markets.append(MarketRecommendationOut.model_validate(market))
        except ValidationError:
            invalid_count += 1

    limitations = list(raw.get("limitations") or [])
    if invalid_count:
        limitations.append(f"{invalid_count} market(s) omitted: malformed data from the agent.")

    return MatchRecommendationOut(
        match=raw.get("match") or {},
        overall=raw.get("overall") or "insufficient_data",
        markets=valid_markets,
        explanation=raw.get("explanation") or "",
        confidence=raw.get("confidence") or "low",
        limitations=limitations,
        prediction_basis=raw.get("prediction_basis") or "unknown",
        invalid_market_count=invalid_count,
        cold_start_risk=bool(raw.get("cold_start_risk", False)),
        feature_completeness=raw.get("feature_completeness"),
        unknown_team=bool(raw.get("unknown_team", False)),
    )
