"""W02: POST /api/recommendations -- request/response models and an
app-owned Pydantic validation layer, wholly independent of the agent's own
extract_recommendation (A28/agent_techspec.md Gaps). A malformed market is
flagged/omitted, not a 500 for the whole request -- the app should never
blindly trust the agent's schema guarantees hold, including for recommendations
that predate A28/A29 (e.g. from a future cache)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.backend.match_info import gate_league
from app.backend.recommendation_cache import RecommendationCache
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_scoped_path
from src.agent import tools as agent_tools
from src.agent.graph import run_agent as _real_run_agent
from src.agent.snapshot_store import league_base_dir, SnapshotMissingError
from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

_LOG = get_logger(__name__)

_cache_singleton: RecommendationCache | None = None
_SANDBOX_CACHE_DB_PATH = sandbox_scoped_path("recommendation_cache.db")
_SANDBOX_SNAPSHOT_BASE_DIR = Path(__file__).parent.parent.parent / "data" / "agent_snapshots" / "sandbox"
_CORPUS_BASE_DIR = Path(__file__).parent.parent.parent / "data" / "agent_snapshots"
_TEAM_MAPPING_PATH = Path(__file__).parent.parent.parent / "config" / "team_mapping.json"


def _composite_match_key(home_team: str, away_team: str, date: str) -> str:
    return f"{home_team}__{away_team}__{date}".replace(" ", "_")


def _lookup_corpus_match_id(home_team: str, away_team: str, date: str, league: str | None) -> str | None:
    """Resolves a fixture to the real raw_matches.match_id the standalone
    agent-snapshot CLI's corpus is keyed by, so the sandbox can replay from
    it directly. Team names come from whatever the fixtures API returned
    (football-data.org/Odds API), not necessarily the ML engine's canonical
    spelling -- mapped through TeamNameMapper first, same tool/pattern
    eod_batch.py's odds_lookup()/match_odds() already use for the identical
    class of problem (W06/BUG-015). Returns None on any kind of miss
    (unmapped team, no matching row, no league) -- never raises; a miss just
    means "no corpus entry, fall through to record," not an error."""
    if not league:
        return None
    try:
        mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
        canonical_home = mapper.map_team(home_team)
        canonical_away = mapper.map_team(away_team)
        db = DuckDBManager()
        with db.connection(read_only=True) as conn:
            row = conn.execute(
                "SELECT match_id FROM raw_matches WHERE league = ? AND date = ? AND home_team = ? AND away_team = ?",
                [league, date, canonical_home, canonical_away],
            ).fetchone()
    except Exception:
        # DuckDBManager()/load_settings() can raise (missing/invalid
        # config.yaml) and conn.execute() can raise a duckdb.Error (e.g. no
        # raw_matches table in this environment yet) -- neither is a
        # "there's no corpus entry" signal worth surfacing as a 500, so this
        # degrades the same as an ordinary lookup miss: log and fall through
        # to record mode.
        _LOG.warning(
            "corpus_match_id_lookup_failed | home=%s | away=%s | date=%s | league=%s",
            home_team, away_team, date, league, exc_info=True,
        )
        return None
    return row[0] if row else None


def _run_agent_in_mode(mode: str, match_info: dict, config, match_id: str, base_dir: Path):
    """Configure the snapshot store for `mode` and run the real agent,
    always resetting the store to live mode afterward regardless of
    outcome."""
    agent_tools.configure_snapshot_store(
        mode, match_id=match_id, match_date=match_info.get("date"), base_dir=base_dir,
    )
    try:
        return _real_run_agent(match_info, config=config)
    finally:
        agent_tools.configure_snapshot_store("live")


def _select_sandbox_snapshot_source(match_info: dict) -> tuple[str, str, Path]:
    """W70: decides record vs replay -- and which namespace to use -- by
    checking disk directly, in priority order: (1) the sandbox's own prior
    recording for this exact match (fixes recordings not surviving a
    backend restart -- previously tracked in an in-memory set that started
    empty every process, so a fresh process always re-recorded and silently
    overwrote whatever was already there); (2) a matching complete entry in
    the standalone agent-snapshot corpus, resolved via
    _lookup_corpus_match_id; (3) otherwise, record fresh into the sandbox's
    own partition, unchanged from before this story. Returns
    (mode, match_id, base_dir)."""
    home_team = match_info.get("home_team")
    away_team = match_info.get("away_team")
    date = match_info.get("date")
    sandbox_match_id = _composite_match_key(home_team, away_team, date)

    if (_SANDBOX_SNAPSHOT_BASE_DIR / sandbox_match_id / "_complete.json").exists():
        return "replay", sandbox_match_id, _SANDBOX_SNAPSHOT_BASE_DIR

    league = match_info.get("league")
    corpus_match_id = _lookup_corpus_match_id(home_team, away_team, date, league)
    if corpus_match_id:
        corpus_league_dir = league_base_dir(league, base_dir=_CORPUS_BASE_DIR)
        if (corpus_league_dir / corpus_match_id / "_complete.json").exists():
            return "replay", corpus_match_id, corpus_league_dir

    return "record", sandbox_match_id, _SANDBOX_SNAPSHOT_BASE_DIR


def run_agent(match_info: dict, config=None):
    """W37/W70: routes through SnapshotStore record/replay when sandbox mode
    is active -- replaying from an existing recording (the sandbox's own, or
    W70's agent-snapshot corpus bridge) whenever one already exists on disk,
    recording fresh into the sandbox partition only when neither does.
    Otherwise passes straight through to the real, live run_agent.

    W43: a replay-mode SnapshotMissingError can happen even for a match
    that's genuinely already recorded -- SnapshotStore's replay lookup key
    is a hash of the tool call's exact input arguments (e.g. an LLM-chosen
    optional follow-up web_search query), and that specific text isn't
    reproducible run-to-run (agent_techspec.md Sec 18.6). Rather than let
    that 500 the request, fall back to one fresh record-mode pass into the
    sandbox partition for this request -- matching this codebase's "never
    assume the agent/its own optimizations hold, degrade gracefully"
    philosophy (W02/W15/W16's validate_and_degrade). Any other exception --
    including a second failure from the record-mode retry itself -- is not
    caught here and propagates uncaught, so this isn't a silent catch-all."""
    if not is_sandbox_mode():
        return _real_run_agent(match_info, config=config)

    mode, match_id, base_dir = _select_sandbox_snapshot_source(match_info)
    try:
        return _run_agent_in_mode(mode, match_info, config, match_id, base_dir)
    except SnapshotMissingError:
        if mode != "replay":
            raise
        _LOG.warning(
            "sandbox_agent_replay_miss | match=%s | retrying_in_record_mode", match_id,
        )
        sandbox_match_id = _composite_match_key(
            match_info.get("home_team"), match_info.get("away_team"), match_info.get("date"),
        )
        return _run_agent_in_mode("record", match_info, config, sandbox_match_id, _SANDBOX_SNAPSHOT_BASE_DIR)


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
