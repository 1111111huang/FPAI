"""W02: POST /api/recommendations -- request/response models and an
app-owned Pydantic validation layer, wholly independent of the agent's own
extract_recommendation (A28/agent_techspec.md Gaps). A malformed market is
flagged/omitted, not a 500 for the whole request -- the app should never
blindly trust the agent's schema guarantees hold, including for recommendations
that predate A28/A29 (e.g. from a future cache)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Literal

from pydantic import BaseModel, ValidationError

from app.backend.match_info import gate_league
from app.backend.recommendation_cache import RecommendationCache
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_scoped_path
from src.agent import tools as agent_tools
from src.agent.graph import run_agent as _real_run_agent
from src.agent.schema import normalize_explanation, reported_teams, teams_match
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
_MODEL_SELECTION_PATH = Path(__file__).parent.parent.parent / "config" / "model_selection.yaml"


def _current_model_selection_hash() -> str:
    """Fingerprint of config/model_selection.yaml's content. BUG-036: both
    known sandbox-snapshot staleness incidents (a stuck cold-start SP1 card,
    43 E0 cards stuck on a disproven 1-target forecast) were caused by this
    exact file changing -- model paths landing in a consistent state, a
    target's model_path getting fixed -- *after* a snapshot had already been
    recorded, with nothing to detect the recording now predates the fix.
    This is the concrete signal both real incidents traced back to, not a
    general "did anything change" check.
    # ponytail: scoped to this one file, the only one either incident
    # implicated. Extend to other config/pipeline inputs only if a future
    # staleness incident actually traces back to one -- not speculative
    # work today."""
    try:
        return hashlib.sha256(_MODEL_SELECTION_PATH.read_bytes()).hexdigest()[:16]
    except FileNotFoundError:
        return "missing"


def _sandbox_marker_is_fresh(marker_path: Path) -> bool:
    """A sandbox-recorded snapshot is only trusted for replay if it was
    recorded under the current model_selection.yaml -- an older recording
    (or one from before this fix, which never wrote a hash at all) can't
    prove that, so it's treated the same as "not recorded yet" and falls
    through to a fresh live record instead of silently replaying
    indefinitely (BUG-036). Deliberately scoped to the sandbox's own
    directory only -- the separate agent-snapshot CLI corpus (main.py's
    run_agent_snapshot, A11) is a historical training/backtest corpus that
    *should* stay pinned to what it recorded regardless of later model
    changes (A09's determinism guarantee), so its own "_complete.json"
    markers and skip logic are untouched by this."""
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return marker.get("model_selection_hash") == _current_model_selection_hash()


def _composite_match_key(home_team: str, away_team: str, date: str) -> str:
    return f"{home_team}__{away_team}__{date}".replace(" ", "_")


_sandbox_recording_locks: dict[str, threading.Lock] = {}
_sandbox_recording_locks_guard = threading.Lock()


def _lock_for_match(match_id: str) -> threading.Lock:
    """One lock per sandbox match_id, created lazily. Guards the whole
    record/replay decision + agent call for a given match so two
    concurrent requests for the SAME match (double-click, two open tabs,
    a live request racing a --precompute run) can't both slip past the
    disk check and both make a duplicate live LLM/Tavily call -- the
    second request blocks, then re-checks disk after the first completes
    and correctly finds the just-written marker (replay), instead of also
    recording. Different matches use different locks and never contend
    with each other. The registry itself grows unboundedly over a long
    process lifetime (one entry per distinct match ever requested in
    sandbox mode) -- acceptable for this app's real scale (single-user,
    local, a sandbox session covers a handful of fixtures at a time), not
    worth adding eviction for."""
    with _sandbox_recording_locks_guard:
        if match_id not in _sandbox_recording_locks:
            _sandbox_recording_locks[match_id] = threading.Lock()
        return _sandbox_recording_locks[match_id]


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
        result = _real_run_agent(match_info, config=config)
        if mode == "record":
            # Mirrors main.py's agent-snapshot CLI convention exactly (the
            # only other writer of this marker) -- without it, a
            # successful record pass is never recognized as "already
            # recorded" on a later check, defeating the whole point of
            # checking disk instead of in-memory state.
            marker = base_dir / match_id / "_complete.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            # BUG-036: records the model_selection.yaml fingerprint this
            # recording was made under, so a later change can be detected
            # and this recording stops being trusted for replay -- see
            # _sandbox_marker_is_fresh().
            marker.write_text(json.dumps({
                "model_selection_hash": _current_model_selection_hash(),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }))
        return result
    finally:
        agent_tools.configure_snapshot_store("live")


def _select_sandbox_snapshot_source(match_info: dict) -> tuple[str, str, Path]:
    """W70: decides record vs replay -- and which namespace to use -- by
    checking disk directly, in priority order: (1) the sandbox's own prior
    recording for this exact match (fixes recordings not surviving a
    backend restart -- previously tracked in an in-memory set that started
    empty every process, so a fresh process always re-recorded and silently
    overwrote whatever was already there) -- BUG-036: only trusted if it's
    still *fresh* (recorded under the current model_selection.yaml, see
    _sandbox_marker_is_fresh); a stale one falls through exactly like a
    missing one, forcing one fresh live re-record instead of replaying
    disproven data forever; (2) a matching complete entry in the standalone
    agent-snapshot corpus, resolved via _lookup_corpus_match_id (no
    freshness check -- that corpus is a separate, deliberately-pinned
    historical/backtest store, see _sandbox_marker_is_fresh's docstring);
    (3) otherwise, record fresh into the sandbox's own partition, unchanged
    from before this story. Returns (mode, match_id, base_dir)."""
    home_team = match_info.get("home_team")
    away_team = match_info.get("away_team")
    date = match_info.get("date")
    sandbox_match_id = _composite_match_key(home_team, away_team, date)

    if _sandbox_marker_is_fresh(_SANDBOX_SNAPSHOT_BASE_DIR / sandbox_match_id / "_complete.json"):
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
    caught here and propagates uncaught, so this isn't a silent catch-all.

    Holds a per-match lock (_lock_for_match) across the whole decision+call
    sequence, so two concurrent requests for the identical match (double-
    click, two open tabs, a live request racing a --precompute run) can't
    both slip past the disk check and both make a duplicate live call --
    the second blocks until the first finishes, then re-checks disk and
    correctly finds the just-written marker (replay)."""
    if not is_sandbox_mode():
        return _real_run_agent(match_info, config=config)

    sandbox_match_id = _composite_match_key(
        match_info.get("home_team"), match_info.get("away_team"), match_info.get("date"),
    )
    with _lock_for_match(sandbox_match_id):
        mode, match_id, base_dir = _select_sandbox_snapshot_source(match_info)
        try:
            return _run_agent_in_mode(mode, match_info, config, match_id, base_dir)
        except SnapshotMissingError:
            if mode != "replay":
                raise
            _LOG.warning(
                "sandbox_agent_replay_miss | match=%s | retrying_in_record_mode", match_id,
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
    # Constrained to the same vocabulary config/prompts/agent_v1.txt already
    # specifies to the LLM (result_3way/btts/total_goals/home_corners/
    # away_corners, home/draw/away/yes/no/over_2.5/under_2.5) -- previously
    # plain `str`, so a non-canonical name the agent invented (observed live:
    # "1X2" for what should be result_3way, "Asian Handicap", team names used
    # as a selection) passed validation instead of being dropped like any
    # other malformed market.
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]
    recommendation_type: str
    current_odds: float | None
    # BUG-032: defaulted for the same reason as src/agent/schema.py's
    # MarketRecommendationModel -- a cached/replayed row missing this key
    # (e.g. any generation predating this fix) must still validate here,
    # not just at the agent layer.
    min_odds: float = 0.0
    ml_probability: float
    implied_probability: float
    value_edge: float
    # W83: agent-side A52 computes this (src/agent/schema.py) for a
    # 'conditional' market -- the price it'd need to reach to clear
    # min_value_edge, or None when not applicable/computable. Defaulted so a
    # pre-A52 cached row (no such key at all) still validates, same
    # W15-established convention as feature_completeness below.
    target_odds: float | None = None


class MatchRecommendationOut(BaseModel):
    match: dict
    overall: str
    markets: list[MarketRecommendationOut]
    # One bullet per aspect, mirroring src/agent/schema.py's MatchRecommendationModel.
    explanation: list[str]
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
    # A82 (agent_user_stories.md): Kelly-derived stake-sizing suggestion for
    # this recommendation's actual pick, as a multiple of an abstract "Unit
    # Bet" -- not a dollar figure. None when there's no priced pick, or
    # absent entirely on a pre-A82 cached row, same convention as
    # target_odds/feature_completeness above.
    unit_bet_multiplier: float | None = None


def validate_and_degrade(
    raw: dict, home_team: str | None = None, away_team: str | None = None
) -> MatchRecommendationOut:
    """Validate a raw MatchRecommendation dict (from run_agent, a cache, or
    anywhere else), dropping any market that fails validation rather than
    raising for the whole request. Top-level fields default safely too, so
    even a badly malformed payload can't crash the endpoint.

    BUG-023/024: the agent's LLM call has been observed hallucinating a
    completely unrelated match's analysis (most often "Manchester City vs
    Liverpool", confirmed on 5/10 fixtures in one live sandbox precompute
    batch) instead of grounding itself in the real requested fixture -- with
    nothing else in the pipeline catching it before this recommendation is
    cached and served. `home_team`/`away_team` are the fixture actually
    requested, so a mismatch against the agent's own self-reported `match`
    field can be caught and degraded here, the one layer already responsible
    for never trusting the agent's output blindly. Optional (mirroring
    extract_recommendation's own home_team/away_team params, src/agent/schema.py):
    `GET /api/recommendations/{match_id}` (main.py) only has match_id/date, not
    a ground-truth fixture to compare against, so it calls this with neither --
    the match-mismatch check is skipped, but the per-market validation below
    still runs, which is what that endpoint actually needs (BUG-028: it used
    to call MatchRecommendationOut.model_validate() directly instead of this
    function at all, so any pre-existing cached row with a market/selection
    that predates the Literal constraints below -- extremely common, since the
    local-model hallucinations this file's other bugs document routinely wrote
    non-canonical values -- raised an uncaught ValidationError, a 500 for a
    plain cache read, not the graceful degrade every other caller already got.)
    """
    if home_team and away_team:
        reported = reported_teams(raw.get("match") or {})
    else:
        reported = None
    if reported is not None and not teams_match((home_team, away_team), reported):
        _LOG.warning(
            "agent_match_mismatch | requested=%s v %s | agent_reported=%s v %s",
            home_team, away_team, reported[0], reported[1],
        )
        return MatchRecommendationOut(
            match={"home_team": home_team, "away_team": away_team},
            overall="insufficient_data",
            markets=[],
            explanation=["The agent's analysis referenced a different match than requested and was discarded."],
            confidence="low",
            limitations=[
                f"Agent output was for {reported[0]} v {reported[1]}, not the requested "
                f"{home_team} v {away_team} -- discarded as a mismatch."
            ],
            prediction_basis="unknown",
            invalid_market_count=len(raw.get("markets") or []),
        )

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
        explanation=normalize_explanation(raw.get("explanation")),
        confidence=raw.get("confidence") or "low",
        limitations=limitations,
        prediction_basis=raw.get("prediction_basis") or "unknown",
        invalid_market_count=invalid_count,
        cold_start_risk=bool(raw.get("cold_start_risk", False)),
        feature_completeness=raw.get("feature_completeness"),
        unknown_team=bool(raw.get("unknown_team", False)),
        unit_bet_multiplier=raw.get("unit_bet_multiplier"),
    )
