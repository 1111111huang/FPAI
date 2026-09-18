"""W10: per-match refresh 30 minutes before kickoff (D2a). Fetches fresh
odds (W07) first, then always re-runs run_agent() -- direct user request
(2026-09-11): T-30 is the one point in the pipeline that runs close enough
to kickoff for research_node's confirmed-starting-lineup web search
(near_kickoff=True, below) to find real, non-speculative team news, so it's
worth the LLM/Tavily cost even when the odds themselves haven't moved --
the lineup confirmation is the new information, not just the price. (Until
2026-09-11 this compared fresh odds against the cached recommendation's own
odds and skipped an unchanged price entirely, purely as a cost-saving
dedup -- already_fresh() is kept for eod_batch.py's own, unrelated dedup
use, just no longer consulted here.)

Best-effort throughout: a fixture whose odds can't be fetched or matched
(W07's credit budget exhausted, or the fixture no longer appears in the
odds feed -- e.g. removed/postponed), or a run_agent() error, leaves the
prior recommendation in place rather than failing the job. No prior cache
entry at all (e.g. tonight's EOD generation errored for this match) is
simply a first generation, not a special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datetime import timezone

from app.backend import recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.eod_batch import (
    LEAGUE_CODE, _TEAM_MAPPING_PATH, add_secondary_odds, has_kicked_off, match_odds, odds_lookup,
)
from app.backend.football_data_client import NormalizedMatch
from app.backend.odds_api_client import OddsAPIClient
from app.backend.odds_sport_keys import ODDS_SPORT_KEY_BY_COMPETITION
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendations import validate_and_degrade
from app.backend.sandbox_clock import sandbox_now
from src.agent.agent_config import AgentConfig
from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

Outcome = Literal["refreshed", "skipped_no_odds", "skipped_error", "skipped_kicked_off"]


@dataclass
class T30RefreshResult:
    match_id: str
    outcome: Outcome


# Deliberately synchronous, unlike run_eod_batch -- there's exactly one
# match to handle per call (no internal concurrency to bound), and this
# runs from APScheduler's own worker thread (RecoverableScheduler), never
# on the FastAPI event loop, so there's nothing to gain from asyncio here
# and an async def would risk a nested asyncio.run() if a caller wrapped
# this in one (as W09's schedule_t30 catch-up path does).
def refresh_match_at_t30(
    fixture: NormalizedMatch,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    date_str: str,
    league: str = LEAGUE_CODE,
) -> T30RefreshResult:
    # W62: `league` defaults to LEAGUE_CODE/"E0", preserving every existing
    # caller's exact behavior -- lets the multi-competition scheduler
    # orchestration call this once per competition instead of it being
    # structurally single-league.
    #
    # This T-30 job was scheduled while `fixture` was still genuinely
    # pre-match, but RecoverableScheduler's own catch-up path (scheduler.py)
    # runs a job immediately if its trigger time has already passed by the
    # time the process actually gets to registering it (e.g. a restart long
    # after the real T-30 instant) -- by then the match itself may have
    # kicked off. Same rationale as eod_batch.has_kicked_off()'s own two
    # call sites: leave the last cached pre-match recommendation as-is
    # rather than replacing it with an "analysis" of a match already live.
    if has_kicked_off(fixture, sandbox_now(timezone.utc)):
        LOGGER.info(
            "T-30 refresh: skipping match_id=%s -- kickoff already passed (stale "
            "catch-up run), leaving the last cached pre-match recommendation in place.",
            fixture.match_id,
        )
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_kicked_off")

    # W58: explicit sport_key from the competition-id mapping, rather than
    # relying on get_odds()'s own "soccer_epl" default parameter.
    odds_events = odds_client.get_odds(sport_key=ODDS_SPORT_KEY_BY_COMPETITION[league]) if odds_client is not None else None
    if odds_events is None:
        LOGGER.info(
            "T-30 refresh: skipping match_id=%s -- no odds available (credit budget "
            "exhausted or no odds client configured).", fixture.match_id,
        )
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_no_odds")

    # W234: a tight 2-name candidate pool (just this fixture's own home/away)
    # lets odds_lookup() resolve a club-type-prefixed odds-side spelling via
    # token-containment matching -- even lower collision risk than
    # eod_batch.py's whole-batch pool, since there are only ever two names to
    # choose between.
    fixture_mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    fixture_candidates = [fixture_mapper.map_team(fixture.home_team), fixture_mapper.map_team(fixture.away_team)]
    odds_by_teams = odds_lookup(odds_events, fixture_candidates)
    fresh_odds = match_odds(fixture, odds_by_teams)
    if fresh_odds is None:
        LOGGER.info(
            "T-30 refresh: skipping match_id=%s -- no matching odds event found "
            "(fixture may have been removed/postponed).", fixture.match_id,
        )
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_no_odds")

    agent_config_hash = compute_agent_config_hash(config)
    match_info = {
        "home_team": fixture.home_team, "away_team": fixture.away_team,
        "date": date_str, "league": league, "odds": fresh_odds,
        # Direct user request (2026-09-07): real starting lineups are
        # typically confirmed only ~T-60 minutes before kickoff -- this is
        # the one call in the whole system whose actual run time lands
        # after that point (every other generation, including this job's
        # own *scheduling* instant, is well before it), so it's the one
        # context where research_node's confirmed-lineup-first query
        # (src/agent/pipeline.py) can find real, non-speculative team news.
        "near_kickoff": True,
    }
    # W164 fixed EOD generation's odds to include totals/btts, not just
    # 1X2 -- t30_refresh.py never got the equivalent call, so every T-30
    # refresh regenerated on 1X2 odds alone, silently reintroducing the
    # exact "most picks are draws" bug W164 fixed (the agent's own prompt
    # rule is "if you don't have a real current price for a market, don't
    # recommend it at all" -- config/prompts/agent_v1.txt -- so no
    # totals/btts odds structurally means no totals/btts picks). Still
    # reuses a cached secondary-odds fetch when h2h is unchanged (its own
    # internal check, independent of the always-refresh change above) --
    # that's a real avoided API call, not a skipped generation.
    add_secondary_odds(
        match_info, fresh_odds, odds_client, cache, fixture, date_str,
        agent_config_hash, ODDS_SPORT_KEY_BY_COMPETITION[league], odds_by_teams,
    )

    try:
        agent_result = recommendations.run_agent(match_info=match_info, config=config)
    except Exception as exc:
        LOGGER.warning("T-30 refresh: run_agent failed for match_id=%s: %s", fixture.match_id, exc)
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_error")

    raw, reasoning_trace, forecast_payload = recommendations.unwrap_agent_result(agent_result)
    degraded = validate_and_degrade(raw, fixture.home_team, fixture.away_team)
    cache.record_generation(
        match_id=fixture.match_id, date=date_str, agent_config_hash=agent_config_hash,
        odds=fresh_odds, recommendation=degraded.model_dump(), triggered_by="scheduled",
        # A107: same tracing agent-train/agent-backtest already persist to
        # agent_telemetry -- see recommendations.run_agent's docstring.
        reasoning_trace=reasoning_trace, forecast_payload=forecast_payload,
    )
    return T30RefreshResult(match_id=fixture.match_id, outcome="refreshed")
