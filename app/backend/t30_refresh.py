"""W10: per-match refresh 30 minutes before kickoff (D2a). Fetches fresh
odds (W07) first and compares them against the odds used for the
currently-cached recommendation (W11) -- only re-runs run_agent() if the
odds actually changed (exact match = skip, logged as "no new data, refresh
skipped"), both saving LLM/Tavily cost and reducing how often the agent's
known run-to-run non-reproducibility (agent_techspec.md sec18.6) gets
exercised for no reason.

Best-effort throughout: a fixture whose odds can't be fetched or matched
(W07's credit budget exhausted, or the fixture no longer appears in the
odds feed -- e.g. removed/postponed), or a run_agent() error, leaves the
prior recommendation in place rather than failing the job. If no prior
cache entry exists at all (e.g. tonight's EOD generation errored for this
match), there's no baseline to compare against and nothing to lose by
generating fresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.backend import recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.eod_batch import LEAGUE_CODE, already_fresh, match_odds, odds_lookup
from app.backend.football_data_client import NormalizedMatch
from app.backend.odds_api_client import OddsAPIClient
from app.backend.odds_sport_keys import ODDS_SPORT_KEY_BY_COMPETITION
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendations import validate_and_degrade
from src.agent.agent_config import AgentConfig
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

Outcome = Literal["refreshed", "skipped_no_change", "skipped_no_odds", "skipped_error"]


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
    # W58: explicit sport_key from the competition-id mapping, rather than
    # relying on get_odds()'s own "soccer_epl" default parameter.
    odds_events = odds_client.get_odds(sport_key=ODDS_SPORT_KEY_BY_COMPETITION[league]) if odds_client is not None else None
    if odds_events is None:
        LOGGER.info(
            "T-30 refresh: skipping match_id=%s -- no odds available (credit budget "
            "exhausted or no odds client configured).", fixture.match_id,
        )
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_no_odds")

    fresh_odds = match_odds(fixture, odds_lookup(odds_events))
    if fresh_odds is None:
        LOGGER.info(
            "T-30 refresh: skipping match_id=%s -- no matching odds event found "
            "(fixture may have been removed/postponed).", fixture.match_id,
        )
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_no_odds")

    agent_config_hash = compute_agent_config_hash(config)
    # W151: shared with eod_batch.py's own EOD-pass dedup check -- one
    # "does this need regenerating" rule for both scheduled entry points.
    if already_fresh(cache, fixture.match_id, date_str, agent_config_hash, fresh_odds):
        LOGGER.info("T-30 refresh: no new data, refresh skipped for match_id=%s.", fixture.match_id)
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_no_change")

    match_info = {
        "home_team": fixture.home_team, "away_team": fixture.away_team,
        "date": date_str, "league": league, "odds": fresh_odds,
    }
    try:
        raw = recommendations.run_agent(match_info=match_info, config=config)
    except Exception as exc:
        LOGGER.warning("T-30 refresh: run_agent failed for match_id=%s: %s", fixture.match_id, exc)
        return T30RefreshResult(match_id=fixture.match_id, outcome="skipped_error")

    degraded = validate_and_degrade(raw, fixture.home_team, fixture.away_team)
    cache.record_generation(
        match_id=fixture.match_id, date=date_str, agent_config_hash=agent_config_hash,
        odds=fresh_odds, recommendation=degraded.model_dump(), triggered_by="scheduled",
    )
    return T30RefreshResult(match_id=fixture.match_id, outcome="refreshed")
