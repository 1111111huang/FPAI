"""W09: end-of-day scheduled batch recommendation generation (D2a). Fetches
the next day's E0 fixtures (W05) and current odds (W07), generates
recommendations concurrently using the same fault-tolerant pattern as
main.py's _run_backtest_concurrent (agent_techspec.md §13 -- bounded
asyncio.Semaphore, asyncio.to_thread per match, skip-and-continue on
error), writing each into W11's cache. Also schedules each fixture's T-30
job (W10) -- for every fixture, regardless of whether EOD generation
itself succeeded for it, since W10's own refresh is independently
best-effort and shouldn't depend on tonight's batch having gone cleanly.

Odds are matched to fixtures by each side's *canonical* team name, via the
same TeamNameMapper/config/team_mapping.json ingestion already uses --
BUG-015: confirmed live against a real Odds API key that The Odds API and
football-data.org spell many clubs differently ("Man United" vs
"Manchester United", "Nottingham" vs "Nottingham Forest", "Tottenham" vs
"Tottenham Hotspur", "Brighton Hove" vs "Brighton and Hove Albion", ...) --
raw case-insensitive string equality silently dropped odds for 6/10 real
fixtures in that check. A fixture with no matching odds event (even after
canonical mapping) proceeds with odds omitted (the agent's existing
no-odds handling) rather than blocking the whole batch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.backend import recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.odds_api_client import NormalizedOdds, OddsAPIClient
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendations import validate_and_degrade
from src.agent.agent_config import AgentConfig
from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

COMPETITION_CODE = "PL"
LEAGUE_CODE = "E0"

_TEAM_MAPPING_PATH = Path(__file__).parent.parent.parent / "config" / "team_mapping.json"


@dataclass
class EodBatchResult:
    fixtures: list[NormalizedMatch] = field(default_factory=list)
    generated: int = 0
    skipped: int = 0


def odds_lookup(odds_events: list[NormalizedOdds]) -> dict[tuple[str, str], NormalizedOdds]:
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    return {(mapper.map_team(o.home_team), mapper.map_team(o.away_team)): o for o in odds_events}


def match_odds(
    fixture: NormalizedMatch, odds_by_teams: dict[tuple[str, str], NormalizedOdds]
) -> dict[str, float] | None:
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    key = (mapper.map_team(fixture.home_team), mapper.map_team(fixture.away_team))
    odds = odds_by_teams.get(key)
    if odds is None or odds.home_odds is None or odds.draw_odds is None or odds.away_odds is None:
        return None
    return {"home": odds.home_odds, "draw": odds.draw_odds, "away": odds.away_odds}


async def run_eod_batch(
    fixtures_client: FootballDataClient,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    schedule_t30: Callable[[NormalizedMatch], None],
    date_str: str,
    concurrency: int = 5,
) -> EodBatchResult:
    fixtures = fixtures_client.get_fixtures(competition_code=COMPETITION_CODE, date_from=date_str, date_to=date_str)

    odds_events = odds_client.get_odds() if odds_client is not None else None
    odds_by_teams = odds_lookup(odds_events or [])

    agent_config_hash = compute_agent_config_hash(config)
    semaphore = asyncio.Semaphore(concurrency)
    result = EodBatchResult(fixtures=list(fixtures))

    async def _generate_one(fixture: NormalizedMatch) -> None:
        match_info = {
            "home_team": fixture.home_team, "away_team": fixture.away_team,
            "date": date_str, "league": LEAGUE_CODE,
        }
        odds = match_odds(fixture, odds_by_teams)
        if odds is not None:
            match_info["odds"] = odds

        async with semaphore:
            try:
                raw = await asyncio.to_thread(recommendations.run_agent, match_info=match_info, config=config)
            except Exception as exc:
                LOGGER.warning(
                    "EOD batch: skipping match_id=%s (%s v %s): %s",
                    fixture.match_id, fixture.home_team, fixture.away_team, exc,
                )
                result.skipped += 1
                return

        degraded = validate_and_degrade(raw)
        cache.record_generation(
            match_id=fixture.match_id, date=date_str, agent_config_hash=agent_config_hash,
            odds=odds or {}, recommendation=degraded.model_dump(), triggered_by="scheduled",
        )
        result.generated += 1

    await asyncio.gather(*[_generate_one(fixture) for fixture in fixtures])

    for fixture in fixtures:
        schedule_t30(fixture)

    return result
