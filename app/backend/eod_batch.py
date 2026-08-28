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

W50: run_eod_batch() also accepts an optional pre-fetched `fixtures` list
(bypassing its own internal fixtures_client.get_fixtures() call, which only
ever queries status=SCHEDULED) and an optional `on_progress` callback -- see
run_eod_batch()'s docstring. Added for scripts/launch_sandbox.py's opt-in
--precompute step, which needs a *past* sandbox date's fixtures
(get_results(), status=FINISHED) and terminal progress output; the real
live scheduler path (scheduler_wiring.py) passes neither and is unaffected.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.backend import recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.odds_api_client import NormalizedOdds, OddsAPIClient
from app.backend.football_data_competition_codes import FOOTBALL_DATA_CODE_BY_LEAGUE
from app.backend.odds_sport_keys import ODDS_SPORT_KEY_BY_COMPETITION
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendations import validate_and_degrade
from app.backend.sandbox_clock import sandbox_now
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
    # W151: fixtures a prior pass (boot pregenerate, or an earlier EOD run)
    # already generated with these exact odds -- distinct from `skipped`
    # (a real run_agent failure) so a batch-result log line tells the two
    # apart without digging through warnings.
    unchanged: int = 0


def odds_lookup(odds_events: list[NormalizedOdds]) -> dict[tuple[str, str], NormalizedOdds]:
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    return {(mapper.map_team(o.home_team), mapper.map_team(o.away_team)): o for o in odds_events}


def already_fresh(
    cache: RecommendationCache, match_id: str, date: str, agent_config_hash: str, odds: dict | None,
) -> bool:
    """W151: True when a cache entry already exists for this exact
    (match, agent_config_hash) with the same odds as `odds` -- shared by
    run_eod_batch (below) and t30_refresh.py's refresh_match_at_t30 so a
    fixture boot-pregenerate/an earlier EOD pass already handled isn't
    silently regenerated (and re-billed against the LLM) again for no
    reason. `odds` may be None (no odds found this pass) -- compared
    against RecommendationCache's own `odds={}` convention for a
    no-odds generation (see run_eod_batch's `odds or {}` below)."""
    cached = cache.get_latest(match_id, date, agent_config_hash)
    return cached is not None and cached.odds == (odds or {})


def _fixture_date(fixture: NormalizedMatch) -> str:
    return fixture.utc_date[:10]


def has_kicked_off(fixture: NormalizedMatch, now: datetime) -> bool:
    """True once kickoff has passed. Shared by run_eod_batch (below, used by
    both the nightly EOD job and main.py's boot-time pregenerate sweep) and
    t30_refresh.py -- a pre-match value-betting recommendation is meaningless
    once a match has actually started, since "current odds" past that point
    reflect in-game state, not a pre-match edge. Time-based rather than
    fixture.status-based deliberately: get_fixtures() (BUG-set found live)
    already had to chase down multiple provider-specific in-progress status
    spellings (IN_PLAY/PAUSED/LIVE) that don't fully overlap between
    football-data.org competitions -- kickoff-time comparison sidesteps that
    whole problem, and also stays correct for a fixture snapshot that's gone
    stale by the time a restart-catch-up run actually reaches it (status
    reflects whenever the fixture was fetched, not right now)."""
    kickoff = datetime.fromisoformat(fixture.utc_date.replace("Z", "+00:00"))
    return now >= kickoff


def matched_odds_event(
    fixture: NormalizedMatch, odds_by_teams: dict[tuple[str, str], NormalizedOdds]
) -> NormalizedOdds | None:
    """The raw matched NormalizedOdds (with event_id, W164), or None on no
    match -- match_odds() (below) builds on this for its own dict-shaped
    return; eod_batch.py's _generate_one uses this directly too, to get at
    event_id for get_event_odds() without a second canonical-name lookup."""
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    key = (mapper.map_team(fixture.home_team), mapper.map_team(fixture.away_team))
    return odds_by_teams.get(key)


def match_odds(
    fixture: NormalizedMatch, odds_by_teams: dict[tuple[str, str], NormalizedOdds]
) -> dict[str, float] | None:
    odds = matched_odds_event(fixture, odds_by_teams)
    if odds is None or odds.home_odds is None or odds.draw_odds is None or odds.away_odds is None:
        return None
    return {"home": odds.home_odds, "draw": odds.draw_odds, "away": odds.away_odds}


def add_secondary_odds(
    match_info: dict,
    odds: dict,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    fixture: NormalizedMatch,
    fixture_date: str,
    agent_config_hash: str,
    sport_key: str,
    odds_by_teams: dict[tuple[str, str], NormalizedOdds],
) -> None:
    """W164/W164a, shared by run_eod_batch (below) and t30_refresh.py's
    refresh_match_at_t30 -- both need the identical "fetch or reuse
    total_goals/btts odds" behavior, and having it live in only one of
    them is exactly how t30_refresh.py silently fell behind W164 the first
    time (T-30 refreshes kept regenerating on 1X2 odds alone, degrading
    straight back into the "20/24 live picks were draws" bug W164 fixed).

    Folds total_goals/btts odds into `match_info` (so the LLM prompt sees
    them, graph.py) and into `odds` (so already_fresh()'s dedup/freshness
    check also reacts to a secondary-market price move, not just a h2h
    move) -- reusing the prior cached secondary-market snapshot when h2h
    odds haven't moved since (W164a: get_event_odds() costs real Odds-API
    credits per call), fetching fresh only when h2h moved or nothing's
    cached yet. Mutates both `match_info` and `odds` in place; caller must
    already have set `match_info["odds"] = odds` and `odds` must already be
    the matched h2h dict (match_odds()'s return)."""
    cached_entry = cache.get_latest(fixture.match_id, fixture_date, agent_config_hash)
    h2h_unchanged = cached_entry is not None and {
        k: cached_entry.odds.get(k) for k in ("home", "draw", "away")
    } == odds
    already_checked_secondary = cached_entry is not None and (
        "total_goals" in cached_entry.odds or "btts" in cached_entry.odds
    )

    if h2h_unchanged and already_checked_secondary:
        if cached_entry.odds.get("total_goals"):
            match_info["total_goals_odds"] = cached_entry.odds["total_goals"]
        if cached_entry.odds.get("btts"):
            match_info["btts_odds"] = cached_entry.odds["btts"]
        odds["total_goals"] = cached_entry.odds.get("total_goals")
        odds["btts"] = cached_entry.odds.get("btts")
    else:
        odds_event = matched_odds_event(fixture, odds_by_teams)
        get_event_odds = getattr(odds_client, "get_event_odds", None)
        if get_event_odds is not None and odds_event is not None and odds_event.event_id:
            secondary = get_event_odds(sport_key=sport_key, event_id=odds_event.event_id)
            if secondary is not None:
                if secondary.total_goals:
                    match_info["total_goals_odds"] = secondary.total_goals
                if secondary.btts:
                    match_info["btts_odds"] = secondary.btts
                odds["total_goals"] = secondary.total_goals
                odds["btts"] = secondary.btts


async def run_eod_batch(
    fixtures_client: FootballDataClient,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    schedule_t30: Callable[[NormalizedMatch], None],
    date_str: str,
    concurrency: int = 5,
    fixtures: list[NormalizedMatch] | None = None,
    on_progress: Callable[[NormalizedMatch, str], None] | None = None,
    league: str = LEAGUE_CODE,
) -> EodBatchResult:
    """W62: `league` (defaults to `LEAGUE_CODE`/"E0", preserving every
    existing caller's exact behavior unchanged) tags every generated
    match_info and selects the matching Odds-API sport_key
    (ODDS_SPORT_KEY_BY_COMPETITION) -- lets the multi-competition scheduler
    orchestration (scheduler_wiring.py) call this once per competition
    instead of it being structurally single-league.

    W50: `fixtures`, when supplied, is used as-is and
    fixtures_client.get_fixtures() is never called. This matters because
    get_fixtures() only ever queries status=SCHEDULED -- fine for the real
    live scheduler path (this always runs for *tomorrow*, still-scheduled
    fixtures), but structurally unable to return anything for a date that's
    already in the past, which W50's sandbox precompute caller needs (a
    SANDBOX_DATE is always historical by construction). This is the exact
    same root cause W45 already fixed for /api/fixtures
    (main.py's _split_fixture_date_range) -- applied here to this module's
    own separate direct FootballDataClient usage, which W45 never touched.
    A sandbox caller sources its fixtures via get_results() (status=FINISHED)
    instead and hands them in through this parameter. Omitting `fixtures`
    (the default) preserves this function's exact pre-W50 behavior.
    `fixtures_client` itself goes otherwise unused in that branch -- it
    remains a required parameter only because the default/live path still
    needs it.

    `on_progress`, when supplied, is called once per fixture as its
    generation finishes (order not guaranteed -- fixtures generate
    concurrently, bounded by `concurrency`), with the fixture and one of
    "generated"/"skipped"/"unchanged" (W151 -- already_fresh() found a
    cache entry with identical odds, no LLM call made). A CLI progress
    hook for W50's sandbox precompute step; the real scheduler path never
    passes it."""
    if fixtures is None:
        # W76: keyed off `league` (already accepted below), not the E0-only
        # COMPETITION_CODE constant -- this fallback is dead in practice for
        # every real caller today (both the scheduler and sandbox precompute
        # always supply `fixtures` explicitly), but shouldn't silently fetch
        # the wrong competition's data if that ever changes.
        competition_code = FOOTBALL_DATA_CODE_BY_LEAGUE.get(league, COMPETITION_CODE)
        fixtures = fixtures_client.get_fixtures(competition_code=competition_code, date_from=date_str, date_to=date_str)

    # W58: explicit sport_key from the competition-id mapping, rather than
    # relying on get_odds()'s own "soccer_epl" default parameter.
    #
    # W54: a true one-day batch (the live scheduler; an exact-date sandbox
    # precompute) has every fixture dated exactly date_str -- preserve the
    # original single get_odds() call unchanged for that case. A sandbox
    # fallback-window batch (W51) can contain fixtures on other dates (even
    # several different ones), and date_str's own odds lookup would find
    # none of them -- fetch per distinct fixture date instead.
    fixture_dates = {_fixture_date(fixture) for fixture in fixtures}
    sport_key = ODDS_SPORT_KEY_BY_COMPETITION[league]
    if odds_client is None:
        odds_by_teams_by_date: dict[str, dict] = {}
    elif fixture_dates <= {date_str}:
        odds_by_teams_by_date = {date_str: odds_lookup(odds_client.get_odds(sport_key=sport_key) or [])}
    else:
        odds_by_teams_by_date = {
            fixture_date: odds_lookup(odds_client.get_odds(sport_key=sport_key, date=fixture_date) or [])
            for fixture_date in fixture_dates
        }

    agent_config_hash = compute_agent_config_hash(config)
    semaphore = asyncio.Semaphore(concurrency)
    result = EodBatchResult(fixtures=list(fixtures))

    async def _generate_one(fixture: NormalizedMatch) -> None:
        if has_kicked_off(fixture, sandbox_now(timezone.utc)):
            LOGGER.info(
                "EOD batch: skipping match_id=%s (%s v %s) -- kickoff already passed, a "
                "pre-match recommendation is no longer meaningful.",
                fixture.match_id, fixture.home_team, fixture.away_team,
            )
            result.skipped += 1
            if on_progress is not None:
                on_progress(fixture, "skipped")
            return

        fixture_date = _fixture_date(fixture)
        match_info = {
            "home_team": fixture.home_team, "away_team": fixture.away_team,
            "date": fixture_date, "league": league,
        }
        odds_by_teams = odds_by_teams_by_date.get(fixture_date, {})
        odds = match_odds(fixture, odds_by_teams)
        if odds is not None:
            match_info["odds"] = odds
            add_secondary_odds(
                match_info, odds, odds_client, cache, fixture, fixture_date,
                agent_config_hash, sport_key, odds_by_teams,
            )

        # W151: a prior pass (boot pregenerate, or tonight's own EOD run
        # catching a fixture pregenerate already covered) already produced
        # a recommendation with these exact odds -- nothing has changed,
        # so skip the redundant (real-money) LLM call. T-30 still gets
        # scheduled for this fixture regardless, unchanged below.
        if already_fresh(cache, fixture.match_id, fixture_date, agent_config_hash, odds):
            result.unchanged += 1
            if on_progress is not None:
                on_progress(fixture, "unchanged")
            return

        async with semaphore:
            try:
                raw = await asyncio.to_thread(recommendations.run_agent, match_info=match_info, config=config)
            except Exception as exc:
                LOGGER.warning(
                    "EOD batch: skipping match_id=%s (%s v %s): %s",
                    fixture.match_id, fixture.home_team, fixture.away_team, exc,
                )
                result.skipped += 1
                if on_progress is not None:
                    on_progress(fixture, "skipped")
                return

        degraded = validate_and_degrade(raw, fixture.home_team, fixture.away_team)
        cache.record_generation(
            match_id=fixture.match_id, date=fixture_date, agent_config_hash=agent_config_hash,
            odds=odds or {}, recommendation=degraded.model_dump(), triggered_by="scheduled",
        )
        result.generated += 1
        if on_progress is not None:
            on_progress(fixture, "generated")

    await asyncio.gather(*[_generate_one(fixture) for fixture in fixtures])

    now = sandbox_now(timezone.utc)
    for fixture in fixtures:
        # A T-30 job for a fixture that's already kicked off would fire
        # immediately via RecoverableScheduler's own catch-up path (its
        # trigger time is already in the past) -- straight back into the
        # same "recommendation for a live match" problem this function's
        # own has_kicked_off() check above just avoided.
        if not has_kicked_off(fixture, now):
            schedule_t30(fixture)

    return result
