"""FastAPI backend for the FPAI web app (W01).

Imports src/agent and src/forecast directly as Python libraries -- no MCP,
no subprocess, no HTTP hop to src/mcp_server.py (that server is for external
third-party agent consumers, not this first-party app)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import dataclasses
from datetime import date, datetime, timezone
import os
import time
from typing import Callable

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from starlette.concurrency import run_in_threadpool

load_dotenv()

from app.backend import bets, eod_batch, recommendations, sandbox_clock
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.bet_tracker import BetTracker
from app.backend.bets import BetFromRecommendationRequest, BetManualRequest, BetOut
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.odds_sport_keys import DEFAULT_SPORT_KEY, ODDS_SPORT_KEY_BY_COMPETITION
from app.backend.sweden_fixtures_client import (
    SwedenFixturesClient,
    historical_results_from_raw_matches,
)
from app.backend.llm_check import check_llm_reachable
from app.backend.recommendation_cache import RecommendationCache
from app.backend.bet_stats import compute_bet_stats
from app.backend.recommendations import MatchRecommendationOut, RecommendationRequest, validate_and_degrade
from app.backend.scheduler import JobRunLog, RecoverableScheduler
from app.backend.scheduler_wiring import build_odds_client, register_eod_job
from app.backend.settlement import settle_open_bets
from src.agent.agent_config import AgentConfig
from src.tools.data_tools import get_data_freshness
from src.tools.model_tools import get_model_status
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_SANDBOX_JOB_RUNS_DB_PATH = sandbox_clock.sandbox_scoped_path("job_runs.db")

_fixtures_client: FootballDataClient | None = None


def get_fixtures_client() -> FootballDataClient:
    """FastAPI dependency -- overridden in tests via patching this function."""
    global _fixtures_client
    if _fixtures_client is None:
        _fixtures_client = FootballDataClient(api_key=os.environ.get("FOOTBALL_DATA_API_KEY", ""))
    return _fixtures_client


_sweden_fixtures_client: SwedenFixturesClient | None = None


def get_sweden_fixtures_client() -> SwedenFixturesClient:
    """W57: Sweden (Allsvenskan) fixtures/results, sourced from The Odds API
    rather than football-data.org -- W55 confirmed football-data.org's free
    tier has no Allsvenskan coverage at all. FastAPI dependency -- overridden
    in tests via patching this function, same pattern as get_fixtures_client."""
    global _sweden_fixtures_client
    if _sweden_fixtures_client is None:
        _sweden_fixtures_client = SwedenFixturesClient(api_key=os.environ.get("ODDS_API_KEY", ""))
    return _sweden_fixtures_client


# W52: football-data.org's free tier (~10 req/min) is shared across every
# request through the single `_fixtures_client` singleton above. Three
# independent frontend call sites (Dashboard, Match Explorer, manual bet
# form) each fetch fixtures fresh on every mount with no de-duplication, so
# normal navigation within a session can burst well past the budget and trip
# a 429. This is a short-lived request-dedup cache (not a data-freshness
# cache) -- 60s is short enough that staleness is a non-issue, long enough to
# absorb that exact repeated-navigation pattern. Module-level state, matching
# the existing `_fixtures_client` singleton pattern already in this file.
_FIXTURE_CACHE_TTL_SECONDS = 60.0
_fixture_cache: dict[tuple[str, str | None, str | None], tuple[float, list[NormalizedMatch]]] = {}

# Code review follow-up (post-21e6bf9): a bare cache dict only de-dupes
# requests that arrive *after* an earlier one has already completed and
# populated the cache -- two genuinely concurrent cache-miss requests for
# the same key (e.g. React StrictMode's double-effect-invocation in dev, or
# two browser tabs loading at once) would both race past the cache check
# and both hit the upstream client for real. This tracks the in-flight
# asyncio.Task for each key so a second concurrent request awaits the same
# task instead of starting its own.
_fixture_cache_pending: dict[tuple[str, str | None, str | None], "asyncio.Task[list[NormalizedMatch]]"] = {}


def _fixture_cache_now() -> float:
    """Split out from _cached_fixture_call so tests can monkeypatch the
    clock -- mirrors this file's existing `_current_real_date()` patchable-
    function pattern -- to deterministically exercise TTL expiry without a
    real 60-second sleep."""
    return time.monotonic()


async def _fetch_and_cache_fixtures(
    cache_key: tuple[str, str | None, str | None],
    fetch: Callable[..., list[NormalizedMatch]],
    fetch_kwargs: dict[str, str | None],
) -> list[NormalizedMatch]:
    """Runs the actual upstream call (in the threadpool) and populates the
    cache on success. A requests.HTTPError raised by `fetch` (e.g.
    football-data.org's 429 rate-limit response) is turned into a clean 503
    HTTPException rather than left to propagate as an unhandled 500 -- and
    is NOT cached, so it doesn't wrongly suppress the next genuine
    request."""
    try:
        matches = await run_in_threadpool(fetch, **fetch_kwargs)
    except requests.exceptions.HTTPError as exc:
        LOGGER.warning("Upstream fixture provider call failed for %s: %s", cache_key, exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                "Fixture data is temporarily unavailable (the upstream provider is rate-limited "
                "or unreachable). Please try again in a minute."
            ),
        ) from exc

    _fixture_cache[cache_key] = (_fixture_cache_now() + _FIXTURE_CACHE_TTL_SECONDS, matches)
    return matches


async def _cached_fixture_call(
    cache_key: tuple[str, str | None, str | None],
    fetch: Callable[..., list[NormalizedMatch]],
    **fetch_kwargs: str | None,
) -> list[NormalizedMatch]:
    """Look up `cache_key` in the module-level TTL cache; on a miss, either
    join an already-in-flight call for the same key (`_fixture_cache_pending`)
    or kick off a new one. The pending task is registered *before* the first
    `await` inside it runs -- since asyncio is single-threaded/cooperative,
    a second concurrent call checking `_fixture_cache_pending` between that
    registration and the task's completion is guaranteed to see it, closing
    the race a bare cache dict would leave open."""
    cached = _fixture_cache.get(cache_key)
    if cached is not None:
        expires_at, matches = cached
        if expires_at > _fixture_cache_now():
            return matches

    pending = _fixture_cache_pending.get(cache_key)
    if pending is None:
        pending = asyncio.ensure_future(_fetch_and_cache_fixtures(cache_key, fetch, fetch_kwargs))
        _fixture_cache_pending[cache_key] = pending

    try:
        return await pending
    finally:
        if _fixture_cache_pending.get(cache_key) is pending:
            del _fixture_cache_pending[cache_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AgentConfig.default()
    if not check_llm_reachable(config):
        LOGGER.warning(
            "LLM provider '%s' (model=%s) does not appear reachable at startup -- "
            "recommendation generation will fail until this is resolved.",
            config.provider, config.model,
        )

    # W08/W09/W10: off by default. Registering the EOD job runs
    # RecoverableScheduler's restart/catch-up check immediately (by design,
    # W08's whole point) -- if wired in unconditionally, every test file's
    # `with TestClient(app)` would non-deterministically trigger a real live
    # EOD batch (real fixtures/odds/LLM calls) depending on whatever wall-clock
    # time tests happen to run at relative to the 23:00 NY trigger. Gated
    # behind an explicit opt-in until this wave is verified live and the app
    # is actually going to production -- see app_user_stories.md W08 sequencing
    # note ("built last, right before going live").
    scheduler: RecoverableScheduler | None = None
    if os.environ.get("ENABLE_SCHEDULER", "").lower() in ("1", "true", "yes"):
        # W29: sandbox mode routes JobRunLog to a scratch path so it never touches real dev data.
        run_log = JobRunLog(db_path=_SANDBOX_JOB_RUNS_DB_PATH) if sandbox_clock.is_sandbox_mode() else None
        scheduler = RecoverableScheduler(run_log=run_log)
        register_eod_job(
            scheduler,
            fixtures_client=get_fixtures_client(),
            odds_client=build_odds_client(),
            cache=recommendations.get_cache(),
            config=config,
            # W62: Sweden (Allsvenskan) processed alongside EPL in the same
            # nightly EOD batch / T-30 refresh, sourced from The Odds API
            # (W55/W57) rather than football-data.org.
            sweden_fixtures_client=get_sweden_fixtures_client(),
        )
        scheduler.start()
        LOGGER.info("W08/W09/W10 scheduler started (ENABLE_SCHEDULER=1).")
    else:
        LOGGER.info("Scheduler disabled -- set ENABLE_SCHEDULER=1 to enable the EOD/T-30 pipeline.")

    yield

    if scheduler is not None:
        scheduler.shutdown()


app = FastAPI(title="FPAI Web App Backend", lifespan=lifespan)

# D7: standard two-process local dev -- Next.js dev server + uvicorn, talking
# over HTTP with CORS rather than a shared process.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/sandbox/status")
def get_sandbox_status() -> dict:
    """W27: lets the frontend and test scripts introspect the active
    sandbox date instead of each needing their own access to the env vars."""
    return sandbox_clock.sandbox_status()


@app.get("/api/status")
def get_status() -> dict:
    """W17: system status surface (data staleness + current model
    selections) -- pure reuse of already-exposed src/tools functions, no
    new engine work."""
    return {"data_freshness": get_data_freshness(), "model_status": get_model_status()}


def _current_real_date() -> date:
    """Genuine wall-clock UTC 'today' -- deliberately NOT
    sandbox_clock.sandbox_now(). football-data.org's own SCHEDULED/FINISHED
    match status reflects whether a match has actually been played in the
    real world, independent of what date SANDBOX_MODE/SANDBOX_DATE is
    pretending "today" is, so the split below must be computed against real
    time, not the app's sandbox-overridable clock."""
    return datetime.now(timezone.utc).date()


def _split_fixture_date_range(
    date_from: str | None, date_to: str | None, today: date,
) -> tuple[tuple[str, str] | None, tuple[str | None, str | None] | None]:
    """Splits a requested [date_from, date_to] range into a (results_range,
    fixtures_range) pair relative to real wall-clock `today`, so the caller
    can source the past portion from get_results() (status=FINISHED) and the
    future portion from get_fixtures() (status=SCHEDULED) -- either element
    is None if that side contributes nothing. A range that includes `today`
    queries `today` from *both* sides, since a same-day match may already be
    FINISHED or still SCHEDULED depending on kickoff time relative to right
    now -- there is no way to know which without asking both.

    Either bound omitted (or unparseable) falls back to the pre-W45
    single-call behavior: everything goes through fixtures_range unchanged,
    since a half-open/unbounded range can't be meaningfully split against a
    boundary, and this preserves however football-data.org's API itself
    already interprets missing date params by default.
    """
    if date_from is None or date_to is None:
        return None, (date_from, date_to)
    try:
        parsed_from = date.fromisoformat(date_from)
        parsed_to = date.fromisoformat(date_to)
    except ValueError:
        return None, (date_from, date_to)

    if parsed_to < today:
        return (date_from, date_to), None
    if parsed_from > today:
        return None, (date_from, date_to)

    # Range includes today: query today from *both* sides. A match kicking
    # off today may already be FINISHED (e.g. an early kickoff, viewed later
    # the same day) or still SCHEDULED, depending on kickoff time relative to
    # right now -- excluding today from get_results() would silently drop
    # already-finished same-day matches, which is exactly the gap this fix
    # exists to close (found in code review of the first version of this
    # function, which stopped the results side at yesterday).
    return (date_from, today.isoformat()), (today.isoformat(), date_to)


@app.get("/api/fixtures")
async def get_fixtures(date_from: str | None = None, date_to: str | None = None) -> list[NormalizedMatch]:
    """Thin wrapper over W05's FootballDataClient -- gives the frontend a real
    fixture list to render (Dashboard/Match Explorer). Still used directly
    even after W09 shipped -- W09 populates the recommendation cache in the
    background, but the frontend still needs a live fixture list to render
    cards for, cached recommendation or not.

    W45: a date range entirely in the past needs get_results()
    (status=FINISHED) -- get_fixtures() (status=SCHEDULED) structurally can
    never return anything for a range that's already happened, regardless of
    real data availability. A range spanning real "today" is split and the
    two sides merged, past-then-future (already chronological, since the
    split is on whole days and each side is independently date-ordered by
    the API)."""
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    results_range, fixtures_range = _split_fixture_date_range(date_from, date_to, _current_real_date())

    def _tag(matches: list[NormalizedMatch], competition: str) -> list[NormalizedMatch]:
        # W64: explicit tagging here (not inside each client's own
        # normalize function) is deliberate -- this is the one place every
        # return path through this endpoint passes through, real or
        # test-mocked, so it's the only place a tag is guaranteed to stick.
        return [dataclasses.replace(m, competition=competition) for m in matches]

    matches: list[NormalizedMatch] = []
    if results_range is not None:
        past_from, past_to = results_range
        matches += _tag(
            await _cached_fixture_call(
                ("results", past_from, past_to), client.get_results, date_from=past_from, date_to=past_to
            ),
            "E0",
        )
        # W71: sourced from raw_matches directly, not sweden_client.get_results()
        # (The Odds API's /scores endpoint can only see the last few real
        # days -- it has no arbitrary-historical-date capability at all,
        # unlike football-data.org's get_results() for E0). Still
        # cache-keyed as "results_swe" -- same TTL-cache slot as before,
        # just backed by a different underlying source.
        matches += _tag(
            await _cached_fixture_call(
                ("results_swe", past_from, past_to),
                historical_results_from_raw_matches, date_from=past_from, date_to=past_to,
            ),
            "SWE",
        )
    if fixtures_range is not None:
        future_from, future_to = fixtures_range
        matches += _tag(
            await _cached_fixture_call(
                ("fixtures", future_from, future_to), client.get_fixtures, date_from=future_from, date_to=future_to
            ),
            "E0",
        )
        matches += _tag(
            await _cached_fixture_call(
                ("fixtures_swe", future_from, future_to), sweden_client.get_fixtures, date_from=future_from, date_to=future_to
            ),
            "SWE",
        )
    return matches


def _fetch_odds_for_manual_request(request: RecommendationRequest, league: str | None) -> dict[str, float] | None:
    """W49: best-effort odds lookup for the manual 'regenerate now' path,
    reusing eod_batch.py's exact odds_client + match_odds team-matching
    logic (BUG-015's canonical-name matching included) rather than
    reimplementing it. Unlike run_eod_batch's build_odds_client() call
    (fetched once for a whole batch of same-day fixtures), this fetches
    fresh odds per manual request -- acceptable here since it's a single
    user-triggered click, not a batch loop.

    `league` is the already-gated competition id (match_info.get("league"),
    W03's gate_league output) -- W58: picks the matching Odds-API sport_key
    instead of relying on get_odds()'s own "soccer_epl" default, so a
    Swedish fixture's fetch actually queries Sweden's odds feed rather than
    silently querying EPL's.

    Every failure mode degrades to None (no odds attached) rather than
    raising: no odds client configured (build_odds_client() returns None
    when no ODDS_API_KEY/sandbox override is set), no matching odds event
    for this fixture, or the odds client call itself raising (e.g. a
    network error). This must never turn a previously-working no-odds
    request into a 500 -- it's strictly additive over the pre-W49
    behavior.

    Cost note (code review, W49): in production this draws from the same
    CreditCounter-guarded monthly Odds API budget the scheduled EOD/T-30
    jobs use (build_odds_client() -> the same on-disk counter path) --
    manual regenerate previously cost zero Odds API credits. would_exceed()
    still protects the safety margin (degrades to no-odds, never errors),
    but frequent manual clicks now measurably compete with the scheduler
    for the same budget within a given month -- worth watching if usage
    patterns change, not something this fix can size in advance."""
    try:
        odds_client = build_odds_client()
        if odds_client is None:
            return None
        sport_key = ODDS_SPORT_KEY_BY_COMPETITION.get(league, DEFAULT_SPORT_KEY)
        odds_events = odds_client.get_odds(sport_key=sport_key)
        odds_by_teams = eod_batch.odds_lookup(odds_events or [])
        fixture = NormalizedMatch(
            match_id=request.effective_match_id(), utc_date="", status="",
            home_team=request.home_team, away_team=request.away_team,
            home_goals=None, away_goals=None,
        )
        return eod_batch.match_odds(fixture, odds_by_teams)
    except Exception:
        LOGGER.warning(
            "Manual recommendation odds fetch failed for %s v %s (%s)",
            request.home_team, request.away_team, request.date, exc_info=True,
        )
        return None


@app.post("/api/recommendations")
async def create_recommendation(
    request: RecommendationRequest,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> MatchRecommendationOut:
    """The explicit 'regenerate now' escape hatch (W11) -- always calls the
    agent and writes the result into the cache, tagged manual_regenerate so
    it's distinguishable from a scheduled (W09/W10) generation.

    W49: unlike the scheduled EOD/T-30 pipeline (eod_batch.py), this path
    used to never fetch odds itself -- match_info["odds"] was purely
    caller-supplied, and neither MatchCard nor MatchAnalysisPage ever
    populate it, leaving the agent to rely on its own web_search (which
    structurally can't succeed for a historical/sandboxed match, and
    frequently fails for real matches too). Now mirrors run_eod_batch's
    odds-fetch-before-run_agent sequence whenever the caller didn't already
    supply odds explicitly -- an explicit request.odds always wins."""
    match_info = request.to_match_info()
    if request.odds is None:
        # build_odds_client()/get_odds() make a real synchronous HTTP or DB
        # call (HistoricalOddsClient/OddsAPIClient) -- off the event loop,
        # same convention as the run_agent call directly below.
        fetched_odds = await run_in_threadpool(_fetch_odds_for_manual_request, request, match_info.get("league"))
        if fetched_odds is not None:
            match_info["odds"] = fetched_odds

    # run_agent is a real ~10-30s synchronous call (LLM + Tavily) -- must run
    # off the event loop or it blocks every other request.
    raw = await run_in_threadpool(recommendations.run_agent, match_info)
    result = validate_and_degrade(raw, request.home_team, request.away_team)

    cache.record_generation(
        match_id=request.effective_match_id(),
        date=request.date,
        agent_config_hash=compute_agent_config_hash(AgentConfig.default()),
        # W49: record what was actually used for generation, not just what
        # the caller explicitly supplied -- match_info["odds"] reflects the
        # server-side fetch too. Recording request.odds alone would persist
        # {} even when real odds were fetched and used, spuriously tricking
        # t30_refresh.py's "odds unchanged, skip regeneration" dedup check
        # into always seeing a change on the next comparison.
        odds=match_info.get("odds", {}),
        recommendation=result.model_dump(),
        triggered_by="manual_regenerate",
    )
    return result


@app.get("/api/recommendations/{match_id}")
async def get_cached_recommendation(
    match_id: str,
    date: str,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> MatchRecommendationOut:
    """Reads exclusively from the cache (W11) -- never calls run_agent. The
    normal path for an already-scheduled fixture; a cache miss means nothing
    has generated a recommendation for this match/date yet.

    BUG-028: routes through validate_and_degrade (no home_team/away_team --
    this endpoint only has match_id/date, so the match-mismatch check is
    skipped, but per-market validation still gracefully drops a malformed
    market) instead of a raw MatchRecommendationOut.model_validate(), which
    raised an uncaught ValidationError (a 500) for any pre-existing cached
    row using a market/selection value that predates BUG-027's Literal
    constraints -- routine for local-model-hallucinated rows."""
    entry = cache.get_latest(match_id, date, compute_agent_config_hash(AgentConfig.default()))
    if entry is None:
        raise HTTPException(status_code=404, detail="No cached recommendation for this match/date yet.")
    return validate_and_degrade(entry.recommendation)


@app.post("/api/bets/from-recommendation")
async def create_bet_from_recommendation(
    request: BetFromRecommendationRequest,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
) -> BetOut:
    """Every field but stake is locked -- derived from the recommendation
    snapshot itself, which is also stored verbatim (recommendations aren't
    reproducible run-to-run, agent_techspec.md sec18.6)."""
    try:
        resolved = bets.resolve_from_recommendation(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    bet = tracker.create_bet(
        match_id=resolved["match_id"], date=resolved["date"],
        home_team=resolved["home_team"], away_team=resolved["away_team"],
        market=resolved["market"], selection=resolved["selection"],
        odds=resolved["odds"], stake=resolved["stake"],
        source="from_recommendation", recommendation_snapshot=request.recommendation,
    )
    return BetOut.from_bet(bet)


@app.post("/api/bets/manual")
async def create_bet_manual(
    request: BetManualRequest,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
) -> BetOut:
    """User-provided fields, but match_id must be a resolved fixture reference
    (enforced by the frontend's Match Explorer search, not free-typed team
    names) -- Pydantic requires it non-empty at minimum."""
    bet = tracker.create_bet(
        match_id=request.match_id, date=request.date,
        home_team=request.home_team, away_team=request.away_team,
        market=request.market, selection=request.selection,
        odds=request.odds, stake=request.stake,
        source="manual", recommendation_snapshot=None,
    )
    return BetOut.from_bet(bet)


@app.get("/api/bets")
async def list_bets(tracker: BetTracker = Depends(bets.get_bet_tracker)) -> list[BetOut]:
    return [BetOut.from_bet(bet) for bet in tracker.list_bets()]


@app.get("/api/bets/stats")
async def get_bet_stats(tracker: BetTracker = Depends(bets.get_bet_tracker)) -> dict:
    """W14: ROI/hit-rate/bankroll summary, computed fresh over settled bets
    on every call -- no persisted running total."""
    return compute_bet_stats(tracker.list_bets())


@app.post("/api/bets/settle-open")
async def settle_open(tracker: BetTracker = Depends(bets.get_bet_tracker)) -> list[BetOut]:
    """On-demand settlement trigger (W13) -- intentionally not folded into
    W08's scheduler; bet settlement isn't tied to recommendation-generation
    timing the way W09/W10 are. Reuses get_fixtures_client (W05's
    FootballDataClient) since results/fixtures share the same API and rate
    limit budget. W57: also consults get_sweden_fixtures_client so a
    Swedish bet's match_id (unknown to football-data.org) can settle too."""
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    settled = await run_in_threadpool(settle_open_bets, tracker, client, sweden_client)
    return [BetOut.from_bet(bet) for bet in settled]
