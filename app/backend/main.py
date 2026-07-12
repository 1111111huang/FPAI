"""FastAPI backend for the FPAI web app (W01).

Imports src/agent and src/forecast directly as Python libraries -- no MCP,
no subprocess, no HTTP hop to src/mcp_server.py (that server is for external
third-party agent consumers, not this first-party app)."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

load_dotenv()

from app.backend import bets, recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.bet_tracker import BetTracker
from app.backend.bets import BetFromRecommendationRequest, BetManualRequest, BetOut
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.llm_check import check_llm_reachable
from app.backend.recommendation_cache import RecommendationCache
from app.backend.bet_stats import compute_bet_stats
from app.backend.recommendations import MatchRecommendationOut, RecommendationRequest, validate_and_degrade
from app.backend.scheduler import RecoverableScheduler
from app.backend.scheduler_wiring import build_odds_client, register_eod_job
from app.backend.settlement import settle_open_bets
from src.agent.agent_config import AgentConfig
from src.tools.data_tools import get_data_freshness
from src.tools.model_tools import get_model_status
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_fixtures_client: FootballDataClient | None = None


def get_fixtures_client() -> FootballDataClient:
    """FastAPI dependency -- overridden in tests via patching this function."""
    global _fixtures_client
    if _fixtures_client is None:
        _fixtures_client = FootballDataClient(api_key=os.environ.get("FOOTBALL_DATA_API_KEY", ""))
    return _fixtures_client


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
        scheduler = RecoverableScheduler()
        register_eod_job(
            scheduler,
            fixtures_client=get_fixtures_client(),
            odds_client=build_odds_client(),
            cache=recommendations.get_cache(),
            config=config,
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


@app.get("/api/status")
def get_status() -> dict:
    """W17: system status surface (data staleness + current model
    selections) -- pure reuse of already-exposed src/tools functions, no
    new engine work."""
    return {"data_freshness": get_data_freshness(), "model_status": get_model_status()}


@app.get("/api/fixtures")
async def get_fixtures(date_from: str | None = None, date_to: str | None = None) -> list[NormalizedMatch]:
    """Thin wrapper over W05's FootballDataClient -- gives the frontend a real
    fixture list to render (Dashboard/Match Explorer). Still used directly
    even after W09 shipped -- W09 populates the recommendation cache in the
    background, but the frontend still needs a live fixture list to render
    cards for, cached recommendation or not."""
    client = get_fixtures_client()
    fixtures = await run_in_threadpool(client.get_fixtures, date_from=date_from, date_to=date_to)
    return fixtures


@app.post("/api/recommendations")
async def create_recommendation(
    request: RecommendationRequest,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> MatchRecommendationOut:
    """The explicit 'regenerate now' escape hatch (W11) -- always calls the
    agent and writes the result into the cache, tagged manual_regenerate so
    it's distinguishable from a scheduled (W09/W10) generation."""
    # run_agent is a real ~10-30s synchronous call (LLM + Tavily) -- must run
    # off the event loop or it blocks every other request.
    raw = await run_in_threadpool(recommendations.run_agent, request.to_match_info())
    result = validate_and_degrade(raw)

    cache.record_generation(
        match_id=request.effective_match_id(),
        date=request.date,
        agent_config_hash=compute_agent_config_hash(AgentConfig.default()),
        odds=request.odds.model_dump() if request.odds else {},
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
    has generated a recommendation for this match/date yet."""
    entry = cache.get_latest(match_id, date, compute_agent_config_hash(AgentConfig.default()))
    if entry is None:
        raise HTTPException(status_code=404, detail="No cached recommendation for this match/date yet.")
    return MatchRecommendationOut.model_validate(entry.recommendation)


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
    limit budget."""
    client = get_fixtures_client()
    settled = await run_in_threadpool(settle_open_bets, tracker, client)
    return [BetOut.from_bet(bet) for bet in settled]
