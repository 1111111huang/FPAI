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

from app.backend import recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.llm_check import check_llm_reachable
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendations import MatchRecommendationOut, RecommendationRequest, validate_and_degrade
from src.agent.agent_config import AgentConfig
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
    yield


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


@app.get("/api/fixtures")
async def get_fixtures(date_from: str | None = None, date_to: str | None = None) -> list[NormalizedMatch]:
    """Thin wrapper over W05's FootballDataClient -- gives the frontend a real
    fixture list to render (Dashboard/Match Explorer). No dedicated story
    covers this narrowly; W09 (built last) will eventually populate fixtures
    via the recommendation cache instead, but the frontend needs something
    real to fetch today."""
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
    it's distinguishable from a scheduled generation (once W09 exists)."""
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
