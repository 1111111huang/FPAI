"""FastAPI backend for the FPAI web app (W01).

Imports src/agent and src/forecast directly as Python libraries -- no MCP,
no subprocess, no HTTP hop to src/mcp_server.py (that server is for external
third-party agent consumers, not this first-party app)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.backend import recommendations
from app.backend.llm_check import check_llm_reachable
from app.backend.recommendations import MatchRecommendationOut, RecommendationRequest, validate_and_degrade
from src.agent.agent_config import AgentConfig
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


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


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/recommendations")
async def create_recommendation(request: RecommendationRequest) -> MatchRecommendationOut:
    # run_agent is a real ~10-30s synchronous call (LLM + Tavily) -- must run
    # off the event loop or it blocks every other request.
    raw = await run_in_threadpool(recommendations.run_agent, request.to_match_info())
    return validate_and_degrade(raw)
