"""FastAPI backend for the FPAI web app (W01).

Imports src/agent and src/forecast directly as Python libraries -- no MCP,
no subprocess, no HTTP hop to src/mcp_server.py (that server is for external
third-party agent consumers, not this first-party app)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.backend.llm_check import check_llm_reachable
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
