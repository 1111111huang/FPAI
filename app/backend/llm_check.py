"""Soft reachability check for the configured LLM provider (W01).

Never raises -- a failed check produces a startup warning, not a hard
failure (per acceptance: recommendation generation is degraded, not the
whole backend)."""

from __future__ import annotations

import os

import requests

from src.agent.agent_config import AgentConfig

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def check_llm_reachable(config: AgentConfig, timeout: float = 2.0) -> bool:
    """Best-effort reachability check for config.provider.

    Ollama: a real probe against the local daemon's own API.
    Anthropic/DeepSeek: presence of the provider's own API key only -- a live
    call would spend real API credits just to check connectivity, not worth
    it for a soft startup check.
    """
    if config.provider == "ollama":
        try:
            response = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False
    if config.provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if config.provider == "deepseek":
        return bool(os.environ.get("DEEPSEEK_API_KEY"))
    return False
