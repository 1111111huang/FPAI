"""Stable hash of an AgentConfig's tunable fields (W11), used as part of the
recommendation cache key -- a config change (model, prompt version,
thresholds, ...) naturally produces a new cache key rather than colliding
with entries generated under a different configuration."""

from __future__ import annotations

import hashlib
import json

from src.agent.agent_config import AgentConfig


def compute_agent_config_hash(config: AgentConfig) -> str:
    payload = json.dumps(
        {
            "model": config.model,
            "provider": config.provider,
            "temperature": config.temperature,
            "max_tool_calls": config.max_tool_calls,
            "min_odds_threshold": config.min_odds_threshold,
            "max_odds_threshold": config.max_odds_threshold,
            "min_conditional_odds_threshold": config.min_conditional_odds_threshold,
            "max_conditional_odds_threshold": config.max_conditional_odds_threshold,
            "min_value_edge": config.min_value_edge,
            "markets": config.markets,
            "system_prompt_version": config.system_prompt_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
