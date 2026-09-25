"""Relocated to src/agent/agent_config_hash.py (A127) -- src/agent needed this
value and must not be depended on by app.backend in reverse. Re-exported here
so every existing caller (eod_batch.py, recommendation_cache.py, main.py,
t30_refresh.py, ...) keeps working with no import-path change."""

from __future__ import annotations

from src.agent.agent_config_hash import compute_agent_config_hash

__all__ = ["compute_agent_config_hash"]
