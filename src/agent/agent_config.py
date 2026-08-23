from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

_REQUIRED = {
    "model", "provider", "temperature", "max_tool_calls",
    "min_odds_threshold", "max_odds_threshold", "min_conditional_odds_threshold",
    "min_value_edge", "markets", "system_prompt_version",
}
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "agent_config.yaml"


@dataclass
class AgentConfig:
    model: str
    provider: Literal["ollama", "anthropic", "groq", "gemini", "deepseek", "qwen"]
    temperature: float
    max_tool_calls: int
    min_odds_threshold: float
    max_odds_threshold: float
    # A66: a 'conditional' market's current_odds below this is too short a
    # price for "wait for it to improve" to be a realistic strategy (or, in
    # the degenerate case, not a real price at all -- e.g. 0.0) -- see
    # src/agent/schema.py's _downgrade_conditional_below_floor.
    min_conditional_odds_threshold: float
    min_value_edge: float
    markets: list[str]
    system_prompt_version: str
    # Posture-driven (aggressive): strips the forecast's entropy/uncertainty
    # diagnostic from the evidence shown to the LLM. Root cause: this field is
    # real data (see src/forecast/uncertainty.py), not prompt-invented, and
    # gives the model a ready-made "high uncertainty" reason to decline a
    # qualifying value_edge no matter how the prompt is worded. Optional --
    # defaults to False (unchanged) for any config that doesn't set it.
    suppress_forecast_uncertainty: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Agent config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        missing = _REQUIRED - data.keys()
        if missing:
            raise ValueError(f"Missing required fields in agent config: {sorted(missing)}")
        kwargs = {k: data[k] for k in _REQUIRED}
        kwargs["suppress_forecast_uncertainty"] = data.get("suppress_forecast_uncertainty", False)
        return cls(**kwargs)

    @classmethod
    def default(cls) -> "AgentConfig":
        return cls.from_yaml(_DEFAULT_CONFIG)
