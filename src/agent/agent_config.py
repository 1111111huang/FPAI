from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

_REQUIRED = {
    "model", "provider", "temperature", "max_tool_calls",
    "min_odds_threshold", "max_odds_threshold", "min_value_edge", "markets", "system_prompt_version",
}
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "agent_config.yaml"


@dataclass
class AgentConfig:
    model: str
    provider: Literal["ollama", "anthropic", "groq", "gemini", "deepseek"]
    temperature: float
    max_tool_calls: int
    min_odds_threshold: float
    max_odds_threshold: float
    min_value_edge: float
    markets: list[str]
    system_prompt_version: str

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
        return cls(**{k: data[k] for k in _REQUIRED})

    @classmethod
    def default(cls) -> "AgentConfig":
        return cls.from_yaml(_DEFAULT_CONFIG)
