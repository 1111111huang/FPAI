"""Tests for AgentConfig (A02)."""
import pytest
import yaml
from pathlib import Path

from src.agent.agent_config import AgentConfig

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "agent_config.yaml"


def test_default_config_loads():
    cfg = AgentConfig.default()
    assert cfg.provider in ("ollama", "anthropic", "groq", "gemini")
    assert cfg.temperature >= 0.0
    assert cfg.max_tool_calls > 0
    assert cfg.min_odds_threshold >= 1.0
    assert cfg.max_odds_threshold > cfg.min_odds_threshold
    assert isinstance(cfg.markets, list)
    assert len(cfg.markets) > 0


def test_default_config_has_a29_widened_odds_bounds():
    """A29: bounds widened to [1.2, 11.0], replacing the old 2.0-only floor."""
    cfg = AgentConfig.default()
    assert cfg.min_odds_threshold == 1.2
    assert cfg.max_odds_threshold == 11.0


def test_from_yaml_missing_file():
    with pytest.raises(FileNotFoundError):
        AgentConfig.from_yaml("/nonexistent/path/config.yaml")


def test_from_yaml_missing_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump({"model": "x", "provider": "ollama"}))
    with pytest.raises(ValueError, match="Missing required fields"):
        AgentConfig.from_yaml(bad)


def test_from_yaml_roundtrip(tmp_path):
    data = {
        "model": "llama3.2:3b",
        "provider": "ollama",
        "temperature": 0.0,
        "max_tool_calls": 5,
        "min_odds_threshold": 1.2,
        "max_odds_threshold": 11.0,
        "min_value_edge": 0.05,
        "markets": ["btts", "result_3way"],
        "system_prompt_version": "v1",
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump(data))
    cfg = AgentConfig.from_yaml(cfg_file)
    assert cfg.model == "llama3.2:3b"
    assert cfg.max_tool_calls == 5
    assert cfg.markets == ["btts", "result_3way"]
