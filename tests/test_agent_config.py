"""Tests for AgentConfig (A02)."""
import pytest
import yaml
from pathlib import Path

from src.agent.agent_config import AgentConfig

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "agent_config.yaml"


def test_default_config_loads():
    cfg = AgentConfig.default()
    assert cfg.provider in ("ollama", "anthropic", "groq", "gemini", "deepseek")
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


def test_default_config_has_a66_conditional_odds_floor():
    """A66: a 'conditional' market below this (decimal, roughly -200
    American) is too short a price for 'wait for it to improve' to be a
    realistic strategy -- downgraded to no_bet instead."""
    cfg = AgentConfig.default()
    assert cfg.min_conditional_odds_threshold == 1.5


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
        "min_conditional_odds_threshold": 1.5,
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


def test_conservative_posture_config_loads_with_own_prompt_and_edge():
    cfg = AgentConfig.from_yaml("config/agent_config_conservative.yaml")
    assert cfg.system_prompt_version == "v1_conservative"
    assert cfg.min_value_edge == 0.06


def test_balanced_posture_config_loads_with_own_prompt_and_edge():
    cfg = AgentConfig.from_yaml("config/agent_config_balanced.yaml")
    assert cfg.system_prompt_version == "v1_balanced"
    assert cfg.min_value_edge == 0.05


def test_aggressive_posture_config_loads_with_own_prompt_and_edge():
    cfg = AgentConfig.from_yaml("config/agent_config_aggressive.yaml")
    assert cfg.system_prompt_version == "v1_aggressive"
    assert cfg.min_value_edge == 0.04


def test_all_three_posture_configs_keep_every_other_field_identical_to_default():
    """2026-08-22 (A71): model/provider are now a deliberate posture difference
    -- DeepSeek showed a strong decline bias not reproduced by gemini/anthropic
    on identical inputs (agent_user_stories.md), so the three posture configs
    moved to gemini-3.6-flash while the shared default config is untouched.
    Everything else (temperature, thresholds, markets) still must match."""
    default = AgentConfig.default()
    for posture in ("conservative", "balanced", "aggressive"):
        cfg = AgentConfig.from_yaml(f"config/agent_config_{posture}.yaml")
        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-3.6-flash"
        assert cfg.temperature == default.temperature
        assert cfg.max_tool_calls == default.max_tool_calls
        assert cfg.min_odds_threshold == default.min_odds_threshold
        assert cfg.max_odds_threshold == default.max_odds_threshold
        assert cfg.min_conditional_odds_threshold == default.min_conditional_odds_threshold
        assert cfg.markets == default.markets
