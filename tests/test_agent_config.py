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
    """A29 widened bounds to [1.2, 11.0]; direct user request (2026-08-28)
    tightened the live production default specifically to [1.71, 4.0]
    (-140/+300 American) -- see test_default_config_has_2026_08_28_odds_range
    below. Only asserting the general shape here now (min >= 1.0, max >
    min); the exact historical [1.2, 11.0] band lives on unchanged in
    test_agent_odds_bounds.py's own default-argument tests."""
    cfg = AgentConfig.default()
    assert cfg.min_odds_threshold >= 1.0
    assert cfg.max_odds_threshold > cfg.min_odds_threshold


def test_default_config_has_2026_08_28_odds_range():
    """Direct user request (2026-08-28): tighten the live production
    default's recommendable odds range to -140/+300 American (decimal
    [1.71, 4.0]) -- both direct_bet's own bounds (A29) and, symmetrically,
    conditional's floor/ceiling (A66 + the new ceiling) all raised/capped
    to the same band, so nothing outside it is ever recommended at all."""
    cfg = AgentConfig.default()
    assert cfg.min_odds_threshold == 1.71
    assert cfg.max_odds_threshold == 4.0
    assert cfg.min_conditional_odds_threshold == 1.71
    assert cfg.max_conditional_odds_threshold == 4.0


def test_default_config_has_a66_conditional_odds_floor():
    """A66: a 'conditional' market below this is too short a price for
    'wait for it to improve' to be a realistic strategy -- downgraded to
    no_bet instead. Exact value covered by
    test_default_config_has_2026_08_28_odds_range above; this just checks
    the floor is still a real, positive number less than the ceiling."""
    cfg = AgentConfig.default()
    assert cfg.min_conditional_odds_threshold > 1.0
    assert cfg.min_conditional_odds_threshold < cfg.max_conditional_odds_threshold


def test_posture_and_backtest_configs_keep_the_original_a29_a66_odds_bounds():
    """The 2026-08-28 tightening above applies to the live production
    default only (direct user choice) -- every other config must be
    completely unaffected, keeping A29/A66's original [1.2, 11.0]/1.5
    band and the (new) unbounded conditional ceiling default."""
    for name in ("conservative", "balanced", "aggressive", "deepseek"):
        cfg = AgentConfig.from_yaml(f"config/agent_config_{name}.yaml")
        assert cfg.min_odds_threshold == 1.2
        assert cfg.max_odds_threshold == 11.0
        assert cfg.min_conditional_odds_threshold == 1.5
        assert cfg.max_conditional_odds_threshold == float("inf")


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
    Everything else (temperature, max_tool_calls, markets) still must match.

    Odds thresholds are the one deliberate exception since 2026-08-28: the
    live production default's own bounds were tightened to -140/+300
    American (see test_default_config_has_2026_08_28_odds_range) while the
    posture configs were explicitly left at A29/A66's original wider band
    (see test_posture_and_backtest_configs_keep_the_original_a29_a66_odds_bounds)
    -- reversing this file's own prior 2026-08-21 "safety rails, not a
    posture dial" framing for these three fields specifically, by direct
    user choice. Not asserted equal-to-default here anymore."""
    default = AgentConfig.default()
    for posture in ("conservative", "balanced", "aggressive"):
        cfg = AgentConfig.from_yaml(f"config/agent_config_{posture}.yaml")
        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-3.6-flash"
        assert cfg.temperature == default.temperature
        assert cfg.max_tool_calls == default.max_tool_calls
        assert cfg.markets == default.markets
