"""BUG-054: config/prompts/agent_v1*.txt's stated value-edge/odds thresholds
must match AgentConfig's real enforced values -- confirmed live that
config/agent_config.yaml's 2026-08-28 tightening of min/max_odds_threshold
(1.2/11.0 -> 1.71/4.0, production only) was never reflected in the prompt
text: the LLM kept reasoning under the stale 1.2/11.0 bounds, correctly
called a market direct_bet by its own (outdated) stated rule, and the
code-enforced downgrade (using the real, tighter config) then silently
rejected it -- a self-contradictory recommendation whose own explanation
argued for a bet the app refused to place."""
from __future__ import annotations

from dataclasses import replace

from src.agent.agent_config import AgentConfig
from src.agent.graph import _load_system_prompt

_BASE = AgentConfig(
    model="m", provider="deepseek", temperature=0.0, max_tool_calls=10,
    min_odds_threshold=1.2, max_odds_threshold=11.0,
    min_conditional_odds_threshold=1.5, min_value_edge=0.05,
    markets=["result_3way"], system_prompt_version="v1",
)


def test_prompt_reflects_configs_own_odds_thresholds_not_a_hardcoded_copy():
    prompt = _load_system_prompt(_BASE)
    assert "odds below 1.2 or above 11" in prompt

    tightened = replace(_BASE, min_odds_threshold=1.71, max_odds_threshold=4.0)
    prompt = _load_system_prompt(tightened)
    assert "odds below 1.71 or above 4" in prompt
    assert "odds below 1.2 or above 11" not in prompt


def test_prompt_reflects_configs_own_value_edge_and_conditional_floor():
    prompt = _load_system_prompt(_BASE)
    assert "value_edge >= 0.05" in prompt
    assert 'conditional" below 1.5' in prompt

    tuned = replace(_BASE, min_value_edge=0.08, min_conditional_odds_threshold=1.71)
    prompt = _load_system_prompt(tuned)
    assert "value_edge >= 0.08" in prompt
    assert 'conditional" below 1.71' in prompt


def test_prompt_mentions_conditional_ceiling_only_when_config_sets_one():
    prompt = _load_system_prompt(_BASE)  # unbounded default (float('inf'))
    assert "never use it above" not in prompt

    capped = replace(_BASE, max_conditional_odds_threshold=4.0)
    prompt = _load_system_prompt(capped)
    assert "never use it above 4" in prompt


def test_prompt_mentions_draw_value_edge_floor_only_when_config_sets_one():
    prompt = _load_system_prompt(_BASE)  # default None
    assert "draw selection specifically needs" not in prompt

    guarded = replace(_BASE, min_value_edge_result_3way_draw=0.15)
    prompt = _load_system_prompt(guarded)
    assert "draw selection specifically needs value_edge >= 0.15" in prompt


def test_no_leftover_placeholder_tokens_in_any_posture_prompt():
    for version in ("v1", "v1_aggressive", "v1_balanced", "v1_conservative"):
        prompt = _load_system_prompt(replace(_BASE, system_prompt_version=version))
        assert "{{" not in prompt, f"unsubstituted placeholder left in {version}"
