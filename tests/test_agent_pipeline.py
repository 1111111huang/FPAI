"""Tests for the deterministic evidence pipeline nodes (A31/A32):
resolve_competition_node, research_node, and forecast_node run before the LLM
ever sees the match, replacing the old LLM-tool-choice path for competition
resolution, baseline research, and the ML forecast."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent import tools as agent_tools
from src.agent.pipeline import resolve_competition_node


@pytest.fixture(autouse=True)
def reset_snapshot_store():
    agent_tools._snapshot_store.set_mode("live")
    yield
    agent_tools._snapshot_store.set_mode("live")


def _base_state(**overrides) -> dict:
    state = {
        "messages": [],
        "match_info": {"home_team": "Man City", "away_team": "Arsenal", "date": "2026-06-21", "league": "E0"},
        "recommendation": None,
        "tool_call_count": 0,
        "competition_resolution": None,
        "research_evidence": None,
        "forecast_payload": None,
    }
    state.update(overrides)
    return state


def test_resolve_competition_node_uses_real_registry_for_known_league():
    result = resolve_competition_node(_base_state())
    assert result["competition_resolution"]["tier"] == "competition_specific"
    assert result["competition_resolution"]["recommended_tool"] == "forecast_league"


def test_resolve_competition_node_defaults_general_purpose_when_no_league_supplied():
    state = _base_state(match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21"})
    result = resolve_competition_node(state)
    assert result["competition_resolution"] == {
        "competition": None,
        "tier": "general_purpose",
        "recommended_tool": "forecast_international",
    }


def test_resolve_competition_node_goes_through_snapshot_store():
    with patch(
        "src.agent.tools._resolve_competition_impl",
        wraps=agent_tools._resolve_competition_impl,
    ) as spy:
        resolve_competition_node(_base_state())
    spy.assert_called_once_with(competition_or_league="E0")
