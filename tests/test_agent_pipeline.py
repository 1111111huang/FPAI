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


from src.agent.pipeline import _parse_odds_from_search_text, research_node


def test_parse_odds_from_search_text_finds_three_plausible_numbers():
    text = "Best odds: Man City 1.45, Draw 4.50, Arsenal 7.00 at time of writing."
    assert _parse_odds_from_search_text(text) == {"home": 1.45, "draw": 4.50, "away": 7.00}


def test_parse_odds_from_search_text_returns_none_with_fewer_than_three_numbers():
    assert _parse_odds_from_search_text("Man City are favourites at 1.45.") is None


def test_parse_odds_from_search_text_returns_none_for_empty_text():
    assert _parse_odds_from_search_text("") is None
    assert _parse_odds_from_search_text(None) is None


def test_research_node_runs_availability_and_form_searches_only_when_odds_supplied():
    with patch("src.agent.tools._dated_web_search", side_effect=["injury text", "form text"]) as mock_search:
        state = _base_state(match_info={
            "home_team": "A", "away_team": "B", "date": "2026-06-21",
            "odds": {"home": 2.0, "draw": 3.0, "away": 3.5},
        })
        result = research_node(state)

    assert result["research_evidence"]["availability"] == "injury text"
    assert result["research_evidence"]["form_context"] == "form text"
    assert result["research_evidence"]["odds_verification"] is None
    assert mock_search.call_count == 2


def test_research_node_also_runs_odds_search_when_caller_supplied_no_odds():
    with patch(
        "src.agent.tools._dated_web_search",
        side_effect=["injury text", "form text", "Man City 1.45 Draw 4.50 Arsenal 7.00"],
    ) as mock_search:
        state = _base_state(match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21"})
        result = research_node(state)

    assert mock_search.call_count == 3
    odds_verification = result["research_evidence"]["odds_verification"]
    assert odds_verification["parsed_odds"] == {"home": 1.45, "draw": 4.50, "away": 7.00}
