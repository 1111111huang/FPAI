"""Regression tests for W15's data-source prerequisite: cold_start_risk,
feature_completeness, and unknown_team must be structured fields on the
final MatchRecommendation, populated deterministically from the forecast
tool's own diagnostics -- not left to the LLM's own prose in `limitations`,
which agent_v1.txt never asks for and cannot be relied on to include
correctly. Extracted from tool call results directly, so the value is
correct "regardless of what prediction_basis says" (or what the model
chooses to write), matching the same code-over-prompt philosophy as
A28/A29."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_config import AgentConfig
from src.agent.graph import _build_recommendation, _extract_forecast_diagnostics


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        model="stub-model", provider="ollama", temperature=0.0, max_tool_calls=5,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_value_edge=0.05,
        markets=["btts"], system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _forecast_tool_message(name: str, cold_start_risk: bool, feature_completeness: float, unknown_team: bool) -> ToolMessage:
    payload = {
        "forecast": {},
        "diagnostics": {"cold_start_risk": cold_start_risk, "feature_completeness": feature_completeness},
        "data_quality": {"prediction_basis": "team_history_and_market", "unknown_team": unknown_team},
    }
    return ToolMessage(content=json.dumps(payload), name=name, tool_call_id="1")


def test_extracts_diagnostics_from_forecast_league_tool_message():
    messages = [
        HumanMessage(content="analyse this match"),
        _forecast_tool_message("forecast_league", cold_start_risk=True, feature_completeness=0.62, unknown_team=True),
    ]
    result = _extract_forecast_diagnostics(messages)
    assert result == {"cold_start_risk": True, "feature_completeness": 0.62, "unknown_team": True}


def test_extracts_diagnostics_from_forecast_international_tool_message():
    messages = [
        _forecast_tool_message("forecast_international", cold_start_risk=False, feature_completeness=1.0, unknown_team=False),
    ]
    result = _extract_forecast_diagnostics(messages)
    assert result == {"cold_start_risk": False, "feature_completeness": 1.0, "unknown_team": False}


def test_defaults_when_no_forecast_tool_was_ever_called():
    messages = [HumanMessage(content="hello"), AIMessage(content="some text, no tools")]
    result = _extract_forecast_diagnostics(messages)
    assert result == {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}


def test_uses_the_most_recent_forecast_call_not_the_first():
    messages = [
        _forecast_tool_message("forecast_league", cold_start_risk=True, feature_completeness=0.5, unknown_team=True),
        AIMessage(content="", tool_calls=[{"name": "forecast_league", "args": {}, "id": "2"}]),
        _forecast_tool_message("forecast_league", cold_start_risk=False, feature_completeness=0.95, unknown_team=False),
    ]
    result = _extract_forecast_diagnostics(messages)
    assert result == {"cold_start_risk": False, "feature_completeness": 0.95, "unknown_team": False}


def test_ignores_non_forecast_tool_messages():
    messages = [
        ToolMessage(content="some web search result text", name="web_search", tool_call_id="1"),
    ]
    result = _extract_forecast_diagnostics(messages)
    assert result == {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}


def test_build_recommendation_enriches_even_when_llm_json_omits_the_fields():
    """Core acceptance: cold_start_risk must be True on the final
    recommendation even though the LLM's own JSON output never mentions it --
    the value comes from the tool call, not from what the model wrote."""
    llm_json = json.dumps({
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "direct_bet",
        "markets": [],
        "explanation": "Looks good.",
        "confidence": "high",
        "limitations": [],
        "prediction_basis": "team_history_and_market",
    })
    cfg = _make_config()
    messages = [
        SystemMessage(content="sys"),
        _forecast_tool_message("forecast_league", cold_start_risk=True, feature_completeness=0.4, unknown_team=False),
        AIMessage(content=llm_json),
    ]

    recommendation = _build_recommendation(
        text=llm_json, match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        messages=messages, config=cfg,
    )

    assert recommendation["cold_start_risk"] is True
    assert recommendation["feature_completeness"] == 0.4
    assert recommendation["overall"] == "direct_bet"  # rest of the LLM's answer preserved


def test_build_recommendation_enriches_the_parse_failure_fallback_too():
    cfg = _make_config()
    messages = [
        _forecast_tool_message("forecast_league", cold_start_risk=True, feature_completeness=0.3, unknown_team=True),
        AIMessage(content="not valid json at all"),
    ]

    recommendation = _build_recommendation(
        text="not valid json at all", match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        messages=messages, config=cfg,
    )

    assert recommendation["overall"] == "insufficient_data"
    assert recommendation["cold_start_risk"] is True
    assert recommendation["unknown_team"] is True
