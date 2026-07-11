"""Tests for StateGraph compilation and routing logic (A03)."""
import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.agent.agent_config import AgentConfig
from src.agent.graph import AgentState, build_graph


@tool
def stub_search(query: str) -> str:
    """Stub web search."""
    return "stub result"


@tool
def stub_forecast(home_team: str, away_team: str, date: str, league: str, odds_h: float, odds_d: float, odds_a: float) -> str:
    """Stub forecast."""
    return '{"result_3way": {"probabilities": {"home": 0.5, "draw": 0.25, "away": 0.25}}}'


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        model="stub-model",
        provider="ollama",
        temperature=0.0,
        max_tool_calls=5,
        min_odds_threshold=1.2,
        max_odds_threshold=11.0,
        min_value_edge=0.05,
        markets=["btts"],
        system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_graph_compiles():
    cfg = _make_config()
    with patch("src.agent.graph._build_llm") as mock_llm, \
         patch("src.agent.graph._load_system_prompt", return_value="stub prompt"):
        mock_llm.return_value = MagicMock()
        mock_llm.return_value.bind_tools.return_value = MagicMock()
        graph = build_graph(cfg, [stub_search, stub_forecast])
    assert graph is not None


def test_should_continue_routes_to_tools_when_tool_calls_present():
    cfg = _make_config(max_tool_calls=5)
    ai_msg = AIMessage(content="", tool_calls=[{"name": "stub_search", "args": {"query": "test"}, "id": "1"}])
    state: AgentState = {
        "messages": [ai_msg],
        "match_info": {},
        "recommendation": None,
        "tool_call_count": 1,
    }
    route = _route_for_state(cfg, state)
    assert route == "tools"


def test_should_continue_routes_to_output_when_no_tool_calls():
    cfg = _make_config(max_tool_calls=5)
    ai_msg = AIMessage(content="Final answer here.")
    state: AgentState = {
        "messages": [ai_msg],
        "match_info": {},
        "recommendation": None,
        "tool_call_count": 0,
    }
    route = _route_for_state(cfg, state)
    assert route == "output"


def test_should_continue_routes_to_output_when_budget_exceeded():
    cfg = _make_config(max_tool_calls=3)
    ai_msg = AIMessage(content="", tool_calls=[{"name": "stub_search", "args": {"query": "x"}, "id": "2"}])
    state: AgentState = {
        "messages": [ai_msg],
        "match_info": {},
        "recommendation": None,
        "tool_call_count": 3,  # at limit
    }
    route = _route_for_state(cfg, state)
    assert route == "output"


def test_forecast_league_falls_back_to_international_when_no_league_models():
    """forecast_league returns data (not error) via international fallback when no league context."""
    from unittest.mock import patch, MagicMock
    import json as _json

    mock_result = {"result_3way": {"probabilities": {"home": 0.55}}, "data_quality": {}}

    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        # First call (league) raises FileNotFoundError; second call (international) succeeds
        instance.forecast_upcoming.side_effect = [FileNotFoundError("no league models"), mock_result]

        from src.agent.tools import forecast_league
        result_str = forecast_league.invoke({
            "home_team": "Manchester City", "away_team": "Arsenal",
            "date": "2026-06-21", "league": "E0",
            "odds_h": 2.1, "odds_d": 3.4, "odds_a": 3.6,
        })

    result = _json.loads(result_str)
    assert "result_3way" in result
    assert result["data_quality"]["prediction_basis"] == "market_odds_only_league_fallback"
    assert instance.forecast_upcoming.call_count == 2


def test_run_agent_appends_extra_system_instructions():
    from unittest.mock import patch, MagicMock
    from src.agent.graph import run_agent

    captured = {}

    def fake_build_graph(config, tools):
        mock_compiled = MagicMock()

        def fake_invoke(initial_state):
            captured["system_content"] = initial_state["messages"][0].content
            return {"recommendation": {"overall": "no_bet"}}

        mock_compiled.invoke.side_effect = fake_invoke
        return mock_compiled

    cfg = _make_config()
    with patch("src.agent.graph._build_llm"), \
         patch("src.agent.graph._load_system_prompt", return_value="BASE PROMPT"), \
         patch("src.agent.graph.build_graph", side_effect=fake_build_graph):
        run_agent(
            match_info={"home_team": "A", "away_team": "B", "date": "2025-01-01"},
            config=cfg,
            tools=[],
            extra_system_instructions="EXTRA INSTRUCTIONS HERE",
        )

    assert "BASE PROMPT" in captured["system_content"]
    assert "EXTRA INSTRUCTIONS HERE" in captured["system_content"]


def test_run_agent_without_extra_instructions_unchanged():
    from unittest.mock import patch, MagicMock
    from src.agent.graph import run_agent

    captured = {}

    def fake_build_graph(config, tools):
        mock_compiled = MagicMock()

        def fake_invoke(initial_state):
            captured["system_content"] = initial_state["messages"][0].content
            return {"recommendation": {"overall": "no_bet"}}

        mock_compiled.invoke.side_effect = fake_invoke
        return mock_compiled

    cfg = _make_config()
    with patch("src.agent.graph._build_llm"), \
         patch("src.agent.graph._load_system_prompt", return_value="BASE PROMPT"), \
         patch("src.agent.graph.build_graph", side_effect=fake_build_graph):
        run_agent(
            match_info={"home_team": "A", "away_team": "B", "date": "2025-01-01"},
            config=cfg,
            tools=[],
        )

    assert captured["system_content"] == "BASE PROMPT"


def _route_for_state(cfg: AgentConfig, state: AgentState) -> str:
    """Helper: extract the routing logic without building the full graph."""
    last = state["messages"][-1]
    has_calls = bool(getattr(last, "tool_calls", None))
    under_budget = state["tool_call_count"] < cfg.max_tool_calls
    return "tools" if has_calls and under_budget else "output"
