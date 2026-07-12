from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.agent_config import AgentConfig
from src.agent.schema import MatchRecommendation, RecommendationParseError, extract_recommendation
from src.utils.logger import get_logger

_FORECAST_TOOL_NAMES = ("forecast_league", "forecast_international")

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"
_LOG = get_logger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    match_info: dict
    recommendation: dict | None
    tool_call_count: int


def _build_llm(config: AgentConfig) -> Any:
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.model, temperature=config.temperature)
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.model, temperature=config.temperature)
    raise ValueError(f"Unknown provider: {config.provider!r}")


def _load_system_prompt(config: AgentConfig) -> str:
    path = _PROMPTS_DIR / f"agent_{config.system_prompt_version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text()


def _extract_forecast_diagnostics(messages: list[BaseMessage]) -> dict:
    """W15: pull cold_start_risk/feature_completeness/unknown_team from the
    most recent forecast_league/forecast_international tool result, rather
    than trusting the LLM to transcribe them into its own JSON -- these are
    engine-computed facts, not something agent_v1.txt even asks the model to
    report, so relying on prose in `limitations` would be unreliable at best.
    """
    for message in reversed(messages):
        if not isinstance(message, ToolMessage) or message.name not in _FORECAST_TOOL_NAMES:
            continue
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        diagnostics = payload.get("diagnostics") or {}
        data_quality = payload.get("data_quality") or {}
        return {
            "cold_start_risk": bool(diagnostics.get("cold_start_risk", False)),
            "feature_completeness": diagnostics.get("feature_completeness"),
            "unknown_team": bool(data_quality.get("unknown_team", False)),
        }
    return {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}


def _build_recommendation(text: str, match_info: dict, messages: list[BaseMessage], config: AgentConfig) -> dict:
    """Extract the LLM's MatchRecommendation JSON (or fall back to an
    insufficient_data placeholder on parse failure), then enrich it with
    forecast-tool diagnostics regardless of which path was taken."""
    try:
        recommendation = extract_recommendation(
            text,
            min_odds_threshold=config.min_odds_threshold,
            max_odds_threshold=config.max_odds_threshold,
        )
        _LOG.info("output_node | parse=success | overall=%s", recommendation.get("overall"))
    except RecommendationParseError as exc:
        _LOG.warning("output_node | parse=failed | reason=%s", exc)
        recommendation = {
            "match": match_info,
            "overall": "insufficient_data",
            "markets": [],
            "explanation": f"Agent did not produce a parseable recommendation. Raw output: {text[:800]}",
            "confidence": "low",
            "limitations": ["Agent output could not be parsed as a structured recommendation"],
            "prediction_basis": "unknown",
        }
    recommendation.update(_extract_forecast_diagnostics(messages))
    return recommendation


def build_graph(config: AgentConfig, tools: list):
    """Compile and return the LangGraph StateGraph for the betting agent."""
    llm = _build_llm(config)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        tool_calls = getattr(response, "tool_calls", []) or []
        new_count = state["tool_call_count"] + len(tool_calls)
        if tool_calls:
            _LOG.info("agent_node | tool_calls=%s | count_after=%d", [tc["name"] for tc in tool_calls], new_count)
        else:
            content = response.content if isinstance(response.content, str) else str(response.content)
            _LOG.info("agent_node | no tool_calls | raw_output_length=%d", len(content))
            _LOG.debug("agent_node | raw_output=%s", content)
        return {"messages": [response], "tool_call_count": new_count}

    def should_continue(state: AgentState) -> Literal["tools", "output"]:
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        under_budget = state["tool_call_count"] < config.max_tool_calls
        route = "tools" if has_calls and under_budget else "output"
        _LOG.info("should_continue | has_tool_calls=%s | tool_call_count=%d | route=%s", has_calls, state["tool_call_count"], route)
        return route

    def output_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)

        if not text.strip():
            # Budget was exhausted — last message is a tool call with no text content.
            # Make one final synthesis call (no tools) so the model can produce its JSON.
            _LOG.info("output_node | empty_content | forcing_synthesis_call")
            synthesis_prompt = (
                "You have reached the tool call limit. "
                "Based on all the information gathered above, output your final JSON recommendation now. "
                "Include all required fields: match, overall, markets, explanation, confidence, limitations, prediction_basis."
            )
            synthesis_response = llm.invoke(state["messages"] + [HumanMessage(content=synthesis_prompt)])
            text = synthesis_response.content if isinstance(synthesis_response.content, str) else str(synthesis_response.content)
            _LOG.info("output_node | synthesis_length=%d | synthesis_output=%s", len(text), text)

        _LOG.info("output_node | raw_output_length=%d", len(text))
        _LOG.info("output_node | raw_output=%s", text)
        recommendation = _build_recommendation(text, state["match_info"], state["messages"], config)
        return {"recommendation": recommendation}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output", output_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "output": "output"})
    graph.add_edge("tools", "agent")
    graph.add_edge("output", END)

    return graph.compile()


def run_agent(
    match_info: dict,
    config: AgentConfig | None = None,
    tools: list | None = None,
    extra_system_instructions: str | None = None,
) -> MatchRecommendation:
    """Run the betting agent for a single match and return a structured recommendation.

    Args:
        match_info: Dict with keys: home_team, away_team, date, and optionally league.
        config: AgentConfig instance. Loads from config/agent_config.yaml if None.
        tools: List of LangChain tools. Loads default tools if None.
        extra_system_instructions: Appended to the loaded system prompt. Used by
            agent-snapshot (A11) to inject snapshot-collection-only rules (e.g.
            "ignore any result mentioning a final score") without forking the
            whole prompt file.
    """
    if config is None:
        config = AgentConfig.default()
    if tools is None:
        from src.agent.tools import get_default_tools
        tools = get_default_tools()

    system_prompt = _load_system_prompt(config)
    if extra_system_instructions:
        system_prompt = f"{system_prompt}\n\n{extra_system_instructions}"

    prompt = (
        f"Analyse the upcoming match: {match_info['home_team']} vs {match_info['away_team']}"
        f" on {match_info['date']}"
    )
    if match_info.get("league"):
        prompt += f" in league {match_info['league']}"
    odds = match_info.get("odds")
    if odds:
        prompt += (
            f". Bookmaker odds: home={odds['home']}, draw={odds['draw']}, away={odds['away']}. "
            "Use these exact odds_h/odds_d/odds_a values when calling the forecast tool."
        )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ],
        "match_info": match_info,
        "recommendation": None,
        "tool_call_count": 0,
    }

    compiled = build_graph(config, tools)
    result = compiled.invoke(initial_state)
    return result["recommendation"]
