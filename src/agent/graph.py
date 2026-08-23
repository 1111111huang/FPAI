from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.agent_config import AgentConfig
from src.agent.pipeline import forecast_node, lessons_node, research_node, resolve_competition_node
from src.agent.schema import (
    MatchRecommendation,
    MatchRecommendationModel,
    RecommendationParseError,
    extract_recommendation,
)
from src.utils.logger import get_logger

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"
_LOG = get_logger(__name__)
# A75: request timeout for the OpenAI-compatible provider branches (deepseek,
# qwen) -- ChatOpenAI's own default is no timeout at all (see _build_llm).
# 90s comfortably covers real observed per-match latency (single digits to
# ~40s on a slow response) with margin, while still failing a genuinely
# hung/blackholed connection fast enough for _invoke_with_retry's 3 attempts
# to matter instead of each one blocking indefinitely.
_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 90


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    match_info: dict
    recommendation: dict | None
    tool_call_count: int
    competition_resolution: dict | None
    research_evidence: dict | None
    forecast_payload: dict | None


def _build_llm(config: AgentConfig) -> Any:
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.model, temperature=config.temperature)
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.model, temperature=config.temperature)
    if config.provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.model, temperature=config.temperature)
    if config.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.model, temperature=config.temperature)
    if config.provider == "deepseek":
        # A42: DeepSeek exposes an OpenAI-compatible chat-completions endpoint,
        # so it needs no dedicated langchain-deepseek package -- ChatOpenAI
        # pointed at DeepSeek's base_url is the standard integration path.
        # Reads DEEPSEEK_API_KEY the same way every other provider here reads
        # its own *_API_KEY (langchain's own env-var convention, not custom).
        # A75: `timeout=` (ChatOpenAI's real field is `request_timeout`, this
        # is its alias) -- ChatOpenAI's own default is None (no timeout at
        # all). Found live: a real backtest run hit a bad connection to a
        # provider on this same OpenAI-compatible code path and hung for
        # over 9 hours on a single match -- _invoke_with_retry (above) only
        # retries on a raised exception, so a request that never returns
        # (rather than erroring) blocks forever, retries or not.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.model, temperature=config.temperature, base_url="https://api.deepseek.com", api_key=os.environ.get("DEEPSEEK_API_KEY"), timeout=_OPENAI_COMPATIBLE_TIMEOUT_SECONDS)
    if config.provider == "qwen":
        # A74: QwenCloud (docs.qwencloud.com) also exposes an OpenAI-compatible
        # endpoint via Alibaba Cloud's DashScope backend -- same ChatOpenAI +
        # custom base_url pattern as the deepseek branch above, no dedicated
        # langchain-qwen package needed. Confirmed live: qwen3.8-max and
        # qwen3.5-flash both respond correctly at this endpoint with a
        # DASHSCOPE_API_KEY (QwenCloud's own key, prefixed sk-ws-, not a
        # generic Alibaba Cloud Model Studio workspace-scoped key -- that's a
        # different product with a different, account-specific base_url).
        # A75: same missing-timeout gap as the deepseek branch -- see its
        # comment above. This is the branch that actually hit it live.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.model, temperature=config.temperature, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", api_key=os.environ.get("DASHSCOPE_API_KEY"), timeout=_OPENAI_COMPATIBLE_TIMEOUT_SECONDS)
    raise ValueError(f"Unknown provider: {config.provider!r}")


def _load_system_prompt(config: AgentConfig) -> str:
    path = _PROMPTS_DIR / f"agent_{config.system_prompt_version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text()


def _extract_text(content: str | list) -> str:
    """BUG-021: some provider integrations (e.g. langchain-google-genai)
    return AIMessage.content as a list of content-block dicts carrying
    per-block metadata (Gemini's 'extras.signature' thought-signature blob)
    rather than a plain string. str()-ing that list renders the metadata
    too, corrupting downstream JSON extraction. Extract only each block's
    'text' field -- never fall back to stringifying the whole structure
    unless it's a genuinely unrecognized shape."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _invoke_with_retry(runnable: Any, messages: list, attempts: int = 3) -> Any:
    """W151/A64: agent_node's calls to the LLM provider were the one
    external call left in this graph with no error handling at all --
    every other tool call already degrades on failure instead of raising
    (_web_search_impl/A53, _forecast_league_impl, resolve_competition).
    A transient provider error (timeout/rate limit/5xx) here used to
    propagate all the way to eod_batch.py's per-match try/except, silently
    skipping the whole match until the next scheduled EOD/T-30 window --
    sometimes days away for an early fixture. Retries the identical call
    up to `attempts` times; the last exception is re-raised unchanged if
    every attempt fails, so callers see exactly the same failure mode as
    before, just less often.
    # ponytail: no backoff between attempts -- batch concurrency is
    # already bounded (eod_batch.py's own semaphore), so this isn't
    # hammering the provider. Add exponential backoff if a real outage
    # (not a one-off blip) starts exhausting all 3 attempts in practice."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return runnable.invoke(messages)
        except Exception as exc:
            last_exc = exc
            _LOG.warning("llm_invoke_failed | attempt=%d/%d | %s", attempt + 1, attempts, exc)
    raise last_exc


_CONFIDENCE_STEPS = ["high", "medium", "low"]
_NO_RESULTS_MARKERS = ("No results found.", "TOOL_PERMANENTLY_UNAVAILABLE")


def _extract_forecast_diagnostics(forecast_payload: dict | None) -> dict:
    """A31: pull cold_start_risk/feature_completeness/unknown_team from the
    deterministic forecast_node's own payload, rather than trusting the LLM
    to transcribe them into its own JSON -- these are engine-computed facts,
    not something agent_v1.txt even asks the model to report."""
    if not forecast_payload or "error" in forecast_payload:
        return {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}
    diagnostics = forecast_payload.get("diagnostics") or {}
    data_quality = forecast_payload.get("data_quality") or {}
    return {
        "cold_start_risk": bool(diagnostics.get("cold_start_risk", False)),
        "feature_completeness": diagnostics.get("feature_completeness"),
        "unknown_team": bool(data_quality.get("unknown_team", False)),
    }


def _apply_a30_backstop(recommendation: dict, forecast_payload: dict | None) -> dict:
    """A30: a recommendation can never claim more evidence than actually
    exists. Keyed purely on the structural presence of a successful
    forecast_payload -- never on parsing the LLM's own explanation text.
    Should be unreachable in the current graph (output_node's early return
    already handles a missing/failed forecast before this ever runs) --
    kept as defense-in-depth against a future graph change reintroducing the
    original Burnley/Bournemouth bug class."""
    if forecast_payload and "error" not in forecast_payload:
        return recommendation
    reason = (forecast_payload or {}).get("error", "no forecast payload available")
    limitations = list(recommendation.get("limitations") or [])
    if recommendation.get("overall") != "insufficient_data":
        limitations.append(f"Forced insufficient_data: {reason}")
    recommendation["overall"] = "insufficient_data"
    recommendation["markets"] = []
    recommendation["prediction_basis"] = "unknown"
    recommendation["limitations"] = limitations
    return recommendation


def _has_no_research_coverage(text: str | None) -> bool:
    if not text:
        return True
    return any(text.startswith(marker) for marker in _NO_RESULTS_MARKERS)


def _apply_research_coverage_downgrade(recommendation: dict, research_evidence: dict | None) -> dict:
    """A32: missing availability/form research coverage downgrades confidence
    by one step per missing category (capped at 'low') and names the gap,
    rather than letting a recommendation claim full confidence off partial
    evidence the LLM never actually received. Odds coverage is handled
    separately (forecast_node blocks the whole recommendation, not just
    confidence, when odds are unavailable) so it's not checked here."""
    if recommendation.get("overall") == "insufficient_data":
        return recommendation
    evidence = research_evidence or {}
    gaps = []
    if _has_no_research_coverage(evidence.get("availability")):
        gaps.append("availability/injury")
    if _has_no_research_coverage(evidence.get("form_context")):
        gaps.append("recent form")
    if not gaps:
        return recommendation
    current = recommendation.get("confidence", "medium")
    idx = _CONFIDENCE_STEPS.index(current) if current in _CONFIDENCE_STEPS else 1
    recommendation["confidence"] = _CONFIDENCE_STEPS[min(idx + len(gaps), len(_CONFIDENCE_STEPS) - 1)]
    limitations = list(recommendation.get("limitations") or [])
    limitations.append(f"Research coverage gap: no results for {', '.join(gaps)}.")
    recommendation["limitations"] = limitations
    return recommendation


def _finalize_recommendation(
    recommendation: dict, forecast_payload: dict | None, research_evidence: dict | None,
) -> dict:
    """Shared normalization pass applied regardless of how `recommendation`
    was produced (schema-constrained structured output, A37, or free-text
    regex extraction below) -- diagnostics/backstop/downgrade are all
    deterministic, pipeline-state-derived, and must apply identically either
    way (A30/A31/A32)."""
    recommendation.update(_extract_forecast_diagnostics(forecast_payload))
    recommendation = _apply_a30_backstop(recommendation, forecast_payload)
    recommendation = _apply_research_coverage_downgrade(recommendation, research_evidence)
    return recommendation


def _structured_output(llm, messages: list) -> dict | None:
    """A37: request a schema-constrained final answer directly from the
    provider via LangChain's with_structured_output(), instead of relying on
    the LLM to freely write JSON that then gets regex-extracted by
    extract_recommendation() below. When the provider/binding actually
    supports it, this guarantees field names/types/enums match
    MatchRecommendationModel -- it cannot return the wrong shape the way
    free text can.

    Returns None on any failure or unexpected return shape (no structured-
    output support, a network error, a loosely-typed integration that
    doesn't raise but also doesn't return the real Pydantic instance) --
    callers fall back to the pre-existing free-text path in that case, so
    this is strictly additive, never a regression."""
    try:
        result = llm.with_structured_output(MatchRecommendationModel).invoke(messages)
    except Exception:
        _LOG.warning("output_node | structured_output_unavailable", exc_info=True)
        return None
    if not isinstance(result, MatchRecommendationModel):
        return None
    return result.model_dump()


def _build_recommendation(
    text: str,
    match_info: dict,
    forecast_payload: dict | None,
    research_evidence: dict | None,
    config: AgentConfig,
) -> dict:
    """Extract the LLM's MatchRecommendation JSON (or fall back to an
    insufficient_data placeholder on parse failure), then enrich/normalize it
    against the deterministic pipeline's own evidence -- never the LLM's
    prose (A30/A31/A32)."""
    try:
        recommendation = extract_recommendation(
            text,
            min_odds_threshold=config.min_odds_threshold,
            max_odds_threshold=config.max_odds_threshold,
            min_conditional_odds_threshold=config.min_conditional_odds_threshold,
            min_value_edge=config.min_value_edge,
            home_team=match_info.get("home_team"),
            away_team=match_info.get("away_team"),
        )
        _LOG.info("output_node | parse=success | overall=%s", recommendation.get("overall"))
    except RecommendationParseError as exc:
        _LOG.warning("output_node | parse=failed | reason=%s", exc)
        recommendation = {
            "match": match_info,
            "overall": "insufficient_data",
            "markets": [],
            "explanation": [f"Agent did not produce a parseable recommendation. Raw output: {text[:800]}"],
            "confidence": "low",
            "limitations": ["Agent output could not be parsed as a structured recommendation"],
            "prediction_basis": "unknown",
        }
    return _finalize_recommendation(recommendation, forecast_payload, research_evidence)


def route_after_forecast(state: AgentState) -> Literal["lessons", "output"]:
    """A31/A33: a successful forecast proceeds to the lessons node (which
    itself no-ops outside live mode) before the LLM ever sees the match; a
    failed/impossible forecast (no odds from any source, or a tool error)
    routes straight to output. Module-level (not nested in build_graph, unlike
    its sibling node/route closures) since it's a pure function of state with
    no dependency on config/llm/tools -- this lets tests call the real
    function directly instead of hand-duplicating its logic, which is what
    let this function's success target silently drift out of sync with a
    test during A33's development (route target changed from "agent" to
    "lessons" but a duplicated test helper kept asserting the old value)."""
    payload = state.get("forecast_payload")
    succeeded = bool(payload) and "error" not in payload
    route = "lessons" if succeeded else "output"
    _LOG.info("route_after_forecast | succeeded=%s | route=%s", succeeded, route)
    return route


def build_graph(config: AgentConfig, tools: list):
    """Compile and return the LangGraph StateGraph for the betting agent.

    A31/A32: resolve_competition -> research -> forecast run first and always,
    deterministically, before the LLM ever sees the match. A failed/impossible
    forecast (no odds available from any source, or a tool error) routes
    straight to output -- the LLM node is never invoked in that case."""
    llm = _build_llm(config)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = _invoke_with_retry(llm_with_tools, state["messages"])
        tool_calls = getattr(response, "tool_calls", []) or []
        new_count = state["tool_call_count"] + len(tool_calls)
        if tool_calls:
            _LOG.info("agent_node | tool_calls=%s | count_after=%d", [tc["name"] for tc in tool_calls], new_count)
        else:
            content = _extract_text(response.content)
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
        forecast_payload = state.get("forecast_payload")
        match_info = state["match_info"]

        if not forecast_payload or "error" in forecast_payload:
            reason = (forecast_payload or {}).get("error", "forecast step did not run")
            _LOG.warning("output_node | no_forecast | reason=%s", reason)
            return {"recommendation": {
                "match": match_info,
                "overall": "insufficient_data",
                "markets": [],
                "explanation": [f"No ML forecast is available for this match: {reason}"],
                "confidence": "low",
                "limitations": [f"Forecast step failed or was skipped: {reason}"],
                "prediction_basis": "unknown",
                "cold_start_risk": False,
                "feature_completeness": None,
                "unknown_team": False,
            }}

        if config.provider == "ollama":
            structured = _structured_output(llm, state["messages"])
            if structured is not None:
                _LOG.info("output_node | structured_output=success | overall=%s", structured.get("overall"))
                recommendation = _finalize_recommendation(
                    structured, forecast_payload, state.get("research_evidence"),
                )
                return {"recommendation": recommendation}
            _LOG.info("output_node | structured_output=unavailable_or_failed | falling_back_to_free_text")

        last = state["messages"][-1]
        text = _extract_text(last.content)

        if not text.strip():
            # Budget was exhausted — last message is a tool call with no text content.
            # Make one final synthesis call (no tools) so the model can produce its JSON.
            _LOG.info("output_node | empty_content | forcing_synthesis_call")
            synthesis_prompt = (
                "You have reached the tool call limit. "
                "Based on all the information gathered above, output your final JSON recommendation now. "
                "Include all required fields: match, overall, markets, explanation, confidence, limitations, prediction_basis. "
                "Output ONLY the JSON block -- no narrative report, no headers, no text before or after it."
            )
            synthesis_response = _invoke_with_retry(llm, state["messages"] + [HumanMessage(content=synthesis_prompt)])
            text = _extract_text(synthesis_response.content)
            _LOG.info("output_node | synthesis_length=%d | synthesis_output=%s", len(text), text)

        _LOG.info("output_node | raw_output_length=%d", len(text))
        _LOG.info("output_node | raw_output=%s", text)
        recommendation = _build_recommendation(
            text, match_info, forecast_payload, state.get("research_evidence"), config,
        )
        return {"recommendation": recommendation}

    def forecast_node_with_config(state: AgentState) -> dict:
        return forecast_node(state, suppress_uncertainty=config.suppress_forecast_uncertainty)

    graph = StateGraph(AgentState)
    graph.add_node("resolve_competition", resolve_competition_node)
    graph.add_node("research", research_node)
    graph.add_node("forecast", forecast_node_with_config)
    graph.add_node("lessons", lessons_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output", output_node)

    graph.set_entry_point("resolve_competition")
    graph.add_edge("resolve_competition", "research")
    graph.add_edge("research", "forecast")
    graph.add_conditional_edges("forecast", route_after_forecast, {"lessons": "lessons", "output": "output"})
    graph.add_edge("lessons", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "output": "output"})
    graph.add_edge("tools", "agent")
    graph.add_edge("output", END)

    return graph.compile()


def run_agent(
    match_info: dict,
    config: AgentConfig | None = None,
    tools: list | None = None,
    extra_system_instructions: str | None = None,
    return_full_state: bool = False,
) -> MatchRecommendation | dict[str, Any]:
    """Run the betting agent for a single match and return a structured recommendation.

    Args:
        match_info: Dict with keys: home_team, away_team, date, and optionally league, odds.
        config: AgentConfig instance. Loads from config/agent_config.yaml if None.
        tools: List of LangChain tools available to the LLM synthesis step (web_search
            by default). Loads default tools if None. Competition resolution and the
            ML forecast are no longer LLM-callable tools -- see src/agent/pipeline.py.
        extra_system_instructions: Appended to the loaded system prompt. Used by
            agent-snapshot (A11) to inject snapshot-collection-only rules (e.g.
            "ignore any result mentioning a final score") without forking the
            whole prompt file.
        return_full_state: A33 -- when True, return the full graph state dict
            (recommendation, competition_resolution, research_evidence,
            forecast_payload, messages, match_info, tool_call_count) instead
            of just the recommendation. Used by agent-train to persist raw
            evidence to DuckDB. Note: messages is a list of LangChain
            BaseMessage objects and is NOT JSON-serializable -- callers should
            pick specific keys out of the dict rather than serialize it whole.
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
        prompt += f". Bookmaker odds: home={odds['home']}, draw={odds['draw']}, away={odds['away']}."
    # 2026-08-21: total_goals is the only secondary market (beyond result_3way)
    # with a real odds source anywhere in this system (raw_matches's own
    # over25_odds/under25_odds, threaded in by backtest.py's _build_match_info
    # for replay; live callers may set this key the same way). btts/corners
    # have no equivalent real column, live or historical -- see
    # documents/agent_techspec.md's "Secondary-market odds coverage" section.
    total_goals_odds = match_info.get("total_goals_odds")
    if total_goals_odds:
        prompt += (
            f" Bookmaker odds for total goals (over/under 2.5): "
            f"over_2.5={total_goals_odds['over_2.5']}, under_2.5={total_goals_odds['under_2.5']}."
        )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ],
        "match_info": match_info,
        "recommendation": None,
        "tool_call_count": 0,
        "competition_resolution": None,
        "research_evidence": None,
        "forecast_payload": None,
    }

    compiled = build_graph(config, tools)
    result = compiled.invoke(initial_state)
    if return_full_state:
        return result
    return result["recommendation"]
