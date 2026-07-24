"""Deterministic evidence pipeline (A31/A32): competition resolution, baseline
web research, and the ML forecast all run here, as required graph nodes, before
the LLM ever sees the match -- replacing the old design where the LLM could
choose (or fail) to call resolve_competition/forecast_league/forecast_international
as tools. See docs/superpowers/specs/2026-07-22-agent-phase11-design.md."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

_ODDS_NUMBER_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,2}\b")


def resolve_competition_node(state: dict) -> dict:
    """A31: deterministic competition-tier lookup. If match_info has no league
    at all (e.g. a genuinely unlabeled international fixture), there's nothing
    to look up -- default straight to general_purpose/forecast_international
    rather than calling the registry with an empty string."""
    league = state["match_info"].get("league")
    if not league:
        return {"competition_resolution": {
            "competition": None,
            "tier": "general_purpose",
            "recommended_tool": "forecast_international",
        }}

    from src.agent.tools import _resolve_competition_impl, get_snapshot_store

    raw = get_snapshot_store().wrap("resolve_competition", _resolve_competition_impl)(
        competition_or_league=league
    )
    return {"competition_resolution": json.loads(raw)}


def _parse_odds_from_search_text(text: str | None) -> dict | None:
    """Best-effort extraction of three decimal odds (home/draw/away) from a
    web search snippet. Deliberately conservative: requires at least three
    plausible decimal-odds-shaped numbers (1.01-50.0) in the text and just
    takes the first three in reading order. This is a heuristic, not a
    guarantee -- forecast_node only ever falls back to it when the caller
    supplied no odds at all, and a low-confidence/failed parse (fewer than 3
    plausible numbers) correctly results in insufficient_data rather than a
    forecast built on a wrong guess."""
    if not text:
        return None
    numbers = [float(m) for m in _ODDS_NUMBER_PATTERN.findall(text)]
    plausible = [n for n in numbers if 1.01 <= n <= 50.0]
    if len(plausible) < 3:
        return None
    home, draw, away = plausible[:3]
    return {"home": home, "draw": draw, "away": away}


def research_node(state: dict) -> dict:
    """A32: guarantees minimum research coverage deterministically instead of
    depending on the LLM choosing to search. Always runs availability and
    recent-form searches; only runs an odds-verification search when the
    caller didn't already supply odds (match_info.get('odds'))."""
    from src.agent.tools import _dated_web_search

    match_info = state["match_info"]
    home, away = match_info["home_team"], match_info["away_team"]

    availability_text = _dated_web_search(f"{home} {away} injury suspension team news")
    form_text = _dated_web_search(f"{home} {away} recent form last 5 matches")

    evidence: dict = {
        "availability": availability_text,
        "form_context": form_text,
        "odds_verification": None,
    }

    if not match_info.get("odds"):
        odds_text = _dated_web_search(f"{home} vs {away} odds")
        evidence["odds_verification"] = {
            "results": odds_text,
            "parsed_odds": _parse_odds_from_search_text(odds_text),
        }

    return {"research_evidence": evidence}


def _format_evidence_message(forecast_payload: dict, research_evidence: dict | None) -> str:
    """The message injected into the LLM's context once the deterministic
    pipeline finishes, replacing the old tool-call results the LLM used to
    see. Explicitly tells the LLM the forecast/competition tools are gone.

    BUG-019: deliberately avoids two things that caused local models (observed
    on both llama3.1:8b and qwen2.5-coder:7b) to not emit the required JSON:
    (1) markdown headers (`##`/`###`) around the evidence, which the model
    pattern-completed with its own document subsections instead of writing
    JSON at all; (2) a fenced ```-code-block JSON-shaped example, which -- once
    (1) was fixed -- the model instead imitated the *shape* of (a single
    wrapper key like `{"recommendation": "..."}`) rather than recalling the
    real schema defined earlier in the system prompt. Evidence is now plain
    UPPERCASE_LABEL: text with no code fence, and the exact required top-level
    JSON keys are restated immediately after the evidence -- closest to the
    model's own next turn, where a schema reminder has the most influence."""
    evidence = research_evidence or {}
    lines = [
        "Reference data for this match below (not a document to write about, and "
        "not an example of your output format -- use it only to inform the JSON "
        "recommendation described further below).",
        "",
        "FORECAST_PAYLOAD: " + json.dumps(forecast_payload, default=str),
        "AVAILABILITY_SEARCH_RESULT: " + (evidence.get("availability") or "No results."),
        "FORM_SEARCH_RESULT: " + (evidence.get("form_context") or "No results."),
    ]
    odds_verification = evidence.get("odds_verification")
    if odds_verification:
        lines.append("ODDS_VERIFICATION_SEARCH_RESULT: " + (odds_verification.get("results") or "No results."))
    lines += [
        "",
        "forecast_league, forecast_international, and resolve_competition are NOT "
        "available as tools -- do not attempt to call them. Use web_search only for "
        "additional follow-up context beyond what's above. Do not summarize or write "
        "prose about the data above.",
        "",
        "Your final answer must be a single JSON object with EXACTLY these top-level "
        "keys, no others: match, overall, markets, explanation, confidence, "
        "limitations, prediction_basis. Do not wrap your answer in any other key "
        "(e.g. not {\"recommendation\": ...} or {\"response\": ...}) -- use these exact "
        "field names at the top level.",
    ]
    return "\n".join(lines)


def forecast_node(state: dict) -> dict:
    """A31: the ML forecast is now a required deterministic step, not
    something the LLM chooses (or fails) to call. Odds are sourced in
    priority order: caller-supplied -> research_node's odds-verification
    parse -> none. There is no third "fallback odds" tier -- ForecastService
    cannot run without real odds (see the plan's technical-correction note),
    so "none" short-circuits to an error payload that routes straight to
    insufficient_data, skipping the LLM entirely."""
    match_info = state["match_info"]
    odds = match_info.get("odds")
    if not odds:
        research_evidence = state.get("research_evidence") or {}
        odds_verification = research_evidence.get("odds_verification") or {}
        odds = odds_verification.get("parsed_odds")

    if not odds:
        return {"forecast_payload": {
            "error": "No odds available: not supplied by caller and odds-verification search found none",
            "status": "no_odds",
        }}

    resolution = state.get("competition_resolution") or {}
    recommended_tool = resolution.get("recommended_tool", "forecast_international")

    from src.agent.tools import _forecast_international_impl, _forecast_league_impl, get_snapshot_store

    store = get_snapshot_store()
    if recommended_tool == "forecast_league":
        raw = store.wrap("forecast_league", _forecast_league_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"], league=match_info.get("league", ""),
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )
    else:
        raw = store.wrap("forecast_international", _forecast_international_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"],
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )

    payload = json.loads(raw)
    if "error" in payload:
        return {"forecast_payload": payload}

    evidence_message = _format_evidence_message(payload, state.get("research_evidence"))
    return {"forecast_payload": payload, "messages": [HumanMessage(content=evidence_message)]}


def lessons_node(state: dict) -> dict:
    """A33: inject reviewer-approved lessons scoped to this match's
    competition/tier as a HumanMessage before the LLM's turn -- same
    injection pattern forecast_node uses for evidence (a node-returned
    "messages" list is appended via AgentState's add_messages reducer).

    Gated on SnapshotStore mode == "live": outside genuine live runs
    (agent-backtest/agent-train replay, or agent-snapshot record), lessons
    are skipped entirely. Injecting lessons approved *after* a historical
    match would leak future information into backtest/train scoring,
    corrupting the A13/A21/A34 baseline methodology agent-backtest and
    agent-train share. Gating here (rather than a config flag) means the
    same compiled graph is correct for every CLI entry point.

    Only imports load_approved_lessons from src.agent.lessons -- see that
    module's docstring and tests/test_agent_lessons.py for why that function
    alone can't read match outcomes or pending/rejected lessons.
    """
    from src.agent.tools import get_snapshot_store

    if get_snapshot_store().mode != "live":
        return {}

    from src.agent.lessons import extract_competition_scope, load_approved_lessons
    from src.utils.db_manager import DuckDBManager

    competition_id, tier = extract_competition_scope(state)
    with DuckDBManager().connection(read_only=True) as conn:
        lessons = load_approved_lessons(conn, competition_id, tier)
    if not lessons:
        return {}
    lessons_text = "Lessons from past evaluated matches:\n" + "\n".join(f"- {lesson}" for lesson in lessons)
    return {"messages": [HumanMessage(content=lessons_text)]}
