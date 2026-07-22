"""Regression tests for A31's forecast_payload-based diagnostics (cold_start_risk,
feature_completeness, unknown_team), A30's backstop (a recommendation can never
claim more evidence than actually exists), and A32's research-coverage confidence
downgrade. All three are applied in graph.py's _build_recommendation,
deterministically from pipeline state -- never from the LLM's own prose, matching
the code-over-prompt philosophy already established by A28/A29."""

from __future__ import annotations

import json

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


def _forecast_payload(cold_start_risk: bool, feature_completeness: float, unknown_team: bool) -> dict:
    return {
        "forecast": {},
        "diagnostics": {"cold_start_risk": cold_start_risk, "feature_completeness": feature_completeness},
        "data_quality": {"prediction_basis": "team_history_and_market", "unknown_team": unknown_team},
    }


def _valid_llm_json(**overrides) -> str:
    data = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "direct_bet",
        "markets": [],
        "explanation": "Looks good.",
        "confidence": "high",
        "limitations": [],
        "prediction_basis": "team_history_and_market",
    }
    data.update(overrides)
    return json.dumps(data)


# --- _extract_forecast_diagnostics ---

def test_extracts_diagnostics_from_a_successful_forecast_payload():
    payload = _forecast_payload(cold_start_risk=True, feature_completeness=0.62, unknown_team=True)
    assert _extract_forecast_diagnostics(payload) == {"cold_start_risk": True, "feature_completeness": 0.62, "unknown_team": True}


def test_defaults_when_forecast_payload_is_none():
    assert _extract_forecast_diagnostics(None) == {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}


def test_defaults_when_forecast_payload_has_an_error():
    payload = {"error": "boom", "status": "tool_error"}
    assert _extract_forecast_diagnostics(payload) == {"cold_start_risk": False, "feature_completeness": None, "unknown_team": False}


# --- _build_recommendation: diagnostics enrichment ---

def test_build_recommendation_enriches_even_when_llm_json_omits_the_fields():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=True, feature_completeness=0.4, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(), match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload, research_evidence={"availability": "x", "form_context": "y"}, config=cfg,
    )

    assert recommendation["cold_start_risk"] is True
    assert recommendation["feature_completeness"] == 0.4
    assert recommendation["overall"] == "direct_bet"


def test_build_recommendation_enriches_the_parse_failure_fallback_too():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=True, feature_completeness=0.3, unknown_team=True)

    recommendation = _build_recommendation(
        text="not valid json at all", match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload, research_evidence=None, config=cfg,
    )

    assert recommendation["overall"] == "insufficient_data"
    assert recommendation["cold_start_risk"] is True
    assert recommendation["unknown_team"] is True


# --- A30 backstop ---

def test_backstop_forces_insufficient_data_when_forecast_payload_is_none():
    """The Burnley/Bournemouth shape that motivated A30: LLM claims a
    confident no_bet with a populated prediction_basis, but no forecast ever
    actually ran."""
    cfg = _make_config()

    recommendation = _build_recommendation(
        text=_valid_llm_json(overall="no_bet", prediction_basis="team_history_and_market"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=None, research_evidence=None, config=cfg,
    )

    assert recommendation["overall"] == "insufficient_data"
    assert recommendation["markets"] == []
    assert recommendation["prediction_basis"] == "unknown"
    assert any("Forced insufficient_data" in note for note in recommendation["limitations"])


def test_backstop_forces_insufficient_data_when_forecast_payload_has_an_error():
    cfg = _make_config()

    recommendation = _build_recommendation(
        text=_valid_llm_json(overall="direct_bet"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload={"error": "no models found", "status": "tool_error"},
        research_evidence=None, config=cfg,
    )

    assert recommendation["overall"] == "insufficient_data"


def test_backstop_leaves_a_genuine_forecast_backed_no_bet_untouched():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)
    llm_json = _valid_llm_json(overall="no_bet", markets=[{
        "market": "btts", "selection": "yes", "recommendation_type": "no_bet",
        "current_odds": None, "min_odds": 1.5, "ml_probability": 0.5,
        "implied_probability": 0.5, "value_edge": 0.0,
    }])

    recommendation = _build_recommendation(
        text=llm_json, match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload, research_evidence={"availability": "x", "form_context": "y"}, config=cfg,
    )

    assert recommendation["overall"] == "no_bet"
    assert len(recommendation["markets"]) == 1


# --- A32 research coverage downgrade ---

def test_research_coverage_downgrade_lowers_confidence_when_availability_missing():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(confidence="high"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload,
        research_evidence={"availability": "No results found.", "form_context": "won last 3"},
        config=cfg,
    )

    assert recommendation["confidence"] == "medium"
    assert any("Research coverage gap" in note for note in recommendation["limitations"])


def test_research_coverage_downgrade_caps_at_low_with_two_gaps():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(confidence="high"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload,
        research_evidence={"availability": "No results found.", "form_context": "TOOL_PERMANENTLY_UNAVAILABLE: no key"},
        config=cfg,
    )

    assert recommendation["confidence"] == "low"


def test_research_coverage_downgrade_leaves_full_coverage_untouched():
    cfg = _make_config()
    payload = _forecast_payload(cold_start_risk=False, feature_completeness=1.0, unknown_team=False)

    recommendation = _build_recommendation(
        text=_valid_llm_json(confidence="high"),
        match_info={"home_team": "A", "away_team": "B", "date": "2026-08-22"},
        forecast_payload=payload,
        research_evidence={"availability": "no injuries", "form_context": "won last 3"},
        config=cfg,
    )

    assert recommendation["confidence"] == "high"
    assert not any("Research coverage gap" in note for note in recommendation["limitations"])
