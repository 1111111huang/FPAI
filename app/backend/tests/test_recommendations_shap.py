"""Tests for _attach_shap_contributions -- deterministic, code-only
attachment of each candidate's own SHAP explanation from forecast_payload,
signed relative to that candidate's own selection so the frontend never has
to reason about which raw class a market's SHAP output was computed
against. Never LLM-authored; wired into unwrap_agent_result so every real
generation call site gets it for free, in one place."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.recommendations import _attach_shap_contributions, unwrap_agent_result


def _forecast_payload(market: str, contributions: list[dict]) -> dict:
    return {"forecast": {market: {"shap_contributions": contributions}}}


def test_positive_selection_gets_contributions_unmodified():
    recommendation = {"candidates": [{"market": "btts", "selection": "yes"}]}
    contributions = [{"feature": "MKT_OVERROUND", "shap_value": 0.1, "value": 1.04}]

    result = _attach_shap_contributions(recommendation, _forecast_payload("btts", contributions))

    assert result["candidates"][0]["shap_contributions"] == contributions


def test_opposite_selection_gets_sign_flipped():
    recommendation = {"candidates": [{"market": "btts", "selection": "no"}]}
    contributions = [{"feature": "MKT_OVERROUND", "shap_value": 0.1, "value": 1.04}]

    result = _attach_shap_contributions(recommendation, _forecast_payload("btts", contributions))

    flipped = result["candidates"][0]["shap_contributions"]
    assert flipped[0]["shap_value"] == -0.1
    assert flipped[0]["feature"] == "MKT_OVERROUND"
    assert flipped[0]["value"] == 1.04  # the raw feature value itself never flips, only the attribution


def test_missing_value_stays_none_through_sign_flip():
    recommendation = {"candidates": [{"market": "btts", "selection": "no"}]}
    contributions = [{"feature": "MKT_LAMBDA_AWAY", "shap_value": -0.05, "value": None}]

    result = _attach_shap_contributions(recommendation, _forecast_payload("btts", contributions))

    assert result["candidates"][0]["shap_contributions"][0]["value"] is None


def test_result_3way_never_gets_shap_attached():
    """3 candidates share one multiclass explanation with no unambiguous
    per-selection sign -- deliberately left out of _SHAP_POSITIVE_SELECTION
    rather than guessed at."""
    recommendation = {"candidates": [{"market": "result_3way", "selection": "home"}]}
    contributions = [{"feature": "MKT_IMPLIED_HOME", "shap_value": 0.2, "value": 0.4}]

    result = _attach_shap_contributions(recommendation, _forecast_payload("result_3way", contributions))

    assert "shap_contributions" not in result["candidates"][0]


def test_market_absent_from_forecast_payload_is_a_noop():
    recommendation = {"candidates": [{"market": "btts", "selection": "yes"}]}

    result = _attach_shap_contributions(recommendation, {"forecast": {}})

    assert "shap_contributions" not in result["candidates"][0]


def test_none_forecast_payload_is_a_noop():
    recommendation = {"candidates": [{"market": "btts", "selection": "yes"}]}

    result = _attach_shap_contributions(recommendation, None)

    assert "shap_contributions" not in result["candidates"][0]


def test_empty_candidates_list_is_a_noop():
    recommendation = {"candidates": []}

    result = _attach_shap_contributions(recommendation, _forecast_payload("btts", [{"feature": "x", "shap_value": 1.0}]))

    assert result["candidates"] == []


def test_unwrap_agent_result_attaches_shap_before_returning():
    full_state = {
        "recommendation": {
            "overall": "no_bet",
            "match": {},
            "candidates": [
                {"market": "btts", "selection": "yes"},
                {"market": "btts", "selection": "no"},
            ],
        },
        "messages": [],
        "forecast_payload": _forecast_payload(
            "btts", [{"feature": "MKT_OVERROUND", "shap_value": 0.3, "value": 1.05}]
        ),
    }

    recommendation, _, _ = unwrap_agent_result(full_state)

    yes_candidate, no_candidate = recommendation["candidates"]
    assert yes_candidate["shap_contributions"][0]["shap_value"] == 0.3
    assert no_candidate["shap_contributions"][0]["shap_value"] == -0.3
