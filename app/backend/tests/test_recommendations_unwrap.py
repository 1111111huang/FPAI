"""Tests for A107: recommendations.unwrap_agent_result() -- splits
run_agent()'s return value into (recommendation, reasoning_trace,
forecast_payload). run_agent() now always requests the agent's full graph
state (return_full_state=True), but every existing test in this codebase's
suite mocks it (or _real_run_agent) with a bare recommendation dict -- this
function must treat both shapes correctly: a real full_state dict
(unwrapped) and a legacy bare recommendation dict (passed through, no trace
available), discriminated by whether "recommendation" is a top-level key
(MatchRecommendation's own schema never uses that name)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.recommendations import unwrap_agent_result

_RECOMMENDATION = {"overall": "no_bet", "candidates": [], "match": {}}


def test_unwraps_a_real_full_state_dict():
    full_state = {
        "recommendation": _RECOMMENDATION,
        "messages": [],
        "forecast_payload": {"result_3way": {"probabilities": {"home": 0.5}}},
        "competition_resolution": {"competition": "E0"},
    }

    recommendation, reasoning_trace, forecast_payload = unwrap_agent_result(full_state)

    assert recommendation == _RECOMMENDATION
    assert reasoning_trace == []
    assert forecast_payload == {"result_3way": {"probabilities": {"home": 0.5}}}


def test_serializes_real_messages_into_the_reasoning_trace():
    from langchain_core.messages import AIMessage, SystemMessage

    full_state = {
        "recommendation": _RECOMMENDATION,
        "messages": [SystemMessage(content="system prompt"), AIMessage(content="reasoning text")],
        "forecast_payload": None,
    }

    _, reasoning_trace, _ = unwrap_agent_result(full_state)

    assert reasoning_trace == [
        {"role": "system", "content": "system prompt"},
        {"role": "ai", "content": "reasoning text"},
    ]


def test_passes_through_a_legacy_bare_recommendation_dict_unchanged():
    """A test (or a not-yet-updated caller) mocking run_agent()/
    _real_run_agent with a bare recommendation dict, the pre-A107 shape."""
    recommendation, reasoning_trace, forecast_payload = unwrap_agent_result(_RECOMMENDATION)

    assert recommendation == _RECOMMENDATION
    assert reasoning_trace == []
    assert forecast_payload is None


def test_missing_messages_key_is_still_treated_as_legacy_shape():
    """"recommendation" alone isn't enough -- MatchRecommendation's own
    schema never uses that key, so this is belt-and-suspenders, not load-
    bearing, but confirms the discriminator checks both keys."""
    almost_full_state = {"recommendation": _RECOMMENDATION}  # no "messages"

    recommendation, reasoning_trace, forecast_payload = unwrap_agent_result(almost_full_state)

    assert recommendation == almost_full_state
    assert reasoning_trace == []
    assert forecast_payload is None
