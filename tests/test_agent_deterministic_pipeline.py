"""Tests for run_deterministic_pipeline (A97).

Found live: agent-snapshot (both its normal recording pass AND
--refresh-model) always called run_agent(), the single all-in-one entry
point that runs the FULL graph through to a real LLM turn (agent_node) --
even though nothing about the snapshot corpus ever reads that LLM output
back. resolve_competition_node/research_node/forecast_node are the only
graph nodes whose output actually gets persisted as a snapshot file
(SnapshotStore.wrap's four tool names: resolve_competition,
forecast_league, forecast_international, web_search -- research_node's own
guaranteed searches share the web_search wrapper); agent_node/output_node
aren't wrapped by anything, so their output was never stored, never
replayed, and every agent-snapshot run (recording or refresh) was paying
for a real LLM API call that served no purpose at all for the corpus's
actual use case (agent-train/backtest always generates a fresh
recommendation live during replay, regardless of what the original
recording run produced).

run_deterministic_pipeline runs just the three deterministic nodes in
their real graph order (resolve_competition -> research -> forecast),
with zero LLM involvement -- exactly enough to write/refresh every
snapshot type that's actually reused, at zero LLM cost.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import run_deterministic_pipeline


def test_calls_nodes_in_real_graph_order_with_threaded_state():
    call_order = []

    def fake_resolve(state):
        call_order.append(("resolve_competition", state))
        return {"competition_resolution": {"competition": "E0", "tier": "competition_specific"}}

    def fake_research(state):
        call_order.append(("research", state))
        return {"research_evidence": {"availability": "ok"}}

    def fake_forecast(state):
        call_order.append(("forecast", state))
        return {"forecast_payload": {"result_3way": {}}}

    match_info = {"home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0"}
    with patch("src.agent.graph.resolve_competition_node", side_effect=fake_resolve), \
         patch("src.agent.graph.research_node", side_effect=fake_research), \
         patch("src.agent.graph.forecast_node", side_effect=fake_forecast):
        run_deterministic_pipeline(match_info)

    assert [name for name, _ in call_order] == ["resolve_competition", "research", "forecast"]
    # research_node must see competition_resolution already threaded in
    assert call_order[1][1]["competition_resolution"]["competition"] == "E0"
    # forecast_node must see both prior nodes' output already threaded in
    assert call_order[2][1]["research_evidence"]["availability"] == "ok"
    assert call_order[2][1]["competition_resolution"]["competition"] == "E0"


def test_returns_merged_state_with_expected_keys():
    match_info = {"home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0"}
    with patch("src.agent.graph.resolve_competition_node", return_value={"competition_resolution": {"competition": "E0"}}), \
         patch("src.agent.graph.research_node", return_value={"research_evidence": {"availability": "ok"}}), \
         patch("src.agent.graph.forecast_node", return_value={"forecast_payload": {"btts": {}}}):
        result = run_deterministic_pipeline(match_info)

    assert result["match_info"] == match_info
    assert result["competition_resolution"] == {"competition": "E0"}
    assert result["research_evidence"] == {"availability": "ok"}
    assert result["forecast_payload"] == {"btts": {}}


def test_never_touches_the_llm():
    """The whole point: no LLM client construction, no graph compilation --
    only the three deterministic node functions run."""
    match_info = {"home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0"}
    with patch("src.agent.graph.resolve_competition_node", return_value={"competition_resolution": {}}), \
         patch("src.agent.graph.research_node", return_value={"research_evidence": {}}), \
         patch("src.agent.graph.forecast_node", return_value={"forecast_payload": {}}), \
         patch("src.agent.graph._build_llm") as mock_build_llm, \
         patch("src.agent.graph.build_graph") as mock_build_graph:
        run_deterministic_pipeline(match_info)

    mock_build_llm.assert_not_called()
    mock_build_graph.assert_not_called()


def test_propagates_a_node_exception_rather_than_swallowing_it():
    match_info = {"home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0"}
    with patch("src.agent.graph.resolve_competition_node", return_value={"competition_resolution": {}}), \
         patch("src.agent.graph.research_node", return_value={"research_evidence": {}}), \
         patch("src.agent.graph.forecast_node", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            run_deterministic_pipeline(match_info)
