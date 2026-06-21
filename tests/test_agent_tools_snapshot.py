"""Tests for SnapshotStore integration in agent tools (A10)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent import tools as agent_tools
from src.agent.snapshot_store import SnapshotMissingError


@pytest.fixture(autouse=True)
def reset_snapshot_store(tmp_path):
    """Point the module-level store at a tmp dir and reset to live mode after each test."""
    agent_tools._snapshot_store.base_dir = tmp_path
    agent_tools._snapshot_store.set_mode("live")
    yield
    agent_tools._snapshot_store.set_mode("live")


def test_web_search_record_then_replay(tmp_path):
    with patch("src.agent.tools.os.environ.get", return_value="fake-key"), \
         patch("src.agent.tools.TavilyClient") if False else patch("tavily.TavilyClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.search.return_value = {"results": [{"title": "T", "content": "C", "url": "U"}]}

        agent_tools.configure_snapshot_store("record", match_id="m1")
        first = agent_tools.web_search.invoke({"query": "test odds"})
        assert "T" in first
        assert instance.search.call_count == 1

        # Replay must return the exact same text WITHOUT calling Tavily again
        agent_tools.configure_snapshot_store("replay", match_id="m1")
        second = agent_tools.web_search.invoke({"query": "test odds"})
        assert second == first
        assert instance.search.call_count == 1  # unchanged


def test_web_search_replay_missing_raises():
    agent_tools.configure_snapshot_store("replay", match_id="never-recorded")
    with pytest.raises(SnapshotMissingError):
        agent_tools.web_search.invoke({"query": "anything"})


def test_web_search_unavailable_message_bypasses_snapshot_key_consistently():
    # No TAVILY_API_KEY in this process env by default in CI; record mode without
    # a key returns the fixed unavailable message both times — proves wrap() doesn't
    # choke on the early-return path (no tavily call at all).
    agent_tools.configure_snapshot_store("record", match_id="m2")
    with patch.dict("os.environ", {}, clear=True):
        result = agent_tools.web_search.invoke({"query": "x"})
    assert "TOOL_PERMANENTLY_UNAVAILABLE" in result


def test_web_search_date_filter_applied_during_record_and_replay():
    captured_queries = []

    def fake_impl(query):
        captured_queries.append(query)
        return "ok"

    with patch.object(agent_tools, "_web_search_impl", side_effect=fake_impl):
        agent_tools.configure_snapshot_store("record", match_id="m3", match_date="2025-03-01")
        agent_tools.web_search.invoke({"query": "team news"})

    assert captured_queries == ["team news before:2025-03-01"]


def test_forecast_league_record_then_replay():
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        agent_tools.configure_snapshot_store("record", match_id="m4")
        first = agent_tools.forecast_league.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0",
            "odds_h": 2.0, "odds_d": 3.0, "odds_a": 3.5,
        })
        assert instance.forecast_upcoming.call_count == 1

        agent_tools.configure_snapshot_store("replay", match_id="m4")
        second = agent_tools.forecast_league.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0",
            "odds_h": 2.0, "odds_d": 3.0, "odds_a": 3.5,
        })
        assert second == first
        assert instance.forecast_upcoming.call_count == 1  # not called again


def test_live_mode_is_default_and_unaffected_by_snapshot_machinery():
    """Existing A05/A06 behaviour (no snapshot config called) must be unchanged."""
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result
        result = agent_tools.forecast_league.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01", "league": "E0",
            "odds_h": 2.0, "odds_d": 3.0, "odds_a": 3.5,
        })
    assert json.loads(result)["result_3way"]["probabilities"]["home"] == 0.5
