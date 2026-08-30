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
    agent_tools._snapshot_store.set_allow_lessons_in_replay(False)
    agent_tools._snapshot_store.set_tool_mode_overrides({})
    yield
    agent_tools._snapshot_store.set_mode("live")
    agent_tools._snapshot_store.set_allow_lessons_in_replay(False)
    agent_tools._snapshot_store.set_tool_mode_overrides({})


def test_configure_snapshot_store_allow_lessons_in_replay_defaults_false():
    assert agent_tools.get_snapshot_store().allow_lessons_in_replay is False


def test_configure_snapshot_store_sets_allow_lessons_in_replay():
    agent_tools.configure_snapshot_store("replay", match_id="m1", allow_lessons_in_replay=True)
    assert agent_tools.get_snapshot_store().allow_lessons_in_replay is True


def test_configure_snapshot_store_allow_lessons_in_replay_is_sticky_if_omitted():
    agent_tools.configure_snapshot_store("replay", match_id="m1", allow_lessons_in_replay=True)
    agent_tools.configure_snapshot_store("live")  # no allow_lessons_in_replay passed
    assert agent_tools.get_snapshot_store().allow_lessons_in_replay is True


def test_configure_snapshot_store_tool_mode_overrides_defaults_empty():
    assert agent_tools.get_snapshot_store().tool_mode_overrides == {}


def test_configure_snapshot_store_sets_tool_mode_overrides():
    agent_tools.configure_snapshot_store("replay", match_id="m1", tool_mode_overrides={"forecast_league": "record"})
    assert agent_tools.get_snapshot_store().tool_mode_overrides == {"forecast_league": "record"}


def test_configure_snapshot_store_tool_mode_overrides_is_sticky_if_omitted():
    agent_tools.configure_snapshot_store("replay", match_id="m1", tool_mode_overrides={"forecast_league": "record"})
    agent_tools.configure_snapshot_store("live")  # no tool_mode_overrides passed
    assert agent_tools.get_snapshot_store().tool_mode_overrides == {"forecast_league": "record"}


def test_configure_snapshot_store_tool_mode_overrides_explicit_empty_clears():
    agent_tools.configure_snapshot_store("replay", match_id="m1", tool_mode_overrides={"forecast_league": "record"})
    agent_tools.configure_snapshot_store("live", tool_mode_overrides={})
    assert agent_tools.get_snapshot_store().tool_mode_overrides == {}


def test_web_search_record_then_replay(tmp_path):
    with patch("src.agent.tools.os.environ.get", return_value="fake-key"), \
         patch("tavily.TavilyClient") as MockClient:
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


def test_web_search_tavily_failure_degrades_instead_of_raising():
    """A53: a real API-side failure (quota exceeded, rate limit, network
    error, ...) previously propagated uncaught all the way to a raw 500 --
    crashing the whole recommendation for a single research call. Must
    degrade to the same TOOL_PERMANENTLY_UNAVAILABLE-style sentinel the
    missing-key case already returns, not raise."""
    from src.agent.tools import _web_search_impl

    with patch("src.agent.tools.os.environ.get", return_value="fake-key"), \
         patch("tavily.TavilyClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.search.side_effect = RuntimeError("This request exceeds your plan's set usage limit.")

        result = _web_search_impl("Arsenal Chelsea injury news")

    assert "TOOL_PERMANENTLY_UNAVAILABLE" in result
    assert "usage limit" in result


def test_web_search_falls_back_to_secondary_key_when_primary_fails():
    """Primary key raises (quota/rate-limit); secondary key succeeds -- the
    call should return real results, not degrade to unavailable."""
    from src.agent.tools import _web_search_impl

    primary = MagicMock()
    primary.search.side_effect = RuntimeError("quota exceeded")
    secondary = MagicMock()
    secondary.search.return_value = {"results": [{"title": "T", "content": "C", "url": "U"}]}

    def fake_client(api_key):
        return primary if api_key == "primary-key" else secondary

    with patch.dict("os.environ", {"TAVILY_API_KEY": "primary-key", "TAVILY_API_KEY_FALLBACK": "fallback-key"}, clear=True), \
         patch("tavily.TavilyClient", side_effect=fake_client):
        result = _web_search_impl("Arsenal Chelsea injury news")

    assert "T" in result
    assert primary.search.call_count == 1
    assert secondary.search.call_count == 1


def test_web_search_both_keys_failing_degrades_instead_of_raising():
    from src.agent.tools import _web_search_impl

    instance = MagicMock()
    instance.search.side_effect = RuntimeError("still exceeds usage limit")

    with patch.dict("os.environ", {"TAVILY_API_KEY": "primary-key", "TAVILY_API_KEY_FALLBACK": "fallback-key"}, clear=True), \
         patch("tavily.TavilyClient", return_value=instance):
        result = _web_search_impl("Arsenal Chelsea injury news")

    assert "TOOL_PERMANENTLY_UNAVAILABLE" in result
    assert instance.search.call_count == 2


def test_web_search_unavailable_message_bypasses_snapshot_key_consistently():
    # No TAVILY_API_KEY in this process env by default in CI; record mode without
    # a key returns the fixed unavailable message both times — proves wrap() doesn't
    # choke on the early-return path (no tavily call at all).
    agent_tools.configure_snapshot_store("record", match_id="m2")
    with patch.dict("os.environ", {}, clear=True):
        result = agent_tools.web_search.invoke({"query": "x"})
    assert "TOOL_PERMANENTLY_UNAVAILABLE" in result


class TestPostMatchResultFilter:
    """A47: leaked post-match content is filtered before it reaches the
    model, at the Tavily-result level. Real examples below are taken
    verbatim (titles) or paraphrased (content) from genuine leaks/false
    positives found during the 2026-07-29 corpus investigation, not
    invented -- see _RESULT_LEAK_MARKERS' docstring in src/agent/tools.py."""

    def test_flags_score_in_title(self):
        from src.agent.tools import _looks_like_post_match_result
        assert _looks_like_post_match_result(
            "Sunderland 2-0 Wolves (Oct 18, 2025) Final Score - ESPN", "score table",
        ) is True

    def test_flags_recap_language_without_score_in_title(self):
        from src.agent.tools import _looks_like_post_match_result
        assert _looks_like_post_match_result(
            "Sunderland maintain fine home form as Wolves lose again - BBC Sport",
            "Sunderland maintained their fine start to the Premier League season "
            "as they condemned Wolves to a sixth defeat",
        ) is True

    def test_flags_match_report_instant_reaction(self):
        from src.agent.tools import _looks_like_post_match_result
        assert _looks_like_post_match_result(
            "Everton 2-1 Crystal Palace: Match Report & Instant Reaction",
            "The biggest miss for Everton was the suspension to Kiernan Dewsbury-Hall.",
        ) is True

    def test_does_not_flag_head_to_head_table(self):
        """False positive from the same investigation: a score coincidentally
        matching elsewhere in a legitimate historical head-to-head table,
        with no score in the title itself and no recap-language marker."""
        from src.agent.tools import _looks_like_post_match_result
        assert _looks_like_post_match_result(
            "Liverpool football club: record v Nottingham Forest",
            "14 Sep 2024 | Liverpool v Nottingham Forest | L | 0-1 | Premier League | "
            "14 Jan 2025 | Nottingham Forest v Liverpool | D | 1-1 | Premier League",
        ) is False

    def test_does_not_flag_prediction_article_referencing_a_past_meeting(self):
        from src.agent.tools import _looks_like_post_match_result
        assert _looks_like_post_match_result(
            "Wolves vs Liverpool Prediction: Team News",
            "The most recent encounter between these two sides was a Premier League "
            "fixture at Anfield, which ended in a 2-1 home win for Liverpool.",
        ) is False

    def test_web_search_impl_drops_leaked_result_keeps_clean_one(self):
        from src.agent.tools import _web_search_impl
        with patch("src.agent.tools.os.environ.get", return_value="fake-key"), \
             patch("tavily.TavilyClient") as MockClient:
            instance = MagicMock()
            MockClient.return_value = instance
            instance.search.return_value = {"results": [
                {"title": "Sunderland 2-0 Wolves (Oct 18, 2025) Final Score - ESPN", "content": "x", "url": "u1"},
                {"title": "Sunderland v Wolves Confirmed Starting Lineups", "content": "team news", "url": "u2"},
            ]}
            result = _web_search_impl("Sunderland Wolves recent form")

        assert "Final Score" not in result
        assert "Confirmed Starting Lineups" in result

    def test_web_search_impl_returns_no_results_found_when_everything_filtered(self):
        from src.agent.tools import _web_search_impl
        with patch("src.agent.tools.os.environ.get", return_value="fake-key"), \
             patch("tavily.TavilyClient") as MockClient:
            instance = MagicMock()
            MockClient.return_value = instance
            instance.search.return_value = {"results": [
                {"title": "Team A 2-0 Team B Final Score - ESPN", "content": "x", "url": "u1"},
            ]}
            result = _web_search_impl("Team A Team B recent form")

        assert result == "No results found."


def test_web_search_date_filter_applied_during_record_and_replay(tmp_path):
    with patch("tavily.TavilyClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        instance.search.return_value = {"results": [{"title": "T", "content": "C", "url": "U"}]}

        with patch("src.agent.tools.os.environ.get", return_value="fake-key"):
            agent_tools.configure_snapshot_store("record", match_id="m5", match_date="2025-03-01")
            first = agent_tools.web_search.invoke({"query": "team news"})
            assert instance.search.call_args.kwargs["query"] == "team news before:2025-03-01"
            assert instance.search.call_count == 1

            # Replay must compute the SAME effective query (so the same SHA-256 key),
            # find the file record mode wrote, and NOT call Tavily again.
            agent_tools.configure_snapshot_store("replay", match_id="m5", match_date="2025-03-01")
            second = agent_tools.web_search.invoke({"query": "team news"})
            assert second == first
            assert instance.search.call_count == 1  # unchanged — proves replay didn't hit Tavily


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


def test_forecast_international_record_then_replay():
    fake_result = {"result_3way": {"probabilities": {"home": 0.4}}, "data_quality": {}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        agent_tools.configure_snapshot_store("record", match_id="m6")
        first = agent_tools.forecast_international.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01",
            "odds_h": 2.0, "odds_d": 3.0, "odds_a": 3.5,
        })
        assert instance.forecast_upcoming.call_count == 1

        agent_tools.configure_snapshot_store("replay", match_id="m6")
        second = agent_tools.forecast_international.invoke({
            "home_team": "A", "away_team": "B", "date": "2025-01-01",
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


def test_get_default_tools_only_exposes_web_search():
    from src.agent.tools import get_default_tools
    assert [t.name for t in get_default_tools()] == ["web_search"]
