"""Tests for the deterministic evidence pipeline nodes (A31/A32):
resolve_competition_node, research_node, and forecast_node run before the LLM
ever sees the match, replacing the old LLM-tool-choice path for competition
resolution, baseline research, and the ML forecast."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent import tools as agent_tools
from src.agent.pipeline import resolve_competition_node


@pytest.fixture(autouse=True)
def reset_snapshot_store():
    agent_tools._snapshot_store.set_mode("live")
    yield
    agent_tools._snapshot_store.set_mode("live")


def _base_state(**overrides) -> dict:
    state = {
        "messages": [],
        "match_info": {"home_team": "Man City", "away_team": "Arsenal", "date": "2026-06-21", "league": "E0"},
        "recommendation": None,
        "tool_call_count": 0,
        "competition_resolution": None,
        "research_evidence": None,
        "forecast_payload": None,
    }
    state.update(overrides)
    return state


def test_resolve_competition_node_uses_real_registry_for_known_league():
    result = resolve_competition_node(_base_state())
    assert result["competition_resolution"]["tier"] == "competition_specific"
    assert result["competition_resolution"]["recommended_tool"] == "forecast_league"


def test_resolve_competition_node_uses_real_registry_for_second_competition_specific_league():
    """A35: SWE (Allsvenskan) is a second real competition_specific registration
    (config/competitions.yaml) alongside E0 -- this must resolve the same way
    E0 does, not fall through to general_purpose just because it isn't the
    only registered competition_specific league."""
    state = _base_state(match_info={"home_team": "Malmo FF", "away_team": "AIK", "date": "2026-07-25", "league": "SWE"})
    result = resolve_competition_node(state)
    assert result["competition_resolution"]["tier"] == "competition_specific"
    assert result["competition_resolution"]["recommended_tool"] == "forecast_league"


def test_resolve_competition_node_defaults_general_purpose_when_no_league_supplied():
    state = _base_state(match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21"})
    result = resolve_competition_node(state)
    assert result["competition_resolution"] == {
        "competition": None,
        "tier": "general_purpose",
        "recommended_tool": "forecast_international",
    }


def test_resolve_competition_node_goes_through_snapshot_store():
    with patch(
        "src.agent.tools._resolve_competition_impl",
        wraps=agent_tools._resolve_competition_impl,
    ) as spy:
        resolve_competition_node(_base_state())
    spy.assert_called_once_with(competition_or_league="E0")


from src.agent.pipeline import _parse_odds_from_search_text, research_node


def test_parse_odds_from_search_text_finds_three_plausible_numbers():
    text = "Best odds: Man City 1.45, Draw 4.50, Arsenal 7.00 at time of writing."
    assert _parse_odds_from_search_text(text) == {"home": 1.45, "draw": 4.50, "away": 7.00}


def test_parse_odds_from_search_text_returns_none_with_fewer_than_three_numbers():
    assert _parse_odds_from_search_text("Man City are favourites at 1.45.") is None


def test_parse_odds_from_search_text_returns_none_for_empty_text():
    assert _parse_odds_from_search_text("") is None
    assert _parse_odds_from_search_text(None) is None


def test_research_node_runs_availability_and_form_searches_only_when_odds_supplied():
    with patch("src.agent.tools._dated_web_search", side_effect=["injury text", "form text"]) as mock_search:
        state = _base_state(match_info={
            "home_team": "A", "away_team": "B", "date": "2026-06-21",
            "odds": {"home": 2.0, "draw": 3.0, "away": 3.5},
        })
        result = research_node(state)

    assert result["research_evidence"]["availability"] == "injury text"
    assert result["research_evidence"]["form_context"] == "form text"
    assert result["research_evidence"]["odds_verification"] is None
    assert mock_search.call_count == 2


def test_research_node_also_runs_odds_search_when_caller_supplied_no_odds():
    with patch(
        "src.agent.tools._dated_web_search",
        side_effect=["injury text", "form text", "Man City 1.45 Draw 4.50 Arsenal 7.00"],
    ) as mock_search:
        state = _base_state(match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21"})
        result = research_node(state)

    assert mock_search.call_count == 3
    odds_verification = result["research_evidence"]["odds_verification"]
    assert odds_verification["parsed_odds"] == {"home": 1.45, "draw": 4.50, "away": 7.00}


from src.agent.pipeline import _format_evidence_message, forecast_node


def test_forecast_node_prefers_caller_supplied_odds_over_research_odds():
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {"prediction_basis": "team_history_and_market"}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        state = _base_state(
            match_info={
                "home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0",
                "odds": {"home": 2.0, "draw": 3.0, "away": 3.5},
            },
            competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
            research_evidence={"availability": "x", "form_context": "y", "odds_verification": None},
        )
        result = forecast_node(state)

    assert "error" not in result["forecast_payload"]
    call_kwargs = instance.forecast_upcoming.call_args.kwargs
    assert (call_kwargs["odds_h"], call_kwargs["odds_d"], call_kwargs["odds_a"]) == (2.0, 3.0, 3.5)
    assert any("FORECAST_PAYLOAD" in m.content for m in result["messages"])


def test_forecast_node_falls_back_to_research_odds_when_caller_supplied_none():
    fake_result = {"result_3way": {"probabilities": {"home": 0.5}}, "data_quality": {}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        state = _base_state(
            match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0"},
            competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
            research_evidence={
                "availability": "x", "form_context": "y",
                "odds_verification": {"results": "...", "parsed_odds": {"home": 1.9, "draw": 3.4, "away": 4.0}},
            },
        )
        result = forecast_node(state)

    call_kwargs = instance.forecast_upcoming.call_args.kwargs
    assert (call_kwargs["odds_h"], call_kwargs["odds_d"], call_kwargs["odds_a"]) == (1.9, 3.4, 4.0)


def test_forecast_node_returns_no_odds_error_when_neither_source_has_odds():
    state = _base_state(
        match_info={"home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0"},
        competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
        research_evidence={"availability": "x", "form_context": "y", "odds_verification": {"results": "no odds mentioned", "parsed_odds": None}},
    )
    result = forecast_node(state)
    assert result["forecast_payload"]["status"] == "no_odds"
    assert "error" in result["forecast_payload"]
    assert "messages" not in result


def test_forecast_node_calls_forecast_international_when_recommended():
    fake_result = {"result_3way": {"probabilities": {"home": 0.4}}, "data_quality": {"prediction_basis": "market_odds_only"}}
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_result

        state = _base_state(
            match_info={
                "home_team": "Real Madrid", "away_team": "Barcelona", "date": "2026-06-21",
                "league": "La Liga", "odds": {"home": 2.1, "draw": 3.4, "away": 3.3},
            },
            competition_resolution={"competition": "La Liga", "tier": "general_purpose", "recommended_tool": "forecast_international"},
        )
        result = forecast_node(state)

    assert instance.forecast_upcoming.call_args.kwargs["match_type"] == "international"
    assert "error" not in result["forecast_payload"]


def test_forecast_node_propagates_tool_error_payload():
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        MockSvc.side_effect = RuntimeError("boom")
        state = _base_state(
            match_info={
                "home_team": "A", "away_team": "B", "date": "2026-06-21", "league": "E0",
                "odds": {"home": 2.0, "draw": 3.0, "away": 3.5},
            },
            competition_resolution={"competition": "E0", "tier": "competition_specific", "recommended_tool": "forecast_league"},
        )
        result = forecast_node(state)

    assert result["forecast_payload"]["status"] == "tool_error"
    assert "messages" not in result


def test_format_evidence_message_includes_forecast_and_research_evidence():
    payload = {"result_3way": {"probabilities": {"home": 0.5}}}
    evidence = {"availability": "no injuries", "form_context": "won last 3", "odds_verification": {"results": "odds text", "parsed_odds": None}}
    message = _format_evidence_message(payload, evidence)

    assert "no injuries" in message
    assert "won last 3" in message
    assert "odds text" in message
    assert "result_3way" in message
    assert "resolve_competition" in message  # tells the LLM not to call it


def test_format_evidence_message_has_no_markdown_headers():
    """BUG-019: markdown headers (##/###) caused local models to pattern-complete
    a narrative report instead of emitting the required JSON."""
    payload = {"result_3way": {"probabilities": {"home": 0.5}}}
    evidence = {"availability": "no injuries", "form_context": "won last 3", "odds_verification": None}
    message = _format_evidence_message(payload, evidence)

    assert "##" not in message
    assert "###" not in message


def test_lessons_node_returns_empty_dict_when_not_live_mode():
    from unittest.mock import patch
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("replay")
    try:
        with patch("src.agent.lessons.load_approved_lessons") as mock_load:
            result = lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}})
        mock_load.assert_not_called()
        assert result == {}
    finally:
        agent_tools._snapshot_store.set_mode("live")


def test_lessons_node_returns_empty_dict_when_no_approved_lessons():
    from unittest.mock import patch, MagicMock
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    with patch("src.agent.lessons.load_approved_lessons", return_value=[]), \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        result = lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}})
    assert result == {}


def test_lessons_node_appends_human_message_with_approved_lessons():
    from unittest.mock import patch, MagicMock
    from langchain_core.messages import HumanMessage
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    with patch("src.agent.lessons.load_approved_lessons", return_value=["Lesson A", "Lesson B"]), \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        result = lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}})

    assert len(result["messages"]) == 1
    message = result["messages"][0]
    assert isinstance(message, HumanMessage)
    assert "Lesson A" in message.content
    assert "Lesson B" in message.content


def test_lessons_node_uses_competition_resolution_from_state():
    from unittest.mock import patch, MagicMock
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    with patch("src.agent.lessons.load_approved_lessons", return_value=[]) as mock_load, \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        lessons_node({"competition_resolution": {"competition": "SP1", "tier": "competition_specific"}})

    args = mock_load.call_args.args
    assert args[1] == "SP1"
    assert args[2] == "competition_specific"


def test_lessons_node_returns_empty_dict_when_db_file_does_not_exist(tmp_path):
    """Critical bug fix: duckdb.connect(..., read_only=True) raises
    duckdb.IOException (not duckdb.CatalogException) when the DB *file*
    itself doesn't exist yet -- distinct from the missing-table case
    load_approved_lessons already handles. This is realistically reachable
    on a fresh deployment's first live run (e.g. an international match,
    where forecast_node never touches DuckDB) before agent-train has ever
    run create_lessons_tables. Uses a real DuckDBManager pointed at a temp
    config so the real read_only connection attempt hits a genuinely
    nonexistent file, rather than mocking DuckDBManager away."""
    import yaml
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools
    from src.utils.db_manager import DuckDBManager

    db_path = tmp_path / "does_not_exist.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    assert not db_path.exists()

    agent_tools._snapshot_store.set_mode("live")
    try:
        with patch("src.utils.db_manager.DuckDBManager", lambda: DuckDBManager(config_path=str(config_path))):
            result = lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}})
    finally:
        agent_tools._snapshot_store.set_mode("live")

    assert result == {}


def test_pipeline_module_never_imports_lesson_write_or_review_functions():
    """A33 acceptance: the live code path (this module) must have no
    function available to it that can write, approve, or reject lessons --
    only load_approved_lessons, which itself can't read outcomes (see
    tests/test_agent_lessons.py)."""
    import pathlib
    source = pathlib.Path("src/agent/pipeline.py").read_text()
    assert "load_approved_lessons" in source
    for forbidden in ("insert_lesson_candidate", "insert_telemetry", "approve_lesson", "reject_lesson"):
        assert forbidden not in source
