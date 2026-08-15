"""Tests for main.py's run_agent_snapshot CLI entry point, specifically
BUG-022's per-league directory partitioning."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from main import run_agent_snapshot
from src.agent.snapshot_store import league_base_dir


def _fake_matches_df(**overrides) -> pd.DataFrame:
    base = dict(
        match_id=["m1"], league=["SWE"], date=[pd.Timestamp("2026-07-01")],
        home_team=["AIK"], away_team=["GAIS"],
        odds_h=[2.0], odds_d=[3.2], odds_a=[3.5],
    )
    base.update(overrides)
    return pd.DataFrame(base)


def test_run_agent_snapshot_uses_league_scoped_base_dir(tmp_path):
    fake_df = _fake_matches_df()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_agent", return_value={"overall": "no_bet"}) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure, \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(from_date="2026-07-01", to_date="2026-07-02", league="SWE", config_path=None, dry_run=False)

    record_call = mock_configure.call_args_list[0]
    assert record_call.args[0] == "record"
    assert record_call.kwargs["base_dir"] == league_base_dir("SWE", base_dir=tmp_path)
    mock_run.assert_called_once()

    marker = league_base_dir("SWE", base_dir=tmp_path) / "m1" / "_complete.json"
    assert marker.exists()


def test_run_agent_snapshot_skips_already_complete_matches_per_league(tmp_path):
    """A match completed under SWE's subdirectory must not be re-processed;
    an identical match_id under a different league's subdirectory (which
    can't happen in practice since match_id already encodes league via the
    source data, but the point is the check is league-scoped) is unaffected."""
    marker_dir = league_base_dir("SWE", base_dir=tmp_path) / "m1"
    marker_dir.mkdir(parents=True)
    (marker_dir / "_complete.json").write_text("{}")

    fake_df = _fake_matches_df()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_agent") as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"), \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(from_date="2026-07-01", to_date="2026-07-02", league="SWE", config_path=None, dry_run=False)

    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# --refresh-model: reprocess already-complete matches to pick up a new model,
# without re-fetching web_search/resolve_competition from scratch.
# ---------------------------------------------------------------------------

def test_refresh_model_reprocesses_already_complete_matches(tmp_path):
    marker_dir = league_base_dir("SWE", base_dir=tmp_path) / "m1"
    marker_dir.mkdir(parents=True)
    (marker_dir / "_complete.json").write_text("{}")

    fake_df = _fake_matches_df()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_agent", return_value={"overall": "no_bet"}) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"), \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(
            from_date="2026-07-01", to_date="2026-07-02", league="SWE",
            config_path=None, dry_run=False, refresh_model=True,
        )

    mock_run.assert_called_once()


def test_refresh_model_uses_replay_with_forecast_tools_overridden_to_record(tmp_path):
    fake_df = _fake_matches_df()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_agent", return_value={"overall": "no_bet"}), \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure, \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(
            from_date="2026-07-01", to_date="2026-07-02", league="SWE",
            config_path=None, dry_run=False, refresh_model=True,
        )

    refresh_call = mock_configure.call_args_list[0]
    assert refresh_call.args[0] == "replay"
    assert refresh_call.kwargs["tool_mode_overrides"] == {
        "forecast_league": "record", "forecast_international": "record",
    }


def test_refresh_model_disables_llm_optional_tools(tmp_path):
    """Mirrors A42's process_match_row fix: the LLM's own optional follow-up
    web_search call (self-generated query text) almost never matches an
    existing recorded key, which would abort the whole match with
    SnapshotMissingError. tools=[] prevents the LLM from exercising it at
    all -- research_node's own deterministic pre-fetch calls still replay
    fine since their query text is template-fixed, unchanged run to run."""
    fake_df = _fake_matches_df()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_agent", return_value={"overall": "no_bet"}) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"), \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(
            from_date="2026-07-01", to_date="2026-07-02", league="SWE",
            config_path=None, dry_run=False, refresh_model=True,
        )

    assert mock_run.call_args.kwargs["tools"] == []


def test_default_refresh_model_false_still_skips_and_uses_full_record():
    """Regression safety: refresh_model defaults to False, so every existing
    caller's behavior (record mode, skip-if-complete) is unchanged."""
    import inspect
    assert inspect.signature(run_agent_snapshot).parameters["refresh_model"].default is False
