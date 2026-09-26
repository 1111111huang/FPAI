"""Tests for main.py's run_agent_snapshot CLI entry point, specifically
BUG-022's per-league directory partitioning.

A97: run_agent_snapshot calls run_deterministic_pipeline (resolve_competition
-> research -> forecast only, no LLM) instead of the full run_agent -- the
LLM turn was never part of the persisted snapshot corpus anyway (nothing
wraps agent_node/output_node), so it was a wasted API call per match in
both plain and --refresh-model modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from main import run_agent_snapshot
from src.agent.backtest import match_in_test_split
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
         patch("src.agent.graph.run_deterministic_pipeline", return_value={}) as mock_run, \
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
         patch("src.agent.graph.run_deterministic_pipeline") as mock_run, \
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
         patch("src.agent.graph.run_deterministic_pipeline", return_value={}) as mock_run, \
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
         patch("src.agent.graph.run_deterministic_pipeline", return_value={}), \
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


def test_default_refresh_model_false_still_skips_and_uses_full_record():
    """Regression safety: refresh_model defaults to False, so every existing
    caller's behavior (record mode, skip-if-complete) is unchanged."""
    import inspect
    assert inspect.signature(run_agent_snapshot).parameters["refresh_model"].default is False


# ---------------------------------------------------------------------------
# --backfill-missing (BUG-072/A56/A115): a corpus recorded before research_node
# grew a new deterministic web_search query is otherwise permanently
# unreplayable -- replay mode hard-fails on ANY missing key, so 100% of
# matches skip even though only one of several web_search calls per match is
# actually new. backfill_missing reprocesses already-complete matches,
# replaying every existing key and live-fetching+recording only ones that
# are missing.
# ---------------------------------------------------------------------------

def test_backfill_missing_reprocesses_already_complete_matches(tmp_path):
    marker_dir = league_base_dir("SWE", base_dir=tmp_path) / "m1"
    marker_dir.mkdir(parents=True)
    (marker_dir / "_complete.json").write_text("{}")

    fake_df = _fake_matches_df()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_deterministic_pipeline", return_value={}) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"), \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(
            from_date="2026-07-01", to_date="2026-07-02", league="SWE",
            config_path=None, dry_run=False, backfill_missing=True,
        )

    mock_run.assert_called_once()


def test_backfill_missing_uses_replay_with_web_search_overridden_to_record_missing(tmp_path):
    fake_df = _fake_matches_df()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_deterministic_pipeline", return_value={}), \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure, \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(
            from_date="2026-07-01", to_date="2026-07-02", league="SWE",
            config_path=None, dry_run=False, backfill_missing=True,
        )

    backfill_call = mock_configure.call_args_list[0]
    assert backfill_call.args[0] == "replay"
    assert backfill_call.kwargs["tool_mode_overrides"] == {"web_search": "record_missing"}


def test_default_backfill_missing_false_still_skips_and_uses_full_record():
    import inspect
    assert inspect.signature(run_agent_snapshot).parameters["backfill_missing"].default is False


# ---------------------------------------------------------------------------
# A111: --split lets recording target only the agent's own train or test
# partition (BacktestHarness's match_in_test_split, A40) instead of always
# recording every match in the date range -- needed so a snapshot-recording
# pass can be scoped to (and its Tavily-call cost budgeted for) just the
# held-out test split, without also touching the much larger train split.
# ---------------------------------------------------------------------------

def test_split_test_only_processes_test_split_matches(tmp_path):
    # Two match_ids that are known (via match_in_test_split's stable hash) to
    # land on opposite sides of the default 0.2 test_fraction.
    ids = [f"m{i}" for i in range(20)]
    test_ids = [m for m in ids if match_in_test_split(m, 0.2)]
    train_ids = [m for m in ids if not match_in_test_split(m, 0.2)]
    assert test_ids and train_ids  # sanity: fixture actually covers both sides

    fake_df = pd.DataFrame({
        "match_id": ids, "league": ["SWE"] * len(ids),
        "date": [pd.Timestamp("2026-07-01")] * len(ids),
        "home_team": ["AIK"] * len(ids), "away_team": ["GAIS"] * len(ids),
        "odds_h": [2.0] * len(ids), "odds_d": [3.2] * len(ids), "odds_a": [3.5] * len(ids),
    })
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df
    processed = []

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_deterministic_pipeline", side_effect=lambda info: processed.append(info) or {}), \
         patch("src.agent.tools.configure_snapshot_store"), \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(
            from_date="2026-07-01", to_date="2026-07-02", league="SWE",
            config_path=None, dry_run=False, split="test",
        )

    processed_ids = {p["match_id"] for p in processed} if processed and "match_id" in processed[0] else None
    # match_info doesn't carry match_id today -- assert via count instead,
    # which is still a real, meaningful regression guard on the filter.
    assert len(processed) == len(test_ids)


def test_split_all_is_the_default_and_processes_every_match(tmp_path):
    import inspect
    assert inspect.signature(run_agent_snapshot).parameters["split"].default == "all"

    ids = [f"m{i}" for i in range(10)]
    fake_df = pd.DataFrame({
        "match_id": ids, "league": ["SWE"] * len(ids),
        "date": [pd.Timestamp("2026-07-01")] * len(ids),
        "home_team": ["AIK"] * len(ids), "away_team": ["GAIS"] * len(ids),
        "odds_h": [2.0] * len(ids), "odds_d": [3.2] * len(ids), "odds_a": [3.5] * len(ids),
    })
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df

    with patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.graph.run_deterministic_pipeline", return_value={}) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"), \
         patch("main.DEFAULT_BASE_DIR", tmp_path):
        MockDB.return_value.connection.return_value.__enter__.return_value = mock_conn

        run_agent_snapshot(from_date="2026-07-01", to_date="2026-07-02", league="SWE", config_path=None, dry_run=False)

    assert mock_run.call_count == len(ids)
