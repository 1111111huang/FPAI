"""Tests for outcome loading and BacktestHarness (A12)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.agent_config import AgentConfig
from src.agent.backtest import (
    BacktestHarness,
    BacktestRecord,
    load_outcome,
    match_in_test_split,
    process_match_row,
)
from src.agent.snapshot_store import SnapshotMissingError


def _make_config(**overrides) -> AgentConfig:
    defaults = dict(
        model="stub-model", provider="ollama", temperature=0.0, max_tool_calls=5,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_value_edge=0.05, markets=["result_3way"],
        system_prompt_version="v1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _row(**overrides) -> pd.Series:
    base = dict(
        match_id="m1", league="E0", date=pd.Timestamp("2025-03-01"),
        home_team="City", away_team="Arsenal",
        odds_h=1.9, odds_d=3.5, odds_a=4.0,
        fthg=2, ftag=1, hc=5.0, ac=4.0,
    )
    base.update(overrides)
    return pd.Series(base)


def test_load_outcome_home_win():
    outcome = load_outcome(_row(fthg=2, ftag=1))
    assert outcome["result"] == "home"
    assert outcome["btts"] == "yes"
    assert outcome["total_goals"] == 3
    assert outcome["total_goals_side"] == "over_2.5"


def test_load_outcome_draw_and_no_btts():
    outcome = load_outcome(_row(fthg=0, ftag=0))
    assert outcome["result"] == "draw"
    assert outcome["btts"] == "no"
    assert outcome["total_goals_side"] == "under_2.5"


def test_load_outcome_away_win():
    outcome = load_outcome(_row(fthg=0, ftag=2))
    assert outcome["result"] == "away"


def test_process_match_row_scores_markets_correctly():
    recommendation = {
        "match": {}, "overall": "direct_bet",
        "markets": [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 1.9, "min_odds": 1.9, "ml_probability": 0.6, "implied_probability": 0.52, "value_edge": 0.08},
            {"market": "btts", "selection": "yes", "recommendation_type": "no_bet", "current_odds": 1.8, "min_odds": 2.0, "ml_probability": 0.5, "implied_probability": 0.55, "value_edge": -0.05},
            {"market": "home_corners", "selection": "over_4.5", "recommendation_type": "direct_bet", "current_odds": 1.9, "min_odds": 1.9, "ml_probability": 0.5, "implied_probability": 0.52, "value_edge": -0.02},
        ],
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure:
        record = process_match_row(_row(fthg=2, ftag=1), _make_config())

    assert isinstance(record, BacktestRecord)
    assert record.actual["result"] == "home"
    by_market = {m["market"]: m for m in record.market_results}
    assert by_market["result_3way"]["correct"] is True
    assert by_market["btts"]["correct"] is True  # actual btts is "yes" (2-1, both scored); selection was "yes"
    assert by_market["home_corners"]["correct"] is None  # unresolvable, documented limitation

    # configure_snapshot_store called with replay then live (record_calls captures the mode transitions)
    modes_used = [call.args[0] for call in mock_configure.call_args_list]
    assert modes_used == ["replay", "live"]
    mock_run.assert_called_once()


def test_process_match_row_uses_league_scoped_base_dir():
    """BUG-022: replay must look in the same per-league directory
    agent-snapshot recorded into, not the old flat shared directory."""
    from src.agent.snapshot_store import league_base_dir

    recommendation = {
        "match": {}, "overall": "no_bet", "markets": [],
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation), \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure:
        process_match_row(_row(league="SWE", fthg=1, ftag=1), _make_config())

    replay_call = mock_configure.call_args_list[0]
    assert replay_call.kwargs["base_dir"] == league_base_dir("SWE")


def test_process_match_row_threads_allow_lessons_in_replay_to_configure_snapshot_store():
    recommendation = {
        "match": {}, "overall": "no_bet", "markets": [],
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation), \
         patch("src.agent.tools.configure_snapshot_store") as mock_configure:
        process_match_row(_row(), _make_config(), allow_lessons_in_replay=True)

    replay_call = mock_configure.call_args_list[0]
    assert replay_call.kwargs["allow_lessons_in_replay"] is True


def test_process_match_row_propagates_snapshot_missing_error():
    with patch("src.agent.graph.run_agent", side_effect=SnapshotMissingError("web_search", "m1", "abc")), \
         patch("src.agent.tools.configure_snapshot_store"):
        with pytest.raises(SnapshotMissingError):
            process_match_row(_row(), _make_config())


def test_process_match_row_captures_full_state_when_requested():
    full_state = {
        "recommendation": {
            "match": {}, "overall": "no_bet", "markets": [],
            "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
        },
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
    }
    with patch("src.agent.graph.run_agent", return_value=full_state) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"):
        record = process_match_row(_row(fthg=1, ftag=1), _make_config(), capture_state=True)

    assert record.full_state == full_state
    assert record.recommendation == full_state["recommendation"]
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["return_full_state"] is True


def test_process_match_row_full_state_none_by_default():
    recommendation = {
        "match": {}, "overall": "no_bet", "markets": [],
        "explanation": "x", "confidence": "high", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    with patch("src.agent.graph.run_agent", return_value=recommendation) as mock_run, \
         patch("src.agent.tools.configure_snapshot_store"):
        record = process_match_row(_row(fthg=1, ftag=1), _make_config())

    assert record.full_state is None
    assert "return_full_state" not in mock_run.call_args.kwargs


def test_backtest_harness_load_matches_filters_by_date_and_league():
    harness = BacktestHarness(config=_make_config())
    fake_df = pd.DataFrame([
        _row(match_id="a", date=pd.Timestamp("2025-01-15")),
        _row(match_id="b", date=pd.Timestamp("2025-02-15")),
    ])
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df
    with patch.object(harness.db, "connection") as mock_connection:
        mock_connection.return_value.__enter__.return_value = mock_conn
        result = harness.load_matches("2025-01-01", "2025-03-01", league="E0")

    assert len(result) == 2
    sql_used = mock_conn.execute.call_args[0][0]
    assert "raw_matches" in sql_used
    assert "league" in sql_used.lower()


def test_backtest_harness_stratified_sample_balances_result_categories():
    harness = BacktestHarness(config=_make_config())
    rows = (
        [_row(match_id=f"h{i}", fthg=2, ftag=0, date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)) for i in range(6)]
        + [_row(match_id=f"d{i}", fthg=1, ftag=1, date=pd.Timestamp("2025-02-01") + pd.Timedelta(days=i)) for i in range(6)]
        + [_row(match_id=f"a{i}", fthg=0, ftag=2, date=pd.Timestamp("2025-03-01") + pd.Timedelta(days=i)) for i in range(6)]
    )
    df = pd.DataFrame(rows)
    sampled = harness._stratified_sample(df, sample=9)
    assert len(sampled) <= 9

    def _result(r):
        return "home" if r["fthg"] > r["ftag"] else ("away" if r["fthg"] < r["ftag"] else "draw")

    counts = sampled.apply(_result, axis=1).value_counts()
    # Balanced: 9 requested from 3 equal groups of 6 -> per_stratum = 3, so each
    # category should contribute exactly 3 rows (not just "present").
    assert set(counts.index) == {"home", "draw", "away"}
    assert counts.to_dict() == {"home": 3, "draw": 3, "away": 3}


def test_match_in_test_split_is_deterministic():
    match_id = "78da66d1356eb6254a5015ec90ffb819a5bd751ca41ba411cf0f6a618663932d"
    assert match_in_test_split(match_id, 0.2) == match_in_test_split(match_id, 0.2)


def test_match_in_test_split_roughly_matches_fraction():
    match_ids = [f"match-{i}" for i in range(2000)]
    test_count = sum(match_in_test_split(m, 0.2) for m in match_ids)
    # Hash-bucketed, not exact -- allow a wide statistical band rather than pinning a count.
    assert 300 <= test_count <= 500


def test_backtest_harness_load_matches_train_test_split_is_disjoint_and_stable():
    harness = BacktestHarness(config=_make_config())
    fake_df = pd.DataFrame(
        [_row(match_id=f"m{i}", date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)) for i in range(200)]
    )
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = fake_df
    with patch.object(harness.db, "connection") as mock_connection:
        mock_connection.return_value.__enter__.return_value = mock_conn
        train = harness.load_matches("2025-01-01", "2025-12-31", split="train", test_fraction=0.2)
        test = harness.load_matches("2025-01-01", "2025-12-31", split="test", test_fraction=0.2)

    train_ids, test_ids = set(train["match_id"]), set(test["match_id"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(fake_df["match_id"])
    assert 0 < len(test_ids) < len(train_ids)  # roughly 20/80, never empty or majority for this size


def test_backtest_harness_load_matches_rejects_invalid_split():
    harness = BacktestHarness(config=_make_config())
    with pytest.raises(ValueError):
        harness.load_matches("2025-01-01", "2025-12-31", split="bogus")


def test_run_agent_backtest_rejects_use_lessons_without_split_test():
    from main import run_agent_backtest

    with pytest.raises(ValueError, match="--split test"):
        run_agent_backtest(
            from_date="2025-01-01", to_date="2025-12-31", league="E0", stake_mode="flat",
            sample=None, concurrency=5, config_path=None, split="all", use_lessons=True,
        )


def test_backtest_harness_run_uses_process_match_row():
    harness = BacktestHarness(config=_make_config())
    fake_df = pd.DataFrame([_row(match_id="only-one")])
    with patch.object(harness, "load_matches", return_value=fake_df), \
         patch("src.agent.backtest.process_match_row") as mock_process:
        mock_process.return_value = "RECORD-SENTINEL"
        records = harness.run("2025-01-01", "2025-12-31")

    assert records == ["RECORD-SENTINEL"]
    mock_process.assert_called_once()
