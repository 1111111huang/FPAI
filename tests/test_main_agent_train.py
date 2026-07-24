"""Tests for main.py's run_agent_train CLI entry point (A33)."""
from __future__ import annotations

import duckdb

from main import _write_train_artifacts
from src.agent.backtest import BacktestRecord


def _record(match_id="m1", league="E0", full_state=None) -> BacktestRecord:
    return BacktestRecord(
        match_id=match_id,
        home_team="City",
        away_team="Arsenal",
        date="2025-03-01",
        league=league,
        recommendation={
            "overall": "no_bet", "confidence": "medium",
            "prediction_basis": "team_history_and_market", "limitations": [],
        },
        actual={"result": "home", "btts": "yes", "total_goals": 3, "total_goals_side": "over_2.5"},
        market_results=[],
        full_state=full_state,
    )


def test_write_train_artifacts_writes_one_lesson_and_telemetry_row_per_record():
    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
    })

    written = _write_train_artifacts(conn, [record], run_id="run-1")

    assert written == 1
    lesson_row = conn.execute(
        "SELECT competition_id, tier, status, source_match_id FROM agent_lessons"
    ).fetchone()
    assert lesson_row == ("E0", "competition_specific", "pending", "m1")

    telemetry_row = conn.execute("SELECT match_id, run_id FROM agent_telemetry").fetchone()
    assert telemetry_row == ("m1", "run-1")


def test_write_train_artifacts_skips_records_without_full_state():
    conn = duckdb.connect(":memory:")
    record = _record(full_state=None)

    written = _write_train_artifacts(conn, [record], run_id="run-1")

    assert written == 0
    assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM agent_telemetry").fetchone()[0] == 0


def test_write_train_artifacts_handles_multiple_records():
    conn = duckdb.connect(":memory:")
    records = [
        _record(match_id="m1", league="E0", full_state={
            "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        }),
        _record(match_id="m2", league="SP1", full_state={
            "competition_resolution": {"competition": "SP1", "tier": "competition_specific"},
        }),
    ]

    written = _write_train_artifacts(conn, records, run_id="run-2")

    assert written == 2
    assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM agent_telemetry").fetchone()[0] == 2
