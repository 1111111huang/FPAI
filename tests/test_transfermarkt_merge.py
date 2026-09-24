"""Tests for the player_market_values dated-snapshot table."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.transfermarkt.merge import insert_market_value_snapshot
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _values_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"fotmob_player_id": 162549, "transfermarkt_player_id": 96148, "market_value_eur": 220_000_000},
    ])


def test_insert_market_value_snapshot_stamps_the_given_date(tmp_path: Path):
    db_manager = _make_db_manager(tmp_path)
    n = insert_market_value_snapshot(_values_df(), db_manager, snapshot_date="2026-09-23")

    assert n == 1
    with db_manager.connection(read_only=True) as conn:
        rows = conn.execute("SELECT fotmob_player_id, snapshot_date, market_value_eur FROM player_market_values").fetchall()
    assert rows == [(162549, "2026-09-23", 220_000_000)]


def test_insert_market_value_snapshot_never_overwrites_an_earlier_snapshot(tmp_path: Path):
    """Point-in-time correctness (design spec's whole reason for a dated-
    snapshot schema instead of a plain upsert) -- inserting a new snapshot
    date must leave an earlier one intact, both rows present."""
    db_manager = _make_db_manager(tmp_path)
    insert_market_value_snapshot(_values_df(), db_manager, snapshot_date="2026-08-01")
    insert_market_value_snapshot(_values_df(), db_manager, snapshot_date="2026-09-23")

    with db_manager.connection(read_only=True) as conn:
        dates = conn.execute(
            "SELECT snapshot_date FROM player_market_values WHERE fotmob_player_id = 162549 ORDER BY snapshot_date"
        ).fetchall()
    assert dates == [("2026-08-01",), ("2026-09-23",)]
