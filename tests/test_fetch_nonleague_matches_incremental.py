"""Regression tests for US#212's scheduled refresh-data step:
run_fetch_nonleague_matches_incremental must fetch only the days since the
last run, not re-scan the full multi-year history every week (that's a
separate, manually-triggered backfill via the fetch-nonleague-matches CLI).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

import main
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def test_first_run_looks_back_a_fixed_window_not_the_full_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fotmob_nonleague_matches rows yet (table doesn't exist at all) --
    must use the small first-run lookback window, not raw_matches' full
    ~10-year span."""
    db_manager = _make_db_manager(tmp_path)
    captured: dict = {}

    def _fake_fetch_and_upsert(db_manager, from_d, to_d, delay=1.0):
        captured["from_d"] = from_d
        captured["to_d"] = to_d
        return {"days_scanned": (to_d - from_d).days + 1, "matches_seen": 0, "upserted": 0}

    monkeypatch.setattr(main, "_fetch_and_upsert_nonleague_matches", _fake_fetch_and_upsert)

    main.run_fetch_nonleague_matches_incremental(db_manager)

    today = date.today()
    assert captured["to_d"] == today
    assert captured["from_d"] == today - timedelta(days=main._NONLEAGUE_MATCHES_FIRST_RUN_LOOKBACK_DAYS)


def test_resumes_from_the_day_after_the_last_stored_match_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once fotmob_nonleague_matches has rows, the next run must pick up
    right after the latest one -- not re-scan from scratch and not re-fetch
    a day it already has."""
    db_manager = _make_db_manager(tmp_path)
    last_date = date.today() - timedelta(days=5)
    with db_manager.connection() as conn:
        conn.execute(
            """
            CREATE TABLE fotmob_nonleague_matches (
                team TEXT, match_date TIMESTAMP, PRIMARY KEY (team, match_date)
            )
            """
        )
        conn.execute(
            "INSERT INTO fotmob_nonleague_matches VALUES ('Chelsea', ?)", [last_date.isoformat()]
        )

    captured: dict = {}

    def _fake_fetch_and_upsert(db_manager, from_d, to_d, delay=1.0):
        captured["from_d"] = from_d
        captured["to_d"] = to_d
        return {"days_scanned": (to_d - from_d).days + 1, "matches_seen": 0, "upserted": 0}

    monkeypatch.setattr(main, "_fetch_and_upsert_nonleague_matches", _fake_fetch_and_upsert)

    main.run_fetch_nonleague_matches_incremental(db_manager)

    assert captured["from_d"] == last_date + timedelta(days=1)
    assert captured["to_d"] == date.today()


def test_already_up_to_date_skips_the_fetch_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the last stored match date is today, there's nothing new to fetch
    -- must not call out at all (no pointless 0-or-1-day API round trip)."""
    db_manager = _make_db_manager(tmp_path)
    with db_manager.connection() as conn:
        conn.execute(
            """
            CREATE TABLE fotmob_nonleague_matches (
                team TEXT, match_date TIMESTAMP, PRIMARY KEY (team, match_date)
            )
            """
        )
        conn.execute(
            "INSERT INTO fotmob_nonleague_matches VALUES ('Chelsea', ?)", [date.today().isoformat()]
        )

    called = []
    monkeypatch.setattr(
        main, "_fetch_and_upsert_nonleague_matches",
        lambda *a, **kw: called.append(True),
    )

    main.run_fetch_nonleague_matches_incremental(db_manager)

    assert called == []
