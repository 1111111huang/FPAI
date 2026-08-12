"""BUG-042: run_fetch_fotmob's default season range (when from_season isn't
given explicitly) used to span a league's *entire* ingested history --
found live when a real refresh-data run against a decade-plus of real SP1
history turned into one real HTTP request per calendar day across ~15
seasons, realistically hours per refresh, for enrichment data that isn't
even required for a forecast to work. Defaults now to the last
FOTMOB_DEFAULT_SEASON_LOOKBACK seasons instead."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main
from src.utils.config_loader import settings as app_settings
from src.utils.db_manager import DuckDBManager


def _manager_with_raw_matches_spanning(tmp_path: Path, first_year: int, last_year: int) -> DuckDBManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE raw_matches (match_id TEXT PRIMARY KEY, date TIMESTAMP)")
    conn.execute(
        "INSERT INTO raw_matches VALUES (?, ?), (?, ?)",
        ["m1", f"{first_year}-09-01", "m2", f"{last_year}-05-01"],
    )
    conn.close()
    return DuckDBManager(config_path=str(config_path))


def _run_with_mocked_fetch(db_manager: DuckDBManager, **kwargs):
    """Runs run_fetch_fotmob with fetch_player_match_stats/upsert_player_match_stats
    mocked out (real HTTP calls are never acceptable in a test), returns
    the list of (season_from, season_to) pairs it was actually asked to fetch."""
    seasons_fetched: list[tuple] = []

    def _fake_fetch(league, season_from, season_to, delay=1.0):
        seasons_fetched.append((season_from, season_to))
        return pd.DataFrame()

    with patch("src.ingestion.fotmob.fetcher.fetch_player_match_stats", side_effect=_fake_fetch), \
         patch("src.ingestion.fotmob.merge.upsert_player_match_stats", return_value={"matched": 0, "unmatched": 0, "players_upserted": 0, "rows_upserted": 0}):
        main.run_fetch_fotmob(app_settings, db_manager, **kwargs)

    return seasons_fetched


def test_default_lookback_is_capped_at_two_seasons_for_a_decade_of_history(tmp_path: Path) -> None:
    db_manager = _manager_with_raw_matches_spanning(tmp_path, first_year=2012, last_year=2026)

    seasons_fetched = _run_with_mocked_fetch(db_manager, league="SP1")

    # Old behavior would have iterated 2011-2025 (15 seasons); new default
    # must be capped to the last 2 (FOTMOB_DEFAULT_SEASON_LOOKBACK).
    assert len(seasons_fetched) == main.FOTMOB_DEFAULT_SEASON_LOOKBACK


def test_default_lookback_does_not_extend_earlier_than_the_real_data_when_history_is_short(tmp_path: Path) -> None:
    """A brand-new competition with only one real season of data must not
    request a season entirely before any real match exists."""
    db_manager = _manager_with_raw_matches_spanning(tmp_path, first_year=2026, last_year=2026)

    seasons_fetched = _run_with_mocked_fetch(db_manager, league="SP1")

    assert len(seasons_fetched) == 1


def test_an_explicit_from_season_is_still_respected_unbounded(tmp_path: Path) -> None:
    """A deliberate one-off deep backfill (explicit --from_season via the
    CLI) must be unaffected by the new default -- this is an opt-in, not
    something the new cap should silently override."""
    db_manager = _manager_with_raw_matches_spanning(tmp_path, first_year=2012, last_year=2026)

    seasons_fetched = _run_with_mocked_fetch(db_manager, league="SP1", from_season=2012)

    assert len(seasons_fetched) == 14  # 2012..2025 inclusive, unbounded by the new default cap
