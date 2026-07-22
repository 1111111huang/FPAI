"""Tests for normalizing + upserting Sweden CSV rows into raw_matches (US#125).

`fetch_sweden_csv()` (US#124) is fetch-and-parse only -- it returns the
Sweden "New Leagues" CSV's own 25 columns, unrenamed. `upsert_sweden_matches`
(`src/ingestion/football_data/sweden_loader.py`) is the mapping step this
story adds: HG/AG -> fthg/ftag, AvgCH/AvgCD/AvgCA -> avgh/avgd/avga (with a
row-level B365CH/CD/CA fallback when AvgC* is blank, the same pattern
BUG-009/US#56 established for pre-2020 EPL odds), league='SWE' tagging, and
NULL for every column this source doesn't provide (shots/corners/cards/xG/
O-U-2.5/AH odds).

DataFrames are built in-memory to match `fetch_sweden_csv`'s output shape
(same column names as `EXPECTED_COLUMNS`) rather than depending on real
network access, plus one end-to-end test that runs the real fixture CSV
through both `fetch_sweden_csv` (HTTP mocked) and `upsert_sweden_matches`.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.football_data.sweden_fetcher import fetch_sweden_csv
from src.ingestion.football_data.sweden_loader import upsert_sweden_matches
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import generate_match_id

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "football_data_sweden_sample.csv"

NULL_COLUMNS = [
    "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
    "xg_h", "xg_a", "over25_odds", "under25_odds",
    "ah_line", "ah_home_odds", "ah_away_odds",
]


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _row(**overrides: object) -> dict:
    base = {
        "Country": "Sweden",
        "League": "Allsvenskan",
        "Season": 2026,
        "Date": "12/07/2026",
        "Time": "13:00",
        "Home": "Malmo FF",
        "Away": "Goteborg",
        "HG": 4,
        "AG": 0,
        "Res": "H",
        "PSCH": None,
        "PSCD": None,
        "PSCA": None,
        "MaxCH": 1.93,
        "MaxCD": 4.0,
        "MaxCA": 3.8,
        "AvgCH": 1.87,
        "AvgCD": 3.69,
        "AvgCA": 3.42,
        "BFECH": 1.99,
        "BFECD": 4.1,
        "BFECA": 3.8,
        "B365CH": 1.91,
        "B365CD": 4.0,
        "B365CA": 3.3,
    }
    base.update(overrides)
    return base


def _fetch_raw_matches_row(db_manager: DuckDBManager, match_id: str) -> tuple:
    with db_manager.connection() as conn:
        return conn.execute(
            """
            SELECT league, home_team, away_team, fthg, ftag, avgh, avgd, avga,
                   hs, "as", hst, ast, hc, ac, hy, ay, hr, ar,
                   xg_h, xg_a, over25_odds, under25_odds,
                   ah_line, ah_home_odds, ah_away_odds
            FROM raw_matches WHERE match_id = ?
            """,
            [match_id],
        ).fetchone()


def test_upsert_maps_columns_and_tags_league_swe(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame([_row()])

    result = upsert_sweden_matches(df, db_manager)

    assert result == {"rows_in": 1, "skipped": 0, "upserted": 1}

    match_id = generate_match_id(date="2026-07-12", home_team="Malmo FF", away_team="Goteborg", league="SWE")
    row = _fetch_raw_matches_row(db_manager, match_id)
    assert row is not None
    league, home_team, away_team, fthg, ftag, avgh, avgd, avga = row[:8]
    assert league == "SWE"
    assert home_team == "Malmo FF"
    assert away_team == "Goteborg"
    assert fthg == 4
    assert ftag == 0
    assert avgh == pytest.approx(1.87)
    assert avgd == pytest.approx(3.69)
    assert avga == pytest.approx(3.42)

    null_values = row[8:]
    assert all(value is None for value in null_values), (
        "hs/as/hst/ast/hc/ac/hy/ay/hr/ar/xg_h/xg_a/over25_odds/under25_odds/ah_* "
        "must be NULL for Sweden rows, not erroring."
    )


def test_upsert_falls_back_to_b365c_when_avgc_blank(tmp_path: Path) -> None:
    """Row-level fallback: AvgCH/CD/CA blank -> use B365CH/CD/CA (BUG-009/US#56 pattern)."""
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame([_row(AvgCH=None, AvgCD=None, AvgCA=None)])

    upsert_sweden_matches(df, db_manager)

    match_id = generate_match_id(date="2026-07-12", home_team="Malmo FF", away_team="Goteborg", league="SWE")
    row = _fetch_raw_matches_row(db_manager, match_id)
    assert row is not None
    avgh, avgd, avga = row[5], row[6], row[7]
    assert avgh == pytest.approx(1.91)  # B365CH fallback
    assert avgd == pytest.approx(4.0)   # B365CD fallback
    assert avga == pytest.approx(3.3)   # B365CA fallback


def test_upsert_partial_avgc_blank_falls_back_per_field(tmp_path: Path) -> None:
    """The fallback is per-field, not all-or-nothing across the row."""
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame([_row(AvgCD=None)])

    upsert_sweden_matches(df, db_manager)

    match_id = generate_match_id(date="2026-07-12", home_team="Malmo FF", away_team="Goteborg", league="SWE")
    row = _fetch_raw_matches_row(db_manager, match_id)
    avgh, avgd, avga = row[5], row[6], row[7]
    assert avgh == pytest.approx(1.87)  # AvgCH present, used as-is
    assert avgd == pytest.approx(4.0)   # AvgCD blank -> B365CD fallback
    assert avga == pytest.approx(3.42)  # AvgCA present, used as-is


def test_upsert_uses_league_aware_match_id(tmp_path: Path) -> None:
    """match_id must be generated via generate_match_id(date, home, away, league='SWE'),
    the US#140 league-aware version -- not the pre-US#140 3-argument form."""
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame([_row(Date="14/04/2012", Home="AIK", Away="Malmo FF", HG=1, AG=1)])

    upsert_sweden_matches(df, db_manager)

    expected_swe_id = generate_match_id(date="2012-04-14", home_team="AIK", away_team="Malmo FF", league="SWE")
    expected_e0_id = generate_match_id(date="2012-04-14", home_team="AIK", away_team="Malmo FF", league="E0")
    assert expected_swe_id != expected_e0_id  # sanity: league genuinely changes the hash

    with db_manager.connection() as conn:
        stored_ids = [r[0] for r in conn.execute("SELECT match_id FROM raw_matches").fetchall()]
    assert stored_ids == [expected_swe_id]


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    """Calling upsert_sweden_matches twice with identical data must not
    duplicate rows or raise -- required since refresh-data calls this
    repeatedly over time as new Sweden rounds are played."""
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame([_row(), _row(Date="14/04/2012", Home="AIK", Away="Malmo FF", HG=1, AG=1)])

    first = upsert_sweden_matches(df, db_manager)
    second = upsert_sweden_matches(df, db_manager)

    assert first == {"rows_in": 2, "skipped": 0, "upserted": 2}
    assert second == {"rows_in": 2, "skipped": 0, "upserted": 0}

    with db_manager.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
    assert count == 2


def test_upsert_skips_unplayed_fixtures_without_erroring(tmp_path: Path) -> None:
    """Rows with blank HG/AG (not yet played) must be skipped, not crash."""
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame(
        [
            _row(),
            _row(Date="19/07/2026", Home="AIK", Away="Hammarby", HG=None, AG=None),
        ]
    )

    result = upsert_sweden_matches(df, db_manager)

    assert result == {"rows_in": 2, "skipped": 1, "upserted": 1}
    with db_manager.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
    assert count == 1


def test_upsert_tolerates_missing_time_values(tmp_path: Path) -> None:
    """Time may be blank/NaN for historical rows; must not block ingestion."""
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame([_row(Time=None), _row(Date="14/04/2012", Home="AIK", Away="Malmo FF", HG=1, AG=1, Time="")])

    result = upsert_sweden_matches(df, db_manager)

    assert result == {"rows_in": 2, "skipped": 0, "upserted": 2}


def test_upsert_does_not_apply_team_name_mapper(tmp_path: Path) -> None:
    """Team names go in as the CSV's own spelling -- no TeamNameMapper/
    config/team_mapping.json fuzzy matching in this story (that's US#126)."""
    db_manager = _make_db_manager(tmp_path)
    df = pd.DataFrame([_row(Home="IFK Goteborg", Away="Djurgardens IF")])

    upsert_sweden_matches(df, db_manager)

    with db_manager.connection() as conn:
        row = conn.execute("SELECT home_team, away_team FROM raw_matches").fetchone()
    assert row == ("IFK Goteborg", "Djurgardens IF")


def test_upsert_empty_dataframe_is_a_noop(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    result = upsert_sweden_matches(pd.DataFrame(), db_manager)
    assert result == {"rows_in": 0, "skipped": 0, "upserted": 0}


def test_end_to_end_fetch_then_upsert_from_real_fixture(tmp_path: Path) -> None:
    """Full pipeline: the real US#124 fixture CSV through fetch_sweden_csv
    (HTTP mocked) then upsert_sweden_matches, confirming raw_matches rows
    come out with the right shape."""
    db_manager = _make_db_manager(tmp_path)
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = fixture_text
    mock_response.content = fixture_text.encode("utf-8-sig")
    mock_response.raise_for_status = MagicMock()

    with patch("src.ingestion.football_data.sweden_fetcher.requests.get", return_value=mock_response):
        fetched = fetch_sweden_csv()

    result = upsert_sweden_matches(fetched, db_manager)

    assert result["rows_in"] == 3
    assert result["skipped"] == 0
    assert result["upserted"] == 3

    with db_manager.connection() as conn:
        rows = conn.execute(
            "SELECT league, home_team, away_team, fthg, ftag FROM raw_matches ORDER BY date"
        ).fetchall()

    assert rows == [
        ("SWE", "AIK", "Malmo FF", 1, 1),
        ("SWE", "Goteborg", "Hammarby", 0, 2),
        ("SWE", "Malmo FF", "Goteborg", 4, 0),
    ]
