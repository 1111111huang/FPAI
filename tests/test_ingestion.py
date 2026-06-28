from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.football_data.loader import CSVLoader


def test_process_v1_csv_skips_rows_with_missing_odds(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,Avg>2.5,Avg<2.5,AHh,AvgAHH,AvgAHA",
                "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5,1.35,3.20,-1.0,1.85,1.97",
                "16/08/2025,Aston Villa,Newcastle,0,0,2.25,3.5,,1.61,2.15,0.0,1.92,1.90",
            ]
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    loader = CSVLoader(config_path=str(config_path))
    loader.process_v1_csv(file_path=str(csv_path), league_code="E0")

    with duckdb.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
        row = conn.execute(
            """
            SELECT league, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a,
                   over25_odds, under25_odds, ah_line, ah_home_odds, ah_away_odds
            FROM raw_matches
            """
        ).fetchone()

    assert count == 1
    assert row is not None
    assert row[:5] == ("E0", "Liverpool", "Bournemouth", 4, 2)
    assert row[5] == pytest.approx(1.3)
    assert row[6] == pytest.approx(6.0)
    assert row[7] == pytest.approx(8.5)
    assert row[8] == pytest.approx(1.35)   # over25_odds
    assert row[9] == pytest.approx(3.20)   # under25_odds
    assert row[10] == pytest.approx(-1.0)  # ah_line
    assert row[11] == pytest.approx(1.85)  # ah_home_odds
    assert row[12] == pytest.approx(1.97)  # ah_away_odds


def test_process_v1_csv_tolerates_missing_over25_columns(tmp_path: Path) -> None:
    """Older CSVs without over/under or AH columns should still ingest cleanly."""
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A",
                "15/08/2019,Arsenal,Chelsea,2,1,2.10,3.40,3.50",
            ]
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "test_legacy.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    loader = CSVLoader(config_path=str(config_path))
    loader.process_v1_csv(file_path=str(csv_path), league_code="E0")

    with duckdb.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]
        row = conn.execute(
            "SELECT over25_odds, under25_odds, ah_line FROM raw_matches"
        ).fetchone()

    assert count == 1
    assert row[0] is None  # over25_odds absent → NULL
    assert row[1] is None  # under25_odds absent → NULL
    assert row[2] is None  # ah_line absent → NULL
