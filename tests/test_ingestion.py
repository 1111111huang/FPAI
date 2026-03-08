from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.data_loader import CSVLoader


def test_process_v1_csv_skips_rows_with_missing_odds(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,Avg>2.5",
                "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5,1.35",
                "16/08/2025,Aston Villa,Newcastle,0,0,2.25,3.5,,1.61",
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
            SELECT league, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a
            FROM raw_matches
            """
        ).fetchone()

    assert count == 1
    assert row is not None
    assert row[:5] == ("E0", "Liverpool", "Bournemouth", 4, 2)
    assert row[5] == pytest.approx(1.3)
    assert row[6] == pytest.approx(6.0)
    assert row[7] == pytest.approx(8.5)
