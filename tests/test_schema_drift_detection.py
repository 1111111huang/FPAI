"""US#156: a source dropping expected columns has always been *tolerated*
by `process_v1_csv` (logged warning, nulls filled) -- correct for a column
that's genuinely always been optional, but there was no way to distinguish
that from a column that *used to be present and just vanished*, a real
signal the upstream source's format changed, worth a human looking at.

`CSVLoader._check_and_update_schema_baseline` closes that gap: a new
`schema_baselines` table, keyed by league (not file_path -- a brand-new
season file should still compare against that league's established
baseline, not start with nothing to compare against just because the file
itself is new), remembers the column set from each league's last
successful ingestion and logs a distinguishable "SCHEMA DRIFT DETECTED"
warning (not the same generic wording as an always-tolerated missing
column) whenever a previously-present column vanishes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.football_data.loader import CSVLoader


def _make_loader(tmp_path: Path) -> CSVLoader:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "raw_data_dir": str(tmp_path / "raw")}}),
        encoding="utf-8",
    )
    return CSVLoader(config_path=str(config_path))


def test_a_column_that_vanishes_after_being_present_logs_a_distinguishable_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    loader = _make_loader(tmp_path)
    csv_with_hc = tmp_path / "week1.csv"
    csv_with_hc.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,HC,AC",
                "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5,6,4",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        loader.process_v1_csv(file_path=str(csv_with_hc), league_code="E0")
    assert "SCHEMA DRIFT DETECTED" not in caplog.text  # first-ever ingestion: nothing to compare against yet

    caplog.clear()
    csv_without_hc = tmp_path / "week2.csv"
    csv_without_hc.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A",
                "16/08/2025,Aston Villa,Newcastle,0,0,2.25,3.5,3.2",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        loader.process_v1_csv(file_path=str(csv_without_hc), league_code="E0")

    assert "SCHEMA DRIFT DETECTED" in caplog.text
    assert "HC" in caplog.text
    assert "AC" in caplog.text


def test_a_column_thats_always_been_absent_for_this_league_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The always-optional case (US#154/US#156's own distinguishing goal):
    a league whose CSVs have simply never had HC/AC must not be flagged
    just because those columns are missing -- there's no prior baseline
    establishing them as ever having been present."""
    loader = _make_loader(tmp_path)
    csv_no_hc_ever = tmp_path / "week1.csv"
    csv_no_hc_ever.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A",
                "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5",
            ]
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        loader.process_v1_csv(file_path=str(csv_no_hc_ever), league_code="E0")

    caplog.clear()
    csv_still_no_hc = tmp_path / "week2.csv"
    csv_still_no_hc.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A",
                "16/08/2025,Aston Villa,Newcastle,0,0,2.25,3.5,3.2",
            ]
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        loader.process_v1_csv(file_path=str(csv_still_no_hc), league_code="E0")

    assert "SCHEMA DRIFT DETECTED" not in caplog.text


def test_baseline_is_scoped_per_league_not_shared_globally(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A column present for one league must not be compared against a
    different league's file -- each league's own baseline is independent."""
    loader = _make_loader(tmp_path)
    e0_csv = tmp_path / "e0.csv"
    e0_csv.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,HC,AC",
                "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5,6,4",
            ]
        ),
        encoding="utf-8",
    )
    loader.process_v1_csv(file_path=str(e0_csv), league_code="E0")

    caplog.clear()
    sp1_csv = tmp_path / "sp1.csv"
    sp1_csv.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A",
                "16/08/2025,Real Madrid,Sevilla,2,1,1.5,4.0,6.5",
            ]
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        loader.process_v1_csv(file_path=str(sp1_csv), league_code="SP1")

    assert "SCHEMA DRIFT DETECTED" not in caplog.text  # SP1's own first-ever ingestion, no baseline yet
