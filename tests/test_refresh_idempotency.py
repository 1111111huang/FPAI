"""US#154: does the weekly scheduled data refresh (`refresh-data`/
`schedule-refresh`, US#109/132/150) actually stay idempotent across repeated
runs, in the shape a real weekly refresh actually takes?

football-data.co.uk doesn't publish a new file per week during an active
season -- the current season's CSV is the *same file*, grown in place as
new matchday rows are appended. That means `CSVLoader`'s file-level
hash-skip (`_is_file_unchanged`) can never fire during the season: every
weekly refresh re-processes the entire grown file from scratch. The
property that actually prevents duplication is one layer down --
`process_v1_csv`'s row-level `INSERT OR IGNORE` (keyed on
`generate_match_id`) -- and nothing had directly verified it holds across
a realistic multi-week sequence, only assumed it from the mechanism
existing at all.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.football_data.loader import CSVLoader


def _row_count(db_path: Path) -> int:
    with duckdb.connect(str(db_path)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0])


def test_process_directory_is_idempotent_when_the_seasons_csv_file_grows(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "football_data").mkdir(parents=True)
    csv_path = raw_root / "football_data" / "E0_2526.csv"

    week1_rows = [
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A",
        "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5",
        "16/08/2025,Aston Villa,Newcastle,0,0,2.25,3.5,3.2",
    ]
    csv_path.write_text("\n".join(week1_rows), encoding="utf-8")

    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "raw_data_dir": str(raw_root)}}),
        encoding="utf-8",
    )

    loader = CSVLoader(config_path=str(config_path))

    # Week 1: first refresh -- 2 real matches added.
    added_week1 = loader.process_directory(pattern="E0_*.csv")
    assert added_week1 == 2
    assert _row_count(db_path) == 2

    # Same week, re-run with no change to the file (e.g. a scheduler
    # catch-up double-fire, W09's own precedent) -- file-hash skip must
    # fire, zero rows re-added.
    added_rerun = loader.process_directory(pattern="E0_*.csv")
    assert added_rerun == 0
    assert _row_count(db_path) == 2

    # Week 2: the upstream source appends 1 new matchday's row to the SAME
    # file -- the real weekly-growth pattern, not a new file. File-level
    # hash-skip cannot fire (the file genuinely changed).
    week2_rows = week1_rows + ["23/08/2025,Chelsea,Fulham,2,1,1.8,3.6,4.5"]
    csv_path.write_text("\n".join(week2_rows), encoding="utf-8")

    added_week2 = loader.process_directory(pattern="E0_*.csv")
    # Only the genuinely new row -- the 2 already-ingested matches must be
    # silently ignored by the row-level INSERT OR IGNORE, not duplicated,
    # even though the whole file was reprocessed from scratch.
    assert added_week2 == 1
    assert _row_count(db_path) == 3

    # Week 3: no further change -- file-hash skip fires again.
    added_week3 = loader.process_directory(pattern="E0_*.csv")
    assert added_week3 == 0
    assert _row_count(db_path) == 3
