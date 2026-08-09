"""US#155: does a real weekly `refresh-data` recover correctly after a
failure partway through its chain (`scrape -> ingest -> fetch-understat ->
fetch-fotmob -> lineup-backfill`)?

`run_refresh_data` (main.py) runs that chain with no try/except grouping
across steps and no explicit checkpoint/resume state -- a raised exception
at any step aborts everything after it in that call. This file answers, by
direct investigation of the real code (not assumption) and a real test
against it:

1. Does a step's already-committed work survive a later step's failure?
   (DuckDBManager.connection() opens/closes its own connection per call,
   auto-committing on close -- `ingest`'s connection is already closed,
   its rows durably on disk, before `fetch-understat` even starts.)
2. Does simply re-running the whole chain (the only recovery mechanism
   that exists -- there is no partial-resume logic) actually complete the
   work a prior partial failure skipped, or does it leave a silent
   permanent gap? `update_raw_matches_xg` (understat/merge.py) answers
   this by construction: it has no "already done" tracking at all -- it
   unconditionally re-matches and UPDATEs every row in raw_matches against
   the freshly re-fetched season range on every single call, so a retry
   naturally re-covers whatever an earlier partial failure left
   unenriched, at the cost of redoing already-correct work too (a real,
   accepted inefficiency, not a correctness gap).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

import main
from src.ingestion.football_data.loader import CSVLoader
from src.utils.db_manager import DuckDBManager


def _seed_ingested_rows(tmp_path: Path) -> tuple[Path, Path]:
    """Real CSV -> real ingest (US#154's own fixture shape), so the rows
    this test cares about are genuinely committed via the real pipeline,
    not fabricated directly in the DB."""
    raw_root = tmp_path / "raw"
    (raw_root / "football_data").mkdir(parents=True)
    csv_path = raw_root / "football_data" / "E0_2526.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A",
                "15/08/2025,Liverpool,Bournemouth,4,2,1.3,6.0,8.5",
                "16/08/2025,Aston Villa,Newcastle,0,0,2.25,3.5,3.2",
            ]
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "raw_data_dir": str(raw_root)}}),
        encoding="utf-8",
    )

    loader = CSVLoader(config_path=str(config_path))
    added = loader.process_directory(pattern="E0_*.csv")
    assert added == 2  # sanity: the seed itself must have worked
    return db_path, config_path


def _xg_values(db_path: Path) -> list[tuple]:
    with duckdb.connect(str(db_path)) as conn:
        return conn.execute(
            "SELECT home_team, away_team, xg_h, xg_a FROM raw_matches ORDER BY home_team"
        ).fetchall()


def test_ingested_rows_survive_a_later_failure_in_the_understat_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, config_path = _seed_ingested_rows(tmp_path)
    db_manager = DuckDBManager(config_path=str(config_path))

    from src.utils.config_loader import settings as app_settings

    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: None)

    def _raise(*a, **kw):
        raise RuntimeError("Understat unreachable (simulated -- BUG-018's own real failure class)")

    monkeypatch.setattr("src.ingestion.understat.fetcher.fetch_seasons_range", _raise)

    # fetch-fotmob/lineup-backfill must never run this call -- the
    # exception above aborts run_refresh_data before it reaches them.
    fotmob_calls: list[object] = []
    monkeypatch.setattr(main, "run_fetch_fotmob", lambda *a, **kw: fotmob_calls.append(1))
    monkeypatch.setattr(
        "src.ingestion.fotmob.lineup.backfill_lineups_from_player_stats",
        lambda db_manager: fotmob_calls.append(1) or 0,
    )

    with pytest.raises(RuntimeError, match="Understat unreachable"):
        main.run_refresh_data(app_settings, db_manager, league="E0")

    # run_refresh_data has no try/except of its own -- the exception must
    # propagate uncaught (matching run_refresh_job's existing log-and-
    # reraise expectation, tests/test_data_refresh_scheduler.py), not be
    # silently swallowed.
    assert fotmob_calls == []  # confirms the chain genuinely aborted, not "failed but continued"

    rows = _xg_values(db_path)
    assert len(rows) == 2  # ingest's rows were not rolled back by the later failure
    assert {r[2] for r in rows} == {None}  # xG correctly still unset -- understat never ran


def test_a_retry_after_the_understat_failure_backfills_the_rows_it_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuation scenario: the *next* scheduled fire (or APScheduler's
    own retry-on-next-trigger, since run_refresh_job re-raises rather than
    swallowing) calls run_refresh_data again. No special "resume" logic
    exists in the source -- this proves the plain full-chain re-run alone
    is enough, because update_raw_matches_xg's unconditional per-call
    UPDATE naturally re-covers rows a prior partial failure left
    unenriched."""
    db_path, config_path = _seed_ingested_rows(tmp_path)
    db_manager = DuckDBManager(config_path=str(config_path))
    from src.utils.config_loader import settings as app_settings

    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_fetch_fotmob", lambda *a, **kw: None)
    monkeypatch.setattr(
        "src.ingestion.fotmob.lineup.backfill_lineups_from_player_stats", lambda db_manager: 0
    )

    # First call: understat genuinely fails, as in the durability test above.
    monkeypatch.setattr(
        "src.ingestion.understat.fetcher.fetch_seasons_range",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Understat unreachable")),
    )
    with pytest.raises(RuntimeError):
        main.run_refresh_data(app_settings, db_manager, league="E0")
    assert {r[2] for r in _xg_values(db_path)} == {None}  # confirmed still unset before the retry

    # Second call (the retry): understat succeeds this time, returning real-
    # shaped data for the same two matches already sitting in raw_matches.
    real_understat_df = pd.DataFrame(
        {
            "date": ["2025-08-15", "2025-08-16"],
            "home_team": ["Liverpool", "Aston Villa"],
            "away_team": ["Bournemouth", "Newcastle"],
            "xg_h": [2.1, 0.8],
            "xg_a": [1.4, 0.9],
        }
    )
    monkeypatch.setattr(
        "src.ingestion.understat.fetcher.fetch_seasons_range",
        lambda *a, **kw: real_understat_df,
    )

    main.run_refresh_data(app_settings, db_manager, league="E0")

    rows = {(r[0], r[1]): (r[2], r[3]) for r in _xg_values(db_path)}
    assert rows[("Liverpool", "Bournemouth")] == pytest.approx((2.1, 1.4))
    assert rows[("Aston Villa", "Newcastle")] == pytest.approx((0.8, 0.9))
