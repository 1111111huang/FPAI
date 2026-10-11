"""W252: found live (2026-10-10) -- compute_rolling_stats() has no league
scoping at all, so register_data_refresh_job's own per-league subprocess
loop was rebuilding the ENTIRE feature store (every league, every season)
11 times a night across 6 leagues (twice per big-five league -- once via
run_ingest, once more via fetch-understat's own redundant rebuild -- plus
once for Sweden), apparently heavy enough that every single one of those
subprocesses was getting OOM-killed in production (confirmed live: every
league exited with code -9, every night).

These tests cover the fix at the main.py layer: skip_feature_rebuild=True
must still do every genuinely per-league thing (CSV loading, scrape,
understat/fotmob fetch, Sweden's own fetch+upsert) but must never
instantiate FeatureFactory or call compute_rolling_stats/save_features.
The default (skip_feature_rebuild=False) must be completely unchanged --
covered by the pre-existing test_refresh_data_sweden.py/
test_refresh_data_la_liga.py/test_refresh_data_three_leagues.py suites,
none of which pass this new parameter at all.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main


class _FakeConnection:
    """Just enough of DuckDBManager's connection() contract for run_ingest's
    own final raw_count/feature_count query -- a real temp DuckDB file is
    more setup than this fix needs; both branches only ever call
    .execute(sql).fetchone()."""

    def execute(self, *args, **kwargs):
        return self

    def fetchone(self):
        return (0,)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeDBManager:
    config_path = "config.yaml"
    default_max_retries = 5
    default_retry_delay_seconds = 1.0

    def connection(self):
        return _FakeConnection()


class _ExplodingFeatureFactory:
    """Stand-in for FeatureFactory that fails the test if it's ever even
    constructed -- compute_rolling_stats/save_features must not run at all
    when skip_feature_rebuild=True, not just "run with no effect"."""

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("FeatureFactory must not be constructed when skip_feature_rebuild=True")


@pytest.fixture
def app_settings():
    from src.utils.config_loader import settings as app_settings

    return app_settings


def test_run_ingest_skips_feature_rebuild_but_still_loads_csvs(
    monkeypatch: pytest.MonkeyPatch, app_settings, tmp_path: Path,
) -> None:
    (tmp_path / "placeholder.csv").write_text("Date,Home,Away\n")
    monkeypatch.setattr(app_settings.paths, "raw_data_dir", str(tmp_path))

    loader_calls: list[tuple] = []

    class _FakeLoader:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def process_directory(self, pattern: str, force: bool) -> None:
            loader_calls.append((pattern, force))

    monkeypatch.setattr(main, "CSVLoader", _FakeLoader)
    monkeypatch.setattr(main, "FeatureFactory", _ExplodingFeatureFactory)

    main.run_ingest(app_settings, _FakeDBManager(), skip_feature_rebuild=True)

    assert loader_calls == [("*.csv", False)]  # CSV loading still ran


def test_run_ingest_default_still_rebuilds_features(
    monkeypatch: pytest.MonkeyPatch, app_settings, tmp_path: Path,
) -> None:
    """Regression guard: the default (False) must be unchanged."""
    (tmp_path / "placeholder.csv").write_text("Date,Home,Away\n")
    monkeypatch.setattr(app_settings.paths, "raw_data_dir", str(tmp_path))
    monkeypatch.setattr(main, "CSVLoader", lambda *a, **kw: type("L", (), {"process_directory": lambda self, **kw: None})())

    rebuild_calls: list[str] = []

    class _FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compute_rolling_stats(self, window: int = 5):
            rebuild_calls.append("computed")
            return pd.DataFrame()

        def save_features(self, df) -> None:
            rebuild_calls.append("saved")

    monkeypatch.setattr(main, "FeatureFactory", _FakeFactory)

    main.run_ingest(app_settings, _FakeDBManager())  # skip_feature_rebuild defaults to False

    assert rebuild_calls == ["computed", "saved"]


def test_run_refresh_data_skip_feature_rebuild_propagates_to_ingest_and_understat(
    monkeypatch: pytest.MonkeyPatch, app_settings,
) -> None:
    """A non-Sweden league with skip_feature_rebuild=True must still scrape/
    fetch everything, but run_ingest gets skip_feature_rebuild=True and
    fetch-understat's own SEPARATE rebuild (rebuild_features=True by
    default) must also be suppressed -- both were independently rebuilding
    the whole feature store before this fix."""
    calls: list[tuple] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: calls.append(("scrape",)))
    monkeypatch.setattr(
        main, "run_ingest", lambda *a, skip_feature_rebuild=False, **kw: calls.append(("ingest", skip_feature_rebuild)),
    )
    monkeypatch.setattr(
        main, "run_fetch_understat",
        lambda *a, rebuild_features=True, **kw: calls.append(("understat", rebuild_features)),
    )
    monkeypatch.setattr(main, "run_fetch_fotmob", lambda *a, **kw: calls.append(("fotmob",)))
    monkeypatch.setattr(
        "src.ingestion.fotmob.lineup.backfill_lineups_from_player_stats",
        lambda db_manager, league=None: calls.append(("lineups",)) or 0,
    )
    monkeypatch.setattr(main, "run_fetch_nonleague_matches_incremental", lambda *a, **kw: calls.append(("nonleague",)))

    main.run_refresh_data(app_settings, _FakeDBManager(), league="E0", skip_feature_rebuild=True)

    assert ("ingest", True) in calls
    assert ("understat", False) in calls


def test_run_refresh_sweden_data_skip_feature_rebuild_still_fetches_and_upserts(
    monkeypatch: pytest.MonkeyPatch, app_settings,
) -> None:
    fetch_calls: list[bool] = []
    upsert_calls: list[pd.DataFrame] = []
    fake_df = pd.DataFrame({"Date": ["01/01/2026"], "Home": ["A"], "Away": ["B"], "HG": [1], "AG": [0]})

    monkeypatch.setattr(
        "src.ingestion.football_data.sweden_fetcher.fetch_sweden_csv",
        lambda *a, **kw: fetch_calls.append(True) or fake_df,
    )
    monkeypatch.setattr(
        "src.ingestion.football_data.sweden_loader.upsert_sweden_matches",
        lambda df, db_manager, overwrite=False: upsert_calls.append(df) or {"rows_in": 1, "skipped": 0, "upserted": 1},
    )
    monkeypatch.setattr(main, "FeatureFactory", _ExplodingFeatureFactory)

    main.run_refresh_sweden_data(app_settings, _FakeDBManager(), skip_feature_rebuild=True)

    assert fetch_calls == [True]
    assert len(upsert_calls) == 1


def test_refresh_data_cli_flag_parses_skip_feature_rebuild() -> None:
    """The --skip-feature-rebuild flag (register_data_refresh_job's own
    subprocess command passes this per league) must actually parse to
    True -- a narrow argparse-only check, not a full main() invocation
    (which constructs a real DuckDBManager before dispatch)."""
    parser = main._build_parser()

    args = parser.parse_args(["refresh-data", "--league", "E0", "--skip-feature-rebuild"])

    assert args.skip_feature_rebuild is True
    assert args.league == "E0"


def test_refresh_data_cli_flag_defaults_to_false() -> None:
    parser = main._build_parser()

    args = parser.parse_args(["refresh-data", "--league", "E0"])

    assert args.skip_feature_rebuild is False
