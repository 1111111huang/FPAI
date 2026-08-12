"""Regression tests for US#132: wire Sweden (Allsvenskan) into
refresh-data/schedule-refresh.

Part 1 (live investigation, 2026-07-21) found football-data.co.uk's
new/SWE.csv was updated same-day as its most recent match, but Allsvenskan
rounds do NOT reliably fall on weekends the way EPL's do -- the 2026 season
data shows a genuine midweek round (Wed 2026-04-22 / Thu 2026-04-23, 4
matches each) sandwiched between two weekend rounds only 3-4 days apart on
either side. A Sunday-only weekly refresh (Phase 17's EPL cadence) would
leave that midweek round's results uningested for a full week -- spanning
past the *next* round's kickoff. This module tests that (a) Sweden's refresh
takes its own fetch+upsert+feature-rebuild path, skipping the EPL-only
scrape/ingest/fetch-understat/fetch-fotmob/lineup-backfill steps (Sweden has
no Understat/FotMob integration, out of scope), and (b) schedule-refresh
uses a tighter, twice-weekly default cadence for Sweden specifically.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main
from src.scheduling.data_refresh_scheduler import (
    DEFAULT_SWEDEN_DAY_OF_WEEK,
    SWEDEN_JOB_ID,
    build_sweden_refresh_scheduler,
)


class _UnusedDBManager:
    """Stand-in passed to run_refresh_data -- Sweden's path never opens a
    live connection through it directly (FeatureFactory/upsert_sweden_matches
    open their own), so any real use of `.connection()` here is a test
    failure. US#155: `.config_path` *is* legitimately read (threaded into
    FeatureFactory so it respects the caller's config instead of silently
    defaulting to config.yaml) -- not a connection, so it's provided here.
    US#159: `.default_max_retries`/`.default_retry_delay_seconds` are read
    the same way, threaded into FeatureFactory's own retry-window config."""

    config_path = "config.yaml"
    default_max_retries = 5
    default_retry_delay_seconds = 1.0

    def connection(self):
        raise AssertionError("Sweden refresh path should not open a connection via the passed-in db_manager")


def _epl_only_step(name: str):
    def _boom(*args, **kwargs):
        raise AssertionError(f"EPL-only step '{name}' must not run for league='SWE'")

    return _boom


def test_run_refresh_data_sweden_skips_epl_pipeline_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "run_scrape", _epl_only_step("run_scrape"))
    monkeypatch.setattr(main, "run_ingest", _epl_only_step("run_ingest"))
    monkeypatch.setattr(main, "run_fetch_understat", _epl_only_step("run_fetch_understat"))
    monkeypatch.setattr(main, "run_fetch_fotmob", _epl_only_step("run_fetch_fotmob"))

    fetch_calls: list[bool] = []
    upsert_calls: list[pd.DataFrame] = []
    feature_calls: list[object] = []

    fake_df = pd.DataFrame({"Date": ["01/01/2026"], "Home": ["A"], "Away": ["B"], "HG": [1], "AG": [0]})

    def _fake_fetch(*args, **kwargs):
        fetch_calls.append(True)
        return fake_df

    def _fake_upsert(df, db_manager, overwrite=False):
        upsert_calls.append(df)
        return {"rows_in": 1, "skipped": 0, "upserted": 1}

    monkeypatch.setattr("src.ingestion.football_data.sweden_fetcher.fetch_sweden_csv", _fake_fetch)
    monkeypatch.setattr("src.ingestion.football_data.sweden_loader.upsert_sweden_matches", _fake_upsert)

    class _FakeFactory:
        def __init__(
            self,
            config_path: str = "config.yaml",
            default_max_retries: int = 5,
            default_retry_delay_seconds: float = 1.0,
        ) -> None:
            pass  # US#155/US#159: real FeatureFactory now takes these; accept and ignore them here.

        def compute_rolling_stats(self, window: int = 5):
            feature_calls.append(window)
            return pd.DataFrame()

        def save_features(self, df):
            feature_calls.append("saved")

    monkeypatch.setattr(main, "FeatureFactory", _FakeFactory)

    from src.utils.config_loader import settings as app_settings

    main.run_refresh_data(app_settings, _UnusedDBManager(), league="SWE")

    assert fetch_calls == [True]
    assert len(upsert_calls) == 1
    assert feature_calls == [app_settings.settings.rolling_window, "saved"]


def test_run_refresh_data_epl_still_runs_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-Sweden leagues must be entirely unaffected by this change."""
    calls: list[str] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: calls.append("scrape"))
    monkeypatch.setattr(main, "run_ingest", lambda *a, **kw: calls.append("ingest"))
    monkeypatch.setattr(main, "run_fetch_understat", lambda *a, **kw: calls.append("understat"))
    monkeypatch.setattr(main, "run_fetch_fotmob", lambda *a, **kw: calls.append("fotmob"))
    monkeypatch.setattr(
        "src.ingestion.fotmob.lineup.backfill_lineups_from_player_stats",
        lambda db_manager: calls.append("lineups") or 0,
    )

    from src.utils.config_loader import settings as app_settings

    main.run_refresh_data(app_settings, _UnusedDBManager(), league="E0")

    assert calls == ["scrape", "ingest", "understat", "fotmob", "lineups"]


def test_build_sweden_refresh_scheduler_registers_one_job_with_sweden_job_id() -> None:
    scheduler = build_sweden_refresh_scheduler(refresh_fn=lambda: None)
    jobs = scheduler.get_jobs()

    assert len(jobs) == 1
    assert jobs[0].id == SWEDEN_JOB_ID


def test_build_sweden_refresh_scheduler_default_cadence_is_twice_weekly() -> None:
    """Justified by Part 1: Allsvenskan rounds can land on non-weekend days as
    close as 3-4 days apart, so a single weekly slot (EPL's Sunday cadence)
    isn't tight enough -- default here must cover two well-separated days."""
    scheduler = build_sweden_refresh_scheduler(refresh_fn=lambda: None)
    trigger = scheduler.get_job(SWEDEN_JOB_ID).trigger

    fields = {f.name: str(f) for f in trigger.fields}
    assert fields["day_of_week"] == DEFAULT_SWEDEN_DAY_OF_WEEK
    assert "," in DEFAULT_SWEDEN_DAY_OF_WEEK  # more than one day -- i.e. actually tighter than weekly
    assert fields["hour"] == "3"
    assert fields["minute"] == "0"


def test_build_sweden_refresh_scheduler_uses_given_schedule_params() -> None:
    scheduler = build_sweden_refresh_scheduler(refresh_fn=lambda: None, day_of_week="mon,thu", hour=4, minute=30)
    trigger = scheduler.get_job(SWEDEN_JOB_ID).trigger

    fields = {f.name: str(f) for f in trigger.fields}
    assert fields["day_of_week"] == "mon,thu"
    assert fields["hour"] == "4"
    assert fields["minute"] == "30"


def test_scheduled_sweden_job_invokes_the_injected_refresh_fn() -> None:
    calls: list[str] = []
    scheduler = build_sweden_refresh_scheduler(refresh_fn=lambda: calls.append("ran"))

    scheduler.get_job(SWEDEN_JOB_ID).func()

    assert calls == ["ran"]


def test_run_schedule_refresh_dispatches_sweden_scheduler_for_swe_league(monkeypatch: pytest.MonkeyPatch) -> None:
    """schedule-refresh --league SWE must build the (tighter) Sweden scheduler,
    not the EPL weekly one, and must use Sweden's own default cadence when the
    caller doesn't explicitly override day-of-week."""
    import src.scheduling.data_refresh_scheduler as scheduler_module

    built: dict[str, object] = {}

    class _FakeScheduler:
        def start(self):
            built["started"] = True

        def shutdown(self):
            built["shutdown"] = True

    def _fake_build_sweden(refresh_fn=None, day_of_week="tue,fri", hour=3, minute=0):
        built["kind"] = "sweden"
        built["day_of_week"] = day_of_week
        built["hour"] = hour
        built["minute"] = minute
        return _FakeScheduler()

    def _fake_build_weekly(refresh_fn=None, day_of_week="sun", hour=3, minute=0, league="E0"):
        built["kind"] = "weekly"
        return _FakeScheduler()

    monkeypatch.setattr(scheduler_module, "build_sweden_refresh_scheduler", _fake_build_sweden)
    monkeypatch.setattr(scheduler_module, "build_weekly_refresh_scheduler", _fake_build_weekly)
    monkeypatch.setattr("time.sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    main.run_schedule_refresh(league="SWE", day_of_week=None)

    assert built["kind"] == "sweden"
    assert built["day_of_week"] == DEFAULT_SWEDEN_DAY_OF_WEEK
