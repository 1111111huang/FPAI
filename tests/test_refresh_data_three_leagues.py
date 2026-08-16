"""Regression tests for US#169: extend refresh-data/schedule-refresh to cover
Serie A (I1), Bundesliga (D1), and Ligue 1 (F1) -- mirrors US#150's La Liga
precedent exactly, one body parametrized across all three rather than
tripling the same test per league.

Live weekday tabulation (2026-08-15, real ingested raw_matches, 10 seasons
each) found all three well clear of Sweden's own 65.1% "genuine outlier"
threshold that justified US#132's tighter twice-weekly cadence: I1 81.1%,
F1 90.0%, D1 93.3% Fri/Sat/Sun -- all *higher* than E0's own 77.7%. No new
scheduling machinery needed; all three fall through to the same generic
weekly-Sunday `build_weekly_refresh_scheduler` path E0/SP1 already use."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main


@pytest.mark.parametrize(
    "league,page_url",
    [
        ("I1", "https://www.football-data.co.uk/italym.php"),
        ("D1", "https://www.football-data.co.uk/germanym.php"),
        ("F1", "https://www.football-data.co.uk/francem.php"),
    ],
)
def test_run_refresh_data_scrapes_the_correct_leagues_own_page(
    league: str, page_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`refresh-data --league <code>` must scrape that league's own
    football-data.co.uk page, not silently fall through to config.yaml's
    E0/englandm.php default -- otherwise the current-season CSV would never
    actually get refreshed for these three (US#150's exact La Liga bug,
    already fixed generically via _SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE, US#161
    added I1/D1/F1's real pages to that same table)."""
    scrape_calls: list[dict] = []
    monkeypatch.setattr(main, "run_scrape", lambda *a, **kw: scrape_calls.append(kw))
    monkeypatch.setattr(main, "run_ingest", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_fetch_understat", lambda *a, **kw: None)
    monkeypatch.setattr(main, "run_fetch_fotmob", lambda *a, **kw: None)
    monkeypatch.setattr(
        "src.ingestion.fotmob.lineup.backfill_lineups_from_player_stats",
        lambda db_manager: 0,
    )

    from src.utils.config_loader import settings as app_settings

    main.run_refresh_data(app_settings, db_manager=None, league=league)

    assert len(scrape_calls) == 1
    assert scrape_calls[0]["league_page_url"] == page_url
    assert scrape_calls[0]["leagues"] == [league]


@pytest.mark.parametrize("league", ["I1", "D1", "F1"])
def test_schedule_refresh_new_leagues_use_the_same_weekly_sunday_default_as_e0(league: str) -> None:
    """No league-specific scheduler branch needed -- mirrors
    test_schedule_refresh_sp1_uses_the_same_weekly_sunday_default_as_e0."""
    from src.scheduling.data_refresh_scheduler import build_weekly_refresh_scheduler

    scheduler = build_weekly_refresh_scheduler(refresh_fn=lambda: None, league=league)
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    fields = {f.name: str(f) for f in jobs[0].trigger.fields}
    assert fields["day_of_week"] == "sun"
